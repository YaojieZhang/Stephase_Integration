# integrated_dataloader.py
# ==============================================================================
# STELLA x scPhase Integration — DataLoader Module
# ==============================================================================
# This module refactors the original scPhase DataLoader to produce tokenized
# inputs compatible with the STELLA (scrna-llm) foundation model.
#
# KEY CHANGE: Instead of outputting raw gene expression matrices [num_cells, num_hvgs],
# it now outputs tokenized tensors: (gene_symbol_ids, gene_expression_ids, attention_mask).
#
# Data Flow:
#   Input:  csr_matrix [num_cells, num_hvgs] from .h5ad
#   Output: (input_ids_gene_symbol  [num_cells, seq_len],
#            input_ids_gene_expression [num_cells, seq_len],
#            attention_mask [num_cells, seq_len],
#            label, batch_label, sample_id)
# ==============================================================================

import os
import sys
import pickle
import logging
import random
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix, issparse
from typing import List, Tuple, Dict, Optional, Any

logger = logging.getLogger(__name__)


# ==============================================================================
# Section 1: Inline Tokenizer — Replicated from stella.tokenizer logic
# ==============================================================================
# We replicate the core tokenization logic here to avoid import-chain issues
# and to keep this module self-contained. The logic follows
# TranscriptomeTokenizerForCellClassification from scrna-llm.

class StellaInlineTokenizer:
    """
    A lightweight, self-contained tokenizer that replicates the core logic of
    stella.tokenizer.TranscriptomeTokenizerForCellClassification.

    It converts an AnnData object's gene expression matrix into:
      - input_ids_gene_symbol:      [num_cells, seq_len] (vocabulary IDs for gene names)
      - input_ids_gene_expression:  [num_cells, seq_len] (bin IDs or continuous floats)
      - attention_mask:             [num_cells, seq_len] (1 for real tokens, 0 for padding)

    This avoids the overhead of the full TranscriptomeTokenizer which was designed
    for pre-training data path management.
    """

    def __init__(
        self,
        gene2id_path: str,
        bin_boundary_path: str,
        input_gene_expr_type: str = "bin",
        max_length: int = 4096,
    ):
        """
        Args:
            gene2id_path:         Path to the gene2id.pkl vocabulary file.
            bin_boundary_path:    Path to the bin_100.pkl bin boundary file.
            input_gene_expr_type: "bin" for discretized expression, "continuous" for raw float values.
            max_length:           Maximum sequence length (genes per cell). Truncates if exceeded.
        """
        # ---- Load gene symbol vocabulary: {gene_name: token_id} ----
        with open(gene2id_path, "rb") as f:
            self.gene2id = pickle.load(f)
        logger.info(f"[StellaInlineTokenizer] Loaded gene2id vocabulary with {len(self.gene2id)} entries.")

        # ---- Load bin boundaries for expression discretization ----
        with open(bin_boundary_path, "rb") as f:
            self.bin_boundary = pickle.load(f)
        self.nbins = len(self.bin_boundary) - 1
        logger.info(f"[StellaInlineTokenizer] Loaded bin boundaries with {self.nbins} bins.")

        self.input_gene_expr_type = input_gene_expr_type
        self.max_length = max_length

        # PAD_TOKEN_ID must match stella.vocab.PAD_TOKEN_ID = 0
        self.pad_token_id = 0

    def tokenize_sample(self, sample_data_csr, global_gene_ids):
        num_cells = sample_data_csr.shape[0]
        all_gene_sym, all_gene_expr, all_lengths = [], [], []
    
        # 极速稀疏矩阵 Binning (操作副本来防缓存污染)
        csr_data_local = sample_data_csr.data.copy()
        if self.input_gene_expr_type == "bin":
            binned_data = np.digitize(csr_data_local, self.bin_boundary, right=False)
            binned_data[binned_data == 0] = 1
            binned_data[binned_data == self.nbins + 1] = self.nbins
            csr_data_local = binned_data.astype(np.int64)
        else:   
            csr_data_local = csr_data_local.astype(np.float32)

        indptr = sample_data_csr.indptr
        indices = sample_data_csr.indices

        for i in range(num_cells):
            start, end = indptr[i], indptr[i+1]
            
            # 取出当前细胞的 基因列索引 和 对应的表达量
            col_indices = indices[start:end]
            cell_expr = csr_data_local[start:end]

            # 核心：将 h5ad 的列索引 转换为 LLM 词表 Token ID
            cell_gene_ids = global_gene_ids[col_indices]
            
            # 使用 -1 掩码过滤掉不在 大模型词表 中的基因
            valid_mask = cell_gene_ids != -1
            nonzero_gene_ids = cell_gene_ids[valid_mask]
            nonzero_expr = cell_expr[valid_mask]

            if len(nonzero_gene_ids) == 0:
                all_gene_sym.append(np.array([self.pad_token_id], dtype=np.int64))
                all_gene_expr.append(np.array([self.pad_token_id], dtype=np.int64))
                all_lengths.append(0)
                continue
            
            if len(nonzero_gene_ids) > self.max_length:
                # 随机采样并排序
                keep_idx = np.random.choice(len(nonzero_gene_ids), self.max_length, replace=False)
                keep_idx.sort()
                nonzero_gene_ids = nonzero_gene_ids[keep_idx]
                nonzero_expr = nonzero_expr[keep_idx]

            all_gene_sym.append(nonzero_gene_ids)
            all_gene_expr.append(nonzero_expr)
            all_lengths.append(len(nonzero_gene_ids))
            

        # ---- Pad all cells to the same length within this sample ----
        max_seq_len = max(max(all_lengths), 1)  # At least length 1

        batch_gene_sym = np.full((num_cells, max_seq_len), self.pad_token_id, dtype=np.int64)
        if self.input_gene_expr_type == "bin":
            batch_gene_expr = np.full((num_cells, max_seq_len), self.pad_token_id, dtype=np.int64)
        else:
            batch_gene_expr = np.zeros((num_cells, max_seq_len), dtype=np.float32)
        batch_attn_mask = np.zeros((num_cells, max_seq_len), dtype=np.int64)

        for cell_idx in range(num_cells):
            length = all_lengths[cell_idx]
            if length > 0:
                batch_gene_sym[cell_idx, :length] = all_gene_sym[cell_idx]
                batch_gene_expr[cell_idx, :length] = all_gene_expr[cell_idx]
                batch_attn_mask[cell_idx, :length] = 1

        return batch_gene_sym, batch_gene_expr, batch_attn_mask


# ==============================================================================
# Section 2: Data Loading — Reuses scPhase's h5ad loading logic
# ==============================================================================

def load_data(config: dict):
    """
    Load and preprocess data from h5ad or cached pickle file.
    
    This function mirrors the original scPhase load_data() but also returns
    the gene names needed for STELLA tokenization.

    Returns:
        DataList:   List of np.ndarray, each [num_cells_i, num_hvgs] for one patient
        DataLabel:  np.ndarray [num_samples] of labels
        DataBatch:  np.ndarray [num_samples] of batch/group IDs
        SampleIDs:  List[str] of sample identifiers
        GeneNames:  np.ndarray of gene names from the h5ad var_names
    """
    path_cfg = config['path_params']
    master_pickle_path = path_cfg.get('master_pickle_file', '')

    # ---- Try loading from cached pickle ----
    if master_pickle_path and os.path.exists(master_pickle_path):
        logger.info(f"Loading preprocessed data from master pickle: {master_pickle_path}")
        with open(master_pickle_path, 'rb') as f:
            master_data = pickle.load(f)
    else:
        logger.info("Processing from h5ad file (no cached pickle found).")
        master_data = _create_master_data(config)

    DataList = master_data['data_list']
    DataLabel = np.array(master_data['labels'])
    DataBatch = np.array(master_data['groups']).astype(int)
    SampleIDs = master_data.get('sample_ids', [])
    GeneNames = master_data.get('gene_names', np.array([]))

    if not SampleIDs:
        logger.warning("Sample IDs were not found in the loaded data.")
    if len(GeneNames) == 0:
        logger.warning("Gene names were not found in the loaded data. "
                       "Tokenization will fail without gene names.")

    return DataList, DataLabel, DataBatch, SampleIDs, GeneNames


def _create_master_data(config: dict) -> dict:
    """
    Parse the h5ad file and create the master data dictionary.
    
    Unlike the original scPhase version, we keep the data in sparse format
    to save memory (each patient may have tens of thousands of cells).
    We also extract gene_names for STELLA tokenization.
    """
    path_cfg = config['path_params']
    data_cfg = config.get('data_params', {})
    run_cfg = config.get('run_params', {})

    h5ad_path = path_cfg['data_h5ad_file']
    task_type = run_cfg.get('task_type', 'classification')
    sample_col = data_cfg.get('sample_col', 'sample_id')
    label_col = data_cfg.get('label_col', 'phenotype')
    batch_col = data_cfg.get('batch_col', 'batch')

    logger.info(f"Processing h5ad file: {h5ad_path}")
    traindata = sc.read_h5ad(h5ad_path)

    # ---- Extract gene names for tokenization ----
    gene_names = traindata.var_names.values.copy()
    logger.info(f"Extracted {len(gene_names)} gene names from h5ad var_names.")

    # ---- Check batch column existence ----
    has_batch_col = batch_col in traindata.obs.columns
    if not has_batch_col:
        logger.warning(f"Batch column '{batch_col}' not found. Assigning default batch=0.")

    # ---- Build label processor ----
    if task_type == 'classification':
        unique_labels = sorted(traindata.obs[label_col].unique())
        label_map = {label: i for i, label in enumerate(unique_labels)}
        logger.info(f"Classification label map: {label_map}")
        label_processor = lambda l: label_map.get(l, -1)
    else:
        logger.info("Regression task: labels will be cast to float.")
        label_processor = float

    # ---- Group by sample and extract per-patient data ----
    data_list = []
    sample_labels = []
    sample_groups = []
    sample_ids_list = []

    grouped = traindata.obs.groupby(sample_col)
    pbar = tqdm(grouped, desc="Processing samples", total=len(grouped))

    for sample_id, sample_obs in pbar:
        sample_ids_list.append(sample_id)
        sample_labels.append(label_processor(sample_obs[label_col].iloc[0]))
        sample_groups.append(sample_obs[batch_col].iloc[0] if has_batch_col else 0)

        # Extract the expression matrix for this sample
        sample_data = traindata[sample_obs.index].X
        if issparse(sample_data):
            data_list.append(sample_data.tocsr())
        else:
            data_list.append(csr_matrix(sample_data))

    master_data = {
        'data_list': data_list,  # List of csr_matrix, one per patient
        'labels': np.array(sample_labels),
        'groups': np.array(sample_groups),
        'sample_ids': sample_ids_list,
        'gene_names': gene_names,  # np.ndarray of gene name strings
    }

    # ---- Optionally cache to pickle ----
    master_pickle_path = path_cfg.get('master_pickle_file', '')
    if master_pickle_path:
        logger.info(f"Saving master data to: {master_pickle_path}")
        os.makedirs(os.path.dirname(master_pickle_path), exist_ok=True)
        with open(master_pickle_path, 'wb') as f:
            pickle.dump(master_data, f)

    logger.info(f"Data processing complete. {len(data_list)} samples loaded.")
    return master_data


# ==============================================================================
# Section 3: Dataset Class — Tokenizes on-the-fly
# ==============================================================================

class StellaScPhaseDataset(Dataset):
    """
    PyTorch Dataset that tokenizes each patient's cell-gene matrix on-the-fly
    using the StellaInlineTokenizer.

    Each __getitem__ call returns the tokenized representation of one patient's
    entire cell bag, ready to be consumed by STELLAModel.

    Output per sample:
        input_ids_gene_symbol:      Tensor [num_cells, seq_len]
        input_ids_gene_expression:  Tensor [num_cells, seq_len]
        attention_mask:             Tensor [num_cells, seq_len]
        label:                      scalar (int for classification, float for regression)
        batch_label:                scalar int (domain/batch identifier)
        sample_id:                  str
    """

    def __init__(
        self,
        data_list: List,
        data_labels: np.ndarray,
        data_batches: np.ndarray,
        sample_ids: List[str],
        gene_names: np.ndarray,
        tokenizer: StellaInlineTokenizer,
        is_train: bool = True
    ):
        """
        Args:
            data_list:    List of sparse matrices [num_cells_i, num_genes], one per patient.
            data_labels:  Array of labels.
            data_batches: Array of batch/domain IDs.
            sample_ids:   List of sample ID strings.
            gene_names:   Array of gene name strings (columns of data_list matrices).
            tokenizer:    StellaInlineTokenizer instance.
            max_instances: Maximum number of cells to process per patient (to prevent FLOPs explosion).
            is_train:     Whether this dataset is for training (random bag dropout) or testing (deterministic uniform sampling).
        """
        super().__init__()
        self.data_list = data_list
        self.data_labels = data_labels
        self.data_batches = data_batches
        self.sample_ids = sample_ids
        self.gene_names = gene_names
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_instances = getattr(tokenizer, 'max_instances', 10000)


        # ==== 新增：全局基因名到词汇 ID 的映射表 ====
        # 这让我们在 __getitem__ 中不需要重复处理字符串匹配
        # 将不需要的基因标为 -1，后续用以过滤
        global_gene_ids = np.full(len(self.gene_names), -1, dtype=np.int64)
        for i, g_name in enumerate(self.gene_names):
            clean_name = g_name.split('.')[0] # 清理 .1 .2 后缀
            if clean_name in self.tokenizer.gene2id:
                global_gene_ids[i] = self.tokenizer.gene2id[clean_name]
        self.global_gene_ids = global_gene_ids # 保存到实例变量中

        # Cache for tokenized results to avoid re-tokenizing the same sample
        # WARNING: This can consume significant memory for large datasets.
        # Set to {} to disable caching, or use an LRU cache for bounded memory.
        self._cache = {}

    def __len__(self) -> int:
        return len(self.data_labels)

    def __getitem__(self, index: int):
        """
        Tokenize and return one patient's bag.

        Returns:
            Tuple of (gene_sym_ids, gene_expr_ids, attn_mask, label, batch_label, sample_id)
        """
        if torch.is_tensor(index):
            index = index.tolist()

        # ---- Check cache ----
        if index in self._cache:
            gene_sym, gene_expr, attn_mask = self._cache[index]
        else:
            # ---- Tokenize from raw data ----
            sample_data = self.data_list[index]  # csr_matrix [num_cells, num_genes]

            # ==== 硬性下采样逻辑 (防 FLOPs 爆炸) ====
            num_cells = sample_data.shape[0]
            max_instances = self.max_instances
            if num_cells > max_instances:
                if getattr(self, 'is_train', True):
                    # 训练集：随机无放回采样 (Bag Dropout 增强)
                    sampled_indices = np.random.choice(num_cells, max_instances, replace=False)
                    sampled_indices = np.sort(sampled_indices) # 保持稀疏矩阵连续性
                else:
                    # 验证/测试集：确定性均匀采样，保证可复现并且覆盖全样本特征
                    sampled_indices = np.linspace(0, num_cells - 1, max_instances, dtype=int)
                
                sample_data = sample_data[sampled_indices]

            gene_sym, gene_expr, attn_mask = self.tokenizer.tokenize_sample(
                sample_data, self.global_gene_ids
            )
            # gene_sym:   np.ndarray [num_cells, seq_len] int64
            # gene_expr:  np.ndarray [num_cells, seq_len] int64 or float32
            # attn_mask:  np.ndarray [num_cells, seq_len] int64

            # Optionally cache (comment out for memory-constrained environments)
            # self._cache[index] = (gene_sym, gene_expr, attn_mask)

        # ---- Convert to tensors ----
        gene_sym_tensor = torch.from_numpy(gene_sym).long()        # [num_cells, seq_len]
        gene_expr_tensor = torch.from_numpy(gene_expr)             # [num_cells, seq_len]
        if gene_expr_tensor.dtype == torch.float32:
            pass  # Keep float for continuous mode
        else:
            gene_expr_tensor = gene_expr_tensor.long()             # int64 for bin mode
        attn_mask_tensor = torch.from_numpy(attn_mask).long()      # [num_cells, seq_len]

        label = self.data_labels[index]
        batch_label = self.data_batches[index]
        sample_id = self.sample_ids[index]

        return gene_sym_tensor, gene_expr_tensor, attn_mask_tensor, label, batch_label, sample_id


# ==============================================================================
# Section 4: Collate Function — Handles variable-size patient bags
# ==============================================================================

def stella_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, Any, str]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Custom collate function for the StellaScPhaseDataset.

    Since batch_size is always 1 in MIL (one patient = one bag), this function
    primarily unpacks the single-element batch. However, it also handles the
    edge case of batch_size > 1 by padding across patients.

    For batch_size=1:
        Input:  List of length 1, each element is (gene_sym, gene_expr, mask, label, batch, sid)
        Output: (gene_sym [num_cells, seq_len],
                 gene_expr [num_cells, seq_len],
                 mask [num_cells, seq_len],
                 labels [1],
                 batches [1],
                 sample_ids [list of 1 str])

    For batch_size > 1 (rare, but supported):
        Pads seq_len across patients to the maximum, then concatenates along cell dimension.
        NOTE: This is NOT recommended for MIL training, but provided for completeness.
    """
    gene_sym_list, gene_expr_list, mask_list = [], [], []
    label_list, batch_list, sample_ids_list = [], [], []

    for gene_sym, gene_expr, attn_mask, label, batch_label, sample_id in batch:
        gene_sym_list.append(gene_sym)
        gene_expr_list.append(gene_expr)
        mask_list.append(attn_mask)
        label_list.append(label)
        batch_list.append(batch_label)
        sample_ids_list.append(sample_id)

    if len(batch) == 1:
        # ---- Fast path: batch_size=1 (standard MIL) ----
        batched_gene_sym = gene_sym_list[0]             # [num_cells, seq_len]
        batched_gene_expr = gene_expr_list[0]           # [num_cells, seq_len]
        batched_mask = mask_list[0]                     # [num_cells, seq_len]
    else:
        # ---- Slow path: batch_size > 1, need to pad seq_len ----
        # Find the maximum seq_len across all patients in this batch
        max_seq_len = max(t.size(1) for t in gene_sym_list)

        padded_gene_sym, padded_gene_expr, padded_mask = [], [], []
        pad_token_id = 0  # PAD_TOKEN_ID

        for i in range(len(batch)):
            seq_len_i = gene_sym_list[i].size(1)
            pad_len = max_seq_len - seq_len_i

            if pad_len > 0:
                num_cells_i = gene_sym_list[i].size(0)
                # Pad gene_sym with PAD_TOKEN_ID
                padded_gene_sym.append(torch.cat([
                    gene_sym_list[i],
                    torch.full((num_cells_i, pad_len), pad_token_id, dtype=torch.long)
                ], dim=1))
                # Pad gene_expr with 0
                padded_gene_expr.append(torch.cat([
                    gene_expr_list[i],
                    torch.zeros((num_cells_i, pad_len), dtype=gene_expr_list[i].dtype)
                ], dim=1))
                # Pad mask with 0
                padded_mask.append(torch.cat([
                    mask_list[i],
                    torch.zeros((num_cells_i, pad_len), dtype=torch.long)
                ], dim=1))
            else:
                padded_gene_sym.append(gene_sym_list[i])
                padded_gene_expr.append(gene_expr_list[i])
                padded_mask.append(mask_list[i])

        # Concatenate all patients' cells along dim=0
        batched_gene_sym = torch.cat(padded_gene_sym, dim=0)
        batched_gene_expr = torch.cat(padded_gene_expr, dim=0)
        batched_mask = torch.cat(padded_mask, dim=0)

    batched_labels = torch.as_tensor(label_list)
    batched_batches = torch.as_tensor(batch_list)

    return (
        batched_gene_sym,       # [total_cells, seq_len]
        batched_gene_expr,      # [total_cells, seq_len]
        batched_mask,           # [total_cells, seq_len]
        batched_labels,         # [batch_size]  (typically [1])
        batched_batches,        # [batch_size]  (typically [1])
        sample_ids_list         # List[str]     (typically len 1)
    )


# ==============================================================================
# Section 5: Factory Function — Convenient dataset creation from config
# ==============================================================================

def create_tokenizer_from_config(config: dict) -> StellaInlineTokenizer:
    """
    Create a StellaInlineTokenizer from the centralized config.

    Args:
        config: The full integration config dict.

    Returns:
        Configured StellaInlineTokenizer instance.
    """
    llm_cfg = config['llm_params']

    tokenizer = StellaInlineTokenizer(
        gene2id_path=llm_cfg['gene2id_path'],
        bin_boundary_path=llm_cfg['bin_boundary_path'],
        input_gene_expr_type=llm_cfg['input_gene_expr_type'],
        max_length=llm_cfg.get('llm_max_seq_len', 4096),
    )
    tokenizer.max_instances = config["mil_params"].get("max_instances", 10000)

    return tokenizer

