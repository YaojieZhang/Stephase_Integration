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
import scipy.sparse
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
        do_normalize: bool = True,
        do_qc: bool = False,
    ):
        """
        Args:
            gene2id_path:         Path to the gene2id.pkl vocabulary file.
            bin_boundary_path:    Path to the bin_100.pkl bin boundary file.
            input_gene_expr_type: "bin" for discretized expression, "continuous" for raw float values.
            max_length:           Maximum sequence length (genes per cell). Truncates if exceeded.
            do_normalize:         Whether to run sc.pp.normalize_total + sc.pp.log1p.
            do_qc:                Whether to run basic QC filtering.
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
        self.do_normalize = do_normalize
        self.do_qc = do_qc

        # PAD_TOKEN_ID must match stella.vocab.PAD_TOKEN_ID = 0
        self.pad_token_id = 0

    def _preprocess_adata(self, adata: sc.AnnData) -> sc.AnnData:
        """
        Preprocess the AnnData object:
          1. Deduplicate gene names (remove .1, .2 suffixes from var_names_make_unique)
          2. Filter to genes present in the STELLA vocabulary
          3. Optionally QC and normalize

        Args:
            adata: Raw AnnData object (subset for one patient/sample).

        Returns:
            Preprocessed AnnData with genes filtered to vocabulary.
        """
        # Step 1: Clean gene names — remove `.1`, `.2` suffixes added by sc.var_names_make_unique()
        adata.var_names = adata.var_names.str.replace(r'\.\d+$', '', regex=True)
        duplicated_genes = adata.var_names.duplicated(keep="first")
        adata = adata[:, ~duplicated_genes].copy()

        # Step 2: Filter to genes present in the STELLA vocabulary
        genes_in_vocab = self.gene2id.keys()
        gene_mask = adata.var_names.isin(genes_in_vocab)
        adata = adata[:, gene_mask].copy()

        if adata.shape[1] == 0:
            logger.warning("[StellaInlineTokenizer] No genes matched the STELLA vocabulary! "
                         "Check if gene names are compatible (e.g., HUGO symbols).")
            return adata

        # Step 3: Optional QC (basic cell/gene filtering)
        if self.do_qc:
            sc.pp.filter_cells(adata, min_genes=500)
            sc.pp.filter_cells(adata, min_counts=1000)

        # Step 4: Normalize + log1p (required before binning)
        if self.do_normalize:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

        return adata

    def _bin_expression(self, adata: sc.AnnData) -> None:
        """
        Discretize expression values into bin IDs using the pre-computed bin boundaries.
        This is an IN-PLACE operation on adata.X.

        The binning logic replicates stella.tokenizer.TranscriptomeTokenizer.bin():
          1. np.digitize to assign bin IDs
          2. Clamp bin 0 → 1 and bin (nbins+1) → nbins
          3. Cast to int16 for memory efficiency

        Args:
            adata: AnnData with normalized+log1p expression in .X
        """
        # Ensure dense matrix for element-wise operations
        if issparse(adata.X):
            adata.X = adata.X.toarray()

        # Track which entries were originally nonzero
        nonzero_mask_before = adata.X != 0

        # Digitize: assign each nonzero expression value to a bin ID
        adata.X[adata.X != 0] = np.digitize(
            adata.X[adata.X != 0], self.bin_boundary, right=False
        )

        # After digitize, some values at the exact left boundary might map to bin 0.
        # We clamp those to bin 1 (the first valid bin).
        nonzero_mask_after = adata.X != 0
        became_zero = np.logical_xor(nonzero_mask_before, nonzero_mask_after)
        adata.X[became_zero] = 1

        # Values exceeding the last boundary map to nbins+1; clamp to nbins
        adata.X[adata.X == self.nbins + 1] = self.nbins

        # Cast to int16 for memory efficiency
        adata.X = adata.X.astype(np.int16)

    def tokenize_sample(
        self, sample_data: np.ndarray, gene_names: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tokenize a single patient's cell-gene matrix.

        This is the core method that converts raw expression data into
        the three tensors required by STELLAModel.forward().

        Args:
            sample_data: Dense or sparse matrix of shape [num_cells, num_original_genes].
                         These are the raw/HVG expression values from the h5ad file.
            gene_names:  Array of gene names corresponding to columns of sample_data.
                         Length = num_original_genes.

        Returns:
            input_ids_gene_symbol:      np.ndarray [num_cells, seq_len] — Gene vocab IDs
            input_ids_gene_expression:  np.ndarray [num_cells, seq_len] — Bin IDs or continuous values
            attention_mask:             np.ndarray [num_cells, seq_len] — 1=real, 0=pad

        Data Flow (per cell):
            1. Extract nonzero gene indices for this cell
            2. Map gene names → vocab IDs (input_ids_gene_symbol)
            3. Map expression values → bin IDs (input_ids_gene_expression)
            4. Truncate to max_length if necessary
            5. Pad all cells to the same seq_len within this sample
        """
        # ---- Create a temporary AnnData for preprocessing ----
        if issparse(sample_data):
            sample_data_dense = sample_data.toarray()
        else:
            sample_data_dense = np.array(sample_data, dtype=np.float32)

        # Build a minimal AnnData for preprocessing
        import pandas as pd
        adata_tmp = sc.AnnData(
            X=csr_matrix(sample_data_dense),
            var=pd.DataFrame(index=gene_names)
        )

        # ---- Preprocess: filter vocab genes, normalize, bin ----
        adata_tmp = self._preprocess_adata(adata_tmp)

        if adata_tmp.shape[1] == 0:
            # No genes matched vocabulary — return empty tensors
            num_cells = sample_data_dense.shape[0]
            logger.warning(f"[tokenize_sample] 0 genes matched vocab for sample with {num_cells} cells.")
            return (
                np.zeros((num_cells, 1), dtype=np.int64),
                np.zeros((num_cells, 1), dtype=np.int64),
                np.zeros((num_cells, 1), dtype=np.int64),
            )

        # After preprocessing, get the gene-to-id mapping for remaining genes
        remaining_gene_names = adata_tmp.var_names

        # ---- Expression binning (only for "bin" type) ----
        if self.input_gene_expr_type == "bin":
            self._bin_expression(adata_tmp)

        # Ensure dense for iteration
        if issparse(adata_tmp.X):
            X_dense = adata_tmp.X.toarray()
        else:
            X_dense = np.array(adata_tmp.X)

        num_cells = X_dense.shape[0]

        # ---- Gene symbol IDs (shared across all cells since genes are the same) ----
        # All cells in a sample share the same gene set after preprocessing
        gene_symbol_ids_full = np.array(
            [self.gene2id[g] for g in remaining_gene_names], dtype=np.int64
        )

        # ---- Per-cell tokenization ----
        all_gene_sym = []
        all_gene_expr = []
        all_lengths = []

        for cell_idx in range(num_cells):
            cell_expr = X_dense[cell_idx]  # [num_genes_in_vocab]

            # Extract nonzero positions — STELLA only processes expressed genes
            nonzero_mask = cell_expr != 0
            nonzero_indices = np.where(nonzero_mask)[0]

            if len(nonzero_indices) == 0:
                # Cell has no expressed genes in vocab — create a minimal token
                all_gene_sym.append(np.array([self.pad_token_id], dtype=np.int64))
                all_gene_expr.append(np.array([self.pad_token_id], dtype=np.int64))
                all_lengths.append(0)
                continue

            # Truncate to max_length
            if len(nonzero_indices) > self.max_length:
                nonzero_indices = np.random.choice(nonzero_indices, self.max_length, replace=False)
                nonzero_indices = np.sort(nonzero_indices)

            # Gene symbol IDs for this cell's expressed genes
            cell_gene_sym = gene_symbol_ids_full[nonzero_indices]

            # Expression values (bin IDs or continuous) for this cell's expressed genes
            cell_gene_expr = cell_expr[nonzero_indices].astype(np.int64 if self.input_gene_expr_type == "bin" else np.float32)

            all_gene_sym.append(cell_gene_sym)
            all_gene_expr.append(cell_gene_expr)
            all_lengths.append(len(nonzero_indices))

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
    ):
        """
        Args:
            data_list:    List of sparse matrices [num_cells_i, num_genes], one per patient.
            data_labels:  Array of labels.
            data_batches: Array of batch/domain IDs.
            sample_ids:   List of sample ID strings.
            gene_names:   Array of gene name strings (columns of data_list matrices).
            tokenizer:    StellaInlineTokenizer instance.
        """
        super().__init__()
        self.data_list = data_list
        self.data_labels = data_labels
        self.data_batches = data_batches
        self.sample_ids = sample_ids
        self.gene_names = gene_names
        self.tokenizer = tokenizer

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

            # ==== 新增：硬性下采样逻辑 (防 FLOPs 爆炸) ====
            num_cells = sample_data.shape[0]
            max_instances = self.tokenizer.max_instances # 需在 tokenizer 初始化时传入
            if num_cells > max_instances:
                # 随机无放回采样 1024 个细胞
                sampled_indices = np.random.choice(num_cells, max_instances, replace=False)
                # 排序以保持稀疏矩阵在内存中的连续性，加速运算
                sampled_indices = np.sort(sampled_indices)
                sample_data = sample_data[sampled_indices]
            
            gene_sym, gene_expr, attn_mask = self.tokenizer.tokenize_sample(
                sample_data, self.gene_names
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
        do_normalize=llm_cfg.get('do_normalize', True),
        do_qc=llm_cfg.get('do_qc', False),
    )
    return tokenizer
