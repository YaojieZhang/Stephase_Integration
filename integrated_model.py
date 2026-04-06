# integrated_model.py
# ==============================================================================
# STELLA x scPhase Integration — Core Model Module
# ==============================================================================
# This module defines the SCMIL_STELLA_AttnMoE class which replaces the original
# scPhase SCMIL_AttnMoE. Instead of using a simple MLP instance encoder,
# it uses the pre-trained STELLA (scrna-llm) Transformer as a frozen feature
# extractor, projecting cell-level embeddings into the MIL hidden space.
#
# Architecture Flow:
#   Input:  (input_ids_gene_symbol, input_ids_gene_expression, attention_mask)
#           each [num_cells, seq_len]
#
#   Stage 1 — STELLA Encoder (frozen or fine-tuned):
#       [num_cells, seq_len] → STELLAModel → [num_cells, seq_len, stella_hidden_size]
#       Chunked processing to prevent OOM on large cell bags.
#
#   Stage 2 — Masked Mean Pooling + Projection:
#       [num_cells, seq_len, stella_hidden_size] → pool → [num_cells, stella_hidden_size]
#       → projector → [num_cells, mil_hidden_dim]
#
#   Stage 3 — Linformer Attention (from scPhase):
#       [num_cells, mil_hidden_dim] → attended → [num_cells, mil_hidden_dim]
#
#   Stage 4 — MoE MIL Aggregation + Classifier:
#       [num_cells, mil_hidden_dim] → bag_features → disease_output [n_classes]
#
#   Optional — Domain Adaptation:
#       bag_features → GRL → domain_output [num_domains]
# ==============================================================================

import sys
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


# ==============================================================================
# Section 0: Import scPhase MIL components
# ==============================================================================
# We import directly from the scPhase modules to preserve the original logic.
# The sys.path manipulation is handled externally or via config['llm_params']['stella_src_path'].

from scPhase.scphase.modules import (
    InstanceDropout,
    MoEMILAggregation,
    AdaptiveMILAggregation,
    GradientReversalLayer,
    DomainClassifier,
    initialize_weights,
)


# ==============================================================================
# Section 1: STELLA imports — Deferred to allow dynamic sys.path injection
# ==============================================================================

def _import_stella(stella_src_path: str):
    """
    Dynamically import STELLAModel and STELLAConfig by injecting stella_src_path
    into sys.path. This avoids hard-coding the import path and allows the
    integration to work across different deployment environments.

    Args:
        stella_src_path: Absolute path to scRNA-LLM/src, e.g. "/data/wuqinhua/scRNA-LLM/src"

    Returns:
        (STELLAModel, STELLAConfig) class references.
    """
    if stella_src_path not in sys.path:
        sys.path.insert(0, stella_src_path)
        logger.info(f"[_import_stella] Injected '{stella_src_path}' into sys.path.")

    from stella.models.modeling_stella import STELLAModel
    from stella.models.configuration_stella import STELLAConfig

    return STELLAModel, STELLAConfig


# ==============================================================================
# Section 2: Core Integrated Model
# ==============================================================================

class SCMIL_STELLA_AttnMoE(nn.Module):
    """
    STELLA-augmented Multi-Instance Learning model for patient-level phenotype
    prediction from single-cell RNA-seq data.

    This model replaces the simple MLP instance encoder in the original scPhase
    SCMIL_AttnMoE with the pre-trained STELLA Transformer, using it as a
    cell-level feature extractor.

    Args:
        config:      Full integration config dict (loaded from config_integration.json).
        num_domains: Number of unique batch/domain categories for domain adaptation.
        device:      Target device for the model (e.g., "cuda:0").
    """

    def __init__(self, config: Dict[str, Any], num_domains: int, device: str = "cuda:0"):
        super(SCMIL_STELLA_AttnMoE, self).__init__()

        # ---- Unpack configuration sections ----
        llm_cfg = config['llm_params']
        mil_cfg = config['mil_params']

        self.device = device
        self.stella_hidden_size = llm_cfg['llm_hidden_size']       # 512
        self.mil_hidden_dim = mil_cfg['mil_hidden_dim']             # 256
        self.max_instances = mil_cfg['max_instances']               # 1024
        self.chunk_size = llm_cfg.get('chunk_size', 256)            # VRAM defense chunk size
        self.freeze_llm = llm_cfg.get('freeze_llm', True)
        self.enable_gradient_checkpointing = llm_cfg.get('enable_gradient_checkpointing', True)
        self.attention_type = mil_cfg.get('attention_type', 'linformer')
        self.use_moe = mil_cfg.get('use_moe', True)
        self.use_domain_adaptation = mil_cfg.get('use_domain_adaptation', True)

        # ==================================================================
        # Stage 1: STELLA Encoder — Pre-trained Transformer Feature Extractor
        # ==================================================================
        STELLAModel, STELLAConfig = _import_stella(llm_cfg['stella_src_path'])

        logger.info(f"[SCMIL_STELLA_AttnMoE] Loading STELLA from: {llm_cfg['pretrained_model_path']}")
        self.stella_encoder = STELLAModel.from_pretrained(llm_cfg['pretrained_model_path'])

        # Optionally freeze STELLA parameters (feature extractor mode)
        if self.freeze_llm:
            logger.info("[SCMIL_STELLA_AttnMoE] Freezing STELLA encoder parameters.")
            for param in self.stella_encoder.parameters():
                param.requires_grad = False

        # Enable gradient checkpointing to reduce VRAM during fine-tuning
        if self.enable_gradient_checkpointing and not self.freeze_llm:
            self.stella_encoder.encoder.gradient_checkpointing = True
            logger.info("[SCMIL_STELLA_AttnMoE] Gradient checkpointing enabled for STELLA encoder.")

        # ==================================================================
        # Stage 2: Projection Layer — stella_hidden_size → mil_hidden_dim
        # ==================================================================
        # Maps the high-dimensional STELLA embeddings (512-d) to the MIL
        # working dimension (256-d), with LayerNorm for training stability.
        self.projector = nn.Sequential(
            nn.Linear(self.stella_hidden_size, self.mil_hidden_dim),
            nn.LayerNorm(self.mil_hidden_dim),
            nn.GELU(),
        )

        # ==================================================================
        # Stage 3: Linformer Attention (retained from scPhase)
        # ==================================================================
        # The Linformer projects K and V to a fixed low-rank dimension,
        # reducing attention complexity from O(n²) to O(n·k).
        hidden_dim = self.mil_hidden_dim
        self.num_heads = mil_cfg.get('num_heads', 8)
        self.head_dim = hidden_dim // self.num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(mil_cfg.get('linformer_dropout', 0.3))

        if self.attention_type == 'linformer':
            self.linformer_k = mil_cfg.get('linformer_k', 128)
            # Project sequence dimension from up to max_instances to linformer_k
            # Using 10000 as the max seq_len to match original scPhase design
            self.E_proj = nn.Linear(10000, self.linformer_k, bias=False)
            self.F_proj = nn.Linear(10000, self.linformer_k, bias=False)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # ==================================================================
        # Stage 4: Instance Dropout + MoE MIL Aggregation
        # ==================================================================
        self.instance_dropout = InstanceDropout(
            dropout_rate=mil_cfg.get('instance_dropout_rate', 0.2)
        )

        if self.use_moe:
            self.mil_aggregator = MoEMILAggregation(
                input_dim=hidden_dim,
                hidden_dim=mil_cfg['classifier_dims'][0],
                num_experts=mil_cfg.get('moe_num_experts', 8),
                dropout_rate=mil_cfg.get('moe_dropout', 0.2),
            )
        else:
            self.mil_aggregator = AdaptiveMILAggregation(
                input_dim=hidden_dim,
                hidden_dim=mil_cfg['classifier_dims'][0],
            )

        # ==================================================================
        # Stage 5: Disease Classifier
        # ==================================================================
        classifier_dims = mil_cfg['classifier_dims']         # [128, 64]
        n_classes = mil_cfg['n_classes']                     # e.g. 4
        classifier_dropout_rates = mil_cfg['classifier_dropout_rates']  # [0.1, 0.1]

        classifier_layers = [
            nn.Linear(hidden_dim, classifier_dims[0]),
            nn.ReLU(),
            nn.Dropout(classifier_dropout_rates[0]),
            nn.Linear(classifier_dims[0], classifier_dims[1]),
            nn.ReLU(),
            nn.Dropout(classifier_dropout_rates[1]),
            nn.Linear(classifier_dims[1], n_classes),
        ]
        self.classifier = nn.Sequential(*classifier_layers)

        # ==================================================================
        # Stage 6: Domain Adaptation (optional)
        # ==================================================================
        if self.use_domain_adaptation:
            self.gradient_reversal = GradientReversalLayer()
            self.domain_classifier = DomainClassifier(
                input_dim=hidden_dim,
                hidden_dim=classifier_dims[0],
                num_domains=num_domains,
            )

        # ---- Initialize MIL-side weights (STELLA weights are pre-trained) ----
        # Only initialize the non-STELLA components
        initialize_weights(self.projector)
        initialize_weights(self.q_proj)
        initialize_weights(self.k_proj)
        initialize_weights(self.v_proj)
        initialize_weights(self.out_proj)
        initialize_weights(self.mil_aggregator)
        initialize_weights(self.classifier)
        if self.use_domain_adaptation:
            initialize_weights(self.domain_classifier)

        logger.info(
            f"[SCMIL_STELLA_AttnMoE] Initialized. "
            f"stella_hidden={self.stella_hidden_size}, mil_dim={self.mil_hidden_dim}, "
            f"freeze_llm={self.freeze_llm}, attention={self.attention_type}, "
            f"use_moe={self.use_moe}, n_classes={n_classes}, num_domains={num_domains}"
        )

    # ==================================================================
    # Forward Phase — VRAM Defense Mechanism
    # ==================================================================

    def process_large_bag_stella(
        self,
        input_ids_gene_symbol: torch.Tensor,     # [num_cells, seq_len]
        input_ids_gene_expression: torch.Tensor,  # [num_cells, seq_len]
        attention_mask: torch.Tensor,              # [num_cells, seq_len]
    ) -> torch.Tensor:
        """
        Process a patient's cell bag through STELLA with chunked forwarding
        to prevent CUDA OOM on large bags.

        When num_cells > chunk_size, the cells are split into multiple chunks.
        Each chunk is independently forwarded through STELLA, pooled, and projected.
        Results are concatenated to restore the full [num_cells, mil_hidden_dim] shape.

        Args:
            input_ids_gene_symbol:      [num_cells, seq_len] int64 — gene vocab IDs
            input_ids_gene_expression:  [num_cells, seq_len] int64 — bin IDs
            attention_mask:             [num_cells, seq_len] int64 — 1=real, 0=pad

        Returns:
            projected_cells: [num_cells, mil_hidden_dim] — cell-level feature embeddings
        """
        num_cells = input_ids_gene_symbol.size(0)

        if num_cells <= self.chunk_size:
            # ---- Small bag: process in one pass ----
            hidden_states = self._stella_forward_chunk(
                input_ids_gene_symbol, input_ids_gene_expression, attention_mask
            )  # [num_cells, seq_len, stella_hidden_size]
            projected_cells = self._pooling_and_projection(
                hidden_states, attention_mask
            )  # [num_cells, mil_hidden_dim]
            return projected_cells

        # ---- Large bag: split into chunks to avoid OOM ----
        sym_chunks = torch.split(input_ids_gene_symbol, self.chunk_size, dim=0)
        expr_chunks = torch.split(input_ids_gene_expression, self.chunk_size, dim=0)
        mask_chunks = torch.split(attention_mask, self.chunk_size, dim=0)

        projected_chunks = []
        for chunk_sym, chunk_expr, chunk_mask in zip(sym_chunks, expr_chunks, mask_chunks):
            hidden_states = self._stella_forward_chunk(
                chunk_sym, chunk_expr, chunk_mask
            )  # [chunk_size, seq_len, stella_hidden_size]
            chunk_projected = self._pooling_and_projection(
                hidden_states, chunk_mask
            )  # [chunk_size, mil_hidden_dim]
            projected_chunks.append(chunk_projected)

        # Restore shape: [num_cells, mil_hidden_dim]
        projected_cells = torch.cat(projected_chunks, dim=0)
        return projected_cells

    def _stella_forward_chunk(
        self,
        chunk_sym: torch.Tensor,    # [chunk_size, seq_len]
        chunk_expr: torch.Tensor,   # [chunk_size, seq_len]
        chunk_mask: torch.Tensor,   # [chunk_size, seq_len]
    ) -> torch.Tensor:
        """
        Forward a single chunk through the STELLA encoder.

        Uses torch.no_grad() when STELLA is frozen for memory efficiency.

        Args:
            chunk_sym:  [chunk_size, seq_len] — gene symbol IDs
            chunk_expr: [chunk_size, seq_len] — gene expression bin IDs
            chunk_mask: [chunk_size, seq_len] — attention mask

        Returns:
            hidden_states: [chunk_size, seq_len, stella_hidden_size]
        """
        if self.freeze_llm:
            with torch.no_grad():
                outputs = self.stella_encoder(
                    input_ids_gene_symbol=chunk_sym,
                    input_ids_gene_expression=chunk_expr,
                    attention_mask=chunk_mask,
                )
        else:
            outputs = self.stella_encoder(
                input_ids_gene_symbol=chunk_sym,
                input_ids_gene_expression=chunk_expr,
                attention_mask=chunk_mask,
            )

        # outputs is STELLAModelOutput; outputs[0] = last_hidden_state
        hidden_states = outputs[0]  # [chunk_size, seq_len, stella_hidden_size]
        return hidden_states

    # ==================================================================
    # Forward Phase — Pooling & Projection
    # ==================================================================

    def _pooling_and_projection(
        self,
        hidden_states: torch.Tensor,   # [chunk_size, seq_len, stella_hidden_size]
        attention_mask: torch.Tensor,   # [chunk_size, seq_len]
    ) -> torch.Tensor:
        """
        Masked Mean Pooling over the sequence dimension, followed by projection.

        Pooling uses the attention_mask to ignore padding tokens:
            cell_embedding = Σ(hidden * mask) / Σ(mask)

        Then projects from stella_hidden_size → mil_hidden_dim.

        Args:
            hidden_states:  [chunk_size, seq_len, stella_hidden_size]
            attention_mask: [chunk_size, seq_len] — 1 for real tokens, 0 for padding

        Returns:
            projected_cells: [chunk_size, mil_hidden_dim]
        """
        # ---- Masked Mean Pooling ----
        # Expand mask [chunk_size, seq_len] → [chunk_size, seq_len, 1] for broadcasting
        mask_expanded = attention_mask.unsqueeze(-1).to(hidden_states.dtype)

        # Sum over sequence dimension, weighted by mask
        sum_embeddings = (hidden_states * mask_expanded).sum(dim=1)
        # [chunk_size, stella_hidden_size]

        # Count valid (non-padded) tokens per cell, clamp to avoid division by zero
        sum_mask = attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-9).to(hidden_states.dtype)
        # [chunk_size, 1]

        # Compute mean over valid tokens only
        cell_embeddings = sum_embeddings / sum_mask
        # [chunk_size, stella_hidden_size]

        # ---- Projection ----
        projected_cells = self.projector(cell_embeddings)
        # [chunk_size, mil_hidden_dim]

        return projected_cells

    # ==================================================================
    # Linformer Attention (retained from scPhase)
    # ==================================================================

    def linformer_attention_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Linformer-style multi-head attention for instance-level feature refinement.

        Projects K and V to a fixed low-rank dimension (linformer_k) to reduce
        the O(n²) attention complexity to O(n·k), enabling scalability to
        large cell bags (10,000+ cells).

        This implementation is adapted from the original scPhase SCMIL_AttnMoE model,
        with the input expected to be 3D [1, num_cells, mil_hidden_dim].

        Args:
            x: [1, num_cells, mil_hidden_dim] — instance features (batched)

        Returns:
            attended: [1, num_cells, mil_hidden_dim] — attention-refined features
        """
        # Squeeze batch dim for compatibility with original per-sample logic
        # x: [1, num_cells, embed_dim] → [num_cells, embed_dim]
        x_2d = x.squeeze(0)
        seq_len, embed_dim = x_2d.size(0), x_2d.size(1)

        # Project Q, K, V and reshape for multi-head attention
        # [seq_len, embed_dim] → [seq_len, num_heads, head_dim] → [num_heads, seq_len, head_dim]
        Q = self.q_proj(x_2d).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        K = self.k_proj(x_2d).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.v_proj(x_2d).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)

        if seq_len > self.linformer_k:
            # ---- Linformer Low-Rank Projection ----
            # Project K and V from [num_heads, seq_len, head_dim] to [num_heads, linformer_k, head_dim]
            K_t = K.transpose(1, 2)  # [num_heads, head_dim, seq_len]
            V_t = V.transpose(1, 2)  # [num_heads, head_dim, seq_len]

            # Adaptive projection matrix slicing/interpolation
            if seq_len <= self.E_proj.weight.size(0):
                E_proj_matrix = self.E_proj.weight[:seq_len, :self.linformer_k]
                F_proj_matrix = self.F_proj.weight[:seq_len, :self.linformer_k]
            else:
                # Dynamically interpolate projection matrices for seq_len > 10000
                E_proj_matrix = F.interpolate(
                    self.E_proj.weight.T.unsqueeze(0).unsqueeze(0),
                    size=(self.linformer_k, seq_len),
                    mode='bilinear', align_corners=False
                ).squeeze(0).squeeze(0).T
                F_proj_matrix = F.interpolate(
                    self.F_proj.weight.T.unsqueeze(0).unsqueeze(0),
                    size=(self.linformer_k, seq_len),
                    mode='bilinear', align_corners=False
                ).squeeze(0).squeeze(0).T

            # E_proj_matrix: [seq_len, linformer_k]
            K = torch.matmul(K_t, E_proj_matrix)  # [num_heads, head_dim, linformer_k]
            V = torch.matmul(V_t, F_proj_matrix)  # [num_heads, head_dim, linformer_k]
            K = K.transpose(1, 2)  # [num_heads, linformer_k, head_dim]
            V = V.transpose(1, 2)  # [num_heads, linformer_k, head_dim]

        # ---- Scaled Dot-Product Attention ----
        attn_weights = F.softmax(
            torch.matmul(Q, K.transpose(-2, -1)) * (self.head_dim ** -0.5),
            dim=-1,
        )
        attn_output = torch.matmul(self.attn_dropout(attn_weights), V)
        # [num_heads, seq_len, head_dim]

        # Reshape back to [seq_len, embed_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        # Restore batch dimension: [seq_len, embed_dim] → [1, seq_len, embed_dim]
        return attn_output.unsqueeze(0)

    # ==================================================================
    # Main Forward Pass
    # ==================================================================

    def forward(
        self,
        input_ids_gene_symbol: torch.Tensor,      # [num_cells, seq_len]
        input_ids_gene_expression: torch.Tensor,   # [num_cells, seq_len]
        attention_mask: torch.Tensor,               # [num_cells, seq_len]
        alpha: float = 1.0,                         # GRL lambda for domain adaptation
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Full forward pass: STELLA encoding → Pooling → Projection → Attention → MIL → Classification.

        Args:
            input_ids_gene_symbol:      [num_cells, seq_len] — gene vocab IDs
            input_ids_gene_expression:  [num_cells, seq_len] — bin IDs
            attention_mask:             [num_cells, seq_len] — 1=real, 0=pad
            alpha:                      GRL reversal strength for domain adaptation (annealed during training)

        Returns:
            bag_features:       [mil_hidden_dim]         — aggregated patient representation
            disease_output:     [n_classes]              — disease classification logits
            domain_output:      [num_domains] or None    — domain classification logits (if DA enabled)
            attention_weights:  [num_cells, 1]           — MIL attention weights per cell

        Dimension Tracking:
            [num_cells, seq_len]
              ↓ process_large_bag_stella (STELLA + pool + project)
            [num_cells, mil_hidden_dim]
              ↓ instance_dropout
            [num_cells_dropped, mil_hidden_dim]
              ↓ unsqueeze(0) → linformer_attention_forward
            [1, num_cells_dropped, mil_hidden_dim]
              ↓ squeeze(0) + residual + norm
            [num_cells_dropped, mil_hidden_dim]
              ↓ mil_aggregator
            bag_features: [mil_hidden_dim], attention_weights: [num_cells_dropped, 1]
              ↓ classifier
            disease_output: [n_classes]
        """
        # ---- Stage 1+2: STELLA Encoding + Pooling + Projection ----
        # [num_cells, seq_len] → [num_cells, mil_hidden_dim]
        instance_features = self.process_large_bag_stella(
            input_ids_gene_symbol, input_ids_gene_expression, attention_mask
        )

        # ---- Instance Dropout (training-time regularization) ----
        instance_features = self.instance_dropout(instance_features)
        # [num_cells_dropped, mil_hidden_dim]

        # ---- Stage 3: Linformer Attention ----
        if self.attention_type == 'linformer':
            # unsqueeze for batch dim: [num_cells, dim] → [1, num_cells, dim]
            attended_instances = self.linformer_attention_forward(
                instance_features.unsqueeze(0)
            )
            # [1, num_cells, mil_hidden_dim] → [num_cells, mil_hidden_dim]
            attended_instances = attended_instances.squeeze(0)
            # Residual connection + LayerNorm (following original scPhase)
            attended_instances = self.norm1(attended_instances + instance_features)
            attended_instances = self.norm2(attended_instances)
        else:
            # Fallback: no attention refinement
            attended_instances = instance_features

        # ---- Stage 4: MIL Aggregation ----
        # MoEMILAggregation expects [num_cells, mil_hidden_dim]
        # Returns bag_features [mil_hidden_dim], attention_weights [num_cells, 1]
        bag_features, attention_weights = self.mil_aggregator(attended_instances)

        # ---- Stage 5: Disease Classification ----
        # bag_features: [mil_hidden_dim] → unsqueeze → [1, mil_hidden_dim]
        disease_output = self.classifier(bag_features.unsqueeze(0)).squeeze(0)
        # disease_output: [n_classes]

        # ---- Stage 6: Domain Adaptation (optional) ----
        domain_output = None
        if self.use_domain_adaptation:
            self.gradient_reversal.set_lambda(alpha)
            domain_features = self.gradient_reversal(bag_features)
            domain_output = self.domain_classifier(
                domain_features.unsqueeze(0)
            ).squeeze(0)
            # domain_output: [num_domains]

        return bag_features, disease_output, domain_output, attention_weights

    # ==================================================================
    # Utility Methods
    # ==================================================================

    def get_trainable_params(self) -> Dict[str, list]:
        """
        Separate model parameters into two groups for differential learning rates:
          - 'llm': STELLA encoder parameters (lower LR or frozen)
          - 'mil': All other MIL pipeline parameters (higher LR)

        Usage in optimizer:
            param_groups = model.get_trainable_params()
            optimizer = torch.optim.AdamW([
                {'params': param_groups['llm'], 'lr': config['training_params']['llm_lr']},
                {'params': param_groups['mil'], 'lr': config['training_params']['mil_lr']},
            ])

        Returns:
            Dict with 'llm' and 'mil' keys, each containing a list of Parameters.
        """
        llm_params = []
        mil_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('stella_encoder'):
                llm_params.append(param)
            else:
                mil_params.append(param)

        logger.info(
            f"[get_trainable_params] LLM params: {len(llm_params)}, "
            f"MIL params: {len(mil_params)}"
        )
        return {'llm': llm_params, 'mil': mil_params}

    def get_param_count(self) -> Dict[str, int]:
        """
        Count total, trainable, and frozen parameters for logging.

        Returns:
            Dict with 'total', 'trainable', 'frozen' keys.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {'total': total, 'trainable': trainable, 'frozen': frozen}
