# run_integrated.py
# ==============================================================================
# STELLA x scPhase Integration — Robust Execution Script
# ==============================================================================
# Main entry point for training the SCMIL_STELLA_AttnMoE model.
#
# This script orchestrates the full training pipeline:
#   1. Load centralized JSON config
#   2. Initialize STELLA-aware DataLoader with inline tokenization
#   3. Instantiate the integrated model (STELLA encoder + MIL pipeline)
#   4. Train with differential learning rates, mixed precision, and OOM defense
#   5. Evaluate with cross-validation (LOGO or StratifiedKFold)
#
# Key Engineering Features:
#   - Differential LRs: 1e-5 for STELLA (LLM), 1e-3 for MIL downstream
#   - Mixed Precision (AMP): GradScaler + autocast for VRAM efficiency
#   - OOM Defense: try-except around forward/backward, per-sample fault isolation
#   - Cosine warmup scheduler for stable convergence
#   - Early stopping with best-model checkpointing
#
# Usage:
#   python run_integrated.py --config ./config_integration.json
# ==============================================================================

import os
import sys
import json
import copy
import logging
import random
import argparse
import traceback
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import (
    LeaveOneGroupOut, StratifiedKFold, KFold, train_test_split,
)
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, mean_squared_error,
    mean_absolute_error, r2_score,
)
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import pearsonr

# ---- Local imports from the integration package ----
from integrated_dataloader import (
    load_data,
    create_tokenizer_from_config,
    StellaScPhaseDataset,
    stella_collate_fn,
)
from integrated_model import SCMIL_STELLA_AttnMoE


# ==============================================================================
# Section 1: Utility Functions
# ==============================================================================

def setup_logging(config: dict) -> logging.Logger:
    """
    Configure dual logging: file + console.
    Mirrors the original scPhase setup_logging from train_utils.py.
    """
    log_dir = config["path_params"]["RESULTS_DIR"]
    log_name = config["path_params"]["LOGNAME"]
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, log_name)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger("STELLA_scPhase_Runner")


def set_seed(seed: int = 3407):
    """Deterministic seed for reproducibility across all RNG sources."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id: int):
    """Ensure each DataLoader worker has a unique but deterministic seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """
    Cosine annealing LR scheduler with linear warmup.
    Applies to all parameter groups in the optimizer uniformly.

    Args:
        optimizer:           The optimizer instance.
        num_warmup_steps:    Number of steps for linear warmup.
        num_training_steps:  Total number of training steps.
        num_cycles:          Number of cosine half-cycles (0.5 = decay to 0).

    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_domain_loss_weight(epoch: int, total_epochs: int) -> float:
    """
    Sigmoid-scheduled domain adaptation loss weight.
    Gradually ramps from ~0 to 0.2, centered at 30% of training.
    """
    p = epoch / total_epochs
    return 0.2 * (1 / (1 + np.exp(-10 * (p - 0.3))))


def calculate_auc_score(y_true, y_probs, n_classes: int) -> float:
    """Calculate AUC for binary or multi-class classification."""
    y_true_array = np.array(y_true)
    y_probs_array = np.array(y_probs)

    actual_classes = len(np.unique(y_true_array))

    if actual_classes == 2 or n_classes == 2:
        if y_probs_array.ndim == 2 and y_probs_array.shape[1] == 2:
            return roc_auc_score(y_true_array, y_probs_array[:, 1])
        else:
            return roc_auc_score(y_true_array, y_probs_array)
    else:
        return roc_auc_score(
            y_true_array, y_probs_array, multi_class="ovr", labels=np.arange(n_classes)
        )


# ==============================================================================
# Section 2: Early Stopping (adapted from scPhase)
# ==============================================================================

class EarlyStopping:
    """
    Early stopping to terminate training when validation metric stops improving.

    Args:
        patience: Number of epochs to wait after last improvement.
        verbose:  Whether to log checkpoint saves.
        delta:    Minimum change to qualify as an improvement.
        path:     Path to save the best model checkpoint.
        mode:     'max' to maximize metric (AUC), 'min' to minimize (loss).
    """

    def __init__(self, patience=15, verbose=True, delta=0.0, path=None, mode="max"):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state_dict = None
        self.logger = logging.getLogger("STELLA_scPhase_Runner")

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(score, model, is_first=True)
            return

        improved = (
            score > self.best_score + self.delta
            if self.mode == "max"
            else score < self.best_score - self.delta
        )

        if improved:
            prev = self.best_score
            self.best_score = score
            self._save_checkpoint(score, model, previous_score=prev)
            self.counter = 0
        else:
            self.counter += 1
            self.logger.info(
                f"EarlyStopping counter: {self.counter}/{self.patience} "
                f"(current: {score:.6f}, best: {self.best_score:.6f})"
            )
            if self.counter >= self.patience:
                self.early_stop = True

    def _save_checkpoint(self, score, model, previous_score=None, is_first=False):
        if self.verbose:
            if is_first:
                self.logger.info(f"Initial validation score: {score:.6f}. Storing best model state...")
            elif previous_score is not None:
                self.logger.info(
                    f"Validation score improved ({previous_score:.6f} --> {score:.6f}). "
                    f"Storing best model state..."
                )
        self.best_model_state_dict = copy.deepcopy(model.state_dict())
        if self.path:
            torch.save(self.best_model_state_dict, self.path)


# ==============================================================================
# Section 3: Per-Fold Training & Evaluation
# ==============================================================================

def train_and_evaluate_fold(
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    DataList: list,
    DataLabel: np.ndarray,
    DataBatch: np.ndarray,
    SampleIDs: list,
    GeneNames: np.ndarray,
    tokenizer,
    config: dict,
    use_domain_adaptation: bool,
) -> dict:
    """
    Train and evaluate the SCMIL_STELLA_AttnMoE model for a single CV fold.

    This function handles:
      - Train/val/test split
      - Model instantiation with differential LRs
      - Mixed precision training loop with OOM defense
      - Validation with early stopping
      - Final test evaluation

    Args:
        fold:                   Fold index (0-based).
        train_idx:              Indices of training samples.
        test_idx:               Indices of test samples.
        DataList:               List of sparse matrices (one per patient).
        DataLabel:              Labels array.
        DataBatch:              Batch/domain IDs array.
        SampleIDs:              Sample ID strings.
        GeneNames:              Gene name strings for tokenization.
        tokenizer:              StellaInlineTokenizer instance.
        config:                 Full integration config dict.
        use_domain_adaptation:  Whether to enable domain adaptation.

    Returns:
        Dict of test metrics for this fold.
    """
    logger = logging.getLogger("STELLA_scPhase_Runner")
    logger.info(f"{'='*60}")
    logger.info(f"--- Starting Fold {fold + 1} ---")
    logger.info(f"{'='*60}")

    # ---- Unpack config sections ----
    run_cfg = config["run_params"]
    train_cfg = config["training_params"]
    mil_cfg = config["mil_params"]
    path_cfg = config["path_params"]

    device = run_cfg["device"]
    task_type = run_cfg["task_type"]
    seed = run_cfg["seed"]
    set_seed(seed)

    # ==================================================================
    # 3.1: Train / Validation / Test Split
    # ==================================================================
    X_train_full = [DataList[i] for i in train_idx]
    y_train_full = DataLabel[train_idx]
    batch_train_full = DataBatch[train_idx]
    sid_train_full = [SampleIDs[i] for i in train_idx]

    X_test = [DataList[i] for i in test_idx]
    y_test = DataLabel[test_idx]
    batch_test = DataBatch[test_idx]
    sid_test = [SampleIDs[i] for i in test_idx]

    logger.info(f"Test set: {len(X_test)} samples, groups: {np.unique(batch_test)}")

    # Stratified train/val split
    stratify_val = y_train_full if task_type == "classification" else None
    train_indices, val_indices = train_test_split(
        np.arange(len(X_train_full)),
        test_size=train_cfg["val_size"],
        random_state=seed,
        stratify=stratify_val,
    )

    train_data = [X_train_full[i] for i in train_indices]
    train_label = y_train_full[train_indices]
    train_batch = batch_train_full[train_indices]
    train_sids = [sid_train_full[i] for i in train_indices]

    valid_data = [X_train_full[i] for i in val_indices]
    valid_label = y_train_full[val_indices]
    valid_batch = batch_train_full[val_indices]
    valid_sids = [sid_train_full[i] for i in val_indices]

    logger.info(
        f"Train: {len(train_data)} | Val: {len(valid_data)} | Test: {len(X_test)} samples"
    )
    if task_type == "classification":
        logger.info(f"Train label dist: {np.bincount(train_label)}")
        logger.info(f"Val   label dist: {np.bincount(valid_label)}")

    # ---- Domain mapping (remap batch IDs to contiguous 0..N-1) ----
    unique_domains = np.unique(batch_train_full)
    domain_mapping = {domain: idx for idx, domain in enumerate(unique_domains)}
    train_batch_mapped = np.array([domain_mapping[b] for b in train_batch])
    valid_batch_mapped = np.array([domain_mapping[b] for b in valid_batch])
    num_domains = len(unique_domains)

    # ==================================================================
    # 3.2: Create Datasets & DataLoaders
    # ==================================================================
    train_dataset = StellaScPhaseDataset(
        train_data, train_label, train_batch_mapped, train_sids, GeneNames, tokenizer, is_train=True
    )
    valid_dataset = StellaScPhaseDataset(
        valid_data, valid_label, valid_batch_mapped, valid_sids, GeneNames, tokenizer, is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],       # Always 1 for MIL
        num_workers=train_cfg["num_workers"],
        collate_fn=stella_collate_fn,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True if train_cfg["num_workers"] > 0 else False,
        prefetch_factor=2 if train_cfg["num_workers"] > 0 else None,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        collate_fn=stella_collate_fn,
        pin_memory=True,
        shuffle=False,
        worker_init_fn=worker_init_fn,
        persistent_workers=True if train_cfg["num_workers"] > 0 else False,
        prefetch_factor=2 if train_cfg["num_workers"] > 0 else None,
    )

    # ==================================================================
    # 3.3: Model Instantiation
    # ==================================================================
    model = SCMIL_STELLA_AttnMoE(
        config=config,
        num_domains=num_domains,
        device=device,
    ).to(device)

    param_count = model.get_param_count()
    logger.info(
        f"Model parameters — Total: {param_count['total']:,} | "
        f"Trainable: {param_count['trainable']:,} | "
        f"Frozen: {param_count['frozen']:,}"
    )

    # ==================================================================
    # 3.4: Optimizer with Differential Learning Rates
    # ==================================================================
    # LLM (STELLA) params get a conservative LR (1e-5) for gentle fine-tuning
    # or 0 updates if frozen. MIL downstream params get a higher LR (1e-3).
    param_groups = model.get_trainable_params()

    optimizer_param_groups = []
    if len(param_groups["llm"]) > 0:
        optimizer_param_groups.append({
            "params": param_groups["llm"],
            "lr": train_cfg["llm_lr"],
            "name": "stella_llm",
        })
    if len(param_groups["mil"]) > 0:
        optimizer_param_groups.append({
            "params": param_groups["mil"],
            "lr": train_cfg["mil_lr"],
            "name": "mil_downstream",
        })

    optimizer = torch.optim.AdamW(
        optimizer_param_groups,
        weight_decay=train_cfg["weight_decay"],
        betas=tuple(train_cfg["betas"]),
    )

    # Log the actual LR for each group
    for pg in optimizer.param_groups:
        logger.info(f"  Optimizer group '{pg.get('name', '?')}': lr={pg['lr']}, #params={len(pg['params'])}")

    # ---- LR Scheduler: Cosine with linear warmup ----
    total_steps = train_cfg["epochs"] * len(train_loader)
    warmup_steps = train_cfg["warmup_epochs"] * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ==================================================================
    # 3.5: Loss Functions
    # ==================================================================
    if task_type == "classification":
        class_weights = torch.tensor(
            compute_class_weight("balanced", classes=np.unique(train_label), y=train_label),
            dtype=torch.float,
        ).to(device)
        disease_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        early_stopping_mode = "max"  # maximize AUC
    else:
        disease_criterion = nn.MSELoss()
        early_stopping_mode = "max"  # maximize R2

    domain_criterion = nn.CrossEntropyLoss()

    # ==================================================================
    # 3.6: Mixed Precision Setup
    # ==================================================================
    use_amp = train_cfg.get("use_amp", True) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    logger.info(f"Mixed Precision (AMP): {'ENABLED' if use_amp else 'DISABLED'}")

    # ==================================================================
    # 3.7: Early Stopping & Checkpoint
    # ==================================================================
    checkpoint_path = os.path.join(
        path_cfg["RESULTS_DIR"],
        f"BestModel_{path_cfg['MODEL_NAME']}_Fold{fold + 1}.pt",
    )
    early_stopping = EarlyStopping(
        patience=train_cfg["early_stopping_patience"],
        verbose=True,
        path=checkpoint_path,
        mode=early_stopping_mode,
    )

    # ==================================================================
    # 3.8: Training Loop
    # ==================================================================
    for epoch in range(train_cfg["epochs"]):
        model.train()
        total_train_loss = 0.0
        total_domain_loss = 0.0
        n_train_samples = 0
        n_skipped_oom = 0

        # GRL alpha: sigmoid schedule from 0 → 1 over training
        alpha = 2.0 / (1.0 + np.exp(-8.0 * float(epoch) / train_cfg["epochs"])) - 1.0
        domain_weight = (
            get_domain_loss_weight(epoch, train_cfg["epochs"])
            if use_domain_adaptation
            else 0.0
        )

        pbar = tqdm(
            train_loader,
            desc=f"Fold {fold+1} | Epoch {epoch+1}/{train_cfg['epochs']} Train",
        )

        for gene_sym, gene_expr, attn_mask, labels, batches, sample_ids in pbar:
            # ==============================================================
            # ⚠️ CRITICAL: OOM & Exception Defense Mechanism
            # Patient cell bags can be extremely large (10,000+ cells × 4096
            # tokens). We wrap forward+backward in try-except to isolate
            # per-sample failures without crashing the training loop.
            # ==============================================================
            try:
                # ---- Move tensors to device ----
                gene_sym = gene_sym.to(device)           # [num_cells, seq_len]
                gene_expr = gene_expr.to(device)         # [num_cells, seq_len]
                attn_mask = attn_mask.to(device)         # [num_cells, seq_len]
                labels = labels.to(device)               # [batch_size]
                batches = batches.to(device)              # [batch_size]

                if task_type == "classification":
                    labels = labels.long()
                else:
                    labels = labels.float().unsqueeze(0)

                optimizer.zero_grad()

                # ---- Forward pass under AMP autocast ----
                with torch.amp.autocast("cuda", enabled=use_amp):
                    bag_features, disease_out, domain_out, attn_weights = model(
                        input_ids_gene_symbol=gene_sym,
                        input_ids_gene_expression=gene_expr,
                        attention_mask=attn_mask,
                        alpha=alpha,
                    )

                    # ---- Compute disease loss ----
                    # disease_out: [n_classes], labels: [1] → unsqueeze for CE
                    disease_loss = disease_criterion(
                        disease_out.unsqueeze(0), labels
                    )
                    loss = disease_loss

                    # ---- Compute domain adaptation loss (optional) ----
                    if domain_weight > 0 and domain_out is not None:
                        domain_loss = domain_criterion(
                            domain_out.unsqueeze(0), batches.long()
                        )
                        loss = loss + domain_loss * domain_weight
                        total_domain_loss += domain_loss.item()

                # ---- Backward pass with AMP scaler ----
                scaler.scale(loss).backward()

                # ---- Gradient clipping (unscale first for correct norm) ----
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    train_cfg["clip_grad_norm"],
                )

                # ---- Optimizer step ----
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_train_loss += disease_loss.item()
                n_train_samples += 1

                # Update progress bar with running loss
                pbar.set_postfix({
                    "loss": f"{disease_loss.item():.4f}",
                    "cells": gene_sym.size(0),
                    "skipped": n_skipped_oom,
                })

            except RuntimeError as e:
                # ---- CUDA OOM Handler ----
                if "CUDA out of memory" in str(e) or "out of memory" in str(e):
                    n_skipped_oom += 1
                    sid = sample_ids[0] if sample_ids else "UNKNOWN"
                    logger.warning(
                        f"[OOM] Skipping sample '{sid}' "
                        f"(cells={gene_sym.size(0) if gene_sym is not None else '?'}, "
                        f"seq_len={gene_sym.size(1) if gene_sym is not None else '?'}). "
                        f"Total OOM skips this epoch: {n_skipped_oom}"
                    )
                    # Free all cached GPU memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Zero gradients to prevent stale gradient accumulation
                    optimizer.zero_grad(set_to_none=True)
                    continue
                else:
                    # Non-OOM RuntimeError — still log and skip
                    sid = sample_ids[0] if sample_ids else "UNKNOWN"
                    logger.error(
                        f"[RuntimeError] Skipping sample '{sid}': {str(e)[:200]}"
                    )
                    logger.error(traceback.format_exc())
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                    continue

            except Exception as e:
                # ---- General Exception Handler ----
                # Catches h5ad parsing errors, sparse matrix issues, tokenization
                # failures, NaN/Inf in loss, etc.
                sid = sample_ids[0] if sample_ids else "UNKNOWN"
                logger.error(
                    f"[Exception] Skipping sample '{sid}': {type(e).__name__}: {str(e)[:300]}"
                )
                logger.error(traceback.format_exc())
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                continue

        # ---- Epoch-level training summary ----
        avg_train_loss = total_train_loss / max(n_train_samples, 1)
        avg_domain_loss = total_domain_loss / max(n_train_samples, 1)
        logger.info(
            f"Epoch {epoch+1} Train Summary | "
            f"Avg Loss: {avg_train_loss:.4f} | "
            f"Domain Loss: {avg_domain_loss:.4f} | "
            f"Samples: {n_train_samples} | "
            f"OOM Skips: {n_skipped_oom}"
        )

        # ==================================================================
        # 3.9: Validation Loop
        # ==================================================================
        model.eval()
        val_loss = 0.0
        val_preds, val_probs, val_true = [], [], []
        n_val_samples = 0

        with torch.no_grad():
            for gene_sym, gene_expr, attn_mask, labels, batches, sample_ids in tqdm(
                valid_loader,
                desc=f"Fold {fold+1} | Epoch {epoch+1}/{train_cfg['epochs']} Valid",
            ):
                try:
                    gene_sym = gene_sym.to(device)
                    gene_expr = gene_expr.to(device)
                    attn_mask = attn_mask.to(device)
                    labels = labels.to(device)

                    if task_type == "classification":
                        labels = labels.long()
                    else:
                        labels = labels.float().unsqueeze(0)

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        _, disease_out, _, _ = model(
                            input_ids_gene_symbol=gene_sym,
                            input_ids_gene_expression=gene_expr,
                            attention_mask=attn_mask,
                            alpha=1.0,
                        )
                        v_loss = disease_criterion(disease_out.unsqueeze(0), labels)

                    val_loss += v_loss.item()
                    n_val_samples += 1

                    if task_type == "classification":
                        probs = F.softmax(disease_out.float(), dim=0)
                        val_preds.append(torch.argmax(probs).cpu().numpy())
                        val_probs.append(probs.cpu().numpy())
                        val_true.append(labels.cpu().numpy())
                    else:
                        val_preds.append(disease_out.float().cpu().numpy())
                        val_true.append(labels.squeeze().cpu().numpy())

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        sid = sample_ids[0] if sample_ids else "UNKNOWN"
                        logger.warning(f"[Val OOM] Skipping sample '{sid}'")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise
                except Exception as e:
                    sid = sample_ids[0] if sample_ids else "UNKNOWN"
                    logger.error(f"[Val Exception] Skipping sample '{sid}': {e}")
                    continue

        # ---- Epoch-level validation metrics & early stopping ----
        avg_val_loss = val_loss / max(n_val_samples, 1)

        if task_type == "classification" and len(val_true) > 0:
            val_auc = calculate_auc_score(val_true, val_probs, mil_cfg["n_classes"])
            val_acc = accuracy_score(val_true, val_preds)
            logger.info(
                f"Epoch {epoch+1} Valid | "
                f"Loss: {avg_val_loss:.4f} | AUC: {val_auc:.4f} | Acc: {val_acc:.4f}"
            )
            early_stopping(val_auc, model)
        elif task_type == "regression" and len(val_true) > 0:
            val_r2 = r2_score(val_true, val_preds)
            logger.info(
                f"Epoch {epoch+1} Valid | Loss: {avg_val_loss:.4f} | R2: {val_r2:.4f}"
            )
            early_stopping(val_r2, model)
        else:
            logger.warning(f"Epoch {epoch+1} Valid | No valid samples evaluated.")

        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}.")
            break

    # ==================================================================
    # 3.10: Final Test Evaluation
    # ==================================================================
    logger.info(f"Loading best model from: {checkpoint_path}")

    # Load best model state dict
    if early_stopping.best_model_state_dict is not None:
        model.load_state_dict(early_stopping.best_model_state_dict)
    elif os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        logger.warning("No best model checkpoint found. Using final epoch model.")

    model.eval()

    # ---- Build test DataLoader ----
    test_batch_mapped = np.array(
        [domain_mapping.get(b, 0) for b in batch_test]
    )
    test_dataset = StellaScPhaseDataset(
        X_test, y_test, test_batch_mapped, sid_test, GeneNames, tokenizer, is_train=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        collate_fn=stella_collate_fn,
        shuffle=False,
    )

    test_preds, test_probs, test_true = [], [], []

    with torch.no_grad():
        for gene_sym, gene_expr, attn_mask, labels, batches, sample_ids in tqdm(
            test_loader, desc=f"Fold {fold+1} Test"
        ):
            try:
                gene_sym = gene_sym.to(device)
                gene_expr = gene_expr.to(device)
                attn_mask = attn_mask.to(device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    _, disease_out, _, _ = model(
                        input_ids_gene_symbol=gene_sym,
                        input_ids_gene_expression=gene_expr,
                        attention_mask=attn_mask,
                        alpha=1.0,
                    )

                if task_type == "classification":
                    probs = F.softmax(disease_out.float(), dim=0)
                    test_preds.append(torch.argmax(probs).cpu().numpy())
                    test_probs.append(probs.cpu().numpy())
                else:
                    test_preds.append(disease_out.float().cpu().numpy().squeeze())

                test_true.append(labels.numpy().squeeze())

            except Exception as e:
                sid = sample_ids[0] if sample_ids else "UNKNOWN"
                logger.error(f"[Test Exception] Skipping sample '{sid}': {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

    # ---- Compute final test metrics ----
    results = {}
    if task_type == "classification" and len(test_true) > 0:
        results["auc"] = calculate_auc_score(test_true, test_probs, mil_cfg["n_classes"])
        results["acc"] = accuracy_score(test_true, test_preds)
        results["precision"] = precision_score(test_true, test_preds, average="weighted", zero_division=0)
        results["recall"] = recall_score(test_true, test_preds, average="weighted", zero_division=0)
        results["f1"] = f1_score(test_true, test_preds, average="weighted", zero_division=0)
        logger.info(
            f"Fold {fold+1} Test | AUC={results['auc']:.4f} | "
            f"Acc={results['acc']:.4f} | F1={results['f1']:.4f}"
        )
    elif task_type == "regression" and len(test_true) > 0:
        results["mse"] = mean_squared_error(test_true, test_preds)
        results["mae"] = mean_absolute_error(test_true, test_preds)
        results["r2"] = r2_score(test_true, test_preds)
        results["pearson"] = pearsonr(np.array(test_true), np.array(test_preds))[0]
        logger.info(
            f"Fold {fold+1} Test | MSE={results['mse']:.4f} | "
            f"R2={results['r2']:.4f} | Pearson={results['pearson']:.4f}"
        )
    else:
        logger.warning(f"Fold {fold+1}: No test samples evaluated successfully.")

    # ---- Clean up GPU memory before next fold ----
    del model, optimizer, scheduler, scaler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ==============================================================================
# Section 4: Cross-Validation Orchestrator
# ==============================================================================

def run_cv_experiment(config_path: str):
    """
    Main entry point: orchestrates the full cross-validation experiment.

    Steps:
      1. Load config → setup logging → set seed
      2. Load data → create tokenizer
      3. Select CV strategy (LOGO or StratifiedKFold)
      4. Train + evaluate each fold
      5. Aggregate and save results
    """
    # ---- 4.1: Load configuration ----
    with open(config_path, "r") as f:
        config = json.load(f)

    logger = setup_logging(config)
    logger.info("=" * 70)
    logger.info(" STELLA x scPhase Integration — Training Pipeline")
    logger.info("=" * 70)
    logger.info(f"Configuration loaded from: {config_path}")
    logger.info(f"Configuration:\n{json.dumps(config, indent=2)}")

    # ---- 4.2: Environment checks ----
    if not torch.cuda.is_available():
        logger.warning("CUDA NOT available. Forcing CPU mode (this will be very slow).")
        config["run_params"]["device"] = "cpu"
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name} | VRAM: {gpu_mem:.1f} GB")

    set_seed(config["run_params"]["seed"])

    # ---- 4.3: Load data and create tokenizer ----
    logger.info("Loading data from h5ad / pickle...")
    DataList, DataLabel, DataBatch, SampleIDs, GeneNames = load_data(config)
    logger.info(
        f"Data loaded: {len(DataList)} samples | "
        f"{len(np.unique(DataLabel))} classes | "
        f"{len(np.unique(DataBatch))} batches | "
        f"{len(GeneNames)} genes"
    )

    logger.info("Initializing STELLA inline tokenizer...")
    tokenizer = create_tokenizer_from_config(config)

    # ---- 4.4: Determine domain adaptation & CV strategy ----
    num_groups = len(np.unique(DataBatch))
    task_type = config["run_params"]["task_type"]

    use_domain_adaptation = (
        config["mil_params"].get("use_domain_adaptation", True)
        and num_groups > 1
        and task_type == "classification"
    )
    if use_domain_adaptation:
        logger.info(f"Domain Adaptation: ENABLED ({num_groups} groups detected)")
    else:
        logger.info(f"Domain Adaptation: DISABLED")

    # CV Strategy selection
    if num_groups > 1:
        logger.info("CV Strategy: Leave-One-Group-Out (LOGO)")
        cv = LeaveOneGroupOut()
        cv_splitter = cv.split(DataList, DataLabel, DataBatch)
    else:
        num_folds = config["run_params"]["num_folds"]
        logger.info(f"CV Strategy: {num_folds}-Fold {'Stratified' if task_type == 'classification' else ''}KFold")
        if task_type == "classification":
            cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=config["run_params"]["seed"])
            cv_splitter = cv.split(DataList, DataLabel)
        else:
            cv = KFold(n_splits=num_folds, shuffle=True, random_state=config["run_params"]["seed"])
            cv_splitter = cv.split(DataList)

    # ---- 4.5: Cross-Validation Loop ----
    all_results = []

    for fold, (train_idx, test_idx) in enumerate(cv_splitter):
        test_group_info = (
            np.unique(DataBatch[test_idx]) if num_groups > 1 else f"Fold {fold + 1}"
        )

        # Check skip_groups
        skip_groups = config["run_params"].get("skip_groups", [])
        if skip_groups:
            if num_groups > 1:
                test_group = test_group_info[0] if isinstance(test_group_info, np.ndarray) else test_group_info
                if test_group in skip_groups:
                    logger.info(f"Skipping fold {fold+1} (test group {test_group_info}) per skip_groups config.")
                    continue
            else:
                if (fold + 1) in skip_groups:
                    logger.info(f"Skipping fold {fold+1} per skip_groups config.")
                    continue
        else:
            # Auto-skip folds where test set doesn't contain all classes
            if num_groups > 1 and task_type == "classification":
                n_classes_in_fold = len(np.unique(DataLabel[test_idx]))
                total_classes = config["mil_params"]["n_classes"]
                if n_classes_in_fold < total_classes:
                    logger.info(
                        f"Skipping fold {fold+1} (test group {test_group_info}): "
                        f"only {n_classes_in_fold}/{total_classes} classes represented."
                    )
                    continue

        fold_results = train_and_evaluate_fold(
            fold=fold,
            train_idx=train_idx,
            test_idx=test_idx,
            DataList=DataList,
            DataLabel=DataLabel,
            DataBatch=DataBatch,
            SampleIDs=SampleIDs,
            GeneNames=GeneNames,
            tokenizer=tokenizer,
            config=config,
            use_domain_adaptation=use_domain_adaptation,
        )

        fold_with_meta = {
            "model_name": config["path_params"]["MODEL_NAME"],
            "fold": fold + 1,
            "test_group": str(test_group_info),
            **fold_results,
        }
        all_results.append(fold_with_meta)

    # ---- 4.6: Aggregate & Save Results ----
    if not all_results:
        logger.warning("No folds were executed. Check data and configuration.")
        return

    results_df = pd.DataFrame(all_results)
    results_dir = config["path_params"]["RESULTS_DIR"]
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(
        results_dir, f"AllFolds_{config['path_params']['MODEL_NAME']}.csv"
    )
    results_df.to_csv(results_path, index=False)
    logger.info(f"Per-fold results saved to: {results_path}")

    # ---- Summary statistics ----
    summary_metrics = [
        col for col in results_df.columns
        if col not in ["model_name", "fold", "test_group"]
    ]
    logger.info("\n" + "=" * 60)
    logger.info(" FINAL EXPERIMENT SUMMARY")
    logger.info("=" * 60)

    summary = {"model_name": config["path_params"]["MODEL_NAME"]}
    for metric in summary_metrics:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        summary[f"mean_{metric}"] = mean_val
        summary[f"std_{metric}"] = std_val
        logger.info(f"  {metric.upper():>10s}: {mean_val:.4f} (±{std_val:.4f})")

    summary_path = os.path.join(
        results_dir, f"Summary_{config['path_params']['MODEL_NAME']}.csv"
    )
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("=" * 60)
    logger.info("CV Experiment finished successfully!")
    logger.info("=" * 60)


# ==============================================================================
# Section 5: Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="STELLA x scPhase Integration — Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_integrated.py --config ./config_integration.json
  python run_integrated.py --config /data/configs/AD_config.json
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/data/home/zhangyaojie/StePhase/config_integration.json",
        help="Path to the JSON configuration file (default: /data/home/zhangyaojie/StePhase/config_integration.json)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    run_cv_experiment(args.config)
