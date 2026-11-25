"""Tier 2B: 100% Leak-Free Multi-Task Residual MLP in PyTorch (CPU).

Trains a 5-head multi-task Residual MLP across all 5 CYP isozymes
(1a2, 2c9, 2c19, 2d6, 3a4) under a strict zero-leakage protocol:
  1. Build a master multi-task table of all SIDs appearing in any training{j}.csv.
  2. For each target isozyme `iso`:
     - Exclude ALL SIDs in test{iso}.csv from the entire 5-head multi-task training pool.
     - Use the exact 5-fold StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
       splits on training{iso}.csv.
     - For each fold k:
       * Exclude ALL fold-k validation SIDs from the entire 5-head training pool.
       * Fit SimpleImputer(median) + QuantileTransformer(normal) strictly on the
         fold-k training pool's continuous descriptors.
       * Train with masked BCEWithLogitsLoss (weight 2.0 on target head, 1.0 on
         the 4 auxiliary heads; NaN labels masked out).
       * Early-stop on target isozyme validation ROC-AUC (patience 8), restore best weights.
       * Average fold models' predictions on test{iso} (5-fold CV ensemble).

Feature variants:
  - MT_ResMLP_RDKit:     log1p(ecfp4_counts) + fcfp4_bits + maccs + rdkit_desc
  - MT_ResMLP_MOE_RDKit: MOE + log1p(ecfp4_counts) + fcfp4_bits + maccs + rdkit_desc
"""

from __future__ import annotations

import argparse
import copy
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    # Keep CPU/RAM polite alongside other running jobs
    torch.set_num_threads(4)

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
PREDS_DIR = RESULTS_DIR / "tier2_mt_preds"

ISOZYMES = ["1a2", "2c9", "2c19", "2d6", "3a4"]
ACTIVITY_THRESHOLD = 40
SPLIT_SEED = 0
N_SPLITS = 5

BATCH_SIZE = 256
MAX_EPOCHS = 45
PATIENCE = 8
LR = 1e-3
WEIGHT_DECAY = 2e-2
TARGET_TASK_WEIGHT = 2.0
AUX_TASK_WEIGHT = 1.0


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(
    y_train: np.ndarray,
    oof_prob: np.ndarray,
    y_test: np.ndarray,
    test_prob: np.ndarray,
) -> dict[str, float]:
    """Compute CV and Test classification & calibration metrics from probabilities."""
    oof_clip = np.clip(oof_prob, 1e-7, 1.0 - 1e-7)
    test_clip = np.clip(test_prob, 1e-7, 1.0 - 1e-7)

    cv_pred = (oof_clip >= 0.5).astype(int)
    test_pred = (test_clip >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()

    return {
        "cv_acc": float(accuracy_score(y_train, cv_pred)),
        "cv_auc": float(roc_auc_score(y_train, oof_clip)),
        "test_acc": float((tp + tn) / (tp + tn + fp + fn)),
        "test_tpr": float(tp / (tp + fn)),
        "test_tnr": float(tn / (tn + fp)),
        "test_auc": float(roc_auc_score(y_test, test_clip)),
        "test_brier": float(brier_score_loss(y_test, test_clip)),
        "test_logloss": float(log_loss(y_test, test_clip)),
    }


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.35) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop1(F.silu(self.ln1(self.fc1(x))))
        h = self.drop2(self.ln2(self.fc2(h)))
        return F.silu(x + h)


class MultiTaskResMLP(nn.Module):
    def __init__(self, in_dim: int, n_tasks: int = 5) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.4),
        )
        self.res_block = ResidualBlock(512, dropout=0.35)
        self.bottleneck = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.3),
        )
        self.head = nn.Linear(256, n_tasks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.res_block(h)
        h = self.bottleneck(h)
        return self.head(h)


def load_all_data() -> dict:
    """Load RDKit features, per-isozyme train/test splits, and the master training table."""
    rdkit_npz = np.load(DATA_DIR / "rdkit_features.npz")
    rdkit_sids = rdkit_npz["sids"].astype(np.int64)
    sid_to_rdkit_idx = {int(s): i for i, s in enumerate(rdkit_sids)}

    # Pre-build fixed fingerprint block: log1p(ecfp4_counts) + fcfp4_bits + maccs
    fp_all = np.concatenate(
        [
            np.log1p(rdkit_npz["ecfp4_counts"].astype(np.float32)),
            rdkit_npz["fcfp4_bits"].astype(np.float32),
            rdkit_npz["maccs"].astype(np.float32),
        ],
        axis=1,
    )
    rdkit_desc_all = rdkit_npz["rdkit_desc"].astype(np.float32)
    rdkit_desc_all = np.where(np.isfinite(rdkit_desc_all), rdkit_desc_all, np.nan)

    iso_data: dict[str, dict] = {}
    master_labels: dict[int, np.ndarray] = {}
    master_moe: dict[int, np.ndarray] = {}

    for j, iso in enumerate(ISOZYMES):
        tr = pd.read_csv(DATA_DIR / f"training{iso}.csv")
        te = pd.read_csv(DATA_DIR / f"test{iso}.csv")

        feat_cols = [c for c in tr.columns if c not in ("SID", "ActivityScore")]
        sid_tr = tr["SID"].to_numpy(dtype=np.int64)
        sid_te = te["SID"].to_numpy(dtype=np.int64)
        y_tr = (tr["ActivityScore"] >= ACTIVITY_THRESHOLD).to_numpy(dtype=np.int32)
        y_te = (te["ActivityScore"] >= ACTIVITY_THRESHOLD).to_numpy(dtype=np.int32)
        moe_tr = tr[feat_cols].to_numpy(dtype=np.float32)
        moe_te = te[feat_cols].to_numpy(dtype=np.float32)

        iso_data[iso] = {
            "sid_train": sid_tr,
            "sid_test": sid_te,
            "y_train": y_tr,
            "y_test": y_te,
            "moe_train": moe_tr,
            "moe_test": moe_te,
            "rdkit_idx_train": np.array([sid_to_rdkit_idx[int(s)] for s in sid_tr], dtype=np.int64),
            "rdkit_idx_test": np.array([sid_to_rdkit_idx[int(s)] for s in sid_te], dtype=np.int64),
        }

        for row_i, sid in enumerate(sid_tr):
            s = int(sid)
            if s not in master_labels:
                master_labels[s] = np.full(len(ISOZYMES), np.nan, dtype=np.float32)
                master_moe[s] = moe_tr[row_i]
            master_labels[s][j] = float(y_tr[row_i])

    master_sids = np.array(sorted(master_labels.keys()), dtype=np.int64)
    master_y = np.vstack([master_labels[int(s)] for s in master_sids]).astype(np.float32)
    master_moe_arr = np.vstack([master_moe[int(s)] for s in master_sids]).astype(np.float32)
    master_rdkit_idx = np.array([sid_to_rdkit_idx[int(s)] for s in master_sids], dtype=np.int64)

    return {
        "fp_all": fp_all,
        "rdkit_desc_all": rdkit_desc_all,
        "iso_data": iso_data,
        "master_sids": master_sids,
        "master_y": master_y,
        "master_moe": master_moe_arr,
        "master_rdkit_idx": master_rdkit_idx,
    }


def preprocess_fold(
    cont_tr: np.ndarray,
    cont_va: np.ndarray,
    cont_te: np.ndarray,
    fp_tr: np.ndarray,
    fp_va: np.ndarray,
    fp_te: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit SimpleImputer(median) + QuantileTransformer(normal) strictly on fold train pool."""
    imp = SimpleImputer(strategy="median")
    cont_tr_imp = imp.fit_transform(cont_tr)
    cont_va_imp = imp.transform(cont_va)
    cont_te_imp = imp.transform(cont_te)

    n_quantiles = min(1000, len(cont_tr_imp))
    qt = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution="normal",
        random_state=SPLIT_SEED,
    )
    cont_tr_s = np.clip(qt.fit_transform(cont_tr_imp), -8.0, 8.0).astype(np.float32)
    cont_va_s = np.clip(qt.transform(cont_va_imp), -8.0, 8.0).astype(np.float32)
    cont_te_s = np.clip(qt.transform(cont_te_imp), -8.0, 8.0).astype(np.float32)

    X_tr = np.concatenate([fp_tr, cont_tr_s], axis=1)
    X_va = np.concatenate([fp_va, cont_va_s], axis=1)
    X_te = np.concatenate([fp_te, cont_te_s], axis=1)
    return X_tr, X_va, X_te


def train_one_model(
    X_tr: np.ndarray,
    y_tr_mat: np.ndarray,
    X_va: np.ndarray,
    y_va_target: np.ndarray,
    X_te: np.ndarray,
    target_idx: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Train one MultiTaskResMLP with masked BCEWithLogitsLoss and early stopping on val AUC."""
    set_seed(seed)

    mask_tr = (~np.isnan(y_tr_mat)).astype(np.float32)
    y_tr_clean = np.where(np.isnan(y_tr_mat), 0.0, y_tr_mat).astype(np.float32)

    X_tr_t = torch.from_numpy(X_tr).to(DEVICE)
    y_tr_t = torch.from_numpy(y_tr_clean).to(DEVICE)
    m_tr_t = torch.from_numpy(mask_tr).to(DEVICE)
    X_va_t = torch.from_numpy(X_va).to(DEVICE)
    X_te_t = torch.from_numpy(X_te).to(DEVICE)

    task_weights = torch.full(
        (1, len(ISOZYMES)), AUX_TASK_WEIGHT, dtype=torch.float32, device=DEVICE
    )
    task_weights[0, target_idx] = TARGET_TASK_WEIGHT

    model = MultiTaskResMLP(in_dim=X_tr.shape[1], n_tasks=len(ISOZYMES)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-5
    )

    n_samples = len(X_tr)
    best_auc = -1.0
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_samples, device=DEVICE)
        for start in range(0, n_samples, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            xb = X_tr_t[idx]
            yb = y_tr_t[idx]
            mb = m_tr_t[idx]

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            raw_loss = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            w = mb * task_weights
            loss = (raw_loss * w).sum() / w.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            va_logits = model(X_va_t)[:, target_idx]
            va_prob = torch.sigmoid(va_logits).cpu().numpy()
        va_auc = float(roc_auc_score(y_va_target, va_prob))

        if va_auc > best_auc + 1e-5:
            best_auc = va_auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(X_va_t)[:, target_idx]).cpu().numpy().astype(np.float64)
        test_prob = torch.sigmoid(model(X_te_t)[:, target_idx]).cpu().numpy().astype(np.float64)

    return val_prob, test_prob, best_auc, best_epoch


def run_variant_for_isozyme(
    data: dict,
    iso: str,
    use_moe: bool,
    n_seeds: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run 5-fold leak-free multi-task CV + 5-fold ensemble test prediction for one isozyme & feature variant."""
    target_idx = ISOZYMES.index(iso)
    iso_d = data["iso_data"][iso]
    sid_tr = iso_d["sid_train"]
    sid_te = iso_d["sid_test"]
    y_tr = iso_d["y_train"]
    rdkit_idx_tr = iso_d["rdkit_idx_train"]
    rdkit_idx_te = iso_d["rdkit_idx_test"]

    fp_all = data["fp_all"]
    rdkit_desc_all = data["rdkit_desc_all"]
    master_sids = data["master_sids"]
    master_y = data["master_y"]
    master_moe = data["master_moe"]
    master_rdkit_idx = data["master_rdkit_idx"]

    # Step 1: Exclude ALL test{iso} SIDs from the entire 5-head multi-task pool
    test_sid_set = set(sid_te.tolist())
    non_test_mask = np.array([int(s) not in test_sid_set for s in master_sids], dtype=bool)
    pool_sids = master_sids[non_test_mask]
    pool_y = master_y[non_test_mask]
    pool_moe = master_moe[non_test_mask]
    pool_rdkit_idx = master_rdkit_idx[non_test_mask]

    assert len(set(pool_sids.tolist()) & test_sid_set) == 0, "LEAKAGE: test SID in pool!"

    # Step 2: Exact 5-fold StratifiedKFold on training{iso}.csv
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_SEED)
    folds = list(skf.split(np.zeros(len(y_tr)), y_tr))

    oof_prob = np.zeros(len(y_tr), dtype=np.float64)
    test_prob = np.zeros(len(sid_te), dtype=np.float64)

    fp_te = fp_all[rdkit_idx_te]
    if use_moe:
        cont_te = np.concatenate([iso_d["moe_test"], rdkit_desc_all[rdkit_idx_te]], axis=1)
    else:
        cont_te = rdkit_desc_all[rdkit_idx_te]

    best_epochs = []
    fold_times = []

    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        t_f = time.time()
        val_sid_set = set(sid_tr[va_idx].tolist())

        # Step 3: Exclude ALL fold-k validation SIDs from ALL 5 heads in training pool
        fold_tr_mask = np.array([int(s) not in val_sid_set for s in pool_sids], dtype=bool)
        f_sids = pool_sids[fold_tr_mask]
        f_y = pool_y[fold_tr_mask]
        f_rdkit_idx = pool_rdkit_idx[fold_tr_mask]

        assert len(set(f_sids.tolist()) & val_sid_set) == 0, "LEAKAGE: val SID in fold pool!"
        assert len(set(f_sids.tolist()) & test_sid_set) == 0, "LEAKAGE: test SID in fold pool!"

        fp_tr = fp_all[f_rdkit_idx]
        fp_va = fp_all[rdkit_idx_tr[va_idx]]

        if use_moe:
            cont_tr = np.concatenate([pool_moe[fold_tr_mask], rdkit_desc_all[f_rdkit_idx]], axis=1)
            cont_va = np.concatenate(
                [iso_d["moe_train"][va_idx], rdkit_desc_all[rdkit_idx_tr[va_idx]]], axis=1
            )
        else:
            cont_tr = rdkit_desc_all[f_rdkit_idx]
            cont_va = rdkit_desc_all[rdkit_idx_tr[va_idx]]

        X_tr, X_va, X_te = preprocess_fold(cont_tr, cont_va, cont_te, fp_tr, fp_va, fp_te)

        va_pred_fold = np.zeros(len(va_idx), dtype=np.float64)
        te_pred_fold = np.zeros(len(sid_te), dtype=np.float64)
        for s in range(n_seeds):
            seed = 100 * fold_i + s
            vp, tp, _, be = train_one_model(
                X_tr, f_y, X_va, y_tr[va_idx], X_te, target_idx, seed=seed
            )
            va_pred_fold += vp / n_seeds
            te_pred_fold += tp / n_seeds
            best_epochs.append(be)

        oof_prob[va_idx] = va_pred_fold
        test_prob += te_pred_fold / N_SPLITS
        fold_times.append(time.time() - t_f)

    info = {
        "pool_size_non_test": int(len(pool_sids)),
        "mean_best_epoch": float(np.mean(best_epochs)),
        "mean_fold_sec": float(np.mean(fold_times)),
    }
    return oof_prob, test_prob, info


def main() -> None:
    global DATA_DIR, RESULTS_DIR, PREDS_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3, help="Random seeds per fold")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Path to data directory")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="Path to results directory")
    parser.add_argument(
        "--isozymes",
        nargs="+",
        default=ISOZYMES,
        choices=ISOZYMES,
        help="Subset of isozymes to run",
    )
    args = parser.parse_args()
    DATA_DIR = args.data_dir
    RESULTS_DIR = args.results_dir
    PREDS_DIR = RESULTS_DIR / "tier2_mt_preds"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"Running on device: {DEVICE}", flush=True)
    print("Loading datasets and building master multi-task table...", flush=True)
    data = load_all_data()
    print(
        f"Master training pool: {len(data['master_sids'])} unique SIDs | "
        f"fp_dim={data['fp_all'].shape[1]}, rdkit_desc={data['rdkit_desc_all'].shape[1]} | "
        f"seeds/fold={args.seeds}",
        flush=True,
    )

    variants = [
        ("RDKit", "MT_ResMLP_RDKit", False),
        ("MOE+RDKit", "MT_ResMLP_MOE_RDKit", True),
    ]

    summary_rows: list[dict] = []

    for iso in args.isozymes:
        t_iso = time.time()
        iso_d = data["iso_data"][iso]
        y_train = iso_d["y_train"]
        y_test = iso_d["y_test"]
        sid_train = iso_d["sid_train"]
        sid_test = iso_d["sid_test"]

        print(f"\n{'=' * 72}\nCYP {iso.upper()} (train={len(y_train)}, test={len(y_test)})\n{'=' * 72}", flush=True)

        npz_payload: dict[str, np.ndarray] = {
            "y_train": y_train,
            "y_test": y_test,
            "sid_train": sid_train,
            "sid_test": sid_test,
        }

        for feat_set, model_name, use_moe in variants:
            t_v = time.time()
            oof, test_p, info = run_variant_for_isozyme(
                data, iso, use_moe=use_moe, n_seeds=args.seeds
            )
            m = compute_metrics(y_train, oof, y_test, test_p)
            row = {
                "isozyme": iso,
                "feature_set": feat_set,
                "model": model_name,
                **m,
            }
            summary_rows.append(row)
            npz_payload[f"oof_{model_name}"] = oof
            npz_payload[f"test_{model_name}"] = test_p

            print(
                f"  [{iso}] {model_name:20s} | CV ACC={m['cv_acc']:.4f} AUC={m['cv_auc']:.4f} "
                f"| TEST ACC={m['test_acc']:.4f} TPR={m['test_tpr']:.4f} TNR={m['test_tnr']:.4f} "
                f"AUC={m['test_auc']:.4f} Brier={m['test_brier']:.4f} LL={m['test_logloss']:.4f} "
                f"| pool={info['pool_size_non_test']} ep={info['mean_best_epoch']:.1f} "
                f"({time.time() - t_v:.1f}s)",
                flush=True,
            )

        np.savez_compressed(PREDS_DIR / f"{iso}.npz", **npz_payload)
        pd.DataFrame(summary_rows).to_csv(
            RESULTS_DIR / "tier2_multitask_summary.csv", index=False
        )
        print(
            f"Saved {PREDS_DIR / f'{iso}.npz'} ({time.time() - t_iso:.1f}s)",
            flush=True,
        )

    print(
        f"\nAll done in {(time.time() - t0) / 60:.2f} min. "
        f"Summary -> {RESULTS_DIR / 'tier2_multitask_summary.csv'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
