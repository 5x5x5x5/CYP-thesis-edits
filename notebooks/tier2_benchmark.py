"""Tier 2A Benchmark: Cheminformatics (RDKit Fingerprints + Descriptors),
Combined MOE+RDKit Models, Hybrid Tanimoto+RBF Kernel SVM, and Scaffold-Split Diagnostic
across all 5 CYP isozymes (1a2, 2c9, 2c19, 2d6, 3a4).

Strict zero-leakage validation protocol:
- Exact Figshare splits: notebooks/data/training{iso}.csv and notebooks/data/test{iso}.csv
- Evaluate on 100% of every test set (missing-SMILES rows have 0 FP bits and train-median-imputed rdkit_desc)
- Binary target: ActivityScore >= 40
- Internal validation: 5-fold StratifiedKFold(n_splits=5, shuffle=True, random_state=0) on train ONLY
- All imputers, scalers, and rare-bit filters fit strictly inside each fold (and on full train for test)
- Never use test set for hyperparameter selection, early stopping, or stacking weights
"""

from __future__ import annotations

import gc
import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from joblib import Parallel, delayed
from scipy.special import logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
PREDS_DIR = RESULTS_DIR / "tier2_preds"

ISOZYMES = ["1a2", "2c9", "2c19", "2d6", "3a4"]
ACTIVITY_THRESHOLD = 40
SEED = 0
N_SPLITS = 5
N_JOBS = 4
MIN_BIT_COUNT = 5
DESC_CLIP = 1e12


class RDKitStore:
    """In-memory compact store for precomputed RDKit features and Murcko scaffolds."""

    def __init__(self) -> None:
        npz = np.load(DATA_DIR / "rdkit_features.npz", allow_pickle=True)
        self.sids = npz["sids"].astype(np.int64)
        self.sid_to_idx = {int(s): i for i, s in enumerate(self.sids)}
        self.ecfp4_bits = npz["ecfp4_bits"].astype(np.uint8)
        self.ecfp4_counts = npz["ecfp4_counts"].astype(np.uint8)
        self.fcfp4_bits = npz["fcfp4_bits"].astype(np.uint8)
        self.maccs = npz["maccs"].astype(np.uint8)
        self.rdkit_fp = npz["rdkit_fp"].astype(np.uint8)

        # Clip extreme finite descriptor values (e.g. Ipc ~ 1e24) while preserving NaNs
        rd = npz["rdkit_desc"].astype(np.float32)
        finite = np.isfinite(rd)
        rd[finite] = np.clip(rd[finite], -DESC_CLIP, DESC_CLIP)
        rd[~finite] = np.nan
        self.rdkit_desc = rd

        sm = pd.read_csv(DATA_DIR / "sid_smiles.csv", usecols=["SID", "ScaffoldSMILES"])
        scaf_map: dict[int, str] = {}
        for sid, scaf in zip(sm["SID"].to_numpy(), sm["ScaffoldSMILES"].to_numpy()):
            sid_int = int(sid)
            if pd.isna(scaf) or str(scaf).strip() == "":
                # Treat acyclic / missing-SMILES molecules as singleton scaffolds
                scaf_map[sid_int] = f"SINGLETON_{sid_int}"
            else:
                scaf_map[sid_int] = str(scaf)
        self.scaf_map = scaf_map

    def idx_for(self, sids: np.ndarray) -> np.ndarray:
        return np.array([self.sid_to_idx[int(s)] for s in sids], dtype=np.int64)

    def scaffolds_for(self, sids: np.ndarray) -> np.ndarray:
        return np.array([self.scaf_map[int(s)] for s in sids], dtype=object)


def load_isozyme_data(isozyme: str, store: RDKitStore) -> dict[str, np.ndarray]:
    """Load exact Figshare splits aligned with RDKit features and scaffolds."""
    tr = pd.read_csv(DATA_DIR / f"training{isozyme}.csv")
    te = pd.read_csv(DATA_DIR / f"test{isozyme}.csv")

    sid_tr = tr["SID"].to_numpy(dtype=np.int64)
    sid_te = te["SID"].to_numpy(dtype=np.int64)
    y_tr = (tr["ActivityScore"] >= ACTIVITY_THRESHOLD).to_numpy(dtype=np.int32)
    y_te = (te["ActivityScore"] >= ACTIVITY_THRESHOLD).to_numpy(dtype=np.int32)

    moe_cols = [c for c in tr.columns if c not in ("SID", "ActivityScore")]
    X_moe_tr = tr[moe_cols].to_numpy(dtype=np.float32)
    X_moe_te = te[moe_cols].to_numpy(dtype=np.float32)

    idx_tr = store.idx_for(sid_tr)
    idx_te = store.idx_for(sid_te)

    return {
        "sid_tr": sid_tr,
        "sid_te": sid_te,
        "y_tr": y_tr,
        "y_te": y_te,
        "moe_tr": X_moe_tr,
        "moe_te": X_moe_te,
        "rd_tr": store.rdkit_desc[idx_tr],
        "rd_te": store.rdkit_desc[idx_te],
        "ecfp4_bits_tr": store.ecfp4_bits[idx_tr],
        "ecfp4_bits_te": store.ecfp4_bits[idx_te],
        "ecfp4_counts_tr": store.ecfp4_counts[idx_tr],
        "ecfp4_counts_te": store.ecfp4_counts[idx_te],
        "fcfp4_bits_tr": store.fcfp4_bits[idx_tr],
        "fcfp4_bits_te": store.fcfp4_bits[idx_te],
        "maccs_tr": store.maccs[idx_tr],
        "maccs_te": store.maccs[idx_te],
        "rdkit_fp_tr": store.rdkit_fp[idx_tr],
        "rdkit_fp_te": store.rdkit_fp[idx_te],
        "scaffolds_tr": store.scaffolds_for(sid_tr),
    }


def compute_metrics(
    y_train: np.ndarray,
    oof_prob: np.ndarray,
    y_test: np.ndarray,
    test_prob: np.ndarray,
) -> dict[str, float]:
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


def compute_cv_only_metrics(y: np.ndarray, oof_prob: np.ndarray) -> dict[str, float]:
    p = np.clip(oof_prob, 1e-7, 1.0 - 1e-7)
    pred = (p >= 0.5).astype(int)
    return {
        "cv_acc": float(accuracy_score(y, pred)),
        "cv_auc": float(roc_auc_score(y, p)),
        "cv_brier": float(brier_score_loss(y, p)),
        "cv_logloss": float(log_loss(y, p)),
    }


def tanimoto_kernel_binary(
    A_uint8: np.ndarray, B_uint8: np.ndarray, is_self: bool = False
) -> np.ndarray:
    """Exact Tanimoto kernel on binary {0, 1} matrices via float32 GEMM (in-place to minimize RAM)."""
    A = A_uint8.astype(np.float32)
    sum_a = A.sum(axis=1, dtype=np.float32)
    if is_self:
        K = A @ A.T
        del A
        denom = sum_a[:, None] + sum_a[None, :]
        denom -= K
    else:
        B = B_uint8.astype(np.float32)
        sum_b = B.sum(axis=1, dtype=np.float32)
        K = A @ B.T
        del A, B
        denom = sum_a[:, None] + sum_b[None, :]
        denom -= K
    mask = denom > 0.0
    np.divide(K, denom, out=K, where=mask)
    K[~mask] = 0.0
    del denom, mask
    if is_self:
        # Ensure exact symmetry and unit diagonal (PSD even for all-zero rows)
        K += K.T
        K *= 0.5
        np.fill_diagonal(K, 1.0)
    gc.collect()
    return K


def _prepare_fold_features(
    fp_tr: np.ndarray | None,
    fp_va: np.ndarray | None,
    desc_tr: np.ndarray | None,
    desc_va: np.ndarray | None,
    moe_tr: np.ndarray | None = None,
    moe_va: np.ndarray | None = None,
    min_bit_count: int = MIN_BIT_COUNT,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit rare-bit filter and median imputer STRICTLY on train fold, transform train & val/test."""
    parts_tr: list[np.ndarray] = []
    parts_va: list[np.ndarray] = []

    if moe_tr is not None and moe_va is not None:
        parts_tr.append(moe_tr.astype(np.float32, copy=False))
        parts_va.append(moe_va.astype(np.float32, copy=False))

    if desc_tr is not None and desc_va is not None:
        imp = SimpleImputer(strategy="median")
        d_tr = imp.fit_transform(desc_tr).astype(np.float32, copy=False)
        d_va = imp.transform(desc_va).astype(np.float32, copy=False)
        parts_tr.append(d_tr)
        parts_va.append(d_va)

    if fp_tr is not None and fp_va is not None:
        keep = (fp_tr > 0).sum(axis=0) >= min_bit_count
        parts_tr.append(fp_tr[:, keep].astype(np.float32, copy=False))
        parts_va.append(fp_va[:, keep].astype(np.float32, copy=False))

    X_tr = np.hstack(parts_tr) if len(parts_tr) > 1 else parts_tr[0]
    X_va = np.hstack(parts_va) if len(parts_va) > 1 else parts_va[0]
    return X_tr, X_va


def run_tanimoto_svm(
    K_tr_tr: np.ndarray,
    K_te_tr: np.ndarray | None,
    y_train: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    C_grid: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Tune C on 5-fold CV with probability=False, then fit winning C with probability=True.
    Processes one fold at a time to keep peak RAM strictly under ~1.2 GiB.
    """
    if C_grid is None:
        C_grid = [0.5, 1.0, 2.0, 5.0]

    # Fold-outer loop: allocate one fold's float64 sub-kernel at a time
    oof_df_map = {C: np.zeros(len(y_train), dtype=np.float64) for C in C_grid}
    for tr_idx, va_idx in folds:
        K_f_tr = np.ascontiguousarray(K_tr_tr[np.ix_(tr_idx, tr_idx)], dtype=np.float64)
        K_f_va = np.ascontiguousarray(K_tr_tr[np.ix_(va_idx, tr_idx)], dtype=np.float64)
        y_f_tr = y_train[tr_idx]
        for C in C_grid:
            svc = SVC(
                C=C,
                kernel="precomputed",
                probability=False,
                cache_size=100,
                random_state=SEED,
            )
            svc.fit(K_f_tr, y_f_tr)
            oof_df_map[C][va_idx] = svc.decision_function(K_f_va)
            del svc
        del K_f_tr, K_f_va
        gc.collect()

    best_auc, best_C = max(
        ((float(roc_auc_score(y_train, scores)), C) for C, scores in oof_df_map.items()),
        key=lambda r: r[0],
    )

    # Sequential over folds with probability=True to prevent multi-fold RAM spikes
    oof = np.zeros(len(y_train), dtype=np.float64)
    for tr_idx, va_idx in folds:
        K_f_tr = np.ascontiguousarray(K_tr_tr[np.ix_(tr_idx, tr_idx)], dtype=np.float64)
        K_f_va = np.ascontiguousarray(K_tr_tr[np.ix_(va_idx, tr_idx)], dtype=np.float64)
        svc = SVC(
            C=best_C,
            kernel="precomputed",
            probability=True,
            cache_size=100,
            random_state=SEED,
        )
        svc.fit(K_f_tr, y_train[tr_idx])
        oof[va_idx] = svc.predict_proba(K_f_va)[:, 1]
        del K_f_tr, K_f_va, svc
        gc.collect()

    test_prob = None
    if K_te_tr is not None:
        K_full = np.ascontiguousarray(K_tr_tr, dtype=np.float64)
        K_te = np.ascontiguousarray(K_te_tr, dtype=np.float64)
        full_svc = SVC(
            C=best_C,
            kernel="precomputed",
            probability=True,
            cache_size=150,
            random_state=SEED,
        )
        full_svc.fit(K_full, y_train)
        test_prob = full_svc.predict_proba(K_te)[:, 1]
        del K_full, K_te, full_svc
        gc.collect()

    params = {"C": best_C, "tune_df_auc": round(best_auc, 5)}
    return oof, test_prob, params


def run_hybrid_kernel_svm(
    K_tan_tr: np.ndarray,
    K_tan_te: np.ndarray,
    cont_tr: np.ndarray,
    cont_te: np.ndarray,
    y_train: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Hybrid kernel K = alpha * K_tanimoto + (1 - alpha) * K_rbf (QuantileTransformer-scaled MOE+rdkit_desc).
    All imputers and scalers are fit strictly inside each fold. Sequential over folds to cap RAM.
    """
    alphas = [0.3, 0.5, 0.7]
    C_grid = [1.0, 3.0, 10.0]
    oof_df_map: dict[tuple[float, float], np.ndarray] = {
        (a, c): np.zeros(len(y_train), dtype=np.float64) for a in alphas for c in C_grid
    }

    def _compute_rbf_fold(X_tr_raw: np.ndarray, X_va_raw: np.ndarray):
        imp = SimpleImputer(strategy="median")
        qt = QuantileTransformer(output_distribution="normal", random_state=SEED)
        X_tr_s = qt.fit_transform(imp.fit_transform(X_tr_raw)).astype(np.float64)
        X_va_s = qt.transform(imp.transform(X_va_raw)).astype(np.float64)
        gamma = 1.0 / (X_tr_s.shape[1] * max(float(X_tr_s.var()), 1e-12))
        K_rbf_tr = rbf_kernel(X_tr_s, X_tr_s, gamma=gamma).astype(np.float32)
        K_rbf_va = rbf_kernel(X_va_s, X_tr_s, gamma=gamma).astype(np.float32)
        return K_rbf_tr, K_rbf_va

    # Fold-outer loop so we only hold ONE fold's RBF kernel in RAM at a time
    for tr_idx, va_idx in folds:
        K_rbf_tr, K_rbf_va = _compute_rbf_fold(cont_tr[tr_idx], cont_tr[va_idx])
        K_tan_f_tr = K_tan_tr[np.ix_(tr_idx, tr_idx)]
        K_tan_f_va = K_tan_tr[np.ix_(va_idx, tr_idx)]
        y_f_tr = y_train[tr_idx]

        for alpha in alphas:
            K_hyb_tr = np.ascontiguousarray(
                alpha * K_tan_f_tr + (1.0 - alpha) * K_rbf_tr, dtype=np.float64
            )
            K_hyb_va = np.ascontiguousarray(
                alpha * K_tan_f_va + (1.0 - alpha) * K_rbf_va, dtype=np.float64
            )
            for C in C_grid:
                svc = SVC(
                    C=C,
                    kernel="precomputed",
                    probability=False,
                    cache_size=100,
                    random_state=SEED,
                )
                svc.fit(K_hyb_tr, y_f_tr)
                oof_df_map[(alpha, C)][va_idx] = svc.decision_function(K_hyb_va)
                del svc
            del K_hyb_tr, K_hyb_va
            gc.collect()

        del K_rbf_tr, K_rbf_va, K_tan_f_tr, K_tan_f_va
        gc.collect()

    best_auc = -1.0
    best_alpha, best_C = 0.5, 3.0
    for (alpha, C), scores in oof_df_map.items():
        auc = float(roc_auc_score(y_train, scores))
        if auc > best_auc:
            best_auc = auc
            best_alpha, best_C = alpha, C

    # Refit winning (alpha, C) with probability=True for OOF (sequential over folds)
    oof = np.zeros(len(y_train), dtype=np.float64)
    for tr_idx, va_idx in folds:
        K_rbf_tr, K_rbf_va = _compute_rbf_fold(cont_tr[tr_idx], cont_tr[va_idx])
        K_hyb_tr = K_rbf_tr.astype(np.float64)
        del K_rbf_tr
        K_hyb_tr *= (1.0 - best_alpha)
        K_hyb_tr += best_alpha * K_tan_tr[np.ix_(tr_idx, tr_idx)]

        K_hyb_va = K_rbf_va.astype(np.float64)
        del K_rbf_va
        K_hyb_va *= (1.0 - best_alpha)
        K_hyb_va += best_alpha * K_tan_tr[np.ix_(va_idx, tr_idx)]

        svc = SVC(
            C=best_C,
            kernel="precomputed",
            probability=True,
            cache_size=100,
            random_state=SEED,
        )
        svc.fit(K_hyb_tr, y_train[tr_idx])
        oof[va_idx] = svc.predict_proba(K_hyb_va)[:, 1]
        del K_hyb_tr, K_hyb_va, svc
        gc.collect()

    # Full-train fit for test predictions (in-place assembly to avoid 2x float64 peak)
    K_rbf_full, K_rbf_te = _compute_rbf_fold(cont_tr, cont_te)
    K_hyb_full = K_rbf_full.astype(np.float64)
    del K_rbf_full
    K_hyb_full *= (1.0 - best_alpha)
    K_hyb_full += best_alpha * K_tan_tr

    K_hyb_te = K_rbf_te.astype(np.float64)
    del K_rbf_te
    K_hyb_te *= (1.0 - best_alpha)
    K_hyb_te += best_alpha * K_tan_te
    gc.collect()

    full_svc = SVC(
        C=best_C,
        kernel="precomputed",
        probability=True,
        cache_size=150,
        random_state=SEED,
    )
    full_svc.fit(K_hyb_full, y_train)
    test_prob = full_svc.predict_proba(K_hyb_te)[:, 1]
    del K_hyb_full, K_hyb_te, full_svc
    gc.collect()

    params = {"alpha": best_alpha, "C": best_C, "tune_df_auc": round(best_auc, 5)}
    return oof, test_prob, params


def run_lightgbm_cv(
    fp_tr: np.ndarray | None,
    fp_te: np.ndarray | None,
    desc_tr: np.ndarray | None,
    desc_te: np.ndarray | None,
    moe_tr: np.ndarray | None,
    moe_te: np.ndarray | None,
    y_train: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    colsample: float = 0.25,
    reg_alpha: float = 0.5,
    reg_lambda: float = 2.0,
    fit_test: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    base_kwargs = dict(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=colsample,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        n_jobs=N_JOBS,
        random_state=SEED,
        verbose=-1,
    )
    oof = np.zeros(len(y_train), dtype=np.float64)
    best_iters: list[int] = []
    for tr_idx, va_idx in folds:
        X_f_tr, X_f_va = _prepare_fold_features(
            fp_tr[tr_idx] if fp_tr is not None else None,
            fp_tr[va_idx] if fp_tr is not None else None,
            desc_tr[tr_idx] if desc_tr is not None else None,
            desc_tr[va_idx] if desc_tr is not None else None,
            moe_tr[tr_idx] if moe_tr is not None else None,
            moe_tr[va_idx] if moe_tr is not None else None,
        )
        clf = lgb.LGBMClassifier(**base_kwargs)
        clf.fit(
            X_f_tr,
            y_train[tr_idx],
            eval_set=[(X_f_va, y_train[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        best_iter = int(clf.best_iteration_ or base_kwargs["n_estimators"])
        best_iters.append(best_iter)
        oof[va_idx] = clf.predict_proba(X_f_va, num_iteration=best_iter)[:, 1]
        del X_f_tr, X_f_va, clf

    full_n_est = max(50, int(round(float(np.median(best_iters)) * 1.1)))
    test_prob = None
    if fit_test:
        X_full, X_te = _prepare_fold_features(
            fp_tr, fp_te, desc_tr, desc_te, moe_tr, moe_te
        )
        full_clf = lgb.LGBMClassifier(**{**base_kwargs, "n_estimators": full_n_est})
        full_clf.fit(X_full, y_train)
        test_prob = full_clf.predict_proba(X_te)[:, 1]
        del X_full, X_te, full_clf
    gc.collect()
    return oof, test_prob, {
        "median_best_iter": int(np.median(best_iters)),
        "full_n_estimators": full_n_est,
    }


def run_xgboost_cv(
    fp_tr: np.ndarray,
    fp_te: np.ndarray,
    desc_tr: np.ndarray,
    desc_te: np.ndarray,
    moe_tr: np.ndarray | None,
    moe_te: np.ndarray | None,
    y_train: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    base_kwargs = dict(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.25,
        reg_alpha=0.5,
        reg_lambda=2.0,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=N_JOBS,
        random_state=SEED,
    )
    oof = np.zeros(len(y_train), dtype=np.float64)
    best_iters: list[int] = []
    for tr_idx, va_idx in folds:
        X_f_tr, X_f_va = _prepare_fold_features(
            fp_tr[tr_idx],
            fp_tr[va_idx],
            desc_tr[tr_idx],
            desc_tr[va_idx],
            moe_tr[tr_idx] if moe_tr is not None else None,
            moe_tr[va_idx] if moe_tr is not None else None,
        )
        clf = XGBClassifier(**base_kwargs, early_stopping_rounds=50)
        clf.fit(
            X_f_tr,
            y_train[tr_idx],
            eval_set=[(X_f_va, y_train[va_idx])],
            verbose=False,
        )
        best_iter = int(clf.best_iteration) + 1
        best_iters.append(best_iter)
        oof[va_idx] = clf.predict_proba(X_f_va)[:, 1]
        del X_f_tr, X_f_va, clf

    full_n_est = max(50, int(round(float(np.median(best_iters)) * 1.1)))
    X_full, X_te = _prepare_fold_features(
        fp_tr, fp_te, desc_tr, desc_te, moe_tr, moe_te
    )
    full_clf = XGBClassifier(**{**base_kwargs, "n_estimators": full_n_est})
    full_clf.fit(X_full, y_train, verbose=False)
    test_prob = full_clf.predict_proba(X_te)[:, 1]
    del X_full, X_te, full_clf
    gc.collect()
    return oof, test_prob, {
        "median_best_iter": int(np.median(best_iters)),
        "full_n_estimators": full_n_est,
    }


def run_catboost_cv(
    fp_tr: np.ndarray,
    fp_te: np.ndarray,
    desc_tr: np.ndarray,
    desc_te: np.ndarray,
    moe_tr: np.ndarray | None,
    moe_te: np.ndarray | None,
    y_train: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    base_kwargs = dict(
        iterations=2000,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=5.0,
        rsm=0.25,
        eval_metric="Logloss",
        thread_count=N_JOBS,
        random_seed=SEED,
        verbose=0,
    )
    oof = np.zeros(len(y_train), dtype=np.float64)
    best_iters: list[int] = []
    for tr_idx, va_idx in folds:
        X_f_tr, X_f_va = _prepare_fold_features(
            fp_tr[tr_idx],
            fp_tr[va_idx],
            desc_tr[tr_idx],
            desc_tr[va_idx],
            moe_tr[tr_idx] if moe_tr is not None else None,
            moe_tr[va_idx] if moe_tr is not None else None,
        )
        clf = CatBoostClassifier(**base_kwargs, early_stopping_rounds=80)
        clf.fit(
            X_f_tr,
            y_train[tr_idx],
            eval_set=(X_f_va, y_train[va_idx]),
            use_best_model=True,
        )
        best_iter = int(clf.get_best_iteration()) + 1
        best_iters.append(best_iter)
        oof[va_idx] = clf.predict_proba(X_f_va)[:, 1]
        del X_f_tr, X_f_va, clf

    full_iters = max(50, int(round(float(np.median(best_iters)) * 1.1)))
    X_full, X_te = _prepare_fold_features(
        fp_tr, fp_te, desc_tr, desc_te, moe_tr, moe_te
    )
    full_clf = CatBoostClassifier(**{**base_kwargs, "iterations": full_iters})
    full_clf.fit(X_full, y_train)
    test_prob = full_clf.predict_proba(X_te)[:, 1]
    del X_full, X_te, full_clf
    gc.collect()
    return oof, test_prob, {
        "median_best_iter": int(np.median(best_iters)),
        "full_iterations": full_iters,
    }


def _safe_logit(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(p, 1e-6, 1.0 - 1e-6))


def run_stack_logreg(
    model_names: list[str],
    oof_dict: dict[str, np.ndarray],
    test_dict: dict[str, np.ndarray],
    y_train: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """L2 LogisticRegressionCV on OOF logits with nested OOF for honest CV."""
    X_oof = np.column_stack([_safe_logit(oof_dict[m]) for m in model_names])
    X_te = np.column_stack([_safe_logit(test_dict[m]) for m in model_names])

    Cs = np.logspace(-3, 2, 20)
    nested_oof = np.zeros(len(y_train), dtype=np.float64)
    for tr_idx, va_idx in folds:
        meta = LogisticRegressionCV(
            Cs=Cs,
            cv=5,
            scoring="neg_log_loss",
            max_iter=1000,
            random_state=SEED,
        )
        meta.fit(X_oof[tr_idx], y_train[tr_idx])
        nested_oof[va_idx] = meta.predict_proba(X_oof[va_idx])[:, 1]

    full_meta = LogisticRegressionCV(
        Cs=Cs,
        cv=5,
        scoring="neg_log_loss",
        max_iter=1000,
        random_state=SEED,
    )
    full_meta.fit(X_oof, y_train)
    test_prob = full_meta.predict_proba(X_te)[:, 1]

    coefs = {m: round(float(c), 4) for m, c in zip(model_names, full_meta.coef_[0])}
    params = {"C": round(float(full_meta.C_[0]), 5), "coefs": coefs}
    return nested_oof, test_prob, params


def benchmark_isozyme(
    isozyme: str, store: RDKitStore
) -> tuple[list[dict], list[dict]]:
    t0 = time.time()
    print(f"\n{'='*76}\nTier 2A Benchmark: CYP {isozyme.upper()}\n{'='*76}", flush=True)
    d = load_isozyme_data(isozyme, store)
    y_tr, y_te = d["y_tr"], d["y_te"]

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(skf.split(d["moe_tr"], y_tr))

    oof_preds: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}
    rows: list[dict] = []

    def _record(
        feature_set: str,
        model_name: str,
        oof: np.ndarray,
        test_p: np.ndarray,
        params: dict,
        dt: float,
    ) -> None:
        oof_preds[model_name] = oof.astype(np.float64)
        test_preds[model_name] = test_p.astype(np.float64)
        m = compute_metrics(y_tr, oof, y_te, test_p)
        rows.append(
            {
                "isozyme": isozyme,
                "feature_set": feature_set,
                "model": model_name,
                **m,
                "best_params": json.dumps(params, sort_keys=True),
            }
        )
        print(
            f"  [{isozyme}] {feature_set:9s} | {model_name:19s} | "
            f"CV ACC={m['cv_acc']:.4f} AUC={m['cv_auc']:.4f} | "
            f"TEST ACC={m['test_acc']:.4f} AUC={m['test_auc']:.4f} "
            f"Brier={m['test_brier']:.4f} LL={m['test_logloss']:.4f} ({dt:.1f}s)",
            flush=True,
        )

    # Precompute exact binary Tanimoto kernel once per isozyme (ecfp4_bits + fcfp4_bits + maccs + rdkit_fp)
    t_k = time.time()
    bin_tr = np.hstack(
        [d["ecfp4_bits_tr"], d["fcfp4_bits_tr"], d["maccs_tr"], d["rdkit_fp_tr"]]
    )
    bin_te = np.hstack(
        [d["ecfp4_bits_te"], d["fcfp4_bits_te"], d["maccs_te"], d["rdkit_fp_te"]]
    )
    K_tan_tr = tanimoto_kernel_binary(bin_tr, bin_tr, is_self=True)
    K_tan_te = tanimoto_kernel_binary(bin_te, bin_tr, is_self=False)
    del bin_tr, bin_te
    gc.collect()
    print(f"  [{isozyme}] Computed Tanimoto kernel {K_tan_tr.shape} in {time.time() - t_k:.1f}s", flush=True)

    # Compact FP blocks (uint8)
    fp_tree_tr = np.hstack([d["ecfp4_counts_tr"], d["fcfp4_bits_tr"], d["maccs_tr"]])
    fp_tree_te = np.hstack([d["ecfp4_counts_te"], d["fcfp4_bits_te"], d["maccs_te"]])
    fp_cb_tr = np.hstack([d["ecfp4_bits_tr"], d["maccs_tr"]])
    fp_cb_te = np.hstack([d["ecfp4_bits_te"], d["maccs_te"]])

    # =========================================================================
    # PART A: 100% Open-Source RDKit-Only
    # =========================================================================
    t = time.time()
    oof, tp, params = run_tanimoto_svm(K_tan_tr, K_tan_te, y_tr, folds)
    _record("RDKit", "Tanimoto_SVM_RDKit", oof, tp, params, time.time() - t)

    t = time.time()
    oof, tp, params = run_lightgbm_cv(
        fp_tree_tr, fp_tree_te, d["rd_tr"], d["rd_te"], None, None, y_tr, folds
    )
    _record("RDKit", "LightGBM_RDKit", oof, tp, params, time.time() - t)

    t = time.time()
    oof, tp, params = run_xgboost_cv(
        fp_tree_tr, fp_tree_te, d["rd_tr"], d["rd_te"], None, None, y_tr, folds
    )
    _record("RDKit", "XGBoost_RDKit", oof, tp, params, time.time() - t)

    t = time.time()
    oof, tp, params = run_catboost_cv(
        fp_cb_tr, fp_cb_te, d["rd_tr"], d["rd_te"], None, None, y_tr, folds
    )
    _record("RDKit", "CatBoost_RDKit", oof, tp, params, time.time() - t)

    t = time.time()
    rdkit_stack_members = [
        "Tanimoto_SVM_RDKit",
        "LightGBM_RDKit",
        "XGBoost_RDKit",
        "CatBoost_RDKit",
    ]
    oof, tp, params = run_stack_logreg(
        rdkit_stack_members, oof_preds, test_preds, y_tr, folds
    )
    _record("RDKit", "Stack_RDKit", oof, tp, params, time.time() - t)

    # =========================================================================
    # PART B: Combined MOE + RDKit
    # =========================================================================
    t = time.time()
    oof, tp, params = run_lightgbm_cv(
        fp_tree_tr, fp_tree_te, d["rd_tr"], d["rd_te"], d["moe_tr"], d["moe_te"], y_tr, folds
    )
    _record("MOE+RDKit", "LightGBM_MOE_RDKit", oof, tp, params, time.time() - t)

    t = time.time()
    oof, tp, params = run_xgboost_cv(
        fp_tree_tr, fp_tree_te, d["rd_tr"], d["rd_te"], d["moe_tr"], d["moe_te"], y_tr, folds
    )
    _record("MOE+RDKit", "XGBoost_MOE_RDKit", oof, tp, params, time.time() - t)

    t = time.time()
    oof, tp, params = run_catboost_cv(
        fp_cb_tr, fp_cb_te, d["rd_tr"], d["rd_te"], d["moe_tr"], d["moe_te"], y_tr, folds
    )
    _record("MOE+RDKit", "CatBoost_MOE_RDKit", oof, tp, params, time.time() - t)

    t = time.time()
    cont_tr = np.hstack([d["moe_tr"], d["rd_tr"]])
    cont_te = np.hstack([d["moe_te"], d["rd_te"]])
    oof, tp, params = run_hybrid_kernel_svm(
        K_tan_tr, K_tan_te, cont_tr, cont_te, y_tr, folds
    )
    del cont_tr, cont_te
    gc.collect()
    _record("MOE+RDKit", "HybridKernel_SVM", oof, tp, params, time.time() - t)

    t = time.time()
    moe_rdkit_stack_members = [
        "LightGBM_MOE_RDKit",
        "XGBoost_MOE_RDKit",
        "CatBoost_MOE_RDKit",
        "HybridKernel_SVM",
        "Tanimoto_SVM_RDKit",
    ]
    oof, tp, params = run_stack_logreg(
        moe_rdkit_stack_members, oof_preds, test_preds, y_tr, folds
    )
    _record("MOE+RDKit", "Stack_MOE_RDKit", oof, tp, params, time.time() - t)

    # Save Tier 2 predictions npz
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    npz_payload = {
        "y_train": y_tr,
        "y_test": y_te,
        "sid_train": d["sid_tr"],
        "sid_test": d["sid_te"],
    }
    for m_name in oof_preds:
        npz_payload[f"oof_{m_name}"] = oof_preds[m_name]
        npz_payload[f"test_{m_name}"] = test_preds[m_name]
    np.savez_compressed(PREDS_DIR / f"{isozyme}.npz", **npz_payload)

    # =========================================================================
    # PART C: Scaffold-Split Generalization Diagnostic (5-fold GroupKFold vs StratifiedKFold)
    # =========================================================================
    t_sc = time.time()
    scaffolds = d["scaffolds_tr"]
    n_scaffolds = int(len(np.unique(scaffolds)))
    gkf = GroupKFold(n_splits=N_SPLITS)
    scaf_folds = list(gkf.split(d["moe_tr"], y_tr, groups=scaffolds))

    scaf_rows: list[dict] = []

    # 1. LightGBM_MOE (random stratified + scaffold)
    oof_lgb_moe_rand, _, _ = run_lightgbm_cv(
        None, None, None, None, d["moe_tr"], None, y_tr, folds,
        colsample=0.6, reg_alpha=0.1, reg_lambda=1.0, fit_test=False,
    )
    oof_lgb_moe_scaf, _, _ = run_lightgbm_cv(
        None, None, None, None, d["moe_tr"], None, y_tr, scaf_folds,
        colsample=0.6, reg_alpha=0.1, reg_lambda=1.0, fit_test=False,
    )

    # 2. LightGBM_RDKit (reuse random OOF from Part A + run scaffold)
    oof_lgb_rd_rand = oof_preds["LightGBM_RDKit"]
    oof_lgb_rd_scaf, _, _ = run_lightgbm_cv(
        fp_tree_tr, None, d["rd_tr"], None, None, None, y_tr, scaf_folds,
        fit_test=False,
    )

    # 3. Tanimoto_SVM_RDKit (reuse random OOF from Part A + run scaffold)
    oof_tan_rand = oof_preds["Tanimoto_SVM_RDKit"]
    oof_tan_scaf, _, _ = run_tanimoto_svm(K_tan_tr, None, y_tr, scaf_folds)

    for m_name, oof_r, oof_s in [
        ("LightGBM_MOE", oof_lgb_moe_rand, oof_lgb_moe_scaf),
        ("LightGBM_RDKit", oof_lgb_rd_rand, oof_lgb_rd_scaf),
        ("Tanimoto_SVM_RDKit", oof_tan_rand, oof_tan_scaf),
    ]:
        mr = compute_cv_only_metrics(y_tr, oof_r)
        ms = compute_cv_only_metrics(y_tr, oof_s)
        scaf_rows.append(
            {
                "isozyme": isozyme,
                "model": m_name,
                "split_type": "random_stratified",
                **mr,
                "n_train": int(len(y_tr)),
                "n_scaffolds": n_scaffolds,
            }
        )
        scaf_rows.append(
            {
                "isozyme": isozyme,
                "model": m_name,
                "split_type": "scaffold_group",
                **ms,
                "n_train": int(len(y_tr)),
                "n_scaffolds": n_scaffolds,
            }
        )
        print(
            f"  [{isozyme}] SCAFFOLD | {m_name:19s} | "
            f"Rand ACC={mr['cv_acc']:.4f} AUC={mr['cv_auc']:.4f} -> "
            f"Scaf ACC={ms['cv_acc']:.4f} AUC={ms['cv_auc']:.4f} "
            f"(dAUC={ms['cv_auc'] - mr['cv_auc']:+.4f})",
            flush=True,
        )

    print(
        f"  [{isozyme}] Scaffold diagnostic done in {time.time() - t_sc:.1f}s | "
        f"Total isozyme time: {time.time() - t0:.1f}s",
        flush=True,
    )

    # Explicitly release large isozyme arrays
    del K_tan_tr, K_tan_te, fp_tree_tr, fp_tree_te, fp_cb_tr, fp_cb_te, d
    gc.collect()

    return rows, scaf_rows


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDS_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    store = RDKitStore()

    all_rows: list[dict] = []
    all_scaf_rows: list[dict] = []

    for iso in ISOZYMES:
        iso_rows, iso_scaf = benchmark_isozyme(iso, store)
        all_rows.extend(iso_rows)
        all_scaf_rows.extend(iso_scaf)
        pd.DataFrame(all_rows).to_csv(RESULTS_DIR / "tier2_summary.csv", index=False)
        pd.DataFrame(all_scaf_rows).to_csv(
            RESULTS_DIR / "scaffold_cv_summary.csv", index=False
        )

    print(
        f"\nAll 5 isozymes finished in {(time.time() - t_start) / 60:.2f} min.",
        flush=True,
    )
    print(f"Summary saved to {RESULTS_DIR / 'tier2_summary.csv'}", flush=True)
    print(f"Scaffold CV saved to {RESULTS_DIR / 'scaffold_cv_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
