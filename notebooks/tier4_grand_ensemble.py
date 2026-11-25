"""Tier 4: Grand Multi-Modal Meta-Stacking, Calibration, & Statistical Significance Suite.

Combines out-of-fold (OOF) and held-out test probability predictions across all
completed benchmark tiers on the exact same 5-fold StratifiedKFold(n_splits=5,
shuffle=True, random_state=0) splits and 100% of every test set:
  - Tier 1:  MOE-only classical, GBDT, MLP, and ensemble baselines (`tier1_preds/`)
  - Tier 2A: RDKit-only & MOE+RDKit GBDTs + Tanimoto/Hybrid Kernel SVMs (`tier2_preds/`)
  - Tier 2B: Leak-free 5-head Multi-Task ResMLP (`tier2_mt_preds/`)
  - Tier 3A: Leak-free 5-head Multi-Task GINE+VN GNN (`tier3_gnn_preds/`)
  - Tier 3B: Leak-free 5-head Multi-Task Fine-Tuned ChemBERTa-77M-MTR (`tier3_chemberta_preds/`)

Builds two strictly nested-CV meta-learners (ZERO test-set leakage):
  1. `Grand_OpenSource_Stack`: 100% open-source (ZERO proprietary MOE descriptors)
     combining RDKit GBDTs + Tanimoto SVM + MT_ResMLP_RDKit + MT_GNN + MT_ChemBERTa
  2. `Grand_AllModal_Stack`: All modalities (MOE + RDKit + Graph + SMILES + Multi-Task)

Also computes:
  - Fast exact DeLong test (two-sided p-value & 95% CI on ΔROC-AUC)
  - Exact McNemar test on Test Accuracy differences
  - Expected Calibration Error (10-bin ECE), Brier score, Log-Loss
  - Saves `notebooks/results/grand_summary.csv`, `notebooks/results/statistical_tests.csv`,
    `notebooks/results/master_comparison.csv`, and `notebooks/results/grand_preds/{iso}.npz`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit
from scipy.stats import binomtest, norm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
GRAND_PREDS_DIR = RESULTS_DIR / "grand_preds"

ISOZYMES = ["1a2", "2c9", "2c19", "2d6", "3a4"]
SEED = 0
N_SPLITS = 5

OPEN_SOURCE_MEMBERS = [
    "Tanimoto_SVM_RDKit",
    "LightGBM_RDKit",
    "XGBoost_RDKit",
    "CatBoost_RDKit",
    "MT_ResMLP_RDKit",
    "MT_GNN_GraphOnly",
    "MT_GNN_GraphPlusDesc",
    "MT_ChemBERTa_77M_MTR",
]

ALL_MODAL_MEMBERS = [
    "Tanimoto_SVM_RDKit",
    "HybridKernel_SVM",
    "LightGBM_MOE_RDKit",
    "XGBoost_MOE_RDKit",
    "CatBoost_MOE_RDKit",
    "MT_ResMLP_RDKit",
    "MT_ResMLP_MOE_RDKit",
    "MT_GNN_GraphOnly",
    "MT_GNN_GraphPlusDesc",
    "MT_ChemBERTa_77M_MTR",
]


def _safe_logit(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(p, 1e-6, 1.0 - 1e-6))


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        cnt = int(mask.sum())
        if cnt > 0:
            ece += (cnt / n) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return float(ece)


def compute_metrics(
    y_train: np.ndarray,
    oof_prob: np.ndarray,
    y_test: np.ndarray,
    test_prob: np.ndarray,
) -> dict[str, float]:
    oof_c = np.clip(oof_prob, 1e-7, 1.0 - 1e-7)
    te_c = np.clip(test_prob, 1e-7, 1.0 - 1e-7)
    cv_pred = (oof_c >= 0.5).astype(int)
    te_pred = (te_c >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, te_pred, labels=[0, 1]).ravel()
    return {
        "cv_acc": float(accuracy_score(y_train, cv_pred)),
        "cv_auc": float(roc_auc_score(y_train, oof_c)),
        "test_acc": float((tp + tn) / (tp + tn + fp + fn)),
        "test_tpr": float(tp / (tp + fn)),
        "test_tnr": float(tn / (tn + fp)),
        "test_auc": float(roc_auc_score(y_test, te_c)),
        "test_brier": float(brier_score_loss(y_test, te_c)),
        "test_logloss": float(log_loss(y_test, te_c)),
        "test_ece": compute_ece(y_test, te_c),
    }


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(N, dtype=np.float64)
    out[J] = T
    return out


def delong_roc_test(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray
) -> dict[str, float]:
    """Fast Sun & Xu (2014) exact DeLong test comparing ROC-AUC(pred_b) - ROC-AUC(pred_a)."""
    y = y_true.astype(int)
    pos_mask = y == 1
    neg_mask = y == 0
    m = int(pos_mask.sum())
    n = int(neg_mask.sum())

    preds = np.vstack([pred_a, pred_b])
    k = preds.shape[0]
    tx = np.empty((k, m), dtype=np.float64)
    ty = np.empty((k, n), dtype=np.float64)
    tz = np.empty((k, m + n), dtype=np.float64)

    for r in range(k):
        pos_s = preds[r, pos_mask]
        neg_s = preds[r, neg_mask]
        tx[r] = _compute_midrank(pos_s)
        ty[r] = _compute_midrank(neg_s)
        tz[r] = _compute_midrank(np.concatenate([pos_s, neg_s]))

    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1.0) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    S = sx / m + sy / n

    diff = float(aucs[1] - aucs[0])
    var_diff = float(S[0, 0] + S[1, 1] - 2.0 * S[0, 1])
    se = float(np.sqrt(max(var_diff, 1e-18)))
    z = diff / se
    p_val = float(2.0 * norm.sf(abs(z)))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "delta_auc": diff,
        "ci95_lo": diff - 1.96 * se,
        "ci95_hi": diff + 1.96 * se,
        "z_stat": z,
        "delong_p": p_val,
    }


def mcnemar_test(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray
) -> dict[str, float]:
    c_a = (pred_a >= 0.5).astype(int) == y_true
    c_b = (pred_b >= 0.5).astype(int) == y_true
    n01 = int((~c_a & c_b).sum())  # B right, A wrong
    n10 = int((c_a & ~c_b).sum())  # A right, B wrong
    n_disc = n01 + n10
    p_val = float(binomtest(n01, n_disc, 0.5).pvalue) if n_disc > 0 else 1.0
    return {
        "acc_a": float(c_a.mean()),
        "acc_b": float(c_b.mean()),
        "delta_acc": float(c_b.mean() - c_a.mean()),
        "b_wins": n01,
        "a_wins": n10,
        "mcnemar_p": p_val,
    }


def load_isozyme_predictions(iso: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load and verify SID-aligned OOF & test predictions across all available tier directories."""
    pred_dirs = [
        RESULTS_DIR / "tier1_preds",
        RESULTS_DIR / "tier2_preds",
        RESULTS_DIR / "tier2_mt_preds",
        RESULTS_DIR / "tier3_gnn_preds",
        RESULTS_DIR / "tier3_chemberta_preds",
    ]
    oof_dict: dict[str, np.ndarray] = {}
    test_dict: dict[str, np.ndarray] = {}
    y_tr = y_te = sid_tr = sid_te = None

    for d in pred_dirs:
        p = d / f"{iso}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        if y_tr is None:
            y_tr, y_te = z["y_train"], z["y_test"]
            sid_tr, sid_te = z["sid_train"], z["sid_test"]
        else:
            assert np.array_equal(sid_tr, z["sid_train"]), f"Train SID mismatch in {p}"
            assert np.array_equal(sid_te, z["sid_test"]), f"Test SID mismatch in {p}"
            assert np.array_equal(y_tr, z["y_train"]), f"Train y mismatch in {p}"
            assert np.array_equal(y_te, z["y_test"]), f"Test y mismatch in {p}"

        for k in z.files:
            if k.startswith("oof_"):
                name = k[len("oof_"):]
                oof_dict[name] = z[k].astype(np.float64)
            elif k.startswith("test_"):
                name = k[len("test_"):]
                test_dict[name] = z[k].astype(np.float64)

    assert y_tr is not None
    return y_tr, y_te, sid_tr, sid_te, oof_dict, test_dict


def run_nested_stack(
    members: list[str],
    oof_dict: dict[str, np.ndarray],
    test_dict: dict[str, np.ndarray],
    y_train: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    present = [m for m in members if m in oof_dict and m in test_dict]
    X_oof = np.column_stack([_safe_logit(oof_dict[m]) for m in present])
    X_te = np.column_stack([_safe_logit(test_dict[m]) for m in present])

    Cs = np.logspace(-3, 2, 25)
    nested_oof = np.zeros(len(y_train), dtype=np.float64)
    for tr_idx, va_idx in folds:
        meta = LogisticRegressionCV(
            Cs=Cs, cv=5, scoring="neg_log_loss", max_iter=1000, random_state=SEED
        )
        meta.fit(X_oof[tr_idx], y_train[tr_idx])
        nested_oof[va_idx] = meta.predict_proba(X_oof[va_idx])[:, 1]

    full_meta = LogisticRegressionCV(
        Cs=Cs, cv=5, scoring="neg_log_loss", max_iter=1000, random_state=SEED
    )
    full_meta.fit(X_oof, y_train)
    test_prob = full_meta.predict_proba(X_te)[:, 1]

    coefs = {m: round(float(c), 4) for m, c in zip(present, full_meta.coef_[0])}
    return nested_oof, test_prob, {"C": round(float(full_meta.C_[0]), 5), "coefs": coefs}


def _fit_simplex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    k = P.shape[1]
    w0 = np.full(k, 1.0 / k)

    def obj(w):
        return float(log_loss(y, np.clip(P @ w, 1e-7, 1.0 - 1e-7)))

    res = minimize(
        obj,
        w0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"ftol": 1e-9, "maxiter": 500},
    )
    w = np.clip(res.x, 0.0, 1.0)
    return w / np.sum(w)


def run_nested_simplex_blend(
    members: list[str],
    oof_dict: dict[str, np.ndarray],
    test_dict: dict[str, np.ndarray],
    y_train: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    present = [m for m in members if m in oof_dict and m in test_dict]
    P_oof = np.column_stack([oof_dict[m] for m in present])
    P_te = np.column_stack([test_dict[m] for m in present])

    nested_oof = np.zeros(len(y_train), dtype=np.float64)
    for tr_idx, va_idx in folds:
        w = _fit_simplex(P_oof[tr_idx], y_train[tr_idx])
        nested_oof[va_idx] = P_oof[va_idx] @ w

    w_full = _fit_simplex(P_oof, y_train)
    test_prob = P_te @ w_full
    weights = {m: round(float(w), 4) for m, w in zip(present, w_full)}
    return nested_oof, test_prob, {"weights": weights}


def main() -> None:
    GRAND_PREDS_DIR.mkdir(parents=True, exist_ok=True)
    grand_rows: list[dict] = []
    stat_rows: list[dict] = []
    master_rows: list[dict] = []

    for iso in ISOZYMES:
        y_tr, y_te, sid_tr, sid_te, oof_d, test_d = load_isozyme_predictions(iso)
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        folds = list(skf.split(np.zeros(len(y_tr)), y_tr))

        print(f"\n{'='*76}\nTier 4 Grand Meta-Stack: CYP {iso.upper()} ({len(oof_d)} base models loaded)\n{'='*76}", flush=True)

        # Build Grand Meta-Learners
        g_os_oof, g_os_te, g_os_p = run_nested_stack(OPEN_SOURCE_MEMBERS, oof_d, test_d, y_tr, folds)
        g_all_oof, g_all_te, g_all_p = run_nested_stack(ALL_MODAL_MEMBERS, oof_d, test_d, y_tr, folds)
        g_bl_oof, g_bl_te, g_bl_p = run_nested_simplex_blend(ALL_MODAL_MEMBERS, oof_d, test_d, y_tr, folds)

        oof_d["Grand_OpenSource_Stack"] = g_os_oof
        test_d["Grand_OpenSource_Stack"] = g_os_te
        oof_d["Grand_AllModal_Stack"] = g_all_oof
        test_d["Grand_AllModal_Stack"] = g_all_te
        oof_d["Grand_AllModal_Blend"] = g_bl_oof
        test_d["Grand_AllModal_Blend"] = g_bl_te

        for fset, mname, oof, tp, params in [
            ("OpenSource_All", "Grand_OpenSource_Stack", g_os_oof, g_os_te, g_os_p),
            ("AllModalities", "Grand_AllModal_Stack", g_all_oof, g_all_te, g_all_p),
            ("AllModalities", "Grand_AllModal_Blend", g_bl_oof, g_bl_te, g_bl_p),
        ]:
            m = compute_metrics(y_tr, oof, y_te, tp)
            grand_rows.append(
                {"isozyme": iso, "feature_set": fset, "model": mname, **m, "best_params": json.dumps(params, sort_keys=True)}
            )
            print(
                f"  [{iso}] {mname:24s} | CV ACC={m['cv_acc']:.4f} AUC={m['cv_auc']:.4f} "
                f"| TEST ACC={m['test_acc']:.4f} AUC={m['test_auc']:.4f} "
                f"Brier={m['test_brier']:.4f} ECE={m['test_ece']:.4f}",
                flush=True,
            )

        # Record EVERY model in master_comparison.csv with ECE included
        for mname in sorted(oof_d.keys()):
            m = compute_metrics(y_tr, oof_d[mname], y_te, test_d[mname])
            master_rows.append({"isozyme": iso, "model": mname, **m})

        # Pairwise DeLong + McNemar statistical significance tests
        pairs = [
            ("SVM_baseline", "Stack_LogReg", "Tier1_BestMOE_vs_OrigSVM"),
            ("RF_baseline", "Stack_LogReg", "Tier1_BestMOE_vs_OrigRF"),
            ("SVM_baseline", "Grand_OpenSource_Stack", "GrandOpenSource_vs_OrigSVM"),
            ("SVM_baseline", "Grand_AllModal_Stack", "GrandAllModal_vs_OrigSVM"),
            ("Stack_LogReg", "Grand_OpenSource_Stack", "GrandOpenSource_vs_Tier1MOE"),
            ("Stack_LogReg", "Grand_AllModal_Stack", "GrandAllModal_vs_Tier1MOE"),
            ("Stack_MOE_RDKit", "Grand_AllModal_Stack", "GrandAllModal_vs_SingleTaskMOERDKit"),
            ("Grand_OpenSource_Stack", "Grand_AllModal_Stack", "GrandAllModal_vs_GrandOpenSource"),
        ]
        for ma, mb, label in pairs:
            if ma in test_d and mb in test_d:
                dl = delong_roc_test(y_te, test_d[ma], test_d[mb])
                mc = mcnemar_test(y_te, test_d[ma], test_d[mb])
                stat_rows.append(
                    {"isozyme": iso, "comparison": label, "model_a": ma, "model_b": mb, **dl, **mc}
                )

        np.savez_compressed(
            GRAND_PREDS_DIR / f"{iso}.npz",
            y_train=y_tr,
            y_test=y_te,
            sid_train=sid_tr,
            sid_test=sid_te,
            oof_Grand_OpenSource_Stack=g_os_oof,
            test_Grand_OpenSource_Stack=g_os_te,
            oof_Grand_AllModal_Stack=g_all_oof,
            test_Grand_AllModal_Stack=g_all_te,
            oof_Grand_AllModal_Blend=g_bl_oof,
            test_Grand_AllModal_Blend=g_bl_te,
        )

    pd.DataFrame(grand_rows).to_csv(RESULTS_DIR / "grand_summary.csv", index=False)
    pd.DataFrame(stat_rows).to_csv(RESULTS_DIR / "statistical_tests.csv", index=False)
    pd.DataFrame(master_rows).to_csv(RESULTS_DIR / "master_comparison.csv", index=False)
    print("\nSaved grand_summary.csv, statistical_tests.csv, and master_comparison.csv!", flush=True)


if __name__ == "__main__":
    main()
