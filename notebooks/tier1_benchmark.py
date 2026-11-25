"""Tier 1 Apples-to-Apples Benchmark on the exact 186 MOE 2D descriptors & Figshare splits.

Strict zero-leakage validation protocol:
- Exact Figshare splits: notebooks/data/training{iso}.csv and notebooks/data/test{iso}.csv
- Binary target: ActivityScore >= 40
- Internal validation: 5-fold StratifiedKFold(n_splits=5, shuffle=True, random_state=0) on train ONLY
- All preprocessing (StandardScaler, QuantileTransformer) inside sklearn Pipelines fit per fold
- All hyperparameter tuning, early stopping, and ensemble weight fitting strictly on train CV
- Final evaluation: fit on full training set, predict probabilities on held-out test set
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.special import logit
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
PREDS_DIR = RESULTS_DIR / "tier1_preds"

ISOZYMES = ["1a2", "2c9", "2c19", "2d6", "3a4"]
ACTIVITY_THRESHOLD = 40
SEED = 0
N_SPLITS = 5
N_JOBS = 4

STACK_MODELS = [
    "RF_500",
    "ExtraTrees_500",
    "SVM_tuned",
    "LightGBM",
    "XGBoost",
    "CatBoost",
    "MLP",
]
BLEND_MODELS = [
    "CatBoost",
    "LightGBM",
    "XGBoost",
    "SVM_tuned",
    "ExtraTrees_500",
    "MLP",
]


def load_isozyme(
    isozyme: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load exact train/test splits and SIDs for one isozyme."""
    tr = pd.read_csv(DATA_DIR / f"training{isozyme}.csv")
    te = pd.read_csv(DATA_DIR / f"test{isozyme}.csv")

    sid_tr = tr["SID"].to_numpy()
    sid_te = te["SID"].to_numpy()

    y_tr = (tr["ActivityScore"] >= ACTIVITY_THRESHOLD).to_numpy(dtype=np.int32)
    y_te = (te["ActivityScore"] >= ACTIVITY_THRESHOLD).to_numpy(dtype=np.int32)

    feat_cols = [c for c in tr.columns if c not in ("SID", "ActivityScore")]
    X_tr = tr[feat_cols].to_numpy(dtype=np.float64)
    X_te = te[feat_cols].to_numpy(dtype=np.float64)
    return X_tr, y_tr, sid_tr, X_te, y_te, sid_te


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


def make_scaler(name: str):
    if name == "standard":
        return StandardScaler()
    if name == "quantile":
        return QuantileTransformer(output_distribution="normal", random_state=SEED)
    raise ValueError(f"Unknown scaler: {name}")


def run_sklearn_cv_and_test(
    builder,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a fresh model per fold for OOF, then fit on full train for test."""
    oof = np.zeros(len(y_train), dtype=np.float64)
    for tr_idx, va_idx in folds:
        model = builder()
        model.fit(X_train[tr_idx], y_train[tr_idx])
        oof[va_idx] = model.predict_proba(X_train[va_idx])[:, 1]

    full_model = builder()
    full_model.fit(X_train, y_train)
    test_prob = full_model.predict_proba(X_test)[:, 1]
    return oof, test_prob


def _eval_svm_config(
    scaler_name: str,
    C: float,
    gamma: float | str,
    scaled_folds: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    n_train: int,
    y_train: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, str, float, float | str]:
    """Evaluate one (scaler, C, gamma) config across pre-scaled train/val folds using decision_function AUC."""
    oof_scores = np.zeros(n_train, dtype=np.float64)
    for (_, va_idx), (X_tr_s, y_tr_f, X_va_s, _) in zip(folds, scaled_folds):
        svc = SVC(
            C=C,
            kernel="rbf",
            gamma=gamma,
            probability=False,
            cache_size=500,
            random_state=SEED,
        )
        svc.fit(X_tr_s, y_tr_f)
        oof_scores[va_idx] = svc.decision_function(X_va_s)
    auc = float(roc_auc_score(y_train, oof_scores))
    return auc, scaler_name, C, gamma


def run_svm_tuned(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Tune (scaler, C, gamma) strictly on 5-fold CV via decision_function ROC-AUC, then fit with probability=True."""
    scaler_names = ["standard", "quantile"]
    C_grid = [1.0, 3.0, 10.0, 30.0]
    gamma_grid: list[float | str] = ["scale", 0.002, 0.005, 0.01]

    # Fit each scaler strictly inside each fold once to avoid redundant QuantileTransformer fits
    scaled_cache: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}
    for s_name in scaler_names:
        s_folds = []
        for tr_idx, va_idx in folds:
            sc = make_scaler(s_name)
            X_tr_s = sc.fit_transform(X_train[tr_idx])
            X_va_s = sc.transform(X_train[va_idx])
            s_folds.append((X_tr_s, y_train[tr_idx], X_va_s, y_train[va_idx]))
        scaled_cache[s_name] = s_folds

    tasks = [
        (s_name, C, gamma)
        for s_name in scaler_names
        for C in C_grid
        for gamma in gamma_grid
    ]
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_eval_svm_config)(
            s_name, C, gamma, scaled_cache[s_name], len(y_train), y_train, folds
        )
        for s_name, C, gamma in tasks
    )

    best_auc, best_scaler, best_C, best_gamma = max(results, key=lambda r: r[0])
    best_params = {
        "scaler": best_scaler,
        "C": best_C,
        "gamma": best_gamma,
        "tune_df_auc": round(best_auc, 5),
    }

    # Refit winning config with probability=True for OOF (parallel across folds) and full train
    def _fit_fold(tr_idx, va_idx):
        pipe = Pipeline(
            [
                ("scaler", make_scaler(best_scaler)),
                (
                    "svc",
                    SVC(
                        C=best_C,
                        kernel="rbf",
                        gamma=best_gamma,
                        probability=True,
                        cache_size=500,
                        random_state=SEED,
                    ),
                ),
            ]
        )
        pipe.fit(X_train[tr_idx], y_train[tr_idx])
        return va_idx, pipe.predict_proba(X_train[va_idx])[:, 1]

    fold_preds = Parallel(n_jobs=N_JOBS)(
        delayed(_fit_fold)(tr_idx, va_idx) for tr_idx, va_idx in folds
    )
    oof = np.zeros(len(y_train), dtype=np.float64)
    for va_idx, probs in fold_preds:
        oof[va_idx] = probs

    full_pipe = Pipeline(
        [
            ("scaler", make_scaler(best_scaler)),
            (
                "svc",
                SVC(
                    C=best_C,
                    kernel="rbf",
                    gamma=best_gamma,
                    probability=True,
                    cache_size=500,
                    random_state=SEED,
                ),
            ),
        ]
    )
    full_pipe.fit(X_train, y_train)
    test_prob = full_pipe.predict_proba(X_test)[:, 1]
    return oof, test_prob, best_params


def run_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    base_kwargs = dict(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=N_JOBS,
        random_state=SEED,
        verbose=-1,
    )
    oof = np.zeros(len(y_train), dtype=np.float64)
    best_iters = []
    for tr_idx, va_idx in folds:
        clf = lgb.LGBMClassifier(**base_kwargs)
        clf.fit(
            X_train[tr_idx],
            y_train[tr_idx],
            eval_set=[(X_train[va_idx], y_train[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        best_iter = int(clf.best_iteration_ or base_kwargs["n_estimators"])
        best_iters.append(best_iter)
        oof[va_idx] = clf.predict_proba(X_train[va_idx], num_iteration=best_iter)[:, 1]

    full_n_est = max(50, int(round(float(np.median(best_iters)) * 1.1)))
    full_kwargs = {**base_kwargs, "n_estimators": full_n_est}
    full_clf = lgb.LGBMClassifier(**full_kwargs)
    full_clf.fit(X_train, y_train)
    test_prob = full_clf.predict_proba(X_test)[:, 1]
    return oof, test_prob, {"median_best_iter": int(np.median(best_iters)), "full_n_estimators": full_n_est}


def run_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    base_kwargs = dict(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=N_JOBS,
        random_state=SEED,
    )
    oof = np.zeros(len(y_train), dtype=np.float64)
    best_iters = []
    for tr_idx, va_idx in folds:
        clf = XGBClassifier(**base_kwargs, early_stopping_rounds=50)
        clf.fit(
            X_train[tr_idx],
            y_train[tr_idx],
            eval_set=[(X_train[va_idx], y_train[va_idx])],
            verbose=False,
        )
        best_iter = int(clf.best_iteration) + 1
        best_iters.append(best_iter)
        oof[va_idx] = clf.predict_proba(X_train[va_idx])[:, 1]

    full_n_est = max(50, int(round(float(np.median(best_iters)) * 1.1)))
    full_clf = XGBClassifier(**{**base_kwargs, "n_estimators": full_n_est})
    full_clf.fit(X_train, y_train, verbose=False)
    test_prob = full_clf.predict_proba(X_test)[:, 1]
    return oof, test_prob, {"median_best_iter": int(np.median(best_iters)), "full_n_estimators": full_n_est}


def run_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    base_kwargs = dict(
        iterations=2000,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=3.0,
        eval_metric="Logloss",
        thread_count=N_JOBS,
        random_seed=SEED,
        verbose=0,
    )
    oof = np.zeros(len(y_train), dtype=np.float64)
    best_iters = []
    for tr_idx, va_idx in folds:
        clf = CatBoostClassifier(**base_kwargs, early_stopping_rounds=80)
        clf.fit(
            X_train[tr_idx],
            y_train[tr_idx],
            eval_set=(X_train[va_idx], y_train[va_idx]),
            use_best_model=True,
        )
        best_iter = int(clf.get_best_iteration()) + 1
        best_iters.append(best_iter)
        oof[va_idx] = clf.predict_proba(X_train[va_idx])[:, 1]

    full_iters = max(50, int(round(float(np.median(best_iters)) * 1.1)))
    full_clf = CatBoostClassifier(**{**base_kwargs, "iterations": full_iters})
    full_clf.fit(X_train, y_train)
    test_prob = full_clf.predict_proba(X_test)[:, 1]
    return oof, test_prob, {"median_best_iter": int(np.median(best_iters)), "full_iterations": full_iters}


def run_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    seeds = [0, 1, 2]
    oof = np.zeros(len(y_train), dtype=np.float64)
    test_prob = np.zeros(len(X_test), dtype=np.float64)

    for seed in seeds:
        def _builder(s=seed):
            return Pipeline(
                [
                    ("qt", QuantileTransformer(output_distribution="normal", random_state=s)),
                    (
                        "mlp",
                        MLPClassifier(
                            hidden_layer_sizes=(256, 128),
                            alpha=1e-2,
                            batch_size=128,
                            learning_rate_init=1e-3,
                            early_stopping=True,
                            n_iter_no_change=15,
                            max_iter=200,
                            random_state=s,
                        ),
                    ),
                ]
            )

        oof_s, test_s = run_sklearn_cv_and_test(_builder, X_train, y_train, X_test, folds)
        oof += oof_s / len(seeds)
        test_prob += test_s / len(seeds)

    return oof, test_prob, {"seeds": seeds}


def _safe_logit(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(p, 1e-6, 1.0 - 1e-6))


def run_stack_logreg(
    oof_dict: dict[str, np.ndarray],
    test_dict: dict[str, np.ndarray],
    y_train: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """L2 LogisticRegressionCV on OOF logits of the 7 improved models (nested OOF for CV)."""
    X_oof = np.column_stack([_safe_logit(oof_dict[m]) for m in STACK_MODELS])
    X_te = np.column_stack([_safe_logit(test_dict[m]) for m in STACK_MODELS])

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

    coefs = {m: round(float(c), 4) for m, c in zip(STACK_MODELS, full_meta.coef_[0])}
    params = {"C": round(float(full_meta.C_[0]), 5), "coefs": coefs}
    return nested_oof, test_prob, params


def _fit_simplex_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Find non-negative weights summing to 1 minimizing binary log-loss."""
    k = P.shape[1]
    w0 = np.full(k, 1.0 / k)

    def obj(w):
        pred = np.clip(P @ w, 1e-7, 1.0 - 1e-7)
        return float(log_loss(y, pred))

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


def run_blends(
    oof_dict: dict[str, np.ndarray],
    test_dict: dict[str, np.ndarray],
    y_train: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[tuple[np.ndarray, np.ndarray, dict], tuple[np.ndarray, np.ndarray, dict]]:
    P_oof = np.column_stack([oof_dict[m] for m in BLEND_MODELS])
    P_te = np.column_stack([test_dict[m] for m in BLEND_MODELS])

    # Equal blend
    oof_eq = P_oof.mean(axis=1)
    test_eq = P_te.mean(axis=1)
    eq_params = {"models": BLEND_MODELS}

    # Optimized blend (nested OOF for honest CV)
    oof_opt = np.zeros(len(y_train), dtype=np.float64)
    for tr_idx, va_idx in folds:
        w_f = _fit_simplex_weights(P_oof[tr_idx], y_train[tr_idx])
        oof_opt[va_idx] = P_oof[va_idx] @ w_f

    w_full = _fit_simplex_weights(P_oof, y_train)
    test_opt = P_te @ w_full
    opt_params = {
        "weights": {m: round(float(w), 4) for m, w in zip(BLEND_MODELS, w_full)}
    }
    return (oof_eq, test_eq, eq_params), (oof_opt, test_opt, opt_params)


def benchmark_isozyme(isozyme: str) -> list[dict]:
    t0 = time.time()
    print(f"\n{'='*70}\nBenchmarking CYP {isozyme.upper()}\n{'='*70}", flush=True)
    X_train, y_train, sid_train, X_test, y_test, sid_test = load_isozyme(isozyme)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(skf.split(X_train, y_train))

    oof_preds: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}
    rows: list[dict] = []

    def _record(model_name: str, oof: np.ndarray, test_p: np.ndarray, params: dict, dt: float):
        oof_preds[model_name] = oof.astype(np.float64)
        test_preds[model_name] = test_p.astype(np.float64)
        m = compute_metrics(y_train, oof, y_test, test_p)
        row = {
            "isozyme": isozyme,
            "model": model_name,
            **m,
            "best_params": json.dumps(params, sort_keys=True),
        }
        rows.append(row)
        print(
            f"  [{isozyme}] {model_name:15s} | CV ACC={m['cv_acc']:.4f} AUC={m['cv_auc']:.4f} "
            f"| TEST ACC={m['test_acc']:.4f} AUC={m['test_auc']:.4f} "
            f"Brier={m['test_brier']:.4f} LL={m['test_logloss']:.4f} ({dt:.1f}s)",
            flush=True,
        )

    # 1. Original baselines
    t = time.time()
    oof, tp = run_sklearn_cv_and_test(
        lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svc",
                    SVC(
                        C=1.0,
                        kernel="rbf",
                        gamma="auto",
                        probability=True,
                        cache_size=500,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        X_train,
        y_train,
        X_test,
        folds,
    )
    _record("SVM_baseline", oof, tp, {"scaler": "standard", "C": 1.0, "gamma": "auto"}, time.time() - t)

    t = time.time()
    oof, tp = run_sklearn_cv_and_test(
        lambda: RandomForestClassifier(
            n_estimators=10, max_features="sqrt", random_state=SEED
        ),
        X_train,
        y_train,
        X_test,
        folds,
    )
    _record("RF_baseline", oof, tp, {"n_estimators": 10, "max_features": "sqrt"}, time.time() - t)

    # 2. Improved classical & modern tabular models
    t = time.time()
    oof, tp = run_sklearn_cv_and_test(
        lambda: RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=N_JOBS,
            random_state=SEED,
        ),
        X_train,
        y_train,
        X_test,
        folds,
    )
    _record("RF_500", oof, tp, {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 1}, time.time() - t)

    t = time.time()
    oof, tp = run_sklearn_cv_and_test(
        lambda: ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            n_jobs=N_JOBS,
            random_state=SEED,
        ),
        X_train,
        y_train,
        X_test,
        folds,
    )
    _record("ExtraTrees_500", oof, tp, {"n_estimators": 500, "max_features": "sqrt"}, time.time() - t)

    t = time.time()
    oof, tp, params = run_svm_tuned(X_train, y_train, X_test, folds)
    _record("SVM_tuned", oof, tp, params, time.time() - t)

    t = time.time()
    oof, tp, params = run_lightgbm(X_train, y_train, X_test, folds)
    _record("LightGBM", oof, tp, params, time.time() - t)

    t = time.time()
    oof, tp, params = run_xgboost(X_train, y_train, X_test, folds)
    _record("XGBoost", oof, tp, params, time.time() - t)

    t = time.time()
    oof, tp, params = run_catboost(X_train, y_train, X_test, folds)
    _record("CatBoost", oof, tp, params, time.time() - t)

    t = time.time()
    oof, tp, params = run_mlp(X_train, y_train, X_test, folds)
    _record("MLP", oof, tp, params, time.time() - t)

    # 3. Ensembles (strictly from OOF)
    t = time.time()
    oof, tp, params = run_stack_logreg(oof_preds, test_preds, y_train, folds)
    _record("Stack_LogReg", oof, tp, params, time.time() - t)

    t = time.time()
    (oof_eq, tp_eq, eq_p), (oof_opt, tp_opt, opt_p) = run_blends(
        oof_preds, test_preds, y_train, folds
    )
    _record("Blend_Equal", oof_eq, tp_eq, eq_p, time.time() - t)
    _record("Blend_Opt", oof_opt, tp_opt, opt_p, time.time() - t)

    # Save predictions npz
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    npz_payload = {
        "y_train": y_train,
        "y_test": y_test,
        "sid_train": sid_train,
        "sid_test": sid_test,
    }
    for m_name in oof_preds:
        npz_payload[f"oof_{m_name}"] = oof_preds[m_name]
        npz_payload[f"test_{m_name}"] = test_preds[m_name]
    np.savez_compressed(PREDS_DIR / f"{isozyme}.npz", **npz_payload)

    print(f"Completed {isozyme} in {time.time() - t0:.1f}s -> saved {PREDS_DIR / f'{isozyme}.npz'}", flush=True)
    return rows


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    t_start = time.time()
    for iso in ISOZYMES:
        iso_rows = benchmark_isozyme(iso)
        all_rows.extend(iso_rows)
        # Write incremental summary CSV
        df = pd.DataFrame(all_rows)
        df.to_csv(RESULTS_DIR / "tier1_summary.csv", index=False)

    print(f"\nAll 5 isozymes finished in {(time.time() - t_start)/60:.2f} min.", flush=True)
    print(f"Summary saved to {RESULTS_DIR / 'tier1_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
