"""Tier 5B: 100% Leak-Free 10-Head Joint Classification + Continuous ActivityScore
Multi-Task Residual MLP fusing 2D Fingerprints + 3D ETKDGv3 Conformer Geometry +
Pretrained 1.1B-Molecule MoLFormer-XL Embeddings (GPU).

Variants:
  - MT_Joint3D_OpenSource:  RDKit FPs + RDKit 2D Desc + 151 3D Conformer Desc + 768-d MoLFormer-XL Emb (0 MOE)
  - MT_Joint3D_AllFeatures: MOE (186) + RDKit FPs + RDKit 2D Desc + 151 3D Conformer Desc + 768-d MoLFormer-XL Emb

Joint 10-head objective:
  L = L_BCE(5 binary inhibition heads) + 0.6 * L_SmoothL1(5 continuous ActivityScore/100 heads)
  with target-isozyme weight 2.0, auxiliary-isozyme weight 1.0, and double-firewall SID isolation.
"""

from __future__ import annotations

import argparse
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
from sklearn.preprocessing import QuantileTransformer, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
PREDS_DIR = RESULTS_DIR / "tier5_joint_preds"

ISOZYMES = ["1a2", "2c9", "2c19", "2d6", "3a4"]
ACTIVITY_THRESHOLD = 40
SPLIT_SEED = 0
N_SPLITS = 5

BATCH_SIZE = 256
MAX_EPOCHS = 50
PATIENCE = 9
LR = 1e-3
WEIGHT_DECAY = 2e-2
TARGET_TASK_WEIGHT = 2.0
AUX_TASK_WEIGHT = 1.0
REG_LOSS_WEIGHT = 0.6
DESC_CLIP = 1e12


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
    }


class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.35) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.net(x))


class JointMultiTaskResMLP(nn.Module):
    def __init__(self, in_dim: int, n_tasks: int = 5) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_dim, 640),
            nn.LayerNorm(640),
            nn.SiLU(),
            nn.Dropout(0.42),
        )
        self.block1 = ResBlock(640, dropout=0.35)
        self.block2 = ResBlock(640, dropout=0.35)
        self.neck = nn.Sequential(
            nn.Linear(640, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.3),
        )
        self.cls_head = nn.Linear(256, n_tasks)
        self.reg_head = nn.Linear(256, n_tasks)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.neck(self.block2(self.block1(self.stem(x))))
        return self.cls_head(h), self.reg_head(h)


def load_all_data(emb_path: Path, d3_path: Path) -> dict:
    rd_npz = np.load(DATA_DIR / "rdkit_features.npz")
    sids = rd_npz["sids"].astype(np.int64)
    sid_to_idx = {int(s): i for i, s in enumerate(sids)}

    fp_all = np.concatenate(
        [
            np.log1p(rd_npz["ecfp4_counts"].astype(np.float32)),
            rd_npz["fcfp4_bits"].astype(np.float32),
            rd_npz["maccs"].astype(np.float32),
        ],
        axis=1,
    )
    rd2d = rd_npz["rdkit_desc"].astype(np.float32)
    fin = np.isfinite(rd2d)
    rd2d[fin] = np.clip(rd2d[fin], -DESC_CLIP, DESC_CLIP)
    rd2d[~fin] = np.nan

    # Align 3D conformer descriptors to rdkit_features sids
    d3_npz = np.load(d3_path)
    d3_map = {int(s): i for i, s in enumerate(d3_npz["sids"].astype(np.int64))}
    d3_raw = d3_npz["desc_3d"].astype(np.float32)
    rd3d = np.vstack([d3_raw[d3_map[int(s)]] for s in sids]).astype(np.float32)

    # Align MoLFormer-XL 768-d embeddings to rdkit_features sids
    emb_npz = np.load(emb_path)
    emb_map = {int(s): i for i, s in enumerate(emb_npz["sids"].astype(np.int64))}
    emb_raw = emb_npz["emb"].astype(np.float32)
    mol_emb = np.vstack([emb_raw[emb_map[int(s)]] for s in sids]).astype(np.float32)

    iso_data: dict[str, dict] = {}
    master_y: dict[int, np.ndarray] = {}
    master_sc: dict[int, np.ndarray] = {}
    master_moe: dict[int, np.ndarray] = {}

    for j, iso in enumerate(ISOZYMES):
        tr = pd.read_csv(DATA_DIR / f"training{iso}.csv")
        te = pd.read_csv(DATA_DIR / f"test{iso}.csv")
        feat_cols = [c for c in tr.columns if c not in ("SID", "ActivityScore")]
        sid_tr = tr["SID"].to_numpy(dtype=np.int64)
        sid_te = te["SID"].to_numpy(dtype=np.int64)
        sc_tr = (tr["ActivityScore"].to_numpy(dtype=np.float32) / 100.0).clip(0.0, 1.2)
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
            "idx_train": np.array([sid_to_idx[int(s)] for s in sid_tr], dtype=np.int64),
            "idx_test": np.array([sid_to_idx[int(s)] for s in sid_te], dtype=np.int64),
        }
        for row_i, sid in enumerate(sid_tr):
            s = int(sid)
            if s not in master_y:
                master_y[s] = np.full(len(ISOZYMES), np.nan, dtype=np.float32)
                master_sc[s] = np.full(len(ISOZYMES), np.nan, dtype=np.float32)
                master_moe[s] = moe_tr[row_i]
            master_y[s][j] = float(y_tr[row_i])
            master_sc[s][j] = float(sc_tr[row_i])

    master_sids = np.array(sorted(master_y.keys()), dtype=np.int64)
    return {
        "fp_all": fp_all,
        "rd2d_all": rd2d,
        "rd3d_all": rd3d,
        "mol_emb_all": mol_emb,
        "iso_data": iso_data,
        "master_sids": master_sids,
        "master_y": np.vstack([master_y[int(s)] for s in master_sids]).astype(np.float32),
        "master_sc": np.vstack([master_sc[int(s)] for s in master_sids]).astype(np.float32),
        "master_moe": np.vstack([master_moe[int(s)] for s in master_sids]).astype(np.float32),
        "master_idx": np.array([sid_to_idx[int(s)] for s in master_sids], dtype=np.int64),
    }


def preprocess_fold(
    cont_tr: np.ndarray,
    cont_va: np.ndarray,
    cont_te: np.ndarray,
    emb_tr: np.ndarray,
    emb_va: np.ndarray,
    emb_te: np.ndarray,
    fp_tr: np.ndarray,
    fp_va: np.ndarray,
    fp_te: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    imp = SimpleImputer(strategy="median")
    c_tr = imp.fit_transform(cont_tr)
    c_va = imp.transform(cont_va)
    c_te = imp.transform(cont_te)
    qt = QuantileTransformer(n_quantiles=min(1000, len(c_tr)), output_distribution="normal", random_state=SPLIT_SEED)
    c_tr = np.clip(qt.fit_transform(c_tr), -8.0, 8.0).astype(np.float32)
    c_va = np.clip(qt.transform(c_va), -8.0, 8.0).astype(np.float32)
    c_te = np.clip(qt.transform(c_te), -8.0, 8.0).astype(np.float32)

    sc = StandardScaler()
    e_tr = sc.fit_transform(emb_tr).astype(np.float32)
    e_va = sc.transform(emb_va).astype(np.float32)
    e_te = sc.transform(emb_te).astype(np.float32)

    return (
        np.concatenate([fp_tr, c_tr, e_tr], axis=1),
        np.concatenate([fp_va, c_va, e_va], axis=1),
        np.concatenate([fp_te, c_te, e_te], axis=1),
    )


def train_one_model(
    X_tr: np.ndarray,
    y_tr_mat: np.ndarray,
    sc_tr_mat: np.ndarray,
    X_va: np.ndarray,
    y_va_target: np.ndarray,
    X_te: np.ndarray,
    target_idx: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    set_seed(seed)
    mask_tr = (~np.isnan(y_tr_mat)).astype(np.float32)
    y_clean = np.where(np.isnan(y_tr_mat), 0.0, y_tr_mat).astype(np.float32)
    s_clean = np.where(np.isnan(sc_tr_mat), 0.0, sc_tr_mat).astype(np.float32)

    X_tr_t = torch.from_numpy(X_tr).to(DEVICE)
    y_tr_t = torch.from_numpy(y_clean).to(DEVICE)
    s_tr_t = torch.from_numpy(s_clean).to(DEVICE)
    m_tr_t = torch.from_numpy(mask_tr).to(DEVICE)
    X_va_t = torch.from_numpy(X_va).to(DEVICE)
    X_te_t = torch.from_numpy(X_te).to(DEVICE)

    task_w = torch.full((1, len(ISOZYMES)), AUX_TASK_WEIGHT, dtype=torch.float32, device=DEVICE)
    task_w[0, target_idx] = TARGET_TASK_WEIGHT

    model = JointMultiTaskResMLP(in_dim=X_tr.shape[1], n_tasks=len(ISOZYMES)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-5)

    n = len(X_tr)
    best_auc = -1.0
    best_ep = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    no_imp = 0

    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for st in range(0, n, BATCH_SIZE):
            idx = perm[st : st + BATCH_SIZE]
            opt.zero_grad(set_to_none=True)
            cls_l, reg_p = model(X_tr_t[idx])
            w = m_tr_t[idx] * task_w
            denom = w.sum().clamp_min(1.0)
            l_cls = (F.binary_cross_entropy_with_logits(cls_l, y_tr_t[idx], reduction="none") * w).sum() / denom
            l_reg = (F.smooth_l1_loss(reg_p, s_tr_t[idx], reduction="none", beta=0.1) * w).sum() / denom
            loss = l_cls + REG_LOSS_WEIGHT * l_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            va_cls, _ = model(X_va_t)
            va_p = torch.sigmoid(va_cls[:, target_idx]).cpu().numpy()
        auc = float(roc_auc_score(y_va_target, va_p))
        if auc > best_auc + 1e-5:
            best_auc = auc
            best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        val_p = torch.sigmoid(model(X_va_t)[0][:, target_idx]).cpu().numpy().astype(np.float64)
        te_p = torch.sigmoid(model(X_te_t)[0][:, target_idx]).cpu().numpy().astype(np.float64)
    return val_p, te_p, best_ep


def run_variant(data: dict, iso: str, use_moe: bool, n_seeds: int) -> tuple[np.ndarray, np.ndarray, dict]:
    target_idx = ISOZYMES.index(iso)
    iso_d = data["iso_data"][iso]
    sid_tr, sid_te = iso_d["sid_train"], iso_d["sid_test"]
    y_tr = iso_d["y_train"]
    idx_tr, idx_te = iso_d["idx_train"], iso_d["idx_test"]

    test_sid_set = set(sid_te.tolist())
    non_test = np.array([int(s) not in test_sid_set for s in data["master_sids"]], dtype=bool)
    pool_sids = data["master_sids"][non_test]
    pool_y = data["master_y"][non_test]
    pool_sc = data["master_sc"][non_test]
    pool_moe = data["master_moe"][non_test]
    pool_idx = data["master_idx"][non_test]
    assert len(set(pool_sids.tolist()) & test_sid_set) == 0, "LEAKAGE: test SID in pool!"

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_SEED)
    folds = list(skf.split(np.zeros(len(y_tr)), y_tr))

    oof = np.zeros(len(y_tr), dtype=np.float64)
    test_p = np.zeros(len(sid_te), dtype=np.float64)

    fp_te = data["fp_all"][idx_te]
    emb_te = data["mol_emb_all"][idx_te]
    cont_te_parts = [data["rd2d_all"][idx_te], data["rd3d_all"][idx_te]]
    if use_moe:
        cont_te_parts = [iso_d["moe_test"]] + cont_te_parts
    cont_te = np.concatenate(cont_te_parts, axis=1)

    eps = []
    for fold_i, (_, va_idx) in enumerate(folds):
        val_sid_set = set(sid_tr[va_idx].tolist())
        f_mask = np.array([int(s) not in val_sid_set for s in pool_sids], dtype=bool)
        assert len(set(pool_sids[f_mask].tolist()) & val_sid_set) == 0
        assert len(set(pool_sids[f_mask].tolist()) & test_sid_set) == 0

        f_idx = pool_idx[f_mask]
        va_gidx = idx_tr[va_idx]

        c_tr_parts = [data["rd2d_all"][f_idx], data["rd3d_all"][f_idx]]
        c_va_parts = [data["rd2d_all"][va_gidx], data["rd3d_all"][va_gidx]]
        if use_moe:
            c_tr_parts = [pool_moe[f_mask]] + c_tr_parts
            c_va_parts = [iso_d["moe_train"][va_idx]] + c_va_parts

        X_tr, X_va, X_te = preprocess_fold(
            np.concatenate(c_tr_parts, axis=1),
            np.concatenate(c_va_parts, axis=1),
            cont_te,
            data["mol_emb_all"][f_idx],
            data["mol_emb_all"][va_gidx],
            emb_te,
            data["fp_all"][f_idx],
            data["fp_all"][va_gidx],
            fp_te,
        )

        vp_f = np.zeros(len(va_idx), dtype=np.float64)
        tp_f = np.zeros(len(sid_te), dtype=np.float64)
        for s in range(n_seeds):
            vp, tp, be = train_one_model(
                X_tr, pool_y[f_mask], pool_sc[f_mask], X_va, y_tr[va_idx], X_te, target_idx, seed=100 * fold_i + s
            )
            vp_f += vp / n_seeds
            tp_f += tp / n_seeds
            eps.append(be)
        oof[va_idx] = vp_f
        test_p += tp_f / N_SPLITS

    return oof, test_p, {"mean_best_epoch": float(np.mean(eps))}


def main() -> None:
    global DATA_DIR, RESULTS_DIR, PREDS_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--emb-path", type=Path, default=None)
    parser.add_argument("--d3-path", type=Path, default=None)
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    RESULTS_DIR = args.results_dir
    PREDS_DIR = RESULTS_DIR / "tier5_joint_preds"
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    emb_path = args.emb_path or (RESULTS_DIR / "molformer_emb.npz")
    d3_path = args.d3_path or (DATA_DIR / "rdkit_3d_features.npz")

    t0 = time.time()
    print(f"Tier 5B Joint Reg+Cls Multi-Task ResMLP (3D + MoLFormer) on {DEVICE} (seeds={args.seeds})", flush=True)
    data = load_all_data(emb_path, d3_path)

    variants = [
        ("OpenSource_3D_MoLFormer", "MT_Joint3D_OpenSource", False),
        ("AllModal_3D_MoLFormer", "MT_Joint3D_AllFeatures", True),
    ]
    rows: list[dict] = []
    for iso in ISOZYMES:
        iso_d = data["iso_data"][iso]
        y_tr, y_te = iso_d["y_train"], iso_d["y_test"]
        print(f"\n{'='*72}\nTier 5B Joint3D: CYP {iso.upper()}\n{'='*72}", flush=True)
        payload = {"y_train": y_tr, "y_test": y_te, "sid_train": iso_d["sid_train"], "sid_test": iso_d["sid_test"]}
        for fset, mname, use_moe in variants:
            tv = time.time()
            oof, tp, info = run_variant(data, iso, use_moe=use_moe, n_seeds=args.seeds)
            m = compute_metrics(y_tr, oof, y_te, tp)
            rows.append({"isozyme": iso, "feature_set": fset, "model": mname, **m})
            payload[f"oof_{mname}"] = oof
            payload[f"test_{mname}"] = tp
            print(
                f"  [{iso}] {mname:24s} | CV ACC={m['cv_acc']:.4f} AUC={m['cv_auc']:.4f} "
                f"| TEST ACC={m['test_acc']:.4f} AUC={m['test_auc']:.4f} "
                f"Brier={m['test_brier']:.4f} (ep={info['mean_best_epoch']:.1f}, {time.time()-tv:.1f}s)",
                flush=True,
            )
        np.savez_compressed(PREDS_DIR / f"{iso}.npz", **payload)
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "tier5_joint_summary.csv", index=False)

    print(f"\nAll 5 isozymes finished in {(time.time() - t0)/60:.2f} min.", flush=True)


if __name__ == "__main__":
    main()
