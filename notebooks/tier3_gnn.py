"""Tier 3A: 100% Leak-Free Multi-Task Bond-Aware GNN (GINE + Virtual Node) in Pure PyTorch (GPU).

Trains a 5-head Multi-Task Graph Isomorphism Network with Edge Features (GINE) +
Virtual Node + Gated Attention/Mean/Max Readout directly on molecular graphs from
`sid_smiles.csv`, under the exact same strict zero-leakage protocol as Tier 2B:
  1. Build a master multi-task table of all SIDs appearing in any training{j}.csv.
  2. For each target isozyme `iso`:
     - Exclude ALL SIDs in test{iso}.csv from the entire 5-head multi-task training pool.
     - Use the exact 5-fold StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
       splits on training{iso}.csv.
     - For each fold k:
       * Exclude ALL fold-k validation SIDs from ALL 5 heads in the training pool.
       * For `MT_GNN_GraphPlusDesc`, fit SimpleImputer(median) + QuantileTransformer(normal)
         strictly on the fold-k training pool.
       * Train with masked BCEWithLogitsLoss (weight 2.0 on target head, 1.0 on
         the 4 auxiliary heads; NaN labels masked out).
       * Early-stop on target isozyme validation ROC-AUC (patience 10), restore best weights.
       * Average fold × seed models' predictions on 100% of test{iso}.

Feature variants:
  - MT_GNN_GraphOnly:     End-to-end 2D molecular graph (atom + bond features ONLY)
  - MT_GNN_GraphPlusDesc: 2D molecular graph + fold-scaled RDKit 2D descriptors (Chemprop-style)
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
from rdkit import Chem, RDLogger
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

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
PREDS_DIR = RESULTS_DIR / "tier3_gnn_preds"

ISOZYMES = ["1a2", "2c9", "2c19", "2d6", "3a4"]
ACTIVITY_THRESHOLD = 40
SPLIT_SEED = 0
N_SPLITS = 5

BATCH_SIZE = 256
MAX_EPOCHS = 40
PATIENCE = 7
LR = 2e-3
WEIGHT_DECAY = 1e-4
TARGET_TASK_WEIGHT = 2.0
AUX_TASK_WEIGHT = 1.0
DESC_CLIP = 1e12

ATOM_LIST = [6, 7, 8, 9, 15, 16, 17, 35, 53, 5, 14]
HYB_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
STEREO_LIST = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _one_hot(val, choices: list) -> list[float]:
    vec = [0.0] * (len(choices) + 1)
    try:
        vec[choices.index(val)] = 1.0
    except ValueError:
        vec[-1] = 1.0
    return vec


def atom_features(atom: Chem.Atom) -> list[float]:
    return (
        _one_hot(atom.GetAtomicNum(), ATOM_LIST)
        + _one_hot(min(atom.GetTotalDegree(), 5), [0, 1, 2, 3, 4, 5])
        + _one_hot(max(-2, min(2, atom.GetFormalCharge())), [-2, -1, 0, 1, 2])
        + _one_hot(atom.GetHybridization(), HYB_LIST)
        + _one_hot(min(atom.GetTotalNumHs(), 4), [0, 1, 2, 3, 4])
        + [
            1.0 if atom.GetIsAromatic() else 0.0,
            1.0 if atom.IsInRing() else 0.0,
            1.0 if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED else 0.0,
            atom.GetMass() / 100.0,
        ]
    )


def bond_features(bond: Chem.Bond) -> list[float]:
    return (
        _one_hot(bond.GetBondType(), BOND_LIST)
        + _one_hot(bond.GetStereo(), STEREO_LIST)
        + [
            1.0 if bond.GetIsConjugated() else 0.0,
            1.0 if bond.IsInRing() else 0.0,
        ]
    )


ATOM_DIM = len(ATOM_LIST) + 1 + 7 + 6 + len(HYB_LIST) + 1 + 6 + 4  # 41
BOND_DIM = len(BOND_LIST) + 1 + len(STEREO_LIST) + 1 + 2  # 13


def smiles_to_graph(smiles: str | float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert SMILES to (node_x [N, F], edge_index [2, E], edge_attr [E, D])."""
    mol = None
    if isinstance(smiles, str) and smiles.strip():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        # Dummy single-node graph for missing/unparseable SMILES (100% test coverage)
        return (
            np.zeros((1, ATOM_DIM), dtype=np.float32),
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0, BOND_DIM), dtype=np.float32),
        )

    x = np.array([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)
    src, dst, eattr = [], [], []
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        src.extend([u, v])
        dst.extend([v, u])
        eattr.append(bf)
        eattr.append(bf)

    if len(src) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, BOND_DIM), dtype=np.float32)
    else:
        edge_index = np.array([src, dst], dtype=np.int64)
        edge_attr = np.array(eattr, dtype=np.float32)
    return x, edge_index, edge_attr


def collate_graphs(
    graphs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    desc: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor | None]:
    """Collate list of graphs into a single disjoint batched graph on DEVICE."""
    xs, eis, eas, batch_ids = [], [], [], []
    offset = 0
    for g_idx, (x, ei, ea) in enumerate(graphs):
        n = x.shape[0]
        xs.append(x)
        if ei.shape[1] > 0:
            eis.append(ei + offset)
            eas.append(ea)
        batch_ids.append(np.full(n, g_idx, dtype=np.int64))
        offset += n

    x_t = torch.from_numpy(np.concatenate(xs, axis=0)).to(DEVICE, non_blocking=True)
    if eis:
        ei_t = torch.from_numpy(np.concatenate(eis, axis=1)).to(DEVICE, non_blocking=True)
        ea_t = torch.from_numpy(np.concatenate(eas, axis=0)).to(DEVICE, non_blocking=True)
    else:
        ei_t = torch.zeros((2, 0), dtype=torch.long, device=DEVICE)
        ea_t = torch.zeros((0, BOND_DIM), dtype=torch.float32, device=DEVICE)
    batch_t = torch.from_numpy(np.concatenate(batch_ids, axis=0)).to(DEVICE, non_blocking=True)
    desc_t = (
        torch.from_numpy(desc).to(DEVICE, non_blocking=True)
        if desc is not None
        else None
    )
    return x_t, ei_t, ea_t, batch_t, len(graphs), desc_t


class GINELayer(nn.Module):
    """Graph Isomorphism Network layer with edge features + residual LayerNorm."""

    def __init__(self, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_h: torch.Tensor,
    ) -> torch.Tensor:
        agg = torch.zeros_like(h)
        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            msg = F.silu(h[src] + edge_h).to(agg.dtype)
            agg.index_add_(0, dst, msg)
        out = self.mlp((1.0 + self.eps) * h + agg)
        return self.norm(h + self.drop(out))


class MultiTaskGINE(nn.Module):
    """4-layer GINE + Virtual Node + Mean/Max/Attention Readout + Optional Global Descriptors."""

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        desc_dim: int = 0,
        n_tasks: int = 5,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.atom_enc = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.bond_enc = nn.Sequential(
            nn.Linear(bond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList(
            [GINELayer(hidden_dim, dropout=dropout) for _ in range(n_layers)]
        )
        # Virtual node updates between message-passing layers
        self.vn_init = nn.Parameter(torch.zeros(1, hidden_dim))
        self.vn_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in range(n_layers - 1)
            ]
        )
        # Gated attention readout
        self.att_gate = nn.Linear(hidden_dim, 1)

        self.desc_dim = desc_dim
        if desc_dim > 0:
            self.desc_enc = nn.Sequential(
                nn.Linear(desc_dim, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            readout_dim = hidden_dim * 3 + 128
        else:
            self.desc_enc = None
            readout_dim = hidden_dim * 3

        self.head = nn.Sequential(
            nn.Linear(readout_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_tasks),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        desc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.atom_enc(x)
        eh = self.bond_enc(edge_attr) if edge_attr.numel() > 0 else edge_attr.new_zeros((0, h.size(-1)))
        vn = self.vn_init.expand(num_graphs, -1)

        counts = torch.zeros(num_graphs, 1, device=h.device, dtype=h.dtype)
        counts.index_add_(0, batch, torch.ones((h.size(0), 1), device=h.device, dtype=h.dtype))
        counts = counts.clamp_min(1.0)

        for i, layer in enumerate(self.layers):
            h = h + vn[batch]
            h = layer(h, edge_index, eh)
            if i < len(self.vn_mlps):
                pooled = torch.zeros(num_graphs, h.size(-1), device=h.device, dtype=h.dtype)
                pooled.index_add_(0, batch, h)
                vn = F.silu(vn + self.vn_mlps[i](vn + pooled / counts))

        # Pooling in float32 for numerical stability under fp16 autocast
        hf = h.float()
        counts_f = counts.float()

        # Mean pool
        mean_pool = torch.zeros(num_graphs, hf.size(-1), device=hf.device, dtype=torch.float32)
        mean_pool.index_add_(0, batch, hf)
        mean_pool = mean_pool / counts_f

        # Max pool (via scatter_reduce amax)
        max_pool = torch.full((num_graphs, hf.size(-1)), -1e9, device=hf.device, dtype=torch.float32)
        max_pool.scatter_reduce_(
            0, batch.unsqueeze(1).expand_as(hf), hf, reduce="amax", include_self=True
        )
        max_pool = torch.where(max_pool < -1e8, torch.zeros_like(max_pool), max_pool)

        # Gated attention pool (softmax per graph)
        att_logits = self.att_gate(h).squeeze(-1).float()
        att_max = torch.full((num_graphs,), -1e9, device=hf.device, dtype=torch.float32)
        att_max.scatter_reduce_(0, batch, att_logits, reduce="amax", include_self=True)
        att_exp = torch.exp(att_logits - att_max[batch])
        att_denom = torch.zeros(num_graphs, device=hf.device, dtype=torch.float32)
        att_denom.index_add_(0, batch, att_exp)
        att_w = (att_exp / att_denom[batch].clamp_min(1e-12)).unsqueeze(-1)
        att_pool = torch.zeros(num_graphs, hf.size(-1), device=hf.device, dtype=torch.float32)
        att_pool.index_add_(0, batch, hf * att_w)

        feats = [mean_pool, max_pool, att_pool]
        if self.desc_enc is not None and desc is not None:
            feats.append(self.desc_enc(desc))
        return self.head(torch.cat(feats, dim=-1))


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


def load_all_data() -> dict:
    t0 = time.time()
    rdkit_npz = np.load(DATA_DIR / "rdkit_features.npz")
    rdkit_sids = rdkit_npz["sids"].astype(np.int64)
    sid_to_idx = {int(s): i for i, s in enumerate(rdkit_sids)}

    rd = rdkit_npz["rdkit_desc"].astype(np.float32)
    finite = np.isfinite(rd)
    rd[finite] = np.clip(rd[finite], -DESC_CLIP, DESC_CLIP)
    rd[~finite] = np.nan

    sm_df = pd.read_csv(DATA_DIR / "sid_smiles.csv", usecols=["SID", "CanonicalSMILES", "SMILES"])
    sid_to_smiles: dict[int, str | float] = {}
    for sid, cs, s in zip(sm_df["SID"], sm_df["CanonicalSMILES"], sm_df["SMILES"]):
        sid_to_smiles[int(sid)] = cs if isinstance(cs, str) and cs.strip() else s

    print("Featurizing 16,725 molecular graphs...", flush=True)
    graphs = [smiles_to_graph(sid_to_smiles.get(int(s), "")) for s in rdkit_sids]
    print(f"Featurized {len(graphs)} graphs in {time.time() - t0:.1f}s", flush=True)

    iso_data: dict[str, dict] = {}
    master_labels: dict[int, np.ndarray] = {}

    for j, iso in enumerate(ISOZYMES):
        tr = pd.read_csv(DATA_DIR / f"training{iso}.csv", usecols=["SID", "ActivityScore"])
        te = pd.read_csv(DATA_DIR / f"test{iso}.csv", usecols=["SID", "ActivityScore"])

        sid_tr = tr["SID"].to_numpy(dtype=np.int64)
        sid_te = te["SID"].to_numpy(dtype=np.int64)
        y_tr = (tr["ActivityScore"] >= ACTIVITY_THRESHOLD).to_numpy(dtype=np.int32)
        y_te = (te["ActivityScore"] >= ACTIVITY_THRESHOLD).to_numpy(dtype=np.int32)

        iso_data[iso] = {
            "sid_train": sid_tr,
            "sid_test": sid_te,
            "y_train": y_tr,
            "y_test": y_te,
            "idx_train": np.array([sid_to_idx[int(s)] for s in sid_tr], dtype=np.int64),
            "idx_test": np.array([sid_to_idx[int(s)] for s in sid_te], dtype=np.int64),
        }
        for row_i, sid in enumerate(sid_tr):
            s = int(sid)
            if s not in master_labels:
                master_labels[s] = np.full(len(ISOZYMES), np.nan, dtype=np.float32)
            master_labels[s][j] = float(y_tr[row_i])

    master_sids = np.array(sorted(master_labels.keys()), dtype=np.int64)
    master_y = np.vstack([master_labels[int(s)] for s in master_sids]).astype(np.float32)
    master_idx = np.array([sid_to_idx[int(s)] for s in master_sids], dtype=np.int64)

    return {
        "graphs": graphs,
        "rdkit_desc": rd,
        "iso_data": iso_data,
        "master_sids": master_sids,
        "master_y": master_y,
        "master_idx": master_idx,
    }


@torch.no_grad()
def predict_graphs(
    model: nn.Module,
    graphs: list,
    indices: np.ndarray,
    desc: np.ndarray | None,
    target_idx: int,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    preds = np.zeros(len(indices), dtype=np.float64)
    use_amp = DEVICE.type == "cuda"
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        b_idx = indices[start:end]
        b_graphs = [graphs[int(i)] for i in b_idx]
        b_desc = desc[start:end] if desc is not None else None
        x, ei, ea, batch, ng, d_t = collate_graphs(b_graphs, b_desc)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x, ei, ea, batch, ng, d_t)[:, target_idx]
        preds[start:end] = torch.sigmoid(logits.float()).cpu().numpy()
    return preds


def train_one_gnn(
    graphs: list,
    tr_gidx: np.ndarray,
    y_tr_mat: np.ndarray,
    desc_tr: np.ndarray | None,
    va_gidx: np.ndarray,
    y_va_target: np.ndarray,
    desc_va: np.ndarray | None,
    te_gidx: np.ndarray,
    desc_te: np.ndarray | None,
    target_idx: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    set_seed(seed)
    mask_tr = (~np.isnan(y_tr_mat)).astype(np.float32)
    y_tr_clean = np.where(np.isnan(y_tr_mat), 0.0, y_tr_mat).astype(np.float32)

    task_weights = torch.full(
        (1, len(ISOZYMES)), AUX_TASK_WEIGHT, dtype=torch.float32, device=DEVICE
    )
    task_weights[0, target_idx] = TARGET_TASK_WEIGHT

    desc_dim = desc_tr.shape[1] if desc_tr is not None else 0
    model = MultiTaskGINE(
        atom_dim=ATOM_DIM,
        bond_dim=BOND_DIM,
        hidden_dim=256,
        n_layers=4,
        desc_dim=desc_dim,
        n_tasks=len(ISOZYMES),
        dropout=0.25,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-5
    )
    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n_samples = len(tr_gidx)
    best_auc = -1.0
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = np.random.permutation(n_samples)
        for start in range(0, n_samples, BATCH_SIZE):
            b = perm[start : start + BATCH_SIZE]
            b_graphs = [graphs[int(tr_gidx[i])] for i in b]
            b_desc = desc_tr[b] if desc_tr is not None else None
            x, ei, ea, batch, ng, d_t = collate_graphs(b_graphs, b_desc)
            yb = torch.from_numpy(y_tr_clean[b]).to(DEVICE, non_blocking=True)
            mb = torch.from_numpy(mask_tr[b]).to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x, ei, ea, batch, ng, d_t)
                raw_loss = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
                w = mb * task_weights
                loss = (raw_loss * w).sum() / w.sum().clamp_min(1.0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        va_prob = predict_graphs(model, graphs, va_gidx, desc_va, target_idx)
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
    val_prob = predict_graphs(model, graphs, va_gidx, desc_va, target_idx)
    test_prob = predict_graphs(model, graphs, te_gidx, desc_te, target_idx)
    return val_prob, test_prob, best_auc, best_epoch


def run_variant_for_isozyme(
    data: dict, iso: str, use_desc: bool, n_seeds: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    target_idx = ISOZYMES.index(iso)
    iso_d = data["iso_data"][iso]
    sid_tr, sid_te = iso_d["sid_train"], iso_d["sid_test"]
    y_tr = iso_d["y_train"]
    idx_tr, idx_te = iso_d["idx_train"], iso_d["idx_test"]

    graphs = data["graphs"]
    rd_all = data["rdkit_desc"]
    master_sids = data["master_sids"]
    master_y = data["master_y"]
    master_idx = data["master_idx"]

    test_sid_set = set(sid_te.tolist())
    non_test_mask = np.array([int(s) not in test_sid_set for s in master_sids], dtype=bool)
    pool_sids = master_sids[non_test_mask]
    pool_y = master_y[non_test_mask]
    pool_idx = master_idx[non_test_mask]

    assert len(set(pool_sids.tolist()) & test_sid_set) == 0, "LEAKAGE: test SID in pool!"

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_SEED)
    folds = list(skf.split(np.zeros(len(y_tr)), y_tr))

    oof_prob = np.zeros(len(y_tr), dtype=np.float64)
    test_prob = np.zeros(len(sid_te), dtype=np.float64)
    best_epochs = []

    for fold_i, (_, va_idx) in enumerate(folds):
        val_sid_set = set(sid_tr[va_idx].tolist())
        fold_tr_mask = np.array([int(s) not in val_sid_set for s in pool_sids], dtype=bool)
        f_sids = pool_sids[fold_tr_mask]
        f_y = pool_y[fold_tr_mask]
        f_idx = pool_idx[fold_tr_mask]

        assert len(set(f_sids.tolist()) & val_sid_set) == 0, "LEAKAGE: val SID in fold!"
        assert len(set(f_sids.tolist()) & test_sid_set) == 0, "LEAKAGE: test SID in fold!"

        va_gidx = idx_tr[va_idx]
        if use_desc:
            imp = SimpleImputer(strategy="median")
            d_tr_i = imp.fit_transform(rd_all[f_idx])
            d_va_i = imp.transform(rd_all[va_gidx])
            d_te_i = imp.transform(rd_all[idx_te])
            qt = QuantileTransformer(
                n_quantiles=min(1000, len(d_tr_i)),
                output_distribution="normal",
                random_state=SPLIT_SEED,
            )
            desc_tr = np.clip(qt.fit_transform(d_tr_i), -8.0, 8.0).astype(np.float32)
            desc_va = np.clip(qt.transform(d_va_i), -8.0, 8.0).astype(np.float32)
            desc_te = np.clip(qt.transform(d_te_i), -8.0, 8.0).astype(np.float32)
        else:
            desc_tr = desc_va = desc_te = None

        va_pred_f = np.zeros(len(va_idx), dtype=np.float64)
        te_pred_f = np.zeros(len(sid_te), dtype=np.float64)
        for s in range(n_seeds):
            vp, tp, _, be = train_one_gnn(
                graphs, f_idx, f_y, desc_tr,
                va_gidx, y_tr[va_idx], desc_va,
                idx_te, desc_te,
                target_idx, seed=100 * fold_i + s,
            )
            va_pred_f += vp / n_seeds
            te_pred_f += tp / n_seeds
            best_epochs.append(be)

        oof_prob[va_idx] = va_pred_f
        test_prob += te_pred_f / N_SPLITS

    return oof_prob, test_prob, {"mean_best_epoch": float(np.mean(best_epochs))}


def main() -> None:
    global DATA_DIR, RESULTS_DIR, PREDS_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=2, help="Random seeds per fold")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--isozymes", nargs="+", default=ISOZYMES, choices=ISOZYMES)
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    RESULTS_DIR = args.results_dir
    PREDS_DIR = RESULTS_DIR / "tier3_gnn_preds"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"Tier 3A Multi-Task GINE GNN on device: {DEVICE} (seeds/fold={args.seeds})", flush=True)
    data = load_all_data()

    variants = [
        ("GraphOnly", "MT_GNN_GraphOnly", False),
        ("Graph+RDKitDesc", "MT_GNN_GraphPlusDesc", True),
    ]
    summary_rows: list[dict] = []

    for iso in args.isozymes:
        iso_d = data["iso_data"][iso]
        y_tr, y_te = iso_d["y_train"], iso_d["y_test"]
        print(f"\n{'='*72}\nTier 3A GNN: CYP {iso.upper()} (train={len(y_tr)}, test={len(y_te)})\n{'='*72}", flush=True)

        npz_payload = {
            "y_train": y_tr,
            "y_test": y_te,
            "sid_train": iso_d["sid_train"],
            "sid_test": iso_d["sid_test"],
        }
        for feat_set, model_name, use_desc in variants:
            t_v = time.time()
            oof, test_p, info = run_variant_for_isozyme(data, iso, use_desc=use_desc, n_seeds=args.seeds)
            m = compute_metrics(y_tr, oof, y_te, test_p)
            summary_rows.append({"isozyme": iso, "feature_set": feat_set, "model": model_name, **m})
            npz_payload[f"oof_{model_name}"] = oof
            npz_payload[f"test_{model_name}"] = test_p
            print(
                f"  [{iso}] {model_name:22s} | CV ACC={m['cv_acc']:.4f} AUC={m['cv_auc']:.4f} "
                f"| TEST ACC={m['test_acc']:.4f} AUC={m['test_auc']:.4f} "
                f"Brier={m['test_brier']:.4f} LL={m['test_logloss']:.4f} "
                f"(ep={info['mean_best_epoch']:.1f}, {time.time() - t_v:.1f}s)",
                flush=True,
            )

        np.savez_compressed(PREDS_DIR / f"{iso}.npz", **npz_payload)
        pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "tier3_gnn_summary.csv", index=False)

    print(f"\nAll 5 isozymes finished in {(time.time() - t0)/60:.2f} min.", flush=True)


if __name__ == "__main__":
    main()
