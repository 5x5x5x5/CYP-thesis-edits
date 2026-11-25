"""Tier 3B: 100% Leak-Free Multi-Task Fine-Tuned ChemBERTa-77M-MTR (SMILES Transformer) on GPU.

End-to-end 5-head Multi-Task fine-tuning of `DeepChem/ChemBERTa-77M-MTR` (pretrained on
77M PubChem molecules) directly on CanonicalSMILES under the exact same strict
zero-leakage protocol as Tier 2B and Tier 3A:
  1. Master multi-task table of all SIDs appearing in any training{j}.csv.
  2. For each target isozyme `iso`:
     - Exclude ALL SIDs in test{iso}.csv from the entire 5-head training pool.
     - Use the exact 5-fold StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
       splits on training{iso}.csv.
     - For each fold k:
       * Exclude ALL fold-k validation SIDs from ALL 5 heads in the training pool.
       * Fine-tune ChemBERTa-77M-MTR + 5-head linear/MLP classifier with masked
         BCEWithLogitsLoss (weight 2.0 on target head, 1.0 on 4 auxiliary heads)
         and fp16 mixed precision on CUDA.
       * Early-stop on target isozyme validation ROC-AUC, restore best weights.
       * Average the 5 fold models' predictions on 100% of test{iso}.

Model saved:
  - MT_ChemBERTa_77M_MTR (100% raw SMILES, zero hand-crafted descriptors or fingerprints)
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
PREDS_DIR = RESULTS_DIR / "tier3_chemberta_preds"

MODEL_ID = "DeepChem/ChemBERTa-77M-MTR"
ISOZYMES = ["1a2", "2c9", "2c19", "2d6", "3a4"]
ACTIVITY_THRESHOLD = 40
SPLIT_SEED = 0
N_SPLITS = 5

MAX_LEN = 128
BATCH_SIZE = 128
MAX_EPOCHS = 12
PATIENCE = 3
LR = 3e-4
WEIGHT_DECAY = 1e-2
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


class MultiTaskChemBERTa(nn.Module):
    def __init__(self, model_id: str = MODEL_ID, n_tasks: int = 5, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_tasks),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls_emb = out[:, 0]
        mask_f = attention_mask.unsqueeze(-1).type_as(out)
        mean_emb = (out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        return self.head(torch.cat([cls_emb, mean_emb], dim=-1))


def load_all_data() -> dict:
    t0 = time.time()
    sm_df = pd.read_csv(DATA_DIR / "sid_smiles.csv", usecols=["SID", "CanonicalSMILES", "SMILES"])
    sids = sm_df["SID"].to_numpy(dtype=np.int64)
    sid_to_idx = {int(s): i for i, s in enumerate(sids)}

    smiles_list: list[str] = []
    for cs, s in zip(sm_df["CanonicalSMILES"], sm_df["SMILES"]):
        if isinstance(cs, str) and cs.strip():
            smiles_list.append(cs.strip())
        elif isinstance(s, str) and s.strip():
            smiles_list.append(s.strip())
        else:
            smiles_list.append("")

    print(f"Tokenizing {len(smiles_list)} SMILES with {MODEL_ID}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    enc = tokenizer(
        smiles_list,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="np",
    )
    input_ids_all = enc["input_ids"].astype(np.int64)
    attn_mask_all = enc["attention_mask"].astype(np.int64)
    print(f"Tokenized shape={input_ids_all.shape} in {time.time() - t0:.1f}s", flush=True)

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
        "input_ids": input_ids_all,
        "attn_mask": attn_mask_all,
        "iso_data": iso_data,
        "master_sids": master_sids,
        "master_y": master_y,
        "master_idx": master_idx,
    }


@torch.no_grad()
def predict_tokens(
    model: nn.Module,
    input_ids_all: np.ndarray,
    attn_mask_all: np.ndarray,
    indices: np.ndarray,
    target_idx: int,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    preds = np.zeros(len(indices), dtype=np.float64)
    use_amp = DEVICE.type == "cuda"
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        b_idx = indices[start:end]
        ids = torch.from_numpy(input_ids_all[b_idx]).to(DEVICE, non_blocking=True)
        msk = torch.from_numpy(attn_mask_all[b_idx]).to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(ids, msk)[:, target_idx]
        preds[start:end] = torch.sigmoid(logits.float()).cpu().numpy()
    return preds


def train_one_fold(
    input_ids_all: np.ndarray,
    attn_mask_all: np.ndarray,
    tr_idx_all: np.ndarray,
    y_tr_mat: np.ndarray,
    va_idx_all: np.ndarray,
    y_va_target: np.ndarray,
    te_idx_all: np.ndarray,
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

    model = MultiTaskChemBERTa(MODEL_ID, n_tasks=len(ISOZYMES), dropout=0.2).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = MAX_EPOCHS * ((len(tr_idx_all) + BATCH_SIZE - 1) // BATCH_SIZE)
    warmup_steps = max(1, int(0.1 * total_steps))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + np.cos(np.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_auc = -1.0
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    no_improve = 0
    n_samples = len(tr_idx_all)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = np.random.permutation(n_samples)
        for start in range(0, n_samples, BATCH_SIZE):
            b = perm[start : start + BATCH_SIZE]
            g_idx = tr_idx_all[b]
            ids = torch.from_numpy(input_ids_all[g_idx]).to(DEVICE, non_blocking=True)
            msk = torch.from_numpy(attn_mask_all[g_idx]).to(DEVICE, non_blocking=True)
            yb = torch.from_numpy(y_tr_clean[b]).to(DEVICE, non_blocking=True)
            mb = torch.from_numpy(mask_tr[b]).to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(ids, msk)
                raw_loss = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
                w = mb * task_weights
                loss = (raw_loss * w).sum() / w.sum().clamp_min(1.0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        va_prob = predict_tokens(model, input_ids_all, attn_mask_all, va_idx_all, target_idx)
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
    val_prob = predict_tokens(model, input_ids_all, attn_mask_all, va_idx_all, target_idx)
    test_prob = predict_tokens(model, input_ids_all, attn_mask_all, te_idx_all, target_idx)
    return val_prob, test_prob, best_auc, best_epoch


def run_for_isozyme(data: dict, iso: str) -> tuple[np.ndarray, np.ndarray, dict]:
    target_idx = ISOZYMES.index(iso)
    iso_d = data["iso_data"][iso]
    sid_tr, sid_te = iso_d["sid_train"], iso_d["sid_test"]
    y_tr = iso_d["y_train"]
    idx_tr, idx_te = iso_d["idx_train"], iso_d["idx_test"]

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

        vp, tp, _, be = train_one_fold(
            data["input_ids"],
            data["attn_mask"],
            f_idx,
            f_y,
            idx_tr[va_idx],
            y_tr[va_idx],
            idx_te,
            target_idx,
            seed=100 * fold_i,
        )
        oof_prob[va_idx] = vp
        test_prob += tp / N_SPLITS
        best_epochs.append(be)

    return oof_prob, test_prob, {"mean_best_epoch": float(np.mean(best_epochs))}


def main() -> None:
    global DATA_DIR, RESULTS_DIR, PREDS_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--isozymes", nargs="+", default=ISOZYMES, choices=ISOZYMES)
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    RESULTS_DIR = args.results_dir
    PREDS_DIR = RESULTS_DIR / "tier3_chemberta_preds"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"Tier 3B Multi-Task {MODEL_ID} on device: {DEVICE}", flush=True)
    data = load_all_data()
    summary_rows: list[dict] = []

    for iso in args.isozymes:
        t_iso = time.time()
        iso_d = data["iso_data"][iso]
        y_tr, y_te = iso_d["y_train"], iso_d["y_test"]
        print(f"\n{'='*72}\nTier 3B ChemBERTa: CYP {iso.upper()} (train={len(y_tr)}, test={len(y_te)})\n{'='*72}", flush=True)

        oof, test_p, info = run_for_isozyme(data, iso)
        m = compute_metrics(y_tr, oof, y_te, test_p)
        model_name = "MT_ChemBERTa_77M_MTR"
        summary_rows.append(
            {"isozyme": iso, "feature_set": "SMILES_Only", "model": model_name, **m}
        )
        print(
            f"  [{iso}] {model_name:22s} | CV ACC={m['cv_acc']:.4f} AUC={m['cv_auc']:.4f} "
            f"| TEST ACC={m['test_acc']:.4f} AUC={m['test_auc']:.4f} "
            f"Brier={m['test_brier']:.4f} LL={m['test_logloss']:.4f} "
            f"(ep={info['mean_best_epoch']:.1f}, {time.time() - t_iso:.1f}s)",
            flush=True,
        )
        np.savez_compressed(
            PREDS_DIR / f"{iso}.npz",
            y_train=y_tr,
            y_test=y_te,
            sid_train=iso_d["sid_train"],
            sid_test=iso_d["sid_test"],
            oof_MT_ChemBERTa_77M_MTR=oof,
            test_MT_ChemBERTa_77M_MTR=test_p,
        )
        pd.DataFrame(summary_rows).to_csv(
            RESULTS_DIR / "tier3_chemberta_summary.csv", index=False
        )

    print(f"\nAll 5 isozymes finished in {(time.time() - t0)/60:.2f} min.", flush=True)


if __name__ == "__main__":
    main()
