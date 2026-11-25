"""Select the Top 300 high-value held-out test compounds for Stage #3 Boltz-2 + Heme 3D study.

Selects 100 compounds each from CYP 1A2, CYP 2D6, and CYP 3A4 held-out test sets
(300 total), stratified into:
  - 50 Novel-Scaffold test compounds (Murcko scaffold absent from that isozyme's train set),
    balanced 25 true inhibitors / 25 true non-inhibitors, ranked by 2D-model uncertainty.
  - 50 Seen-Scaffold hard/uncertain test compounds, balanced 25 / 25, ranked by 2D-model uncertainty.

Saves `notebooks/results/boltz_top300_manifest.csv` with SID, isozyme, CanonicalSMILES,
y_test, ActivityScore, is_novel_scaffold, grand_prob, uncertainty.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "notebooks" / "data"
RESULTS_DIR = ROOT / "notebooks" / "results"

TARGET_ISOS = ["1a2", "2d6", "3a4"]
PER_ISO = 100


def main() -> None:
    sm = pd.read_csv(DATA_DIR / "sid_smiles.csv", usecols=["SID", "CanonicalSMILES", "ScaffoldSMILES"])
    sm_map = sm.set_index("SID").to_dict(orient="index")

    rows = []
    for iso in TARGET_ISOS:
        tr = pd.read_csv(DATA_DIR / f"training{iso}.csv", usecols=["SID", "ActivityScore"])
        te = pd.read_csv(DATA_DIR / f"test{iso}.csv", usecols=["SID", "ActivityScore"])
        gp = np.load(RESULTS_DIR / "grand_preds" / f"{iso}.npz")
        grand_prob = gp["test_Grand_AllModal_Stack"].astype(np.float64)
        assert np.array_equal(te["SID"].to_numpy(), gp["sid_test"])

        train_scafs = set()
        for s in tr["SID"].to_numpy():
            sc = sm_map[int(s)]["ScaffoldSMILES"]
            if isinstance(sc, str) and sc.strip():
                train_scafs.add(sc.strip())

        recs = []
        for idx, (sid, score, prob) in enumerate(
            zip(te["SID"].to_numpy(), te["ActivityScore"].to_numpy(), grand_prob)
        ):
            info = sm_map[int(sid)]
            smi = info["CanonicalSMILES"]
            scaf = info["ScaffoldSMILES"]
            if not isinstance(smi, str) or not smi.strip() or len(smi) > 160:
                continue
            novel = (not isinstance(scaf, str)) or (not scaf.strip()) or (scaf.strip() not in train_scafs)
            y = int(score >= 40)
            unc = 1.0 - 2.0 * abs(float(prob) - 0.5)  # 1.0 at p=0.5, 0.0 at p=0 or 1
            recs.append(
                {
                    "isozyme": iso,
                    "test_idx": idx,
                    "SID": int(sid),
                    "CanonicalSMILES": smi.strip(),
                    "ScaffoldSMILES": scaf.strip() if isinstance(scaf, str) else "",
                    "ActivityScore": float(score),
                    "y_test": y,
                    "is_novel_scaffold": bool(novel),
                    "grand_prob": round(float(prob), 6),
                    "uncertainty": round(unc, 6),
                }
            )
        df = pd.DataFrame(recs)

        picked = []
        for novel_flag in [True, False]:
            for y_val in [1, 0]:
                sub = df[(df["is_novel_scaffold"] == novel_flag) & (df["y_test"] == y_val)]
                sub = sub.sort_values("uncertainty", ascending=False).head(PER_ISO // 4)
                picked.append(sub)
        iso_df = pd.concat(picked, ignore_index=True)
        print(
            f"{iso}: selected {len(iso_df)} compounds "
            f"(novel={iso_df['is_novel_scaffold'].sum()}, pos={iso_df['y_test'].sum()}, "
            f"mean_unc={iso_df['uncertainty'].mean():.3f})"
        )
        rows.append(iso_df)

    out = pd.concat(rows, ignore_index=True)
    out_path = RESULTS_DIR / "boltz_top300_manifest.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} compounds -> {out_path}")


if __name__ == "__main__":
    main()
