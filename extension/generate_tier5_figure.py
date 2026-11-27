"""Generate Figure 8: Tier 5 Frontier Uplift (MoLFormer-XL + 3D Conformers + Continuous ActivityScore)
and Stage #3 Boltz-2 [CYP 1A2 + HEM + Ligand] 3D Structural Rescue on the 100 Hardest Test Compounds.

Saves `extension/img/fig8_tier5_and_boltz2.png` at 300 DPI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "notebooks" / "results"
OUT_DIR = ROOT / "extension" / "img"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ISOZYMES = ["1a2", "2c9", "2c19", "2d6", "3a4"]
PALETTE = {
    "ink": "#2b2b29",
    "slate": "#4d6473",
    "blue": "#2a78d6",
    "gold": "#d99b26",
    "orange": "#eb6834",
    "teal": "#1baf7a",
    "purple": "#7a5195",
    "crimson": "#c0392b",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11.2,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)


def main() -> None:
    m_old = pd.read_csv(RESULTS / "master_comparison.csv")
    t5_mf = pd.read_csv(RESULTS / "tier5_molformer_summary.csv")
    t5_jt = pd.read_csv(RESULTS / "tier5_joint_summary.csv")
    t5_gr = pd.read_csv(RESULTS / "tier5_grand_summary.csv")
    boltz = pd.read_csv(RESULTS / "boltz2_1a2_2d6_results.csv")

    # Panel A: Tier 4 -> Tier 5 mean ROC-AUC milestones
    milestones = [
        ("2015 SVM\n(MOE)", m_old[m_old.model == "SVM_baseline"]["test_auc"].mean(), PALETTE["slate"]),
        ("Tier 1\nMOE Stack", m_old[m_old.model == "Stack_LogReg"]["test_auc"].mean(), PALETTE["blue"]),
        ("Tier 3B\nChemBERTa", m_old[m_old.model == "MT_ChemBERTa_77M_MTR"]["test_auc"].mean(), PALETTE["gold"]),
        ("Stage #1\nMoLFormer-XL", t5_mf["test_auc"].mean(), PALETTE["orange"]),
        ("Stage #2\nMT-Joint3D\n(0 MOE)", t5_jt[t5_jt.model == "MT_Joint3D_OpenSource"]["test_auc"].mean(), PALETTE["purple"]),
        ("Tier 4\nGrand Stack", m_old[m_old.model == "Grand_AllModal_Stack"]["test_auc"].mean(), "#4db6ac"),
        ("Tier 5\nOpen-Source\n(0 MOE)", t5_gr[t5_gr.model == "Tier5_OpenSource_Stack"]["test_auc"].mean(), "#159a6b"),
        ("Tier 5\nAll-Modal\n(2026 Peak)", t5_gr[t5_gr.model == "Tier5_AllModal_Stack"]["test_auc"].mean(), PALETTE["teal"]),
    ]

    # Panel B: Boltz-2 on the 100 hardest CYP 1A2 held-out test compounds
    b1a2 = boltz[boltz["isozyme"] == "1a2"].copy()
    t5_1a2 = np.load(RESULTS / "tier5_grand_preds" / "1a2.npz")
    sid_to_t5 = dict(zip(t5_1a2["sid_test"].tolist(), t5_1a2["test_Tier5_AllModal_Stack"].tolist()))
    b1a2["t5_prob"] = [sid_to_t5[int(s)] for s in b1a2["SID"]]

    # Zero-fit rank average of Tier 5 + Boltz-2 ligand_ipTM + Boltz-2 (-d_fe_het)
    r_t5 = rankdata(b1a2["t5_prob"]) / len(b1a2)
    r_iptm = rankdata(b1a2["ligand_iptm"]) / len(b1a2)
    r_fe = rankdata(-b1a2["d_fe_het"]) / len(b1a2)
    b1a2["hybrid_3d_rank"] = 0.4 * r_t5 + 0.4 * r_iptm + 0.2 * r_fe
    b1a2["neg_fe"] = -b1a2["d_fe_het"]

    slices = [
        ("All 100 Hardest\nCYP 1A2", b1a2),
        ("50 Novel\nMurcko Scaffolds", b1a2[b1a2["is_novel_scaffold"]]),
        ("50 Seen-Scaffold\nBorderline", b1a2[~b1a2["is_novel_scaffold"]]),
    ]
    b_series = [
        ("2D+SMILES Tier 5 Stack", "t5_prob", PALETTE["slate"]),
        ("Pure 3D −d(Lig, Heme-Fe)", "neg_fe", PALETTE["orange"]),
        ("Zero-Shot Boltz-2 ipTM", "ligand_iptm", PALETTE["blue"]),
        ("Tier 5 + Boltz-2 3D Fusion", "hybrid_3d_rank", PALETTE["teal"]),
    ]

    print("=== BOLTZ-2 CYP 1A2 HARD-100 EVALUATION ===")
    for s_lbl, sdf in slices:
        row_str = f"  {s_lbl.replace(chr(10), ' '):28s} | "
        for lbl, col, _ in b_series:
            auc = roc_auc_score(sdf["y_test"], sdf[col])
            row_str += f"{lbl.split()[0]}={auc:.4f}  "
        print(row_str)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.2), gridspec_kw={"width_ratios": [1.25, 1.0]})

    # Plot Panel A
    xs = np.arange(len(milestones))
    vals = [v for _, v, _ in milestones]
    cols = [c for _, _, c in milestones]
    bars = ax1.bar(xs, vals, width=0.62, color=cols, edgecolor="white")
    for b, v in zip(bars, vals):
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            v + 0.0012,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.1,
            fontweight="bold",
            color=PALETTE["ink"],
        )
    ax1.set_xticks(xs, [lbl for lbl, _, _ in milestones], fontsize=7.9)
    ax1.set_ylim(0.83, 0.892)
    ax1.set_ylabel("Mean Held-Out Test ROC-AUC (5 Isozymes)")
    ax1.set_title("A. Stage #1 & #2 Frontier Lift: Reaching 0.8810 Mean AUC")
    clean_axes(ax1)

    # Plot Panel B
    x2 = np.arange(len(slices))
    w = 0.19
    for i, (lbl, col, color) in enumerate(b_series):
        aucs = [roc_auc_score(sdf["y_test"], sdf[col]) for _, sdf in slices]
        off = (i - 1.5) * w
        bs = ax2.bar(x2 + off, aucs, width=w * 0.92, label=lbl, color=color, edgecolor="white")
        for b, v in zip(bs, aucs):
            ax2.text(
                b.get_x() + b.get_width() / 2.0,
                v + 0.006,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.3,
                fontweight="bold",
                color=PALETTE["ink"],
            )
    ax2.axhline(0.5, color="#888888", ls="--", lw=1.0, label="Random (0.500)")
    ax2.set_xticks(x2, [s_lbl for s_lbl, _ in slices], fontsize=8.8)
    ax2.set_ylim(0.45, 0.78)
    ax2.set_ylabel("Held-Out Test ROC-AUC on Hardest Slice")
    ax2.set_title("B. Stage #3 Boltz-2 [CYP 1A2 + HEM + Ligand] Rescue")
    clean_axes(ax2)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#dddddd", fontsize=7.8)

    fig.tight_layout(w_pad=2.0)
    out_path = OUT_DIR / "fig8_tier5_and_boltz2.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
