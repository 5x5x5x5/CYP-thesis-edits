"""Generate Figure 7: Multi-Modal Architecture & Zero-Leakage Multi-Task Firewall Schematic.

Writes `extension/img/fig7_architecture_and_leakage.png` at 300 DPI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "extension" / "img"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "ink": "#2b2b29",
    "gray": "#8a8a85",
    "slate": "#4d6473",
    "blue": "#2a78d6",
    "gold": "#d99b26",
    "orange": "#eb6834",
    "teal": "#1baf7a",
    "purple": "#7a5195",
    "red": "#c0392b",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
    }
)


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str = "",
    facecolor: str = "#f7f8fa",
    edgecolor: str = "#4d6473",
    title_color: str = "#2b2b29",
    lw: float = 1.4,
    fontsize: float = 8.6,
) -> None:
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        zorder=3,
    )
    ax.add_patch(box)
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.64,
            title,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=title_color,
            zorder=4,
        )
        ax.text(
            x + w / 2,
            y + h * 0.28,
            subtitle,
            ha="center",
            va="center",
            fontsize=fontsize - 1.3,
            color="#444444",
            zorder=4,
        )
    else:
        ax.text(
            x + w / 2,
            y + h / 2,
            title,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=title_color,
            zorder=4,
        )


def arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, color: str = "#555555") -> None:
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.32", color=color, lw=1.3),
        zorder=2,
    )


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.6, 5.8), gridspec_kw={"width_ratios": [1.38, 1.0]}
    )

    # =========================================================================
    # PANEL A: Multi-Modal Four-Branch Architecture -> Nested Meta-Stack
    # =========================================================================
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    ax1.set_title(
        "A. Multi-Modal CYP Modeling Architecture (Tiers 1–4)",
        fontsize=11.5,
        fontweight="bold",
        color=PALETTE["ink"],
        pad=10,
    )

    # Input molecule box
    draw_box(
        ax1,
        0.02,
        0.38,
        0.19,
        0.24,
        "Compound\n(PubChem SID)",
        "SMILES + 186 MOE\n(100% test coverage)",
        facecolor="#eef3f8",
        edgecolor=PALETTE["slate"],
        fontsize=8.6,
    )

    # Four orthogonal representation + model branches
    branches = [
        (
            0.78,
            "1. Tabular & Fingerprint GBDTs",
            "LightGBM · XGBoost · CatBoost (MOE / RDKit)",
            "#edf5ff",
            PALETTE["blue"],
            "Local CPU",
        ),
        (
            0.55,
            "2. Similarity & Hybrid Kernels",
            "Exact Tanimoto SVM · Hybrid (α·Tan + (1−α)·RBF)",
            "#fef8ec",
            PALETTE["gold"],
            "Local CPU",
        ),
        (
            0.32,
            "3. 5-Head Multi-Task Deep Nets",
            "MT-ResMLP (FP+Desc) · MT-GINE+VN (2D Graph)",
            "#fff2ec",
            PALETTE["orange"],
            "Colab T4 GPU",
        ),
        (
            0.09,
            "4. Pretrained Chemical LM",
            "MT-ChemBERTa-77M-MTR (Raw SMILES Only)",
            "#f5eef8",
            PALETTE["purple"],
            "Colab T4 GPU",
        ),
    ]

    for y_b, title, sub, fc, ec, hw in branches:
        draw_box(ax1, 0.28, y_b, 0.43, 0.16, title, sub, facecolor=fc, edgecolor=ec, title_color=ec, fontsize=8.5)
        # Hardware pill
        ax1.text(
            0.70,
            y_b + 0.150,
            hw,
            ha="right",
            va="top",
            fontsize=6.6,
            fontweight="bold",
            color=ec,
        )
        # Arrow from input to branch
        arrow(ax1, 0.215, 0.50, 0.275, y_b + 0.08, color=ec)
        # Arrow from branch to stacker
        arrow(ax1, 0.715, y_b + 0.08, 0.775, 0.50, color=ec)

    # Nested meta-stacker box
    draw_box(
        ax1,
        0.78,
        0.33,
        0.20,
        0.34,
        "Nested OOF\nMeta-Stacker",
        "L2 LogRegCV on logits\n+ Simplex Blend\n\nMean AUC: 0.8792\nMean ECE: 0.0216",
        facecolor="#eaf8f2",
        edgecolor=PALETTE["teal"],
        title_color="#117a53",
        lw=1.8,
        fontsize=8.6,
    )

    # Bottom note
    ax1.text(
        0.50,
        0.015,
        "Open-Source Stack (Branches 1–4 without MOE descriptors): 0.8777 Mean AUC (Δ = −0.0015 vs. All-Modal, ns on all 5 isozymes)",
        ha="center",
        va="bottom",
        fontsize=7.6,
        style="italic",
        color="#444444",
    )

    # =========================================================================
    # PANEL B: Zero-Leakage Multi-Task Pool Firewall
    # =========================================================================
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.set_title(
        "B. Zero-Leakage Multi-Task Pool Firewall",
        fontsize=11.5,
        fontweight="bold",
        color=PALETTE["ink"],
        pad=10,
    )

    # Step 1: Master pool
    draw_box(
        ax2,
        0.06,
        0.79,
        0.88,
        0.14,
        "Master Multi-Task Table (16,089 Unique Training SIDs)",
        "Sparse 5-head label matrix Y ∈ {0, 1, NaN}⁵ across 1A2, 2C9, 2C19, 2D6, 3A4",
        facecolor="#eef3f8",
        edgecolor=PALETTE["slate"],
        fontsize=8.3,
    )

    # Firewall 1: Test purge
    arrow(ax2, 0.50, 0.785, 0.50, 0.715, color=PALETTE["red"])
    draw_box(
        ax2,
        0.06,
        0.58,
        0.88,
        0.13,
        "FIREWALL 1: Purge ALL test{iso} SIDs Across All 5 Heads",
        "assert len(pool_SIDs ∩ test{iso}_SIDs) == 0   (zero cross-isozyme test leakage)",
        facecolor="#fdeeee",
        edgecolor=PALETTE["red"],
        title_color=PALETTE["red"],
        lw=1.6,
        fontsize=8.2,
    )

    # Firewall 2: Fold purge
    arrow(ax2, 0.50, 0.575, 0.50, 0.505, color=PALETTE["orange"])
    draw_box(
        ax2,
        0.06,
        0.37,
        0.88,
        0.13,
        "FIREWALL 2: Per-Fold Purge of val_k{iso} SIDs (All 5 Heads)",
        "assert len(fold_k_SIDs ∩ val_k_SIDs) == 0   ·   Fit Imputer & QuantileScaler on fold k only",
        facecolor="#fff4eb",
        edgecolor=PALETTE["orange"],
        title_color="#b84a1b",
        lw=1.6,
        fontsize=8.1,
    )

    # Masked multi-task training
    arrow(ax2, 0.50, 0.365, 0.50, 0.295, color=PALETTE["blue"])
    draw_box(
        ax2,
        0.06,
        0.16,
        0.88,
        0.13,
        "Masked Multi-Task Optimization (Target Weight 2.0, Aux 1.0)",
        "Early-stop strictly on target val_k ROC-AUC  →  5-fold OOF & test ensemble",
        facecolor="#edf5ff",
        edgecolor=PALETTE["blue"],
        title_color=PALETTE["blue"],
        fontsize=8.2,
    )

    # Final held-out eval
    arrow(ax2, 0.50, 0.155, 0.50, 0.105, color=PALETTE["teal"])
    draw_box(
        ax2,
        0.06,
        0.02,
        0.88,
        0.08,
        "100% Held-Out Test Evaluation (DeLong + McNemar + ECE)",
        "",
        facecolor="#eaf8f2",
        edgecolor=PALETTE["teal"],
        title_color="#117a53",
        fontsize=8.0,
    )

    fig.tight_layout(w_pad=1.8)
    fig.savefig(OUT_DIR / "fig7_architecture_and_leakage.png")
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'fig7_architecture_and_leakage.png'}")


if __name__ == "__main__":
    main()
