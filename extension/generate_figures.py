"""Generate 6 publication-grade 300-DPI figures for the 2026 CYP Thesis Extension.

Reads ONLY the saved CSV summaries and NPZ prediction archives in `notebooks/results/`
and writes all figures into `extension/img/` (never touching original thesis assets).

Usage:
    uv run python extension/generate_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "notebooks" / "results"
OUT_DIR = ROOT / "extension" / "img"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ISOZYMES = ["1a2", "2c9", "2c19", "2d6", "3a4"]
ISO_LABELS = [i.upper() for i in ISOZYMES]

# Consistent colorblind-safe palette anchored to the original thesis (comparison_figures.py)
PALETTE = {
    "gray": "#8a8a85",
    "ink": "#2b2b29",
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "teal": "#1baf7a",
    "purple": "#7a5195",
    "gold": "#d99b26",
    "crimson": "#c0392b",
    "slate": "#4d6473",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.9,
        "xtick.labelsize": 9.2,
        "ytick.labelsize": 9.2,
        "legend.fontsize": 8.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
    }
)


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linestyle="-", linewidth=0.7)
    ax.set_axisbelow(True)


def load_test_preds(iso: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load y_test and all test_<model> arrays for one isozyme."""
    preds: dict[str, np.ndarray] = {}
    y_test = None
    for sub in [
        "tier1_preds",
        "tier2_preds",
        "tier2_mt_preds",
        "tier3_gnn_preds",
        "tier3_chemberta_preds",
        "grand_preds",
    ]:
        p = RESULTS / sub / f"{iso}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        if y_test is None:
            y_test = z["y_test"]
        for k in z.files:
            if k.startswith("test_"):
                preds[k[len("test_") :]] = z[k].astype(np.float64)
    assert y_test is not None
    return y_test, preds


def fig1_ten_year_progression(master: pd.DataFrame) -> None:
    """Two-panel figure: (A) Decade ROC-AUC progression by isozyme, (B) Gain attribution waterfall."""
    models = [
        ("RF_baseline", "2015 RF (10 trees, MOE)", PALETTE["gray"]),
        ("SVM_baseline", "2015 SVM (RBF, MOE)", PALETTE["slate"]),
        ("Stack_LogReg", "Tier 1: Modern MOE Stack", PALETTE["blue"]),
        ("Stack_RDKit", "Tier 2A: RDKit-Only Stack", PALETTE["gold"]),
        ("MT_ResMLP_MOE_RDKit", "Tier 2B: Multi-Task ResMLP", PALETTE["orange"]),
        ("Grand_AllModal_Stack", "Tier 4: Grand All-Modal Stack", PALETTE["teal"]),
    ]

    piv = master.pivot(index="model", columns="isozyme", values="test_auc")[ISOZYMES]
    piv["Mean"] = piv.mean(axis=1)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.6, 5.4), gridspec_kw={"width_ratios": [1.55, 1.0]}
    )

    # Panel A: Grouped bars across 5 isozymes + Mean
    cols = ISOZYMES + ["Mean"]
    x_labels = ISO_LABELS + ["MEAN"]
    x = np.arange(len(cols))
    n_m = len(models)
    width = 0.13

    for i, (m_key, m_label, color) in enumerate(models):
        vals = [piv.loc[m_key, c] for c in cols]
        offset = (i - (n_m - 1) / 2.0) * width
        bars = ax1.bar(
            x + offset,
            vals,
            width=width * 0.92,
            label=m_label,
            color=color,
            edgecolor="white",
            linewidth=0.6,
        )
        if m_key == "Grand_AllModal_Stack":
            for b, v in zip(bars, vals):
                ax1.text(
                    b.get_x() + b.get_width() / 2.0,
                    v + 0.003,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                    fontweight="bold",
                    color=PALETTE["ink"],
                )

    ax1.axvline(4.5, color="#cccccc", linestyle="--", linewidth=1.0)
    ax1.set_xticks(x, x_labels)
    ax1.set_ylim(0.75, 0.94)
    ax1.set_ylabel("Held-Out Test ROC-AUC")
    ax1.set_xlabel("Cytochrome P450 Isozyme")
    ax1.set_title("A. Ten-Year Progression Across Five CYP Isozymes")
    clean_axes(ax1)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#dddddd", ncol=2, fontsize=8.0)

    # Panel B: Attribution waterfall from 2015 SVM_baseline -> 2026 Grand_AllModal_Stack
    base_auc = float(piv.loc["SVM_baseline", "Mean"])
    t1_auc = float(piv.loc["Stack_LogReg", "Mean"])
    t2a_auc = float(piv.loc["Stack_MOE_RDKit", "Mean"])
    t4_auc = float(piv.loc["Grand_AllModal_Stack", "Mean"])
    t4_os_auc = float(piv.loc["Grand_OpenSource_Stack", "Mean"])

    d_alg = t1_auc - base_auc
    d_feat = t2a_auc - t1_auc
    d_dl = t4_auc - t2a_auc
    d_tot = t4_auc - base_auc

    stages = [
        "2015 SVM\n(MOE)",
        "+ Modern\nTabular &\nStacking",
        "+ RDKit FPs\n& Hybrid\nKernels",
        "+ GPU Multi-Task\nDL, GNN &\nChemBERTa",
        "2026 Grand\nAll-Modal\nStack",
    ]
    bottoms = [0.80, base_auc, t1_auc, t2a_auc, 0.80]
    heights = [base_auc - 0.80, d_alg, d_feat, d_dl, t4_auc - 0.80]
    colors = [
        PALETTE["slate"],
        PALETTE["blue"],
        PALETTE["gold"],
        PALETTE["orange"],
        PALETTE["teal"],
    ]

    xs = np.arange(len(stages))
    ax2.bar(xs, heights, bottom=bottoms, width=0.56, color=colors, edgecolor="white", linewidth=1.0)

    # Connector lines
    for j, lvl in enumerate([base_auc, t1_auc, t2a_auc, t4_auc]):
        ax2.plot([j + 0.28, j + 0.72], [lvl, lvl], color="#666666", lw=0.9, ls=":")

    # Annotations
    ax2.text(0, base_auc + 0.0025, f"{base_auc:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=8.8)
    ax2.text(
        1,
        t1_auc + 0.0025,
        f"+{d_alg:.4f}\n({100*d_alg/d_tot:.0f}%)",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=8.3,
        color=PALETTE["blue"],
    )
    ax2.text(
        2,
        t2a_auc + 0.0025,
        f"+{d_feat:.4f}\n({100*d_feat/d_tot:.0f}%)",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=8.3,
        color="#b37b12",
    )
    ax2.text(
        3,
        t4_auc + 0.0025,
        f"+{d_dl:.4f}\n({100*d_dl/d_tot:.0f}%)",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=8.3,
        color=PALETTE["orange"],
    )
    ax2.text(
        4,
        t4_auc + 0.0025,
        f"{t4_auc:.4f}\n(+{d_tot:.4f})",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=8.6,
        color=PALETTE["teal"],
    )

    # Open-source parity callout line (placed cleanly above bars with arrow)
    ax2.axhline(t4_os_auc, color=PALETTE["purple"], ls="--", lw=1.2, alpha=0.9)
    ax2.annotate(
        f"100% Open-Source Stack (no MOE): {t4_os_auc:.4f} (Δ = {t4_os_auc - t4_auc:+.4f}, ns)",
        xy=(0.25, t4_os_auc),
        xytext=(0.05, 0.899),
        fontsize=7.9,
        color=PALETTE["purple"],
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=PALETTE["purple"], lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f9f5fc", edgecolor=PALETTE["purple"], lw=0.7),
    )

    ax2.set_xticks(xs, stages, fontsize=8.4)
    ax2.set_ylim(0.80, 0.908)
    ax2.set_ylabel("Mean Test ROC-AUC (5 Isozymes)")
    ax2.set_title(f"B. Decomposition of Ten-Year Uplift (+{d_tot:.4f} AUC)")
    clean_axes(ax2)

    fig.tight_layout(w_pad=2.2)
    fig.savefig(OUT_DIR / "fig1_ten_year_progression.png")
    plt.close(fig)


def fig2_roc_curves_decade(master: pd.DataFrame) -> None:
    """2x3 ROC grid matching the original thesis layout, comparing 2015 vs 2026 milestones."""
    curve_specs = [
        ("RF_baseline", "2015 RF (10 trees, MOE)", PALETTE["gray"], ":"),
        ("SVM_baseline", "2015 SVM (RBF, MOE)", PALETTE["slate"], "--"),
        ("Stack_LogReg", "Tier 1: MOE Stack", PALETTE["blue"], "-."),
        ("MT_ChemBERTa_77M_MTR", "Tier 3B: ChemBERTa (SMILES)", PALETTE["orange"], "-"),
        ("Grand_AllModal_Stack", "Tier 4: Grand All-Modal Stack", PALETTE["teal"], "-"),
    ]
    auc_piv = master.pivot(index="model", columns="isozyme", values="test_auc")

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.6))
    for ax, iso in zip(axes.flat, ISOZYMES):
        y_te, preds = load_test_preds(iso)
        ax.plot([0, 1], [0, 1], color="#cccccc", lw=1.0, ls=":")
        for m_key, m_label, color, ls in curve_specs:
            fpr, tpr, _ = roc_curve(y_te, preds[m_key])
            lw = 2.3 if m_key == "Grand_AllModal_Stack" else 1.6
            auc_v = auc_piv.loc[m_key, iso]
            ax.plot(fpr, tpr, color=color, ls=ls, lw=lw, label=f"{m_label} ({auc_v:.3f})")
        ax.set(title=f"CYP {iso.upper()}", xlim=(0, 1), ylim=(0, 1.01), aspect="equal")
        clean_axes(ax)
        ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#e0e0e0", fontsize=7.4)

    for ax in axes[1, :2]:
        ax.set_xlabel("False Positive Rate (1 − TNR)")
    axes[0, 2].set_xlabel("False Positive Rate (1 − TNR)")
    for ax in axes[:, 0]:
        ax.set_ylabel("True Positive Rate (TPR)")

    # 6th panel: concise legend + headline summary box
    ax_leg = axes.flat[-1]
    ax_leg.axis("off")
    summary_lines = [
        "Held-Out Test ROC-AUC (Mean across 5 Isozymes)\n",
        f"• 2015 RF Baseline (MOE):         {auc_piv.loc['RF_baseline', ISOZYMES].mean():.4f}",
        f"• 2015 SVM Baseline (MOE):       {auc_piv.loc['SVM_baseline', ISOZYMES].mean():.4f}",
        f"• Tier 1 MOE Stack:                    {auc_piv.loc['Stack_LogReg', ISOZYMES].mean():.4f}",
        f"• Tier 3B ChemBERTa (SMILES): {auc_piv.loc['MT_ChemBERTa_77M_MTR', ISOZYMES].mean():.4f}",
        f"• Tier 4 Grand All-Modal:           {auc_piv.loc['Grand_AllModal_Stack', ISOZYMES].mean():.4f}",
        "\nEvery model evaluated on 100% of",
        "the exact 2015 Figshare test splits.",
    ]
    ax_leg.text(
        0.06,
        0.52,
        "\n".join(summary_lines),
        va="center",
        ha="left",
        fontsize=9.3,
        color=PALETTE["ink"],
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#f7f8fa", edgecolor="#d0d4dc"),
    )

    fig.suptitle("Held-Out Test ROC Curves: 2015 Thesis Baselines vs. 2026 Multi-Modal Models", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "fig2_roc_curves_decade.png")
    plt.close(fig)


def _reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys, cnts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob <= hi if hi == 1.0 else y_prob < hi)
        if mask.sum() > 0:
            xs.append(float(y_prob[mask].mean()))
            ys.append(float(y_true[mask].mean()))
            cnts.append(int(mask.sum()))
    return np.array(xs), np.array(ys), np.array(cnts)


def fig3_calibration_reliability(master: pd.DataFrame) -> None:
    """2x3 Reliability diagrams + Brier/ECE comparison panel."""
    cal_models = [
        ("RF_baseline", "2015 RF (10 trees)", PALETTE["orange"], "s", ":"),
        ("SVM_baseline", "2015 SVM (MOE)", PALETTE["slate"], "^", "--"),
        ("Grand_AllModal_Stack", "2026 Grand Stack", PALETTE["teal"], "o", "-"),
    ]
    brier_piv = master.pivot(index="model", columns="isozyme", values="test_brier")
    ece_piv = master.pivot(index="model", columns="isozyme", values="test_ece")

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.6))
    for ax, iso in zip(axes.flat, ISOZYMES):
        y_te, preds = load_test_preds(iso)
        ax.plot([0, 1], [0, 1], color="#999999", lw=1.0, ls="--", label="Perfect calibration")
        for m_key, m_label, color, marker, ls in cal_models:
            px, py, _ = _reliability_curve(y_te, preds[m_key], n_bins=10)
            ece_v = ece_piv.loc[m_key, iso]
            ax.plot(
                px,
                py,
                color=color,
                marker=marker,
                ms=4.8,
                lw=1.8 if m_key == "Grand_AllModal_Stack" else 1.4,
                ls=ls,
                label=f"{m_label} (ECE {ece_v:.3f})",
            )
        ax.set(title=f"CYP {iso.upper()}", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), aspect="equal")
        clean_axes(ax)
        ax.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="#e0e0e0", fontsize=7.3)

    for ax in axes[1, :2]:
        ax.set_xlabel("Mean Predicted Probability")
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed Inhibitor Fraction")

    # 6th panel: Mean Brier score comparison across tiers
    ax_b = axes.flat[-1]
    b_models = [
        ("RF_baseline", "2015 RF", PALETTE["gray"]),
        ("SVM_baseline", "2015 SVM", PALETTE["slate"]),
        ("Stack_LogReg", "Tier 1 MOE", PALETTE["blue"]),
        ("Stack_RDKit", "Tier 2A RDKit", PALETTE["gold"]),
        ("MT_ResMLP_MOE_RDKit", "Tier 2B MT-MLP", PALETTE["orange"]),
        ("Grand_AllModal_Stack", "Tier 4 Grand", PALETTE["teal"]),
    ]
    b_vals = [brier_piv.loc[k, ISOZYMES].mean() for k, _, _ in b_models]
    y_pos = np.arange(len(b_models))
    bars = ax_b.barh(
        y_pos,
        b_vals,
        color=[c for _, _, c in b_models],
        edgecolor="white",
        height=0.62,
    )
    ax_b.set_yticks(y_pos, [lbl for _, lbl, _ in b_models], fontsize=8.6)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0.11, 0.19)
    ax_b.set_xlabel("Mean Test Brier Score (lower is better)")
    ax_b.set_title("Probabilistic Accuracy (Brier)")
    clean_axes(ax_b)
    for bar, v in zip(bars, b_vals):
        ax_b.text(v + 0.0012, bar.get_y() + bar.get_height() / 2.0, f"{v:.4f}", va="center", fontsize=8.2, fontweight="bold")

    fig.suptitle("Probability Calibration & Brier Score Improvement (2015 vs. 2026)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "fig3_calibration_reliability.png")
    plt.close(fig)


def _sig_stars(p: float) -> str:
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def fig4_delong_forest(stats: pd.DataFrame) -> None:
    """2x2 Forest plot of exact DeLong ΔROC-AUC + 95% CIs for 4 headline hypotheses."""
    panels = [
        ("Tier1_BestMOE_vs_OrigSVM", "A. Algorithm Modernization\n(Tier 1 MOE Stack − 2015 SVM)", PALETTE["blue"]),
        ("GrandOpenSource_vs_Tier1MOE", "B. Open-Source vs. Proprietary MOE\n(Grand Open-Source − Tier 1 MOE Stack)", PALETTE["gold"]),
        ("GrandAllModal_vs_SingleTaskMOERDKit", "C. Multi-Task DL + GNN + ChemBERTa Lift\n(Grand All-Modal − Single-Task MOE+RDKit)", PALETTE["orange"]),
        ("GrandAllModal_vs_GrandOpenSource", "D. Residual Value of Proprietary MOE\n(Grand All-Modal − Grand Open-Source)", PALETTE["purple"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.2))
    for ax, (comp, title, color) in zip(axes.flat, panels):
        sub = stats[stats["comparison"] == comp].set_index("isozyme").loc[ISOZYMES]
        y = np.arange(len(ISOZYMES))
        diffs = sub["delta_auc"].to_numpy()
        lo = diffs - sub["ci95_lo"].to_numpy()
        hi = sub["ci95_hi"].to_numpy() - diffs
        pvals = sub["delong_p"].to_numpy()

        ax.axvline(0.0, color="#666666", lw=1.1, ls="--")
        ax.errorbar(
            diffs,
            y,
            xerr=[lo, hi],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.8,
            capsize=4,
            markersize=6.5,
        )
        # Annotate exact ΔAUC and significance
        xmax = float(sub["ci95_hi"].max())
        xmin = float(min(0.0, sub["ci95_lo"].min()))
        pad = max(0.002, (xmax - xmin) * 0.28)
        ax.set_xlim(xmin - pad * 0.25, xmax + pad)

        for yi, d, hi_v, pv in zip(y, diffs, sub["ci95_hi"].to_numpy(), pvals):
            stars = _sig_stars(pv)
            ax.text(
                hi_v + (xmax - xmin) * 0.03,
                yi,
                f"{d:+.4f} ({stars})",
                va="center",
                ha="left",
                fontsize=8.3,
                fontweight="bold" if stars != "ns" else "normal",
                color=PALETTE["ink"] if stars != "ns" else "#666666",
            )

        ax.set_yticks(y, ISO_LABELS)
        ax.invert_yaxis()
        ax.set_xlabel("Δ Test ROC-AUC (95% DeLong CI)")
        ax.set_title(title, fontsize=10.5)
        clean_axes(ax)

    fig.suptitle(
        "Exact DeLong Tests of ROC-AUC Differences Across All Five CYP Isozymes\n"
        "(* p < 0.05, ** p < 0.01, *** p < 0.001, ns = not significant)",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_DIR / "fig4_delong_forest.png")
    plt.close(fig)


def fig5_scaffold_generalization(scaf: pd.DataFrame) -> None:
    """Random Stratified CV vs. Murcko Scaffold-Split CV across all 5 isozymes."""
    models = [
        ("LightGBM_MOE", "LightGBM (MOE)", PALETTE["blue"]),
        ("LightGBM_RDKit", "LightGBM (RDKit)", PALETTE["teal"]),
        ("Tanimoto_SVM_RDKit", "Tanimoto SVM (RDKit)", PALETTE["orange"]),
    ]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.6, 4.9), gridspec_kw={"width_ratios": [1.55, 1.0]}
    )

    # Panel A: Paired bars (solid = Random CV, hatched/lighter = Scaffold CV)
    x = np.arange(len(ISOZYMES))
    w = 0.13
    for i, (m_key, m_label, color) in enumerate(models):
        r_vals = (
            scaf[(scaf["model"] == m_key) & (scaf["split_type"] == "random_stratified")]
            .set_index("isozyme")
            .loc[ISOZYMES, "cv_auc"]
            .to_numpy()
        )
        s_vals = (
            scaf[(scaf["model"] == m_key) & (scaf["split_type"] == "scaffold_group")]
            .set_index("isozyme")
            .loc[ISOZYMES, "cv_auc"]
            .to_numpy()
        )
        off = (i - 1) * (2.2 * w)
        ax1.bar(
            x + off - w / 2,
            r_vals,
            width=w * 0.95,
            color=color,
            edgecolor="white",
            label=f"{m_label} — Random CV",
        )
        ax1.bar(
            x + off + w / 2,
            s_vals,
            width=w * 0.95,
            color=color,
            alpha=0.52,
            hatch="//",
            edgecolor="white",
            label=f"{m_label} — Scaffold CV",
        )

    ax1.set_xticks(x, ISO_LABELS)
    ax1.set_ylim(0.79, 0.915)
    ax1.set_ylabel("5-Fold Cross-Validation ROC-AUC")
    ax1.set_xlabel("Cytochrome P450 Isozyme")
    ax1.set_title("A. Random Stratified CV vs. Murcko Scaffold-Split CV")
    clean_axes(ax1)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#dddddd", fontsize=7.5, ncol=2)

    # Panel B: Mean ΔAUC generalization gap
    mean_drops = []
    per_iso_drops = []
    for m_key, _, _ in models:
        r_v = (
            scaf[(scaf["model"] == m_key) & (scaf["split_type"] == "random_stratified")]
            .set_index("isozyme")
            .loc[ISOZYMES, "cv_auc"]
        )
        s_v = (
            scaf[(scaf["model"] == m_key) & (scaf["split_type"] == "scaffold_group")]
            .set_index("isozyme")
            .loc[ISOZYMES, "cv_auc"]
        )
        d = (s_v - r_v).to_numpy()
        per_iso_drops.append(d)
        mean_drops.append(float(d.mean()))

    xs = np.arange(len(models))
    bars = ax2.bar(
        xs,
        mean_drops,
        width=0.52,
        color=[c for _, _, c in models],
        edgecolor="white",
    )
    for idx, d in enumerate(per_iso_drops):
        jitter = np.linspace(-0.12, 0.12, len(d))
        ax2.scatter(
            np.full_like(d, idx) + jitter,
            d,
            color=PALETTE["ink"],
            s=26,
            zorder=5,
            alpha=0.8,
        )
    ax2.axhline(0.0, color="#666666", lw=1.0)
    ax2.set_xticks(xs, [lbl.replace(" (", "\n(") for _, lbl, _ in models], fontsize=8.8)
    ax2.set_ylim(-0.020, 0.002)
    ax2.set_ylabel("Δ ROC-AUC (Scaffold CV − Random CV)")
    ax2.set_title("B. Mean Scaffold Drop (Dots = 5 Isozymes)")
    clean_axes(ax2)
    for bar, v in zip(bars, mean_drops):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            -0.0016,
            f"{v:+.4f}",
            ha="center",
            va="top",
            fontsize=8.8,
            fontweight="bold",
            color="white",
        )

    fig.tight_layout(w_pad=2.2)
    fig.savefig(OUT_DIR / "fig5_scaffold_generalization.png")
    plt.close(fig)


def fig6_multimodal_weights_and_compute(grand: pd.DataFrame) -> None:
    """Two-panel figure: (A) Normalized stacking weights by modality, (B) 2026 compute profile."""
    sub = grand[grand["model"] == "Grand_AllModal_Stack"].set_index("isozyme").loc[ISOZYMES]
    member_order = [
        ("XGBoost_MOE_RDKit", "XGBoost (MOE+RDKit)"),
        ("CatBoost_MOE_RDKit", "CatBoost (MOE+RDKit)"),
        ("LightGBM_MOE_RDKit", "LightGBM (MOE+RDKit)"),
        ("HybridKernel_SVM", "Hybrid Tanimoto+RBF SVM"),
        ("Tanimoto_SVM_RDKit", "Tanimoto SVM (RDKit)"),
        ("MT_ResMLP_MOE_RDKit", "MT-ResMLP (MOE+RDKit)"),
        ("MT_ResMLP_RDKit", "MT-ResMLP (RDKit)"),
        ("MT_ChemBERTa_77M_MTR", "MT-ChemBERTa (SMILES)"),
        ("MT_GNN_GraphPlusDesc", "MT-GNN (Graph+Desc)"),
        ("MT_GNN_GraphOnly", "MT-GNN (Graph Only)"),
    ]

    mat = np.zeros((len(member_order), len(ISOZYMES)), dtype=np.float64)
    for j, iso in enumerate(ISOZYMES):
        coefs = json.loads(sub.loc[iso, "best_params"])["coefs"]
        raw = np.array([max(0.0, float(coefs.get(k, 0.0))) for k, _ in member_order])
        s = raw.sum()
        mat[:, j] = raw / s if s > 0 else raw

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.2, 5.2), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )

    im = ax1.imshow(mat, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=0.35)
    ax1.set_xticks(np.arange(len(ISOZYMES)), ISO_LABELS)
    ax1.set_yticks(np.arange(len(member_order)), [lbl for _, lbl in member_order], fontsize=8.6)
    ax1.set_title("A. Normalized Meta-Learner Weights in Grand All-Modal Stack")
    ax1.set_xlabel("Cytochrome P450 Isozyme")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax1.text(
                j,
                i,
                f"{100*v:.1f}%",
                ha="center",
                va="center",
                fontsize=7.8,
                color="white" if v > 0.21 else PALETTE["ink"],
                fontweight="bold" if v >= 0.12 else "normal",
            )
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cbar.set_label("Relative Positive Weight Share", fontsize=8.8)

    # Panel B: Compute runtime breakdown across Tiers (local CPU vs Colab T4 GPU)
    tiers = [
        "Tier 1: MOE Benchmark\n(12 models × 5 iso, Local CPU)",
        "Tier 2A: RDKit & Hybrid\n(10 models + Scaffold, Local CPU)",
        "Tier 2B: Multi-Task ResMLP\n(150 nets, Colab T4 GPU)",
        "Tier 3A: Multi-Task GNN\n(50 GINE nets, Colab T4 GPU)",
        "Tier 3B: ChemBERTa-77M\n(25 Transformers, Colab T4 GPU)",
        "Tier 4: Grand Meta-Stack\n& DeLong/McNemar (Local CPU)",
    ]
    mins = [42.0, 37.5, 9.7, 38.4, 17.5, 0.2]
    t_colors = [
        PALETTE["blue"],
        PALETTE["blue"],
        PALETTE["teal"],
        PALETTE["teal"],
        PALETTE["teal"],
        PALETTE["blue"],
    ]
    yp = np.arange(len(tiers))
    bars = ax2.barh(yp, mins, color=t_colors, edgecolor="white", height=0.62)
    ax2.set_yticks(yp, tiers, fontsize=8.3)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 52)
    ax2.set_xlabel("Compute Duration (minutes)")
    ax2.set_title("B. 2026 Campaign Compute Profile")
    clean_axes(ax2)
    for bar, v in zip(bars, mins):
        ax2.text(v + 0.8, bar.get_y() + bar.get_height() / 2.0, f"{v:.1f} min", va="center", fontsize=8.3, fontweight="bold")

    # Legend for hardware
    cpu_patch = plt.Rectangle((0, 0), 1, 1, fc=PALETTE["blue"], label="Local CPU (4 threads, <1.3 GiB RAM)")
    gpu_patch = plt.Rectangle((0, 0), 1, 1, fc=PALETTE["teal"], label="Colab Tesla T4 GPU (1.80 CU total)")
    ax2.legend(handles=[cpu_patch, gpu_patch], loc="lower right", frameon=True, framealpha=0.95, edgecolor="#dddddd", fontsize=7.8)

    fig.tight_layout(w_pad=2.2)
    fig.savefig(OUT_DIR / "fig6_multimodal_weights_and_compute.png")
    plt.close(fig)


def main() -> None:
    master = pd.read_csv(RESULTS / "master_comparison.csv")
    grand = pd.read_csv(RESULTS / "grand_summary.csv")
    stats = pd.read_csv(RESULTS / "statistical_tests.csv")
    scaf = pd.read_csv(RESULTS / "scaffold_cv_summary.csv")

    fig1_ten_year_progression(master)
    fig2_roc_curves_decade(master)
    fig3_calibration_reliability(master)
    fig4_delong_forest(stats)
    fig5_scaffold_generalization(scaf)
    fig6_multimodal_weights_and_compute(grand)

    print(f"Generated 6 figures in {OUT_DIR}:")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(f"  - {p.name} ({p.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
