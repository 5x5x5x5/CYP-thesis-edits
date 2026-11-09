"""Regenerate the thesis method-comparison bar charts and table rows.

Python results come from results/<isozyme>.csv (written by the Model*.ipynb
notebooks). MOE Binary QSAR (20 principal components) results come from the
MOE models reported in the thesis tables.

    uv run python comparison_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cyp_models import CLASSIFIERS, LINESTYLES, SERIES_COLORS, load_split, scores

HERE = Path(__file__).parent
IMG_DIR = HERE.parent / "img"
ISOZYMES = ["1a2", "2c9", "2c19", "2d6", "3a4"]

# MOE Binary QSAR, 20 principal components: (training, test) total accuracy.
MOE = {
    "1a2": (0.741, 0.748),
    "2c9": (0.710, 0.685),
    "2c19": (0.705, 0.698),
    "2d6": (0.705, 0.683),
    "3a4": (0.705, 0.686),
}
METHODS = ["MOE 20PC", "k-NN", "Random Forest", "SVM"]
# Color follows the method: gray for the MOE reference, then the fixed
# categorical order used in the ROC and calibration plots.
COLORS = {
    "MOE 20PC": "#8a8a85",
    "k-NN": "#2a78d6",
    "Random Forest": "#eb6834",
    "SVM": "#1baf7a",
}
HATCHES = {"MOE 20PC": "", "k-NN": "/", "Random Forest": ".", "SVM": "x"}
INK = "#2b2b29"
plt.rcParams["hatch.linewidth"] = 0.8


def accuracy_table(which: str) -> pd.DataFrame:
    """Accuracy by isozyme (rows) and method (columns) for 'train' or 'test'."""
    rows = {}
    for iso in ISOZYMES:
        r = pd.read_csv(HERE / "results" / f"{iso}.csv")
        subset = r[r.Set == "Test"] if which == "test" else r[r.Set != "Test"]
        acc = subset.set_index("Method")["Accuracy"]
        rows[iso] = {"MOE 20PC": MOE[iso][which == "test"], **acc.to_dict()}
    return pd.DataFrame(rows).T[METHODS]


def plot(table: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    width, gap = 0.2, 0.02
    x = np.arange(len(table))
    for i, m in enumerate(METHODS):
        ax.bar(
            x + (i - 1.5) * width,
            table[m],
            width - gap,
            label=m,
            color=COLORS[m],
            hatch=HATCHES[m],
            edgecolor="white",
            linewidth=1,
        )
    ax.axhline(0.6, color=INK, lw=1, ls="--", zorder=0, label="0.6 accuracy threshold")
    ax.set_xticks(x, [i.upper() for i in table.index])
    ax.set(
        ylim=(0, 1), ylabel="Accuracy", xlabel="Cytochrome P450 isozyme", title=title
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_roc_grid(path: Path) -> None:
    """One ROC panel per isozyme: models fit on the training set, scored on the test set."""
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.6))
    for ax, iso in zip(axes.flat, ISOZYMES):
        X_train, y_train, X_test, y_test = load_split(iso)
        ax.plot([0, 1], [0, 1], color=COLORS["MOE 20PC"], lw=1)
        for (name, clf), color, ls in zip(
            CLASSIFIERS.items(), SERIES_COLORS, LINESTYLES
        ):
            model = make_pipeline(StandardScaler(), clone(clf)).fit(X_train, y_train)
            fpr, tpr, _ = roc_curve(y_test, scores(model, X_test))
            ax.plot(fpr, tpr, color=color, ls=ls, lw=2, label=name)
        ax.set(title=iso.upper(), xlim=(0, 1), ylim=(0, 1.01), aspect="equal")
        ax.grid(alpha=0.3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("False positive rate")
    for ax in axes[:, 0]:
        ax.set_ylabel("True positive rate")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    chance = plt.Line2D([], [], color=COLORS["MOE 20PC"], lw=1)
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    legend_ax.legend(
        handles + [chance], labels + ["Chance"], loc="center", frameon=False
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_roc_grid(IMG_DIR / "roc_curves_test_set.png")
    for which, title, name in [
        (
            "train",
            "Method comparison on the training set (Python: 5-fold CV)",
            "method_comparison_training_set.png",
        ),
        ("test", "Method comparison on the test set", "method_comparison_test_set.png"),
    ]:
        t = accuracy_table(which)
        plot(t, title, IMG_DIR / name)
        print(which)
        print(t.round(3).to_string())
