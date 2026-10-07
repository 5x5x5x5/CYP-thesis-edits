"""Train and evaluate k-NN, random forest and SVM classifiers for one CYP isozyme.

Validation protocol:
- Training set: 5-fold cross-validation (internal validation).
- Test set: each model is fit on the full training set and predicts the
  held-out test set, which plays no part in fitting the scaler or the models
  (external validation).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA_DIR = Path(__file__).parent / "data"
ACTIVITY_THRESHOLD = 40  # Activity Score >= 40 is an inhibitor; everything below is not
SEED = 0

# Categorical series colors, in fixed order (validated for color-vision deficiency)
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
LINESTYLES = ["-", "--", ":"]

# Hyperparameters match the scikit-learn defaults the thesis used in 2015
# (v0.15). Later releases changed the SVC gamma and random forest
# n_estimators defaults, so they are pinned here.
CLASSIFIERS = {
    "k-NN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(
        n_estimators=10, max_features="sqrt", random_state=SEED
    ),
    "SVM": SVC(C=1.0, kernel="rbf", gamma="auto"),
}


def load_split(isozyme: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return X_train, y_train, X_test, y_test for an isozyme such as '1a2'."""
    train = pd.read_csv(DATA_DIR / f"training{isozyme}.csv", index_col="SID")
    test = pd.read_csv(DATA_DIR / f"test{isozyme}.csv", index_col="SID")

    def xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        y = (df["ActivityScore"] >= ACTIVITY_THRESHOLD).to_numpy(dtype=int)
        X = df.drop(columns="ActivityScore").to_numpy(dtype=float)
        return X, y

    return (*xy(train), *xy(test))


def rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy, true positive rate and true negative rate."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": (tp + tn) / (tp + tn + fp + fn),
        "TPR": tp / (tp + fn),
        "TNR": tn / (tn + fp),
    }


def evaluate(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Cross-validate on the training set, then fit on it and score the test set.

    Standardization sits inside the pipeline, so its mean and variance come
    only from the data each model is fit on.
    """
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, test_confusion = [], {}
    for name, clf in CLASSIFIERS.items():
        model = make_pipeline(StandardScaler(), clone(clf))
        cv_pred = cross_val_predict(model, X_train, y_train, cv=folds)
        test_pred = model.fit(X_train, y_train).predict(X_test)
        test_confusion[name] = confusion_matrix(y_test, test_pred, labels=[0, 1])
        rows += [
            {"Method": name, "Set": "Training (5-fold CV)", **rates(y_train, cv_pred)},
            {"Method": name, "Set": "Test", **rates(y_test, test_pred)},
        ]
    return pd.DataFrame(rows).round(3), test_confusion


def scores(model, X: np.ndarray) -> np.ndarray:
    """Continuous inhibitor score: probability where available, else SVM margin."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def calibration_table(y_true: np.ndarray, prob: np.ndarray) -> pd.DataFrame:
    """Observed inhibitor frequency at each distinct predicted probability.

    With a 10-tree random forest, predicted probabilities are multiples of 0.1.
    """
    df = pd.DataFrame({"pred_prob": prob.round(2), "inhibitor": y_true})
    return (
        df.groupby("pred_prob")["inhibitor"]
        .agg(count="size", observed_freq="mean")
        .reset_index()
    )


def rf_calibration(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> dict[str, pd.DataFrame]:
    """Random forest calibration on training-set CV predictions and on the test set."""
    model = make_pipeline(StandardScaler(), clone(CLASSIFIERS["Random Forest"]))
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_prob = cross_val_predict(
        model, X_train, y_train, cv=folds, method="predict_proba"
    )[:, 1]
    test_prob = model.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    return {
        "Training (5-fold CV)": calibration_table(y_train, cv_prob),
        "Test": calibration_table(y_test, test_prob),
    }


def plot_calibration(tables: dict[str, pd.DataFrame], baseline: float, title: str):
    """Predicted probability vs observed inhibitor frequency; marker area ~ count."""
    fig, axes = plt.subplots(
        1, len(tables), figsize=(5 * len(tables), 4.5), sharey=True
    )
    for ax, (name, t) in zip(np.atleast_1d(axes), tables.items()):
        ax.plot([0, 1], [0, 1], color="#8a8a85", lw=1, label="Perfect calibration")
        ax.axhline(
            baseline, color="#8a8a85", lw=1, ls="--", label="Inhibitor base rate"
        )
        ax.scatter(
            t["pred_prob"],
            t["observed_freq"],
            s=t["count"] / t["count"].max() * 300,
            color=SERIES_COLORS[0],
            edgecolor="white",
            linewidth=1,
            zorder=3,
            label="Random forest",
        )
        ax.set(
            xlim=(-0.05, 1.05),
            ylim=(-0.05, 1.05),
            title=name,
            xlabel="Predicted probability of inhibition",
        )
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("Observed fraction of inhibitors")
    np.atleast_1d(axes)[0].legend(loc="upper left", frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def roc_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    title: str,
) -> tuple[pd.DataFrame, plt.Figure]:
    """Fit each classifier on the training set; ROC curves and AUC on the test set."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="#8a8a85", lw=1, label="Chance")
    rows = []
    for (name, clf), color, ls in zip(CLASSIFIERS.items(), SERIES_COLORS, LINESTYLES):
        model = make_pipeline(StandardScaler(), clone(clf)).fit(X_train, y_train)
        s = scores(model, X_test)
        auc = roc_auc_score(y_test, s)
        fpr, tpr, _ = roc_curve(y_test, s)
        ax.plot(fpr, tpr, color=color, ls=ls, lw=2, label=f"{name} (AUC {auc:.3f})")
        rows.append({"Method": name, "Test AUC": round(auc, 3)})
    ax.set(
        xlabel="False positive rate (1 - TNR)",
        ylabel="True positive rate (TPR)",
        title=title,
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    return pd.DataFrame(rows), fig
