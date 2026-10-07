"""Train and evaluate k-NN, random forest and SVM classifiers for one CYP isozyme.

Validation protocol:
- Training set: 5-fold cross-validation (internal validation).
- Test set: each model is fit on the full training set and predicts the
  held-out test set, which plays no part in fitting the scaler or the models
  (external validation).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA_DIR = Path(__file__).parent / "data"
ACTIVITY_THRESHOLD = 40  # Activity Score >= 40 is an inhibitor; everything below is not
SEED = 0

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
