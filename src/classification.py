"""Interpretable classifier (logistic regression) for dispatch_attention."""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import CLF_TARGET, DataError, FEATURES, SEED, supervised_xy, write_json

TEST_SIZE = 0.2
THRESHOLD = 0.5  # fixed in advance; never tuned on the test set

COST_NOTE = (
    "A false negative (a consignment that needs attention but is not flagged) is the more costly error: "
    "it can leave a late or risky consignment unnoticed, causing spoilage, missed dispatch slots or lost "
    "farmer income. A false positive only costs a short manual check. We therefore use class_weight='balanced' "
    "to favour recall and report recall alongside precision. A human should always review flagged and unflagged "
    "borderline cases."
)


def run_classification(df, out_dir, models_dir, seed=SEED):
    X, y, _ = supervised_xy(df, CLF_TARGET)
    y = y.astype(int)
    counts = np.bincount(y, minlength=2)
    if counts.min() == 0:
        raise DataError("dispatch_attention has only one class in the usable rows; cannot train a classifier.")
    stratified = counts.min() >= 2 and int(round(len(y) * TEST_SIZE)) >= 2
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed,
                                              stratify=y if stratified else None)
    except ValueError:
        stratified = False
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)
    # scaler lives inside the pipeline so it is fitted on training data only
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed))])
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    cm = confusion_matrix(yte, pred, labels=[0, 1])
    coefs = pipe.named_steps["clf"].coef_[0]
    out = {
        "random_seed": seed, "model": "LogisticRegression(C=1.0, class_weight='balanced')",
        "stratified_split": bool(stratified), "test_size": TEST_SIZE, "threshold": THRESHOLD,
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "class_counts_all": {"0": int(counts[0]), "1": int(counts[1])},
        "confusion_matrix": {"labels": [0, 1], "matrix": cm.tolist(), "layout": "rows=actual, columns=predicted",
                             "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "accuracy": float(accuracy_score(yte, pred)),
        "precision": float(precision_score(yte, pred, zero_division=0)),
        "recall": float(recall_score(yte, pred, zero_division=0)),
        "f1": float(f1_score(yte, pred, zero_division=0)),
        "coefficients_standardised": dict(zip(FEATURES, coefs)),
        "intercept": float(pipe.named_steps["clf"].intercept_[0]),
        "costlier_error": "false negative",
        "cost_explanation": COST_NOTE,
    }
    write_json(f"{out_dir}/classification_metrics.json", out)
    joblib.dump(pipe, f"{models_dir}/classifier.joblib")
    fig, ax = plt.subplots(figsize=(4.8, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["No attention (0)", "Attention (1)"], yticklabels=["No attention (0)", "Attention (1)"])
    ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/confusion_matrix.png", dpi=150)
    plt.close(fig)
    return out
