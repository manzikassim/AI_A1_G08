"""K-means clustering on standardised INPUT features only (no targets, no identifier)."""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .common import DataError, FEATURES, ID_COL, SEED, write_json

K_RANGE = range(2, 6)
CAUTION = ("Clusters are exploratory statistical groupings of similar operating profiles. They are not verified "
           "real-world categories and should not be used to label farms or people without further validation.")


def run_clustering(df, out_dir, models_dir, seed=SEED):
    feats = df[FEATURES].copy()
    imputed_cells = int(feats.isna().sum().sum())
    if feats.isna().all().any():
        raise DataError("A feature column is entirely missing; cannot cluster.")
    X = feats.fillna(feats.median()).to_numpy(dtype=float)  # median imputation, inputs only
    Xs = StandardScaler().fit(X)
    Z = Xs.transform(X)
    scores, models = {}, {}
    for k in K_RANGE:
        if k >= len(Z):
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        models[k] = km
        scores[k] = float(silhouette_score(Z, km.labels_)) if len(set(km.labels_)) > 1 else float("nan")
    valid = {k: s for k, s in scores.items() if np.isfinite(s)}
    if not valid:
        raise DataError("Could not compute silhouette scores for k in 2..5.")
    best = max(sorted(valid), key=lambda k: valid[k])
    labels = models[best].labels_
    result = df[[ID_COL] + FEATURES].copy()
    result["cluster_label"] = labels
    result.to_csv(f"{out_dir}/clusters.csv", index=False)
    profile = result.groupby("cluster_label")[FEATURES].mean()
    out = {
        "random_seed": seed, "features_used": FEATURES, "excluded": [ID_COL, "actual_yield_kg", "dispatch_attention"],
        "scaling": "StandardScaler (unsupervised; no labels involved)", "imputed_feature_cells_median": imputed_cells,
        "k_evaluated": list(scores), "silhouette_by_k": scores, "selected_k": int(best),
        "justification": f"k={best} has the highest silhouette score ({valid[best]:.3f}) among k={list(scores)}; "
                         "higher means tighter, better-separated clusters.",
        "n_records_labelled": int(len(labels)),
        "cluster_sizes": {str(c): int((labels == c).sum()) for c in sorted(set(labels))},
        "cluster_feature_means": profile.to_dict(orient="index"),
        "caution": CAUTION,
    }
    write_json(f"{out_dir}/clustering_metrics.json", out)
    joblib.dump(Xs, f"{models_dir}/cluster_scaler.joblib")
    joblib.dump(models[best], f"{models_dir}/cluster_model.joblib")
    P = PCA(n_components=2, random_state=seed).fit_transform(Z)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for c in sorted(set(labels)):
        m = labels == c
        ax.scatter(P[m, 0], P[m, 1], s=22, alpha=0.75, label=f"Cluster {c} (n={m.sum()})")
    ax.set(xlabel="PCA component 1", ylabel="PCA component 2", title=f"K-means clusters (k={best}), exploratory")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/cluster_plot.png", dpi=150)
    plt.close(fig)
    return out
