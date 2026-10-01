"""Linear regression with batch gradient descent, implemented from first principles with NumPy."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import DataError, FEATURES, REG_TARGET, SEED, supervised_xy, write_json

LEARNING_RATE = 0.05
EPOCHS = 2000
TEST_SIZE = 0.2


def split_indices(n, test_size, seed):
    idx = np.random.default_rng(seed).permutation(n)
    n_test = max(1, int(round(n * test_size)))
    return idx[n_test:], idx[:n_test]


def fit_scaler(X):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd


def add_bias(Z):
    return np.hstack([np.ones((Z.shape[0], 1)), Z])


def gradient_descent(Xb, y, lr, epochs):
    """Minimise MSE: L(w) = mean((Xw - y)^2); gradient = (2/n) X^T (Xw - y)."""
    w = np.zeros(Xb.shape[1])
    history = []
    n = len(y)
    for _ in range(epochs):
        err = Xb @ w - y
        history.append(float(np.mean(err ** 2)))
        w -= lr * (2.0 / n) * (Xb.T @ err)
        if not np.all(np.isfinite(w)):
            raise DataError("Gradient descent diverged; lower LEARNING_RATE in src/regression.py.")
    return w, history


def predict_yield(model, X):
    """model: dict with mean, std, weights (bias first), y_mean, y_std."""
    Z = (np.asarray(X, dtype=float) - np.array(model["mean"])) / np.array(model["std"])
    yz = add_bias(Z) @ np.array(model["weights"])
    return yz * model["y_std"] + model["y_mean"]


def metrics(y, p):
    err = y - p
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "r2": float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else None}


def run_regression(df, out_dir, models_dir, seed=SEED):
    X, y, _ = supervised_xy(df, REG_TARGET)
    tr, te = split_indices(len(y), TEST_SIZE, seed)
    # scalers use TRAINING data only (no leakage); target is also standardised for stable descent
    mu, sd = fit_scaler(X[tr])
    y_mu, y_sd = float(y[tr].mean()), float(y[tr].std()) or 1.0
    Xtr = add_bias((X[tr] - mu) / sd)
    w, hist = gradient_descent(Xtr, (y[tr] - y_mu) / y_sd, LEARNING_RATE, EPOCHS)
    model = {"feature_names": FEATURES, "mean": mu, "std": sd, "weights": w, "y_mean": y_mu, "y_std": y_sd}
    ptr, pte = predict_yield(model, X[tr]), predict_yield(model, X[te])
    # sanity check against the closed-form least-squares solution (NumPy, not an estimator)
    w_ne = np.linalg.lstsq(Xtr, (y[tr] - y_mu) / y_sd, rcond=None)[0]
    m_ne = predict_yield({**model, "weights": w_ne}, X[te])
    out = {
        "random_seed": seed, "learning_rate": LEARNING_RATE, "epochs": EPOCHS, "test_size": TEST_SIZE,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "loss_definition": "MSE on standardised target (train set)",
        "initial_loss": hist[0], "final_loss": hist[-1],
        "train_metrics": metrics(y[tr], ptr), "test_metrics": metrics(y[te], pte),
        "baseline_test_rmse_predict_train_mean": float(np.sqrt(np.mean((y[te] - y[tr].mean()) ** 2))),
        "closed_form_test_rmse_for_comparison": metrics(y[te], m_ne)["rmse"],
        "weights_standardised": dict(zip(["bias"] + FEATURES, w)),
        "test_predictions_sample": [{"actual": float(a), "predicted": float(b)} for a, b in zip(y[te][:10], pte[:10])],
    }
    write_json(f"{out_dir}/regression_metrics.json", out)
    write_json(f"{models_dir}/regression_model.json", model)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(hist) + 1), hist)
    ax.set(xlabel="Epoch", ylabel="Training MSE (standardised target)", title="Gradient descent loss", yscale="log")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/regression_loss.png", dpi=150)
    plt.close(fig)
    return out
