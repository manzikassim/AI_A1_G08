# Batch gradient descent for linear regression: derivation note

Author: Manzi Kassim (Member 2, Regression engineer), Group 8 (AI-G08)
File this explains: `src/regression.py`

## 1. Model

For one record with standardised features `z` (length p) and a bias term, the prediction is

    y_hat = w0 + w1*z1 + ... + wp*zp = x . w

In matrix form, with a column of ones added on the left (`add_bias`), `X` is n x (p+1) and

    y_hat = X w

## 2. Loss (mean squared error)

    L(w) = (1/n) * sum_i (x_i . w - y_i)^2 = (1/n) * ||X w - y||^2

Squaring makes large mistakes count more and gives a smooth bowl-shaped surface with one minimum.

## 3. Gradient

Let `e = X w - y` (the error vector). Then

    dL/dw = (2/n) * X^T e

This is the code line `Xb.T @ err` multiplied by `2.0 / n`. Each weight's gradient is the average of
(error x that feature), so a weight moves in the direction that reduces the errors it influences.

## 4. Update rule (batch gradient descent)

    w  <-  w - lr * (2/n) * X^T (X w - y)

"Batch" means every training row is used in every step. We start from `w = 0` and repeat for `EPOCHS`
steps. `lr` (the learning rate, `LEARNING_RATE`) is the step size.

## 5. Why we scale first

- Features are standardised with the **training** mean and standard deviation only (`fit_scaler`), so test
  data never influences the scaler (no leakage). If a feature has zero variance, its std is set to 1.
- The target is standardised the same way. This keeps the loss values small and the steps stable.
- Predictions are converted back to kilograms in `predict_yield`: `y_hat * y_std + y_mean`.

With standardised features, all weights have comparable scale, so one learning rate works for all of them.
Without scaling, a feature measured in hundreds (rainfall in mm) would dominate one measured in single
digits (soil pH) and descent would zig-zag or diverge.

## 6. Choosing the learning rate

For this loss, descent converges when `lr < 1 / lambda_max(X^T X / n)`, where `lambda_max` is the largest
eigenvalue of that matrix. With standardised features it is a modest number, so `lr = 0.05` is safe here.

- **Too small:** the loss falls slowly, and 2000 epochs may stop before the minimum.
- **Too large:** steps overshoot, the loss oscillates and then explodes. `gradient_descent` detects
  non-finite weights and raises a clear `DataError` telling the user to lower `LEARNING_RATE`.

## 7. Checks built into the code

- **Closed-form comparison.** `np.linalg.lstsq` solves the same least-squares problem exactly. If gradient
  descent has converged, its test RMSE matches `closed_form_test_rmse_for_comparison`.
- **Convergence record.** `convergence_info` stores the relative change in the loss over the last 100
  epochs in `regression_metrics.json`. A value near zero with `converged: true` means descent has settled.
- **Baseline.** `baseline_test_rmse_predict_train_mean` is the RMSE of always predicting the training mean.
  A useful model must beat it.

## 8. What changes if the assessor edits something

| Change | Expected effect |
|---|---|
| Lower `LEARNING_RATE` (for example 0.01) | Loss curve falls more slowly; final loss may be higher after 2000 epochs; `converged` may become false |
| Raise `LEARNING_RATE` a lot | Loss oscillates, then diverges and the `DataError` appears |
| Raise `EPOCHS` | Loss keeps falling until it reaches the closed-form solution, then flattens |
| Change `SEED` | Different train/test shuffle, so test MAE, RMSE and R-squared change slightly. Weights start at zero, so the descent itself is deterministic given the split |
| Change `TEST_SIZE` | Fewer training rows (larger test set) usually gives noisier estimates |

All numbers above are described qualitatively. The real values come from running `run_all.py` on the
lecturer-issued dataset.
