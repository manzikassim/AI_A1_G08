"""Predict for one record: python predict.py --record '{"plot_area_ha":1.2,"rainfall_mm":81,"soil_ph":5.7,"seed_kg":210,"distance_km":14,"arrival_hour":9}'"""
import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np

from src.common import FEATURES, MODEL_VERSION, RANGES
from src.regression import predict_yield

MODELS = Path(__file__).resolve().parent / "models"


def fail(errors):
    print(json.dumps({"status": "error", "errors": errors}, indent=2))
    sys.exit(1)


def validate(text):
    try:
        rec = json.loads(text)
    except json.JSONDecodeError as exc:
        fail([f"Input is not valid JSON: {exc}"])
    if not isinstance(rec, dict):
        fail(["Input must be a JSON object with the six feature fields."])
    errs = [f"Missing field: {f}" for f in FEATURES if f not in rec]
    errs += [f"Unexpected field: {k}" for k in rec if k not in FEATURES]
    for f in FEATURES:
        if f not in rec:
            continue
        v = rec[f]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            errs.append(f"Field '{f}' must be a finite number, got {v!r}")
            continue
        lo, hi, lo_inc, hi_inc = RANGES[f]
        if (lo is not None and (v < lo or (v == lo and not lo_inc))) or (hi is not None and (v > hi or (v == hi and not hi_inc))):
            errs.append(f"Field '{f}' value {v} is outside the plausible range {lo}..{hi}")
    if errs:
        fail(errs)
    return np.array([[float(rec[f]) for f in FEATURES]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True, help="one JSON object using the documented schema")
    x = validate(ap.parse_args().record)
    try:
        reg = json.loads((MODELS / "regression_model.json").read_text())
        clf = joblib.load(MODELS / "classifier.joblib")
        scaler, km = joblib.load(MODELS / "cluster_scaler.joblib"), joblib.load(MODELS / "cluster_model.joblib")
        meta = json.loads((MODELS / "meta.json").read_text())
    except FileNotFoundError:
        fail(["Trained models not found in models/. Run run_all.py first."])
    proba = float(clf.predict_proba(x)[0, 1])
    print(json.dumps({
        "status": "ok",
        "regression_prediction_kg": round(float(predict_yield(reg, x)[0]), 3),
        "classification_prediction": int(proba >= 0.5),
        "classification_probability": round(proba, 4),
        "cluster_label": int(km.predict(scaler.transform(x))[0]),
        "group_code": meta["group_code"],
        "model_version": meta.get("model_version", MODEL_VERSION),
    }, indent=2))


if __name__ == "__main__":
    main()
