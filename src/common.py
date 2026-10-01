"""Shared constants, data loading/validation and reporting helpers."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
MODEL_VERSION = "1.0.0"
ID_COL = "record_id"
FEATURES = ["plot_area_ha", "rainfall_mm", "soil_ph", "seed_kg", "distance_km", "arrival_hour"]
REG_TARGET = "actual_yield_kg"
CLF_TARGET = "dispatch_attention"
REQUIRED = [ID_COL] + FEATURES + [REG_TARGET, CLF_TARGET]
# (min, max, min_inclusive, max_inclusive) used by predict.py for plausibility checks
RANGES = {
    "plot_area_ha": (0, None, False, True),
    "rainfall_mm": (0, None, True, True),
    "soil_ph": (0, 14, True, True),
    "seed_kg": (0, None, True, True),
    "distance_km": (0, None, True, True),
    "arrival_hour": (0, 24, True, False),
}
MIN_ROWS = 20


class DataError(Exception):
    """Raised for any problem with the input data that the user can fix."""


def sha256_file(path):
    if not Path(path).is_file():
        raise DataError(f"Data file not found: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    return o


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2)


def load_data(path):
    """Load and validate the CSV. Returns (clean_df, info). Nothing is hard-coded about row count."""
    p = Path(path)
    if not p.is_file():
        raise DataError(f"Data file not found: {p}")
    try:
        raw = pd.read_csv(p)
    except Exception as exc:
        raise DataError(f"Could not read CSV: {exc}")
    missing_cols = [c for c in REQUIRED if c not in raw.columns]
    if missing_cols:
        raise DataError(f"Missing required column(s): {missing_cols}. Expected: {REQUIRED}")
    df = raw[REQUIRED].copy()
    df[ID_COL] = df[ID_COL].apply(lambda v: v.strip() if isinstance(v, str) else v)
    non_numeric = {}
    for c in FEATURES + [REG_TARGET, CLF_TARGET]:
        conv = pd.to_numeric(df[c], errors="coerce")
        non_numeric[c] = int((conv.isna() & df[c].notna()).sum())
        df[c] = conv
    df = df.replace([np.inf, -np.inf], np.nan)
    bad_target = df[CLF_TARGET].notna() & ~df[CLF_TARGET].isin([0, 1])
    non_numeric[CLF_TARGET] += int(bad_target.sum())  # values other than 0/1 are treated as invalid
    df.loc[bad_target, CLF_TARGET] = np.nan
    info = {
        "rows_raw": int(len(raw)),
        "extra_columns_ignored": [c for c in raw.columns if c not in REQUIRED],
        "missing_values": {c: int(df[c].isna().sum()) for c in REQUIRED},
        "non_numeric_or_invalid_values": non_numeric,
        "duplicate_full_rows": int(df.duplicated().sum()),
    }
    dup_id = df[ID_COL].notna() & df.duplicated(subset=[ID_COL], keep="first")
    info["duplicate_record_ids_removed"] = int(dup_id.sum())
    df = df[~dup_id].reset_index(drop=True)
    info["rows_after_dedup"] = int(len(df))
    if len(df) < MIN_ROWS:
        raise DataError(f"Only {len(df)} usable rows; at least {MIN_ROWS} are required.")
    return df, info


def supervised_xy(df, target):
    """Rows with complete features and target. Identifier and other target are never included."""
    sub = df.dropna(subset=FEATURES + [target])
    if len(sub) < MIN_ROWS:
        raise DataError(f"Only {len(sub)} complete rows for target '{target}'; need {MIN_ROWS}.")
    return sub[FEATURES].to_numpy(dtype=float), sub[target].to_numpy(dtype=float), sub


def build_report(df, info, group, sha, seed, out_path):
    numeric = FEATURES + [REG_TARGET, CLF_TARGET]
    report = dict(info)
    report.update({
        "group_code": group,
        "dataset_sha256": sha,
        "random_seed": seed,
        "row_count": info["rows_raw"],
        "feature_count": len(FEATURES),
        "feature_names": FEATURES,
        "identifier_column_excluded_from_features": ID_COL,
        "targets": [REG_TARGET, CLF_TARGET],
        "descriptive_statistics": df[numeric].describe().to_dict(),
        "rows_complete_for_regression": int(df.dropna(subset=FEATURES + [REG_TARGET]).shape[0]),
        "rows_complete_for_classification": int(df.dropna(subset=FEATURES + [CLF_TARGET]).shape[0]),
    })
    write_json(out_path, report)
    return report
