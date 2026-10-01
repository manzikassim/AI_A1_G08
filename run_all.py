"""Run the full pipeline: python run_all.py --data data/AI_A1_G08.csv --output artifacts/ --group AI-G08"""
import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

from src.classification import run_classification
from src.clustering import run_clustering
from src.common import MODEL_VERSION, SEED, DataError, build_report, load_data, sha256_file, write_json
from src.regression import run_regression

MODELS_DIR = Path(__file__).resolve().parent / "models"


def main():
    ap = argparse.ArgumentParser(description="Musanze HarvestLink decision pipeline")
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--group", required=True, help="e.g. AI-G08")
    a = ap.parse_args()
    if not a.group.strip():
        ap.error("--group must not be empty")
    if not re.fullmatch(r"AI-G\d{2}", a.group):
        print(f"WARNING: group code '{a.group}' does not look like AI-GXX", file=sys.stderr)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        sha = sha256_file(a.data)
        print(f"Group code     : {a.group}\nDataset SHA-256: {sha}\nRandom seed    : {SEED}")
        df, info = load_data(a.data)
        rep = build_report(df, info, a.group, sha, SEED, out / "data_report.json")
        print(f"Rows: {rep['row_count']} raw, {info['rows_after_dedup']} after de-duplication; features: {rep['feature_count']}")
        r = run_regression(df, out, MODELS_DIR)
        print(f"Regression    : test MAE={r['test_metrics']['mae']:.2f} RMSE={r['test_metrics']['rmse']:.2f} R2={r['test_metrics']['r2']}")
        c = run_classification(df, out, MODELS_DIR)
        print(f"Classification: acc={c['accuracy']:.3f} precision={c['precision']:.3f} recall={c['recall']:.3f} f1={c['f1']:.3f}")
        k = run_clustering(df, out, MODELS_DIR)
        print(f"Clustering    : silhouette by k={k['silhouette_by_k']} -> selected k={k['selected_k']}")
        write_json(MODELS_DIR / "meta.json", {"group_code": a.group, "model_version": MODEL_VERSION, "random_seed": SEED,
                                              "dataset_sha256": sha, "trained_at": datetime.now().isoformat(timespec="seconds")})
    except DataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    print(f"Done. Artifacts written to {out.resolve()}; models to {MODELS_DIR}")


if __name__ == "__main__":
    main()
