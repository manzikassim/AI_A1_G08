# Musanze HarvestLink Decision Lab (SWE 3513, Assignment 1)

- Group: 8 | Group code: AI-G08 | Leader: TODO | Repo: TODO | Final commit: TODO
- Dataset SHA-256: printed by `run_all.py` (copy it here and into Moodle)

## Members and roles
| Member | Role | Main files |
|---|---|---|
| TODO | 1 Data and UX lead | src/common.py, UI/UX PDF |
| TODO | 2 Regression engineer | src/regression.py |
| TODO | 3 Classification engineer | src/classification.py |
| TODO | 4 Clustering and QA engineer | src/clustering.py, evidence/TEST_LOG.pdf |
| TODO | 5 Reproducibility and release lead | run_all.py, predict.py, README, requirements.txt |

## Setup (tested with Python 3.12; re-check on your machine and update this line)
```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
Place the lecturer-issued file, unmodified, at `data/AI_A1_G08.csv`, then:
```
python run_all.py --data data/AI_A1_G08.csv --output artifacts/ --group AI-G08
python predict.py --record '{"plot_area_ha":1.2,"rainfall_mm":81,"soil_ph":5.7,"seed_kg":210,"distance_km":14,"arrival_hour":9}'
```
Windows PowerShell: wrap the JSON in single quotes and escape inner double quotes, or use cmd with `"{\"plot_area_ha\":1.2,...}"`.

## Expected outputs
`artifacts/`: data_report.json, regression_metrics.json, regression_loss.png, classification_metrics.json,
confusion_matrix.png, clustering_metrics.json, clusters.csv, cluster_plot.png. `models/`: saved models and scalers.
Everything is regenerated (overwritten) on each run. Random seed 42 is recorded in every metrics file.

## Method summary
- **Data:** exact column check, numeric coercion, missing and duplicate reports, record_id never used as a feature.
- **Regression:** NumPy batch gradient descent on standardised features (train-fitted), 80/20 split, MAE/RMSE/R2, closed-form check.
- **Classification:** logistic regression (balanced classes), scaler inside the pipeline, stratified split, fixed 0.5 threshold.
- **Clustering:** inputs only, standardised, k-means k=2..5, choice by silhouette. Clusters are exploratory, not verified categories.
- **predict.py:** validates all six fields (presence, number type, plausible range) and returns JSON or a clear error (exit code 1).

## Known limitations
Linear models only; clusters have no verified meaning; rows with missing values are dropped for supervised training
(median-imputed for clustering only); predict.py needs `run_all.py` to have been run first.
