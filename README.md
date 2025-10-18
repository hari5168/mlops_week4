# Week 4 — Iris MLOps project

This folder contains the Week 4 submission: a small Iris classification pipeline with training, inference, tests, DVC metadata and CI integration.

## Contents & purpose

- [train.py](week_4/train.py)  
  Script to train a RandomForest on `data/iris.csv`, evaluate on a test split, and save the trained model to `models/model.joblib`.

- [inference.py](week_4/inference.py)  
  Simple inference/evaluation script that loads `models/model.joblib`, runs predictions on `data/data.csv`, and prints sample predictions and accuracy.

- [requirements.txt](week_4/requirements.txt)  
  Minimal Python dependencies required to run training, inference and tests (scikit-learn, pandas, dvc, pytest, ...).

- data/
  - [data/iris.csv.dvc](week_4/data/iris.csv.dvc) — DVC pointer tracking the canonical `iris.csv` dataset.
  - [data/.gitignore](week_4/data/.gitignore) — prevents checked-in raw data file (`/iris.csv`).

- models/
  - [models/model.joblib.dvc](week_4/models/model.joblib.dvc) — DVC-tracked metadata for the saved model artifact.
  - [models/.gitignore](week_4/models/.gitignore) — prevents committing the large binary model file (`/model.joblib`) to git.

- tests/
  - [tests/test_model_validation.py](week_4/tests/test_model_validation.py) — CI test that loads the trained model and checks accuracy; contains function [`test_model_accuracy`](week_4/tests/test_model_validation.py).
  - [tests/test_data_validation.py](week_4/tests/test_data_validation.py) — dataset validation tests; contains [`test_data_shape`](week_4/tests/test_data_validation.py), [`test_no_missing_values`](week_4/tests/test_data_validation.py) and [`test_valid_species`](week_4/tests/test_data_validation.py).

- DVC & CI config
  - [.dvc/config](week_4/.dvc/config) — DVC remote configuration (GCS remote used for data/model storage).
  - [.dvc/.gitignore](week_4/.dvc/.gitignore) — DVC local files to ignore.
  - [.dvcignore](week_4/.dvcignore) — patterns for DVC to ignore.
  - [.github/workflows/ci-main.yml](week_4/.github/workflows/ci-main.yml) — GitHub Actions CI workflow that installs dependencies, authenticates to GCP, runs `dvc pull`, and runs the pytest suite. The CI step uses the test run output to comment on PRs and fail when tests fail.




## Notes
- Model and dataset binaries are intended to be tracked via DVC — see [models/model.joblib.dvc](week_4/models/model.joblib.dvc) and [data/iris.csv.dvc](week_4/data/iris.csv.dvc).
- CI workflow (see [.github/workflows/ci-main.yml](week_4/.github/workflows/ci-main.yml)) expects a Google Cloud service account secret (`GCP_SERVICE_ACCOUNT_KEY`) to authenticate and run.
