# Australia Weather — Rain Prediction

**Project:** Australia Weather — Rain Prediction using GitHub Actions, CircleCI and GitLab CI

A complete end-to-end example project for forecasting the probability of rain in Australia based on historical weather data. The repository demonstrates data ingestion, feature engineering for time-series and tabular weather data, model training (classical ML + simple deep models), evaluation, packaging, CI/CD pipelines (GitHub Actions, CircleCI, GitLab CI), and a lightweight deployment (FastAPI + Docker / Streamlit demo).

---

## Table of Contents

* [Project Overview](#project-overview)
* [Key Features](#key-features)
* [Data](#data)
* [Getting Started](#getting-started)

  * [Requirements](#requirements)
  * [Install](#install)
  * [Download Data](#download-data)
  * [Quickstart — Train & Evaluate](#quickstart---train--evaluate)
* [Modeling Approach](#modeling-approach)

  * [Feature Engineering](#feature-engineering)
  * [Candidate Models](#candidate-models)
  * [Evaluation Metrics](#evaluation-metrics)
* [CI/CD Pipelines](#cicd-pipelines)

  * [GitHub Actions](#github-actions)
  * [CircleCI](#circleci)
  * [GitLab CI](#gitlab-ci)
* [Deployment](#deployment)

  * [Docker / FastAPI](#docker--fastapi)
  * [Streamlit Demo (Optional)](#streamlit-demo-optional)
* [Reproducibility & Experiments](#reproducibility--experiments)
* [Project Structure](#project-structure)
* [Usage Examples](#usage-examples)
* [Contributing](#contributing)
* [License](#license)

---

## Project Overview

This repo provides a production-minded example for solving a binary classification problem: *will it rain tomorrow?* using historical weather observations from Australia. It includes an end-to-end workflow that is useful for interviews and production demonstrations:

* data ingestion and validation
* feature engineering (temporal and meteorological features)
* model training (scikit-learn & light deep learning option)
* model evaluation and calibration
* CI/CD pipelines implemented for GitHub Actions, CircleCI and GitLab CI
* containerized API and optional interactive demo

The code is modular so you can reuse the pipelines for other tabular/time-series problems.

---

## Key Features

* Clean, documented notebooks for EDA and feature engineering
* Reproducible training scripts (`train.py`) and evaluation (`evaluate.py`)
* Example models: Logistic Regression, Random Forest, XGBoost, simple LSTM
* Model serialization (`joblib` / `torch.save`) and versioning hooks
* Unit tests and data checks run in CI pipelines
* Example GitHub Actions, CircleCI and GitLab CI configuration files
* Dockerfile and FastAPI app for serving predictions
* Streamlit demo for quick manual testing and visualization

---

## Data

This project assumes you have access to a cleaned Australian weather dataset. A commonly used public dataset is the *Australian Weather* dataset (Kaggle):

* **Kaggle:** [https://www.kaggle.com/jsphyg/weather-dataset-rattle-package](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)

**Important columns (typical):**

* `Date` — observation date
* `Location` — station name
* `MinTemp`, `MaxTemp`, `Rainfall`, `Evaporation`, `Sunshine`
* `WindGustSpeed`, `WindSpeed9am`, `WindSpeed3pm`
* `Humidity9am`, `Humidity3pm`, `Pressure9am`, `Pressure3pm`
* `Cloud9am`, `Cloud3pm`, `Temp9am`, `Temp3pm`
* `RainToday` (target feature for same-day experiments)
* `RainTomorrow` (binary target — **Yes/No**)

> In this repo we use a preprocessed and anonymized subset to show the pipeline. See `data/` for scripts to download and preprocess raw CSV files.

---

## Getting Started

### Requirements

* Python 3.9+ (recommend 3.10)
* Git
* Docker (optional — for deployment)
* Optional: GPU if you plan to train deep models

Python packages (example):

```
pip install -r requirements.txt
```

`requirements.txt` includes: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `torch`, `tensorflow` (opt), `joblib`, `fastapi`, `uvicorn`, `streamlit` (optional), `pydantic`, `pytest`.

### Install

```bash
git clone https://github.com/mahmedahmed3355/australia-weather-rain-prediction.git
cd australia-weather-rain-prediction
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Download Data

```bash
python scripts/download_data.py --target-path data/raw/australia_weather.csv
python scripts/preprocess_data.py --input data/raw/australia_weather.csv --output data/processed/data_for_model.csv
```

### Quickstart — Train & Evaluate

Train baseline model (XGBoost):

```bash
python src/train.py --config conf/xgb_config.yaml
```

Evaluate:

```bash
python src/evaluate.py --model-path models/xgb/latest.joblib --test-data data/processed/test.csv
```

A successful run saves model artifacts under `models/` and metrics under `reports/`.

---

## Modeling Approach

### Feature Engineering

* Handle missing values with domain-aware imputation (forward-fill for sensors, median for static fields)
* Create temporal features: day-of-week, month, lag features (`Rainfall_lag_1`, `Rainfall_lag_7`), rolling means and stds
* Meteorological derived features: dew point approximation, relative humidity changes, pressure deltas
* Categorical encoding: target/mean encoding for `Location` or one-hot if small cardinality

### Candidate Models

* **Baseline:** Logistic Regression with calibrated probabilities
* **Tree ensemble:** Random Forest, XGBoost (recommended)
* **Neural:** simple LSTM or Temporal CNN for sequence modeling
* **Ensembling:** stacking probabilities of multiple models

### Evaluation Metrics

* **Primary:** ROC-AUC, PR-AUC (useful on imbalanced sets)
* **Secondary:** Accuracy, Precision, Recall, F1, Brier Score (calibration)
* **Operational:** expected cost from false positives/negatives if deployed (optional)

Calibration plots and confusion matrices are generated in `src/visualize.py`.

---

## CI/CD Pipelines

This repo includes example pipeline definitions for three popular CI providers. Each pipeline runs unit tests, linting, basic data validation and a quick smoke training job (small sample) to ensure reproducibility.

### GitHub Actions

File: `.github/workflows/ci.yml`

* Steps included:

  * Checkout
  * Set up Python
  * Install requirements
  * Run unit tests (`pytest`) and linters (`flake8`)
  * Run a small training smoke test using a sampled CSV
  * Upload model artifact to actions/artifacts or push to model registry

Snippet (example):

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: python -m pip install -r requirements.txt
      - run: pytest -q
      - run: python scripts/smoke_train.py --sample data/sample.csv
```

### CircleCI

File: `.circleci/config.yml`

* Similar stages: checkout, install, test, smoke-train. Example uses CircleCI orbs for caching and python setup.

### GitLab CI

File: `.gitlab-ci.yml`

* Stages: `test`, `build`, `deploy` (deploy is optional and gated)
* Use GitLab runner with Docker image containing Python and dependencies

> Each CI file contains minimal examples to validate the project on PRs. For production, expand to include artifact promotion, model quality gates and security scans.

---

## Deployment

### Docker / FastAPI

A simple FastAPI application is provided:

```
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Dockerfile builds the API image and exposes `/predict` endpoint that accepts JSON payload with recent sensor/feature values and returns probability of rain tomorrow.

### Streamlit Demo (Optional)

Run a lightweight Streamlit app for exploration and manual testing:

```
streamlit run app/streamlit_app.py
```

---

## Reproducibility & Experiments

* Use `conf/` folder for experiment configuration (YAML files)
* `scripts/run_experiment.sh` executes a full experiment with fixed seed
* `reports/` contains training logs, metrics, and artifacts

---

## Project Structure

```
├─ .github/
├─ .circleci/
├─ .gitlab-ci.yml
├─ conf/
├─ data/
│  ├─ raw/
│  └─ processed/
├─ models/
├─ reports/
├─ scripts/
├─ src/
│  ├─ data/
│  ├─ features/
│  ├─ models/
│  └─ api/
├─ app/
└─ README.md
```

---

## Usage Examples

1. Predict with saved model using Python:

```python
from joblib import load
import pandas as pd
model = load('models/xgb/latest.joblib')
X = pd.read_csv('data/processed/sample_features.csv')
probs = model.predict_proba(X)[:,1]
print(probs[:10])
```

2. Request to the API:

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"features": {"Rainfall_lag_1": 0.0, "Humidity3pm": 60}}'
```

---

## Contributing

Contributions are welcome. Please open an issue for a feature request or bug, then submit a PR against `develop` branch. Include tests and update `CHANGELOG.md`.

---

## License

This project is licensed under the MIT License.

---

*If you want, I can also generate a ready-to-commit `README.md` file in your repo, or tailor the CI snippets to the exact workflows you use (GitHub Actions job names, CircleCI orbs, GitLab runners). Just tell me which one you prefer.*
