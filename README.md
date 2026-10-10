# Telecom Customer Churn Prediction

An end-to-end machine learning project that predicts which telecom customers are likely to cancel their subscription. It includes a reproducible training pipeline, a Streamlit app, a FastAPI prediction service (with Docker), an automated test suite with CI, and an interactive customer risk network.

![CI](https://github.com/HmzaTa/customer-churn-prediction/actions/workflows/ci.yml/badge.svg)

> Update the badge URL above to match your repository name, or delete the line.

## Project overview

Customer churn is when a customer stops using a service. Finding at-risk customers before they leave is cheaper than acquiring new ones. This project approaches the problem from two angles:

| Tool | Level | Question it answers |
|---|---|---|
| ML model (Streamlit app, API, CLI) | Individual | What is the probability that *this* customer will churn? |
| Risk network graph | Population | Which customer segments are at risk, and who do they look like? |

## Results

Dataset: [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), 7,043 customers and 21 columns.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** (selected) | 80.4% | 66.4% | 52.7% | 58.8% | **0.8398** |
| Random Forest | 79.6% | 65.4% | 49.2% | 56.1% | 0.8394 |
| Gradient Boosting | 78.7% | 62.6% | 48.7% | 54.8% | 0.8375 |

All metrics are on a held-out, stratified 20% test set (1,405 customers) at the default 0.5 decision threshold. The model with the highest ROC-AUC is selected. Exact accuracy and recall can shift by about 0.002 between scikit-learn versions; ROC-AUC for Logistic Regression is stable at 0.8398.

Logistic Regression won, which suggests the churn signal in this dataset is largely linear. The three models are within 0.003 ROC-AUC of each other, so the simplest one is also the easiest to explain and serve.

## How the pipeline works

```
Raw data (7,043 rows, 21 columns)
  -> Cleaning: drop customerID, fill 11 blank TotalCharges with 0 (new customers),
               encode target, drop 22 duplicate rows  ->  7,021 rows
  -> Feature engineering: 5 new features
       AverageChargePerMonth, TenureGroup_encoded (New / Medium / Long),
       HasProtection, NumAddonServices, ChargePerService
  -> Encoding: Yes/No columns -> 0/1, multi-category columns -> one-hot  ->  46 features
  -> Stratified 80/20 split (5,616 train / 1,405 test, random_state=42)
  -> StandardScaler on 4 continuous columns, fit on the training set only
  -> Train and compare 3 models, select by ROC-AUC, save the bundle to models/churn_model.pkl
```

The saved bundle contains the model, the fitted scaler and the feature list, so inference applies exactly the same transformations as training.

**Models**
- Logistic Regression: `C=1.0`, `max_iter=1000`
- Random Forest: 200 trees, `max_depth=10`, `min_samples_leaf=5`
- Gradient Boosting: 200 trees, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`. If XGBoost is installed, an `XGBClassifier` with similar settings is used in its place, so numbers can differ slightly from the table above.

## What drives churn

Churn rates measured on the raw dataset (overall 26.5%):

| Segment | Churn rate |
|---|---|
| Month-to-month contract | 42.7% |
| One-year contract | 11.3% |
| Two-year contract | 2.8% |
| Tenure of 12 months or less | 47.4% (vs 17.1% for longer tenure) |
| Monthly charges above $70 | 35.4% (vs 17.4% at or below $70) |
| Fiber optic internet | 41.9% (vs 19.0% for DSL) |
| No Online Security / no Tech Support | about 42% (vs about 15% with the service) |

Month-to-month customers churn about 4 times as often as one-year customers and about 15 times as often as two-year customers.

## Quick start

```bash
pip install -r requirements.txt
```

1. Download the dataset from Kaggle and place it at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
   If the file is missing, the training script generates a synthetic demo dataset instead, and the results will be much weaker. Do not report metrics from synthetic data.
2. Train and save the model:
   ```bash
   python src/train_model.py
   ```
3. Launch the Streamlit app:
   ```bash
   streamlit run app/streamlit_app.py
   ```

Other tools:

```bash
python src/predict.py --examples                                   # three example customers
python src/predict.py --input data/new_customers.csv --output data/predictions.csv   # batch
python src/generate_report.py                                      # HTML performance report
```

## Prediction API (FastAPI)

The API loads the saved model once at startup and validates every request with Pydantic (allowed categories, non-negative numbers, no unknown fields).

```bash
pip install -r requirements-api.txt
uvicorn api.main:app --reload
```

Open http://localhost:8000/docs for interactive Swagger documentation, or call it directly:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @api/example_customer.json
```

```json
{"churn_probability": 0.757, "prediction": "High Risk", "threshold": 0.5}
```

| Endpoint | Description |
|---|---|
| `GET /health` | `200` when the model is loaded, `503` otherwise |
| `POST /predict` | Churn probability and risk label for one customer. Optional `?threshold=0.4` sets the probability at which a customer counts as High Risk. `TotalCharges` is optional and defaults to `MonthlyCharges x tenure`. |

Invalid input (an unknown contract type, a negative tenure, a missing field) returns `422`.

## Docker

The image contains only the code needed for inference. The trained model is mounted at run time, not baked into the image.

```bash
docker build -t churn-api .

# Linux / macOS / Git Bash
docker run -p 8000:8000 -v "$(pwd)/models:/app/models:ro" churn-api

# Windows PowerShell
docker run -p 8000:8000 -v "${PWD}/models:/app/models:ro" churn-api
```

The model file is a pickle, so the scikit-learn version in the image must match the one used for training. Check yours with `pip show scikit-learn` and pin it in `requirements-api.txt` if the container reports a version warning.

## Tests and CI

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check api src/inference.py tests
```

The 25 tests cover feature engineering, cleaning, scaler leakage (the scaler is fit on training data only), a stratified split, feature alignment for unseen categories, threshold behaviour, agreement between the API code and the original CLI code, and every API path including validation errors and a missing model. The tests build a small model from the project's synthetic data generator, so they need neither the Kaggle file nor the real model.

GitHub Actions (`.github/workflows/ci.yml`) runs lint and tests on every push and builds the Docker image.

## Customer Risk Network

An interactive network of 48 behavioural customer segments, built with pyvis and NetworkX.

- **Nodes:** segments defined by contract type, internet service, tenure group and churn status. Node size is the segment population.
- **Edges:** cosine similarity on 7 behavioural features, keeping the 3 most similar neighbours per segment (99 edges).
- **Colour:** a composite risk score from 0 to 1 combining churn status, contract, tenure and monthly charge.

```bash
cd network
pip install pyvis networkx scipy
python telco_risk_network.py   # then open telco_risk_network.html
```

## Project structure

```
api/                    FastAPI service (main.py, example_customer.json)
app/streamlit_app.py    Streamlit web app
src/
  data_preprocessing.py   loading, cleaning, encoding, scaling, splitting
  feature_engineering.py  engineered features
  train_model.py          training pipeline (entry point)
  evaluate_model.py       metrics and plots
  inference.py            lightweight prediction code shared by the API and tests
  predict.py              command-line predictions
  generate_report.py      HTML report
tests/                  pytest suite
network/                risk network script and outputs
notebooks/              EDA, feature engineering, model training
visualization/output/   generated plots
models/                 saved model bundle
Dockerfile, requirements*.txt, ruff.toml, .github/workflows/ci.yml
```

## Known limitations

- **Recall is 52.7% at the default 0.5 threshold:** the model catches about half of the customers who churn. Lowering the threshold raises recall at the cost of precision. The right value depends on what a retention offer costs compared with a lost customer, and the API accepts a custom threshold.
- **Model selection uses the test set:** the best model is chosen by test-set ROC-AUC, and hyperparameters are set by hand (no tuning, no cross-validation). The three models are very close, so the effect is small, but a nested or cross-validated comparison would be cleaner.
- **Duplicate rows are dropped after `customerID` is removed:** 22 rows that are identical on every feature are treated as duplicates, even though they could be different customers.
- **Single-record API:** there is no batch endpoint (use `python src/predict.py --input ...` for batches), and the API does not check cross-field consistency (for example, internet add-ons for a customer with no internet service).
- **Static training insights:** the "key business insights" printed at the end of `train_model.py` are fixed text, not computed from the data. Use the table above for measured rates.
- **Public dataset:** the Telco dataset is a well-known sample, so this project demonstrates the engineering workflow, not a business deployment.

## Tech stack

pandas, NumPy, scikit-learn, (optional) XGBoost, matplotlib, seaborn, joblib, Streamlit, FastAPI, Pydantic, Uvicorn, pytest, ruff, Docker, GitHub Actions, pyvis, NetworkX, SciPy.
