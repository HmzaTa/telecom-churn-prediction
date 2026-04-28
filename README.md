# Telecom Customer Churn Prediction

A complete, production-quality machine learning system that predicts customer churn for a telecommunications company, combining a real-time prediction application with an interactive customer risk network visualization.

---

## Project Overview

Customer churn occurs when a customer stops using a company's service. Identifying at-risk customers **before** they leave is significantly cheaper than acquiring new ones. This project addresses that problem from two complementary angles:

| Tool | Level | Question it answers |
|------|-------|-------------------|
| **ML Prediction App** | Individual | *What is the probability THIS specific customer will churn?* |
| **Risk Network Graph** | Population | *Which customer segments are at risk and who do they look like?* |

Together they form a complete retention intelligence system, the network tells you **where to look**, the model tells you **who to call**.

---

## Project Structure

```
customer-churn-prediction/
│
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   ← Kaggle dataset
│   └── processed/
│       └── telco_processed.csv                      ← After preprocessing
│
├── src/
│   ├── data_preprocessing.py    ← Loading, cleaning, encoding, scaling
│   ├── feature_engineering.py   ← Feature creation
│   ├── train_model.py           ← Training pipeline (main entry point)
│   ├── evaluate_model.py        ← Metrics, ROC curves, confusion matrix
│   ├── predict.py               ← CLI prediction tool
│   └── generate_report.py       ← HTML report generator
│
├── network/
│   ├── telco_risk_network.py    ← Network graph script
│   ├── telco_nodes.csv          ← Customer segment nodes
│   ├── telco_edges.csv          ← Behavioural similarity edges
│   └── telco_risk_network.html  ← Generated interactive network
│
├── visualization/
│   ├── plots.py                 ← EDA visualization functions
│   └── output/                  ← Generated PNG plots
│
├── models/
│   └── churn_model.pkl          ← Saved model bundle
│
├── app/
│   └── streamlit_app.py         ← Interactive web application
│
├── notebooks/
│   ├── 01_eda.ipynb             ← Exploratory data analysis
│   ├── 02_feature_engineering.ipynb  ← Feature engineering walkthrough
│   └── 03_model_training.ipynb  ← Model training and evaluation
│
├── reports/
│   └── model_report.html        ← Auto-generated performance report
│
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add the dataset

Download from Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place the CSV at:
```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

> **Note:** If the file is not present, the pipeline automatically generates a synthetic dataset for demonstration purposes — though results will be significantly weaker than with the real data.

### 3. Train the model

```bash
python src/train_model.py
```

### 4. Launch the app

```bash
streamlit run app/streamlit_app.py
```

### 5. Generate the network graph (optional)

```bash
cd network
python telco_risk_network.py
```

Then open `network/telco_risk_network.html` in any browser.

---

## Machine Learning App

### What it Does

An end-to-end machine learning system that takes a telecom customer's profile as input and outputs the probability that this customer will cancel their subscription, along with a risk label and personalized retention recommendations.

### The Pipeline

```
Raw Data (7,043 customers, 21 columns)
        ↓
Data Cleaning
→ Remove customerID (non-predictive)
→ Fix TotalCharges empty strings (100 new customers with tenure = 0)
→ Encode target: Yes → 1, No → 0
        ↓
Feature Engineering (5 new features created)
→ AverageChargePerMonth  = TotalCharges / tenure
→ TenureGroup            = New (0–12m) / Medium (13–36m) / Long (36m+)
→ HasProtection          = 1 if OnlineSecurity or TechSupport = Yes
→ NumAddonServices       = count of 6 optional services subscribed
→ ChargePerService       = MonthlyCharges / (NumAddonServices + 1)
        ↓
Encoding + Scaling
→ Binary columns (Yes/No) → Label Encoding (0/1)
→ Multi-category columns  → One-Hot Encoding
→ 20 features → 46 features after encoding
→ StandardScaler on continuous columns (mean=0, std=1)
        ↓
Train / Test Split
→ 80% training (5,634 customers)
→ 20% testing  (1,409 customers)
→ Stratified — preserves 26% churn rate in both sets
→ random_state=42 — fully reproducible
        ↓
Model Training & Selection
→ Three algorithms trained and evaluated
→ Winner selected by ROC-AUC
→ Saved to models/churn_model.pkl
        ↓
Deployment
→ Streamlit web application
→ Real-time predictions on new customer profiles
```

### The Three Models

**Logistic Regression — Linear Model**

Computes a weighted linear combination of all 46 features (Z score), passes it through the sigmoid function to produce a probability between 0 and 1, then minimizes binary cross-entropy loss through iterative weight optimization until convergence.

```
Z = β₀ + β₁x₁ + β₂x₂ + ... + β₄₆x₄₆
P(Churn) = 1 / (1 + e⁻ᶻ)
```

**Random Forest — Parallel Ensemble**

200 decision trees each trained on a different bootstrap sample of customers, considering a random subset of features at each split. Final prediction = average probability across all 200 trees. Individual tree errors cancel out through averaging.

**Gradient Boosting — Sequential Ensemble**

200 decision trees built one after another, each tree specifically corrects the residuals (errors) of the previous one. Starts with a naive baseline prediction and stacks 200 targeted corrections with a learning rate of 0.05.

```
Final probability = baseline + Σ(learning_rate × tree_correction)
```

### Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| **Logistic Regression** | **80.4%** | **66.4%** | **52.7%** | **58.8%** | **0.8398 ✓** |
| Random Forest | 79.6% | 65.4% | 49.2% | 56.1% | 0.8394 |
| Gradient Boosting | 78.7% | 62.6% | 48.7% | 54.8% | 0.8375 |

**Selection criterion: ROC-AUC** — measures whether the model consistently assigns higher churn probabilities to customers who actually churn. A score of 0.84 means that given one customer who churned and one who didn't, the model correctly ranks the churner as higher risk 84% of the time.

Logistic Regression won, the simplest model outperformed both ensemble methods. This indicates the churn patterns in this dataset are sufficiently linear that a simple decision boundary captures them effectively. **Data quality matters more than algorithmic complexity.**

### Key Churn Drivers

Based on feature importance analysis:

- **Contract type** — month-to-month customers churn 3–4× more than two-year contract holders
- **Tenure** — customers in their first 12 months are the highest-risk segment
- **Monthly charges** — above $70/month significantly increases churn probability
- **Internet service** — fiber optic users churn more than DSL users
- **Support services** — customers without TechSupport or OnlineSecurity are more likely to leave

### Streamlit App Features

- **Predict Churn** — fill in 18 customer attributes, get an instant churn probability with risk gauge and retention recommendations
- **EDA Visualizations** — all charts generated during training
- **About** — project overview and model information

### CLI Tools

**Interactive single prediction:**
```bash
python src/predict.py
```

**Batch prediction from CSV:**
```bash
python src/predict.py --input data/new_customers.csv --output data/predictions.csv
```

**Example profiles:**
```bash
python src/predict.py --examples
```

**Generate HTML report:**
```bash
python src/generate_report.py
```

---

## Customer Risk Network

### What it Does

Visualizes the entire customer base as an interactive network of 48 behavioral segments — showing which segments are at risk and how risk propagates between behaviorally similar groups.

### How it Works

**Nodes :** each node is a customer segment defined by 4 attributes:
- Contract type (Month-to-month / One year / Two year)
- Internet service (DSL / Fiber optic / No)
- Tenure group (Short / Medium / Long)
- Churn status (Active / Churned)

Node size is proportional to segment population. The largest segment contains 637 customers.

**Edges :** connections between segments are computed using **cosine similarity** on 7 behavioral features: average monthly charge, average tenure, number of add-on services, protection rate, senior citizen rate, partner rate and dependent rate. Only the top 3 most similar neighbors per segment are connected, giving 99 edges total. This is a k-nearest-neighbor graph.

**Color :** a continuous composite risk score between 0 and 1:
```
Risk = 0.40 × churn status
     + 0.25 × contract risk
     + 0.20 × tenure risk
     + 0.15 × normalised monthly charge
```
Color gradient runs from deep blue (low risk) through teal → gold → orange to deep red (critical risk).

**Layout :** deliberate risk axis:
- X-axis → risk score (low risk left, high risk right)
- Y-axis → average tenure (short tenure bottom, long tenure top)

The most dangerous segments — high risk and short tenure — cluster in the top-right corner. The safest segments cluster in the bottom-left.

### Running the Network

```bash
cd network
pip install pyvis networkx scipy
python telco_risk_network.py
```

Open `telco_risk_network.html` in any browser. Hover over any node or edge for full details.

### Key Insight from the Network

The highest-risk segment — Month-to-month, Fiber optic, Short tenure, Churned — contains **637 customers** with a risk score of **0.97**. Critically, this segment has thick edges connecting it to active segments with similar behavioral profiles — meaning there are current customers who look almost identical to confirmed churners. Those are the highest-priority retention targets.

---

## EDA Visualizations

Generated automatically during training and saved to `visualization/output/`:

| File | Content |
|------|---------|
| `01_churn_distribution.png` | Overall churn rate — bar + pie chart |
| `02_churn_by_contract.png` | Churn rate by contract type |
| `03_churn_vs_monthly_charges.png` | Monthly charges KDE + box plot by churn |
| `04_churn_vs_tenure.png` | Tenure distribution by churn status |
| `05_correlation_heatmap.png` | Top 15 feature correlations |
| `06_churn_by_internet_service.png` | Churn rate by internet service type |
| `roc_curves_comparison.png` | ROC curves for all three models |
| `*_feature_importance.png` | Top 20 feature importances for best model |

---

## Workflow

```
First time only:
    python src/train_model.py          ← trains and saves model

Every time after:
    streamlit run app/streamlit_app.py ← loads saved model, no retraining

Optional:
    python src/predict.py --examples   ← CLI predictions
    python src/generate_report.py      ← HTML performance report
    python network/telco_risk_network.py ← regenerate network graph
```

The model is fully deterministic, `random_state=42` is set throughout. Running `train_model.py` multiple times produces identical results.

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical computation |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `matplotlib` | Plots and visualizations |
| `seaborn` | Statistical visualizations |
| `joblib` | Model serialization |
| `streamlit` | Interactive web application |
| `pyvis` | Interactive network visualization |
| `networkx` | Network graph construction |
| `scipy` | Cosine similarity computation |

---

## Requirements

Python 3.10+ recommended (tested on Python 3.12.10 and 3.14).

```bash
pip install -r requirements.txt
```

---

## Business Insights

1. **Month-to-month contracts** are the single strongest churn predictor — customers with no long-term commitment have no financial reason to stay
2. **The first 12 months are critical** — new customers who don't build loyalty early are the highest-risk segment
3. **Fiber optic + high charges + no support services** is the most dangerous combination
4. **Two-year contract customers** have a churn probability below 5% — contract upgrades are the most impactful retention lever
5. **The simplest model won** — Logistic Regression outperformed both ensemble methods, confirming that clean real-world data produces strong linear signals that don't require complex modeling

---

## License

MIT License — free for personal and commercial use.
