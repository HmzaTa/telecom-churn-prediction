"""
src/generate_report.py
----------------------
Generates a comprehensive HTML model performance report.

Usage:
    python src/generate_report.py

Output:
    reports/model_report.html
"""

import os
import sys
import base64
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.metrics import (
    roc_curve, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from train_model import load_model
from data_preprocessing import full_preprocessing_pipeline

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "churn_model.pkl")
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "reports")
RAW_DATA     = os.path.join(PROJECT_ROOT, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")


# HELPERS
def fig_to_b64(fig) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def img_tag(b64: str, width: str = "100%") -> str:
    return f'<img src="data:image/png;base64,{b64}" style="width:{width};border-radius:8px;">'


# PLOT HELPERS
BG = "#f8fafc"
TEXT = "#0f172a"

def make_roc_plot(y_test, y_prob, model_name: str):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", linewidth=1.2, label="Random (0.50)")
    ax.plot(fpr, tpr, color="#2563eb", linewidth=2.5, label=f"{model_name} (AUC={auc:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#2563eb")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(linestyle=":", alpha=0.5)
    plt.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


def make_cm_plot(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(BG)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"],
                cbar=False, linewidths=0.5, linecolor="white",
                annot_kws={"size": 16, "weight": "bold"}, ax=ax)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


def make_feat_imp_plot(model, feature_names: list[str], top_n: int = 20):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
    else:
        return None

    idx     = np.argsort(imp)[::-1][:top_n]
    feats   = [feature_names[i] for i in idx]
    vals    = imp[idx]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    norm   = plt.Normalize(vals.min(), vals.max())
    colors = plt.cm.Blues(norm(vals))
    ax.barh(range(top_n), vals[::-1], color=colors[::-1], edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(feats[::-1], fontsize=9)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


def make_churn_dist_plot(y_test):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    counts = pd.Series(y_test).value_counts().sort_index()
    ax.bar(["No Churn", "Churn"], counts.values,
           color=["#2563eb", "#dc2626"], edgecolor="white", linewidth=1.5, width=0.5)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, f"{v:,}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_title("Test Set Distribution", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


# HTML TEMPLATE
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Churn Prediction — Model Report</title>
<style>
  :root {{
    --blue:   #2563eb;
    --red:    #dc2626;
    --green:  #16a34a;
    --gray:   #64748b;
    --bg:     #f1f5f9;
    --card:   #ffffff;
    --text:   #0f172a;
    --border: #e2e8f0;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); }}
  header {{
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
    color: white; padding: 40px 48px;
  }}
  header h1 {{ font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; }}
  header p  {{ font-size: 1rem; opacity: 0.85; margin-top: 6px; }}
  .badge {{
    display: inline-block; background: rgba(255,255,255,0.2);
    border-radius: 20px; padding: 3px 14px; font-size: 0.82rem;
    margin-top: 12px; border: 1px solid rgba(255,255,255,0.3);
  }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 40px 24px; }}
  .section-title {{
    font-size: 1.25rem; font-weight: 700; color: var(--text);
    margin: 40px 0 16px; padding-bottom: 8px;
    border-bottom: 3px solid var(--blue);
  }}
  .card {{
    background: var(--card); border-radius: 12px;
    padding: 24px; box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    border: 1px solid var(--border); margin-bottom: 20px;
  }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .grid-4 {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; }}
  .metric-card {{
    background: var(--card); border-radius: 10px;
    padding: 20px 18px; border: 1px solid var(--border);
    border-top: 4px solid var(--blue);
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    text-align: center;
  }}
  .metric-card.green {{ border-top-color: var(--green); }}
  .metric-card.red   {{ border-top-color: var(--red); }}
  .metric-card.gray  {{ border-top-color: var(--gray); }}
  .metric-value {{ font-size: 2rem; font-weight: 800; color: var(--blue); }}
  .metric-card.green .metric-value {{ color: var(--green); }}
  .metric-card.red   .metric-value {{ color: var(--red); }}
  .metric-card.gray  .metric-value {{ color: var(--gray); }}
  .metric-label {{ font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--gray); margin-top: 4px; }}
  table {{ width:100%; border-collapse:collapse; font-size:0.93rem; }}
  thead th {{ background: #eff6ff; padding: 10px 14px; text-align:left; font-weight:600; color: var(--blue); }}
  tbody td {{ padding: 9px 14px; border-bottom: 1px solid var(--border); }}
  tbody tr:last-child td {{ border-bottom: none; }}
  tbody tr:hover {{ background: #f8fafc; }}
  .best-row td {{ background: #eff6ff; font-weight: 600; }}
  .insight {{
    background: #eff6ff; border-left: 4px solid var(--blue);
    border-radius: 6px; padding: 12px 16px;
    margin-bottom: 10px; font-size: 0.94rem; color: #1e3a8a;
  }}
  .insight::before {{ content: "💡 "; }}
  footer {{ text-align:center; color: var(--gray); font-size:0.85rem; padding: 32px; }}
</style>
</head>
<body>
<header>
  <h1>📡 Telecom Customer Churn Prediction</h1>
  <p>Machine Learning Model Performance Report</p>
  <span class="badge">Generated: {generated_at}</span>
  <span class="badge" style="margin-left:8px">Best Model: {best_model_name}</span>
  <span class="badge" style="margin-left:8px">ROC-AUC: {best_auc}</span>
</header>

<div class="container">

  <!-- KEY METRICS -->
  <div class="section-title">📊 Model Performance — {best_model_name}</div>
  <div class="grid-4">
    <div class="metric-card">
      <div class="metric-value">{accuracy}</div>
      <div class="metric-label">Accuracy</div>
    </div>
    <div class="metric-card green">
      <div class="metric-value">{precision}</div>
      <div class="metric-label">Precision</div>
    </div>
    <div class="metric-card red">
      <div class="metric-value">{recall}</div>
      <div class="metric-label">Recall</div>
    </div>
    <div class="metric-card gray">
      <div class="metric-value">{f1}</div>
      <div class="metric-label">F1-Score</div>
    </div>
  </div>

  <!-- DATASET SUMMARY -->
  <div class="section-title">📋 Dataset Summary</div>
  <div class="card">
    <table>
      <thead><tr><th>Property</th><th>Value</th></tr></thead>
      <tbody>
        <tr><td>Total customers</td><td>{total_customers:,}</td></tr>
        <tr><td>Training samples</td><td>{train_size:,}</td></tr>
        <tr><td>Test samples</td><td>{test_size:,}</td></tr>
        <tr><td>Churn rate (overall)</td><td>{churn_rate}</td></tr>
        <tr><td>Number of features</td><td>{num_features}</td></tr>
      </tbody>
    </table>
  </div>

  <!-- MODEL COMPARISON TABLE -->
  <div class="section-title">🏆 Model Comparison</div>
  <div class="card">
    <table>
      <thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th></tr></thead>
      <tbody>{comparison_rows}</tbody>
    </table>
  </div>

  <!-- ROC + CM -->
  <div class="section-title">📈 ROC Curve & Confusion Matrix</div>
  <div class="grid-2">
    <div class="card">{roc_img}</div>
    <div class="card">{cm_img}</div>
  </div>

  <!-- FEATURE IMPORTANCE -->
  <div class="section-title">🔍 Feature Importance</div>
  <div class="card">{feat_imp_img}</div>

  <!-- DISTRIBUTION -->
  <div class="section-title">📊 Test Set Distribution</div>
  <div class="grid-2">
    <div class="card">{dist_img}</div>
    <div class="card">
      <p style="font-weight:600;font-size:1rem;margin-bottom:12px;">Confusion Matrix Breakdown</p>
      <table>
        <thead><tr><th>Metric</th><th>Value</th></tr></thead>
        <tbody>
          <tr><td>True Positives (correctly predicted churn)</td><td>{tp}</td></tr>
          <tr><td>True Negatives (correctly predicted no churn)</td><td>{tn}</td></tr>
          <tr><td>False Positives (predicted churn, actually stayed)</td><td>{fp}</td></tr>
          <tr><td>False Negatives (missed churners)</td><td>{fn}</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- INSIGHTS -->
  <div class="section-title">💡 Key Business Insights</div>
  <div class="card">
    <div class="insight">Month-to-month contract customers churn 3–4× more frequently than customers on annual or two-year contracts. Contract upgrades are the single most impactful retention lever.</div>
    <div class="insight">Customers with monthly charges above $70 show significantly elevated churn probability. Consider loyalty discounts or bundle plans for this segment.</div>
    <div class="insight">New customers (tenure &lt; 12 months) form the highest-risk cohort. An onboarding programme or early-engagement campaign can meaningfully reduce early churn.</div>
    <div class="insight">Customers without TechSupport or OnlineSecurity are more likely to leave. Proactively offering trial access to these services may improve retention.</div>
    <div class="insight">Fiber optic internet users churn more than DSL users — likely due to price sensitivity. Targeted retention offers or competitive pricing reviews are recommended for this group.</div>
  </div>

</div>
<footer>Telecom Churn Prediction | Generated {generated_at} | Python · scikit-learn · Gradient Boosting</footer>
</body>
</html>"""


# REPORT BUILDER
def generate_report(model_path: str = MODEL_PATH, raw_data: str = RAW_DATA,
                    output_dir: str = REPORTS_DIR) -> str:
    print("[INFO] Loading model …")
    bundle = load_model(model_path)
    model  = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]

    print("[INFO] Loading data …")
    X_train, X_test, y_train, y_test, _, _ = full_preprocessing_pipeline(raw_data)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)

    model_name = type(model).__name__

    # CM values
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Comparison table (only one model; extend if re-running multi-model)
    comp_row = f"""<tr class="best-row">
      <td>★ {model_name}</td>
      <td>{acc:.4f}</td><td>{prec:.4f}</td>
      <td>{rec:.4f}</td><td>{f1:.4f}</td><td>{auc:.4f}</td>
    </tr>"""

    # Plots
    print("[INFO] Generating plots …")
    roc_b64  = make_roc_plot(y_test, y_prob, model_name)
    cm_b64   = make_cm_plot(y_test, y_pred)
    imp_b64  = make_feat_imp_plot(model, feature_names)
    dist_b64 = make_churn_dist_plot(y_test)

    # Build HTML
    html = HTML_TEMPLATE.format(
        generated_at   = datetime.now().strftime("%Y-%m-%d %H:%M"),
        best_model_name= model_name,
        best_auc       = f"{auc:.4f}",
        accuracy       = f"{acc:.1%}",
        precision      = f"{prec:.1%}",
        recall         = f"{rec:.1%}",
        f1             = f"{f1:.1%}",
        total_customers= len(X_train) + len(X_test),
        train_size     = len(X_train),
        test_size      = len(X_test),
        churn_rate     = f"{y_test.mean():.1%}",
        num_features   = len(feature_names),
        comparison_rows= comp_row,
        roc_img        = img_tag(roc_b64),
        cm_img         = img_tag(cm_b64),
        feat_imp_img   = img_tag(imp_b64) if imp_b64 else "<p>Not available for this model type.</p>",
        dist_img       = img_tag(dist_b64),
        tp=tp, tn=tn, fp=fp, fn=fn,
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "model_report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[SAVED] Report → {out_path}")
    return out_path


if __name__ == "__main__":
    generate_report()
