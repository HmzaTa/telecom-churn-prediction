"""
evaluate_model.py
-----------------
Model evaluation utilities: metrics, confusion matrix, ROC curves,
and feature importance plots.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, classification_report
)


# PALETTE

PALETTE = {
    "primary":   "#1e40af",   # deep blue
    "danger":    "#dc2626",   # red
    "success":   "#16a34a",   # green
    "warning":   "#d97706",   # amber
    "neutral":   "#6b7280",   # gray
    "bg":        "#f8fafc",
    "bg_dark":   "#1e293b",
    "text":      "#0f172a",
}

CMAP = sns.diverging_palette(220, 20, as_cmap=True)


# CORE METRICS

def evaluate_model(model, X_test, y_test, model_name: str = "Model",
                   threshold: float = 0.5) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Compute classification metrics for a fitted model.

    Returns
    -------
    (metrics_dict, y_pred, y_prob)
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_prob),
    }

    print(f"\n── {model_name} ──")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    return metrics, y_pred, y_prob


# CONFUSION MATRIX

def plot_confusion_matrix(y_test, y_pred, model_name: str = "Model",
                          save_dir: str | None = None) -> None:
    """Plot and optionally save a styled confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        linewidths=0.5, linecolor="white",
        ax=ax, cbar=False,
        annot_kws={"size": 16, "weight": "bold"}
    )

    ax.set_xlabel("Predicted Label", fontsize=12, color=PALETTE["text"])
    ax.set_ylabel("True Label", fontsize=12, color=PALETTE["text"])
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14,
                 fontweight="bold", color=PALETTE["text"], pad=14)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = model_name.lower().replace(" ", "_") + "_confusion_matrix.png"
        fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")
        print(f"[SAVED] {fname}")
    plt.close(fig)


# ROC CURVES (all models on one plot)

def plot_roc_curves(results: dict, save_dir: str | None = None) -> None:
    """
    Plot ROC curves for all models on a single axes.

    Parameters
    ----------
    results : dict
        Output of train_all_models: {model_name: {"model", "y_prob", ...}}
    """
    colors = [PALETTE["primary"], PALETTE["danger"], PALETTE["success"],
              PALETTE["warning"], PALETTE["neutral"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    # Diagonal reference
    ax.plot([0, 1], [0, 1], linestyle="--", color=PALETTE["neutral"],
            linewidth=1.2, label="Random Classifier (AUC = 0.50)")

    for i, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(res["y_test"] if "y_test" in res else _get_y_test(res),
                                 res["y_prob"])
        auc = res["metrics"]["roc_auc"]
        ax.plot(fpr, tpr, color=colors[i % len(colors)],
                linewidth=2.2, label=f"{name}  (AUC = {auc:.4f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14,
                 fontweight="bold", color=PALETTE["text"], pad=14)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, "roc_curves_comparison.png"),
                    dpi=150, bbox_inches="tight")
        print("[SAVED] roc_curves_comparison.png")
    plt.close(fig)


def _get_y_test(res):
    """Helper — results dict stores y_prob; y_test is captured externally."""
    # This is only needed if y_test is embedded; handled by caller.
    raise ValueError("y_test not found in results dict")


# FEATURE IMPORTANCE

def plot_feature_importance(model, feature_names: list[str],
                             model_name: str = "Model",
                             top_n: int = 20,
                             save_dir: str | None = None) -> None:
    """
    Plot horizontal bar chart of feature importances.
    Works for tree-based models (feature_importances_) and
    Logistic Regression (coef_).
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title_suffix = "Feature Importances (Gini)"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        title_suffix = "Feature Coefficients (|β|)"
    else:
        print(f"[WARN] Model type {type(model).__name__} has no feature importance attribute.")
        return

    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_values = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    # Color gradient: most important = primary blue
    norm = plt.Normalize(top_values.min(), top_values.max())
    colors = plt.cm.Blues(norm(top_values))

    bars = ax.barh(range(top_n), top_values[::-1], color=colors[::-1],
                   edgecolor="white", linewidth=0.4)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=10)
    ax.set_xlabel(title_suffix, fontsize=11)
    ax.set_title(f"{model_name} — Top {top_n} {title_suffix}",
                 fontsize=14, fontweight="bold", color=PALETTE["text"], pad=14)
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    # Value labels
    for bar, val in zip(bars, top_values[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=8,
                color=PALETTE["text"])

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = model_name.lower().replace(" ", "_") + "_feature_importance.png"
        fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")
        print(f"[SAVED] {fname}")
    plt.close(fig)


# FULL EVALUATION REPORT

def full_evaluation_report(results: dict, y_test, save_dir: str | None = None) -> None:
    """
    Generate confusion matrices and feature importance plots for all models.
    """
    for name, res in results.items():
        plot_confusion_matrix(y_test, res["y_pred"], model_name=name,
                              save_dir=save_dir)
