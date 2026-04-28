"""
visualization/plots.py
----------------------
Exploratory Data Analysis visualizations for the Telco Churn dataset.
All functions accept a cleaned DataFrame and produce publication-quality plots.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd

# STYLE CONFIG
PALETTE = {
    "churn_yes":  "#dc2626",   # red – churned
    "churn_no":   "#2563eb",   # blue – retained
    "primary":    "#1e40af",
    "bg":         "#f8fafc",
    "text":       "#0f172a",
    "grid":       "#e2e8f0",
}

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.facecolor":   PALETTE["bg"],
    "figure.facecolor": PALETTE["bg"],
    "axes.edgecolor":   PALETTE["grid"],
    "grid.color":       PALETTE["grid"],
    "text.color":       PALETTE["text"],
    "axes.labelcolor":  PALETTE["text"],
    "xtick.color":      PALETTE["text"],
    "ytick.color":      PALETTE["text"],
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save(fig, filename: str, save_dir: str | None = None) -> None:
    d = save_dir or OUTPUT_DIR
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[SAVED] {path}")
    plt.close(fig)


# 1. CHURN DISTRIBUTION

def plot_churn_distribution(df: pd.DataFrame, save_dir: str | None = None) -> None:
    """Bar + pie chart showing overall churn distribution."""
    churn_counts = df["Churn"].value_counts()
    labels = ["Retained (No Churn)", "Churned"]
    sizes  = [churn_counts.get(0, 0), churn_counts.get(1, 0)]
    colors = [PALETTE["churn_no"], PALETTE["churn_yes"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("Customer Churn Distribution", fontsize=16,
                 fontweight="bold", color=PALETTE["text"], y=1.02)

    # Bar chart
    ax = axes[0]
    bars = ax.bar(labels, sizes, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Number of Customers", fontsize=12)
    ax.set_title("Count", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{val:,}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(sizes) * 1.15)

    # Pie chart
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 12}
    )
    for at in autotexts:
        at.set_fontsize(13)
        at.set_fontweight("bold")
        at.set_color("white")
    ax2.set_title("Proportion", fontsize=13, fontweight="bold")

    plt.tight_layout()
    _save(fig, "01_churn_distribution.png", save_dir)


# 2. CHURN vs CONTRACT TYPE

def plot_churn_by_contract(df: pd.DataFrame, save_dir: str | None = None) -> None:
    """Grouped bar chart: churn rate by contract type."""
    # Work with raw (pre-encoded) data – contract col may be string
    col = "Contract"
    if col not in df.columns:
        print(f"[WARN] Column '{col}' not found. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(PALETTE["bg"])

    ct = pd.crosstab(df[col], df["Churn"], normalize="index") * 100
    ct.columns = ["Retained %", "Churn %"]
    ct[["Retained %", "Churn %"]].plot(
        kind="bar", ax=ax,
        color=[PALETTE["churn_no"], PALETTE["churn_yes"]],
        edgecolor="white", linewidth=0.8, width=0.6
    )

    ax.set_title("Churn Rate by Contract Type", fontsize=15, fontweight="bold",
                 color=PALETTE["text"], pad=12)
    ax.set_xlabel("Contract Type", fontsize=12)
    ax.set_ylabel("Percentage of Customers (%)", fontsize=12)
    ax.yaxis.set_major_formatter(mtick_pct())
    ax.legend(["Retained", "Churned"], fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    _save(fig, "02_churn_by_contract.png", save_dir)


# 3. CHURN vs MONTHLY CHARGES

def plot_churn_by_monthly_charges(df: pd.DataFrame, save_dir: str | None = None) -> None:
    """KDE + box plots comparing monthly charges for churned vs retained."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("Monthly Charges vs Churn", fontsize=15,
                 fontweight="bold", color=PALETTE["text"])

    col = "MonthlyCharges"
    retained = df[df["Churn"] == 0][col]
    churned  = df[df["Churn"] == 1][col]

    # KDE
    ax = axes[0]
    ax.set_facecolor(PALETTE["bg"])
    retained.plot.kde(ax=ax, color=PALETTE["churn_no"], linewidth=2.2, label="Retained")
    churned.plot.kde(ax=ax, color=PALETTE["churn_yes"], linewidth=2.2, label="Churned")
    ax.set_xlabel("Monthly Charges ($)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Distribution (KDE)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(linestyle=":", alpha=0.5)

    # Box plot
    ax2 = axes[1]
    ax2.set_facecolor(PALETTE["bg"])
    data = [retained.values, churned.values]
    bp = ax2.boxplot(data, labels=["Retained", "Churned"], patch_artist=True,
                     medianprops={"color": "white", "linewidth": 2.5})
    bp["boxes"][0].set_facecolor(PALETTE["churn_no"])
    bp["boxes"][1].set_facecolor(PALETTE["churn_yes"])
    ax2.set_ylabel("Monthly Charges ($)", fontsize=12)
    ax2.set_title("Box Plot", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    _save(fig, "03_churn_vs_monthly_charges.png", save_dir)


# 4. CHURN vs TENURE

def plot_churn_by_tenure(df: pd.DataFrame, save_dir: str | None = None) -> None:
    """Histogram of tenure split by churn status."""
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    bins = range(0, 75, 3)
    retained = df[df["Churn"] == 0]["tenure"]
    churned  = df[df["Churn"] == 1]["tenure"]

    ax.hist(retained, bins=bins, color=PALETTE["churn_no"], alpha=0.65,
            label="Retained", edgecolor="white", linewidth=0.3)
    ax.hist(churned, bins=bins, color=PALETTE["churn_yes"], alpha=0.80,
            label="Churned", edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Tenure (months)", fontsize=12)
    ax.set_ylabel("Number of Customers", fontsize=12)
    ax.set_title("Tenure Distribution by Churn Status", fontsize=15,
                 fontweight="bold", color=PALETTE["text"], pad=12)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    _save(fig, "04_churn_vs_tenure.png", save_dir)


# 5. CORRELATION HEATMAP

def plot_correlation_heatmap(df: pd.DataFrame, save_dir: str | None = None) -> None:
    """Heatmap of numeric feature correlations."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    # Keep only top correlated with Churn
    if "Churn" in corr.columns:
        top_cols = corr["Churn"].abs().nlargest(15).index.tolist()
        corr = corr.loc[top_cols, top_cols]

    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor(PALETTE["bg"])

    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        center=0, square=True, linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8}, ax=ax,
        annot_kws={"size": 9}
    )
    ax.set_title("Feature Correlation Heatmap (Top 15 × Churn)",
                 fontsize=14, fontweight="bold", color=PALETTE["text"], pad=14)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    _save(fig, "05_correlation_heatmap.png", save_dir)


# 6. CHURN vs INTERNET SERVICE

def plot_churn_by_internet_service(df: pd.DataFrame, save_dir: str | None = None) -> None:
    col = "InternetService"
    if col not in df.columns:
        print(f"[WARN] Column '{col}' not found. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    ct = pd.crosstab(df[col], df["Churn"], normalize="index") * 100
    ct.columns = ["Retained %", "Churn %"]
    ct[["Retained %", "Churn %"]].plot(
        kind="barh", ax=ax,
        color=[PALETTE["churn_no"], PALETTE["churn_yes"]],
        edgecolor="white", linewidth=0.8, width=0.55
    )
    ax.set_title("Churn Rate by Internet Service Type", fontsize=14,
                 fontweight="bold", color=PALETTE["text"], pad=12)
    ax.set_xlabel("Percentage of Customers (%)", fontsize=11)
    ax.set_ylabel("")
    ax.legend(["Retained", "Churned"], fontsize=10)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    _save(fig, "06_churn_by_internet_service.png", save_dir)


# HELPER

def mtick_pct():
    import matplotlib.ticker as mtick
    return mtick.PercentFormatter()


# RUN ALL EDA PLOTS

def run_all_eda(df: pd.DataFrame, save_dir: str | None = None) -> None:
    """Generate all EDA visualizations."""
    print("\n[EDA] Generating visualizations …")
    plot_churn_distribution(df, save_dir)
    plot_churn_by_contract(df, save_dir)
    plot_churn_by_monthly_charges(df, save_dir)
    plot_churn_by_tenure(df, save_dir)
    plot_correlation_heatmap(df, save_dir)
    plot_churn_by_internet_service(df, save_dir)
    print("[EDA] All plots saved.")
