"""
plots.py - Hàm vẽ biểu đồ dùng chung

Module này cung cấp các hàm vẽ biểu đồ cho:
- EDA (phân phối, tương quan, missing)
- Clustering (scatter, silhouette)
- Classification (confusion matrix, ROC, PR curve)
- Comparison (bảng so sánh mô hình)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os


# Style mặc định
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
FIGSIZE = (10, 6)
SAVE_DPI = 150


def save_fig(fig, name, output_dir="outputs/figures"):
    """Lưu figure vào thư mục outputs."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"💾 Saved: {path}")


# =============================================================
# EDA PLOTS
# =============================================================

def plot_missing_values(df, save=True):
    """Biểu đồ missing values."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        print("✅ No missing values!")
        return
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    missing_pct = (missing / len(df) * 100)
    bars = ax.barh(missing.index, missing_pct, color=sns.color_palette("Reds_r", len(missing)))
    ax.set_xlabel("Missing %")
    ax.set_title("Missing Values by Column")
    
    for bar, val in zip(bars, missing_pct):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center")
    
    plt.tight_layout()
    if save:
        save_fig(fig, "missing_values")
    return fig


def plot_target_distribution(df, target_col="target", save=True):
    """Biểu đồ phân phối target."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count plot
    counts = df[target_col].value_counts()
    colors = ["#2ecc71", "#e74c3c"]
    axes[0].bar(counts.index.astype(str), counts.values, color=colors)
    axes[0].set_title("Target Distribution (Count)")
    axes[0].set_xlabel("Target")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")
    
    # Pie chart
    labels = ["No Disease (0)", "Disease (1)"]
    axes[1].pie(counts.values, labels=labels, autopct="%1.1f%%",
                colors=colors, startangle=90, explode=(0, 0.05))
    axes[1].set_title("Target Distribution (%)")
    
    plt.tight_layout()
    if save:
        save_fig(fig, "target_distribution")
    return fig


def plot_numeric_distributions(df, cols=None, save=True):
    """Biểu đồ phân phối các biến số."""
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()
        cols = [c for c in cols if c not in ["id", "target", "num"]]
    
    n_cols_per_row = 3
    n_rows = (len(cols) + n_cols_per_row - 1) // n_cols_per_row
    fig, axes = plt.subplots(n_rows, n_cols_per_row, figsize=(5*n_cols_per_row, 4*n_rows))
    if n_rows == 1 and n_cols_per_row == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()
    
    colors = sns.color_palette("husl", len(cols))
    for i, col in enumerate(cols):
        if i < len(axes):
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color=colors[i])
            axes[i].set_title(f"{col}")
            axes[i].set_xlabel("")
    
    # Ẩn axes thừa
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle("Numeric Feature Distributions", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save:
        save_fig(fig, "numeric_distributions")
    return fig


def plot_correlation_matrix(df, cols=None, save=True):
    """Ma trận tương quan."""
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()
    
    corr = df[cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Matrix", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    if save:
        save_fig(fig, "correlation_matrix")
    return fig


def plot_features_vs_target(df, target_col="target", cols=None, save=True):
    """Biểu đồ so sánh features theo target."""
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()
        cols = [c for c in cols if c not in ["id", "target", "num"]]
    
    n_cols = 3
    n_rows = (len(cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(cols):
        if i < len(axes):
            sns.boxplot(x=target_col, y=col, data=df, ax=axes[i],
                       palette=["#2ecc71", "#e74c3c"])
            axes[i].set_title(f"{col} by Target")
    
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle("Features vs Target", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save:
        save_fig(fig, "features_vs_target")
    return fig


# =============================================================
# CLUSTERING PLOTS
# =============================================================

def plot_elbow_silhouette(k_range, inertias, silhouettes, save=True):
    """Biểu đồ Elbow + Silhouette Score."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(k_range, inertias, "bo-", linewidth=2)
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method")
    
    axes[1].plot(k_range, silhouettes, "ro-", linewidth=2)
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score")
    
    best_k = k_range[np.argmax(silhouettes)]
    axes[1].axvline(x=best_k, color="green", linestyle="--", label=f"Best k={best_k}")
    axes[1].legend()
    
    plt.tight_layout()
    if save:
        save_fig(fig, "elbow_silhouette")
    return fig


def plot_cluster_profiles(df, cluster_col="cluster", save=True):
    """Biểu đồ radar/bar cho profile từng cluster."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != cluster_col]
    
    cluster_means = df.groupby(cluster_col)[numeric_cols].mean()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    cluster_means.T.plot(kind="bar", ax=ax, colormap="Set2")
    ax.set_title("Cluster Profiles (Mean Values)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Features")
    ax.set_ylabel("Mean Value")
    ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    if save:
        save_fig(fig, "cluster_profiles")
    return fig


# =============================================================
# CLASSIFICATION PLOTS
# =============================================================

def plot_confusion_matrix(cm, labels=None, title="Confusion Matrix", save=True, filename="confusion_matrix"):
    """Biểu đồ Confusion Matrix."""
    if labels is None:
        labels = ["No Disease", "Disease"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels, ax=ax, linewidths=1, linecolor="white")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    if save:
        save_fig(fig, filename)
    return fig


def plot_roc_pr_curves(results_dict, save=True):
    """
    Vẽ ROC và PR curves cho nhiều mô hình.
    
    Parameters
    ----------
    results_dict : dict
        {model_name: {"fpr": ..., "tpr": ..., "roc_auc": ...,
                       "precision": ..., "recall": ..., "pr_auc": ...}}
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for name, result in results_dict.items():
        if "fpr" in result:
            axes[0].plot(result["fpr"], result["tpr"],
                        label=f"{name} (AUC={result['roc_auc']:.3f})", linewidth=2)
        if "precision" in result:
            axes[1].plot(result["recall"], result["precision"],
                        label=f"{name} (PR-AUC={result['pr_auc']:.3f})", linewidth=2)
    
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves")
    axes[0].legend()
    
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves")
    axes[1].legend()
    
    plt.tight_layout()
    if save:
        save_fig(fig, "roc_pr_curves")
    return fig


def plot_model_comparison(results_df, metric_cols=None, save=True):
    """
    Bảng so sánh các mô hình.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame với index=model names, columns=metrics.
    """
    if metric_cols is None:
        metric_cols = results_df.columns.tolist()
    
    fig, ax = plt.subplots(figsize=(12, max(4, len(results_df) * 0.8)))
    
    results_df[metric_cols].plot(kind="barh", ax=ax, colormap="Set2")
    ax.set_xlabel("Score")
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1))
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    if save:
        save_fig(fig, "model_comparison")
    return fig
