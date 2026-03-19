"""
run_pipeline.py - Chạy toàn bộ pipeline end-to-end

Usage:
    python scripts/run_pipeline.py
"""

import sys
import os

# Thêm project root vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.data.loader import load_raw_data, load_params, check_schema, print_data_dictionary
from src.data.cleaner import full_cleaning_pipeline
from src.features.builder import discretize_for_apriori, select_features_for_modeling
from src.mining.association import run_apriori, format_rules_table, interpret_rules
from src.mining.clustering import find_optimal_k, run_kmeans, run_hierarchical, profile_clusters, get_cluster_insights
from src.models.supervised import train_and_evaluate, cross_validate_models, save_model
from src.models.semi_supervised import learning_curve_by_label_ratio, analyze_pseudo_labels, create_partial_labels, run_self_training
from src.evaluation.metrics import classification_metrics, generate_actionable_insights, analyze_error_patterns
from src.evaluation.report import save_results_table, create_summary_report
from src.visualization.plots import (
    plot_missing_values, plot_target_distribution, plot_numeric_distributions,
    plot_correlation_matrix, plot_features_vs_target,
    plot_elbow_silhouette, plot_cluster_profiles,
    plot_confusion_matrix, plot_roc_pr_curves, plot_model_comparison
)
from sklearn.preprocessing import StandardScaler


def main():
    params = load_params()
    seed = params["seed"]
    np.random.seed(seed)
    
    # =========================================================
    # BƯỚC 1: LOAD DATA & EDA
    # =========================================================
    print("\n" + "=" * 70)
    print("📌 BƯỚC 1: LOAD DATA & EDA")
    print("=" * 70)
    
    df = load_raw_data()
    print_data_dictionary()
    schema = check_schema(df)
    
    # Vẽ biểu đồ EDA
    plot_missing_values(df)
    
    # =========================================================
    # BƯỚC 2: PREPROCESSING & FEATURE ENGINEERING
    # =========================================================
    print("\n" + "=" * 70)
    print("📌 BƯỚC 2: PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 70)
    
    df_clean = full_cleaning_pipeline(
        df,
        drop_cols=params["preprocessing"]["drop_cols"],
        missing_strategy=params["preprocessing"]["missing_strategy"],
        missing_threshold=params["preprocessing"]["missing_threshold"],
        encode_method=params["preprocessing"]["encode_method"],
        handle_outliers=params["preprocessing"]["handle_outliers"],
    )
    
    # Lưu processed data
    os.makedirs(os.path.dirname(params["paths"]["processed_data"]), exist_ok=True)
    df_clean.to_csv(params["paths"]["processed_data"], index=False)
    print(f"💾 Saved: {params['paths']['processed_data']}")
    
    # Biểu đồ sau tiền xử lý
    plot_target_distribution(df_clean)
    plot_numeric_distributions(df_clean)
    plot_correlation_matrix(df_clean)
    plot_features_vs_target(df_clean)
    
    # =========================================================
    # BƯỚC 3: MINING - ASSOCIATION RULES & CLUSTERING
    # =========================================================
    print("\n" + "=" * 70)
    print("📌 BƯỚC 3a: ASSOCIATION RULE MINING (Apriori)")
    print("=" * 70)
    
    # Rời rạc hoá cho Apriori
    df_disc = discretize_for_apriori(df_clean)
    
    # Chạy Apriori
    freq_items, rules, rules_heart = run_apriori(
        df_disc,
        min_support=params["apriori"]["min_support"],
        min_confidence=params["apriori"]["min_confidence"],
        min_lift=params["apriori"]["min_lift"],
    )
    
    # Lưu kết quả
    rules_table = format_rules_table(rules_heart)
    if len(rules_table) > 0:
        save_results_table(rules_table, "association_rules_heart")
    
    # Diễn giải
    interpretations = interpret_rules(rules_heart)
    for interp in interpretations:
        print(interp)
    
    print("\n" + "=" * 70)
    print("📌 BƯỚC 3b: CLUSTERING")
    print("=" * 70)
    
    # Chuẩn bị data cho clustering
    X_cluster, _ = select_features_for_modeling(df_clean)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Tìm k tối ưu
    k_results = find_optimal_k(X_scaled, range(params["clustering"]["k_range_min"],
                                                params["clustering"]["k_range_max"] + 1), seed)
    plot_elbow_silhouette(k_results["k_range"], k_results["inertias"], k_results["silhouettes"])
    
    # Chạy KMeans
    best_k = k_results["best_k"]
    kmeans_labels, _, kmeans_sil, kmeans_dbi = run_kmeans(X_scaled, best_k, seed)
    
    # Chạy Hierarchical
    hier_labels, _, hier_sil, hier_dbi = run_hierarchical(X_scaled, best_k)
    
    # Profile clusters
    profile = profile_clusters(df_clean, kmeans_labels)
    insights_cluster = get_cluster_insights(df_clean, kmeans_labels)
    
    # Vẽ cluster profiles
    df_cluster_plot = df_clean.copy()
    df_cluster_plot["cluster"] = kmeans_labels
    plot_cluster_profiles(df_cluster_plot)
    
    # =========================================================
    # BƯỚC 4: CLASSIFICATION
    # =========================================================
    print("\n" + "=" * 70)
    print("📌 BƯỚC 4: CLASSIFICATION")
    print("=" * 70)
    
    X, y = select_features_for_modeling(df_clean)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["split"]["test_size"],
        stratify=y, random_state=seed
    )
    
    results, results_df = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        use_smote=params["classification"]["use_smote"],
        random_state=seed
    )
    
    # Cross-validation
    cv_results = cross_validate_models(X, y, cv=params["classification"]["cv_folds"],
                                       random_state=seed)
    save_results_table(results_df, "classification_results")
    save_results_table(cv_results, "cross_validation_results")
    
    # Vẽ biểu đồ
    plot_roc_pr_curves(results)
    plot_model_comparison(results_df, ["F1-Score", "ROC-AUC", "PR-AUC"])
    
    # Confusion matrix cho mô hình tốt nhất
    best_name = results_df.index[0]
    plot_confusion_matrix(results[best_name]["confusion_matrix"],
                         title=f"Confusion Matrix - {best_name}",
                         filename=f"cm_{best_name.replace(' ', '_').lower()}")
    
    # Save best model
    save_model(results[best_name]["model"], results[best_name].get("scaler"),
               best_name)
    
    # =========================================================
    # BƯỚC 5: SEMI-SUPERVISED
    # =========================================================
    print("\n" + "=" * 70)
    print("📌 BƯỚC 5: SEMI-SUPERVISED LEARNING")
    print("=" * 70)
    
    scaler_ss = StandardScaler()
    X_train_scaled = scaler_ss.fit_transform(X_train)
    X_test_scaled = scaler_ss.transform(X_test)
    
    lc_results = learning_curve_by_label_ratio(
        X_train_scaled, y_train.values, X_test_scaled, y_test.values,
        label_ratios=params["semi_supervised"]["label_ratios"],
        random_state=seed
    )
    save_results_table(lc_results, "semi_supervised_learning_curve")
    
    # =========================================================
    # BƯỚC 6: EVALUATION & REPORT
    # =========================================================
    print("\n" + "=" * 70)
    print("📌 BƯỚC 6: EVALUATION & FINAL REPORT")
    print("=" * 70)
    
    # Error pattern analysis (Tiêu chí G)
    best_name = results_df.index[0]
    best_result = results[best_name]
    error_analysis = analyze_error_patterns(
        X_test, y_test, best_result["y_pred"],
        feature_names=X.columns.tolist()
    )
    
    # Actionable insights (>= 7 insights)
    insights = generate_actionable_insights(results, df_clean)
    print("\n📋 ACTIONABLE INSIGHTS:")
    for insight in insights:
        print(f"   {insight}")
    
    # Tổng hợp báo cáo
    create_summary_report(
        eda_stats={"n_rows": df.shape[0], "n_cols": df.shape[1],
                   "total_missing": df.isnull().sum().sum()},
        association_rules_count=len(rules_heart),
        clustering_results={"best_k": best_k, "silhouette": kmeans_sil},
        classification_results=results_df,
        semi_supervised_results=lc_results,
    )
    
    print("\n" + "=" * 70)
    print("🎉 PIPELINE HOÀN TẤT!")
    print(f"   Kết quả lưu tại: {params['paths']['outputs']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
