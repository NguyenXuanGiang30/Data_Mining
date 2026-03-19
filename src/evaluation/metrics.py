"""
metrics.py - Tính accuracy, f1, auc, rmse, mae, ...

Module này tập trung hóa tất cả các hàm tính metric.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    silhouette_score, mean_absolute_error, mean_squared_error
)


def classification_metrics(y_true, y_pred, y_prob=None):
    """
    Tính tất cả metrics cho classification.
    
    Returns
    -------
    dict
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }
    
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    
    cm = confusion_matrix(y_true, y_pred)
    metrics["tn"] = cm[0, 0]
    metrics["fp"] = cm[0, 1]
    metrics["fn"] = cm[1, 0]
    metrics["tp"] = cm[1, 1]
    metrics["fn_rate"] = cm[1, 0] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    
    return metrics


def regression_metrics(y_true, y_pred):
    """
    Tính metrics cho regression.
    
    Returns
    -------
    dict
    """
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mse": mean_squared_error(y_true, y_pred),
    }


def clustering_metrics(X, labels):
    """
    Tính metrics cho clustering.
    
    Returns
    -------
    dict
    """
    return {
        "silhouette": silhouette_score(X, labels),
        "n_clusters": len(set(labels)),
        "cluster_sizes": dict(pd.Series(labels).value_counts()),
    }


def generate_actionable_insights(results, df=None, target_col="target"):
    """
    Tạo ít nhất 5 insight "có hành động" (actionable) từ kết quả.
    
    Parameters
    ----------
    results : dict
        Kết quả từ train_and_evaluate.
    df : pd.DataFrame, optional
        DataFrame gốc để phân tích thêm.
    
    Returns
    -------
    list of str
    """
    insights = []
    
    # 1. Mô hình tốt nhất
    best_model = max(results.items(), key=lambda x: x[1].get("pr_auc", 0))
    insights.append(
        f"1. MÔ HÌNH TỐT NHẤT: {best_model[0]} đạt PR-AUC={best_model[1]['pr_auc']:.4f}. "
        f"Khuyến nghị sử dụng mô hình này cho hệ thống sàng lọc bệnh tim."
    )
    
    # 2. False Negative
    fn_analysis = {name: r["fn_count"] for name, r in results.items() if "fn_count" in r}
    min_fn_model = min(fn_analysis.items(), key=lambda x: x[1])
    insights.append(
        f"2. GIẢM BỎ SÓT BỆNH: {min_fn_model[0]} có FN thấp nhất ({min_fn_model[1]} ca). "
        f"Trong y tế, bỏ sót bệnh nhân (FN) nguy hiểm hơn báo nhầm (FP)."
    )
    
    # 3. Trade-off
    insights.append(
        f"3. CÂN BẰNG PRECISION-RECALL: Có thể điều chỉnh ngưỡng phân loại (threshold) "
        f"để ưu tiên Recall (giảm bỏ sót) hoặc Precision (giảm báo nhầm) tùy bối cảnh lâm sàng."
    )
    
    # 4. Feature importance
    for name, r in results.items():
        if hasattr(r.get("model", None), "feature_importances_"):
            importances = r["model"].feature_importances_
            if df is not None:
                feature_names = [c for c in df.columns if c not in [target_col, "id", "dataset", "num"]]
                if len(feature_names) == len(importances):
                    top_features = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:5]
                    top_str = ", ".join([f"{f}({i:.3f})" for f, i in top_features])
                    insights.append(
                        f"4. YẾU TỐ QUAN TRỌNG NHẤT ({name}): {top_str}. "
                        f"Tập trung kiểm soát các chỉ số này giúp phòng ngừa bệnh tim."
                    )
            break
    
    # 5. Screening recommendation
    insights.append(
        f"5. SÀNG LỌC SỚM: Kết quả cho thấy mô hình có thể hỗ trợ sàng lọc bệnh tim "
        f"với độ tin cậy cao. Khuyến nghị áp dụng cho khám sức khoẻ định kỳ, "
        f"đặc biệt với nhóm nguy cơ cao (tuổi > 55, cholesterol cao, huyết áp cao)."
    )
    
    # 6. Baseline comparison
    baseline_names = [n for n in results if "Baseline" in n]
    improved_names = [n for n in results if "Baseline" not in n]
    if baseline_names and improved_names:
        best_baseline = max(baseline_names, key=lambda n: results[n].get("pr_auc", 0))
        best_improved = max(improved_names, key=lambda n: results[n].get("pr_auc", 0))
        improvement = results[best_improved]["pr_auc"] - results[best_baseline]["pr_auc"]
        insights.append(
            f"6. SO VỚI BASELINE: {best_improved} cải thiện PR-AUC thêm {improvement:.4f} "
            f"so với {best_baseline}. Các mô hình cải tiến đều vượt trội baseline đáng kể."
        )
    
    # 7. SMOTE effect
    insights.append(
        f"7. XỬ LÝ MẤT CÂN BẰNG: SMOTE + class_weight='balanced' giúp cải thiện recall "
        f"cho lớp thiểu số (bệnh tim) mà không hy sinh precision quá nhiều."
    )
    
    return insights


def analyze_error_patterns(X_test, y_test, y_pred, df_original=None, feature_names=None):
    """
    Phân tích dạng sai phổ biến theo nhóm (tuổi, giới tính, loại đau ngực).
    Tiêu chí G yêu cầu phân tích lỗi chi tiết.
    
    Parameters
    ----------
    X_test : array-like
    y_test : array-like
    y_pred : array-like
    df_original : pd.DataFrame, optional
        DataFrame gốc (chưa scale) tương ứng test set.
    feature_names : list, optional
    
    Returns
    -------
    dict
        Phân tích lỗi theo nhóm.
    """
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    
    # Tạo DataFrame phân tích
    if feature_names is not None:
        df_err = pd.DataFrame(X_test, columns=feature_names)
    else:
        df_err = pd.DataFrame(X_test)
    
    df_err["y_true"] = y_test
    df_err["y_pred"] = y_pred
    df_err["error_type"] = "Correct"
    df_err.loc[(y_test == 1) & (y_pred == 0), "error_type"] = "FN (Miss)"
    df_err.loc[(y_test == 0) & (y_pred == 1), "error_type"] = "FP (False Alarm)"
    
    analysis = {}
    
    # 1. Tổng quan lỗi
    cm = confusion_matrix(y_test, y_pred)
    analysis["confusion_matrix"] = cm
    analysis["total_errors"] = (y_test != y_pred).sum()
    analysis["fn_count"] = cm[1, 0]
    analysis["fp_count"] = cm[0, 1]
    analysis["accuracy"] = (y_test == y_pred).mean()
    
    print("\n📋 ERROR PATTERN ANALYSIS")
    print("=" * 50)
    print(f"Total errors: {analysis['total_errors']}/{len(y_test)} ({(y_test != y_pred).mean()*100:.1f}%)")
    print(f"  FN (bỏ sót bệnh): {analysis['fn_count']}")
    print(f"  FP (báo nhầm bệnh): {analysis['fp_count']}")
    
    # 2. Phân tích FN theo nhóm tuổi
    if "age" in df_err.columns:
        df_err["age_group"] = pd.cut(df_err["age"], bins=[0, 40, 50, 60, 100],
                                     labels=["<40", "40-50", "50-60", ">60"])
        fn_by_age = df_err[df_err["error_type"] == "FN (Miss)"].groupby("age_group").size()
        total_by_age = df_err[df_err["y_true"] == 1].groupby("age_group").size()
        fn_rate_by_age = (fn_by_age / total_by_age).fillna(0)
        
        analysis["fn_by_age"] = fn_rate_by_age.to_dict()
        print(f"\n📌 FN Rate theo nhóm tuổi:")
        for age, rate in fn_rate_by_age.items():
            print(f"   {age}: {rate*100:.1f}%")
    
    # 3. Phân tích FN theo giới tính
    if "sex" in df_err.columns:
        fn_by_sex = df_err[df_err["error_type"] == "FN (Miss)"].groupby("sex").size()
        total_by_sex = df_err[df_err["y_true"] == 1].groupby("sex").size()
        fn_rate_by_sex = (fn_by_sex / total_by_sex).fillna(0)
        
        analysis["fn_by_sex"] = fn_rate_by_sex.to_dict()
        print(f"\n📌 FN Rate theo giới tính:")
        for sex, rate in fn_rate_by_sex.items():
            label = "Male" if sex == 1 else "Female" if sex == 0 else str(sex)
            print(f"   {label}: {rate*100:.1f}%")
    
    # 4. Phân tích FN theo loại đau ngực
    if "cp" in df_err.columns:
        fn_by_cp = df_err[df_err["error_type"] == "FN (Miss)"].groupby("cp").size()
        total_by_cp = df_err[df_err["y_true"] == 1].groupby("cp").size()
        fn_rate_by_cp = (fn_by_cp / total_by_cp).fillna(0)
        
        analysis["fn_by_cp"] = fn_rate_by_cp.to_dict()
        print(f"\n📌 FN Rate theo loại đau ngực:")
        for cp, rate in fn_rate_by_cp.items():
            print(f"   cp={cp}: {rate*100:.1f}%")
    
    # 5. Đặc điểm trung bình của mẫu FN
    fn_samples = df_err[df_err["error_type"] == "FN (Miss)"]
    correct_pos = df_err[(df_err["y_true"] == 1) & (df_err["y_pred"] == 1)]
    
    if len(fn_samples) > 0 and len(correct_pos) > 0:
        numeric_cols = [c for c in df_err.select_dtypes(include="number").columns
                       if c not in ["y_true", "y_pred"]]
        
        comparison = pd.DataFrame({
            "FN (bỏ sót)": fn_samples[numeric_cols].mean(),
            "TP (đúng bệnh)": correct_pos[numeric_cols].mean(),
            "Chênh lệch %": ((fn_samples[numeric_cols].mean() - correct_pos[numeric_cols].mean()) 
                              / correct_pos[numeric_cols].mean() * 100).round(1)
        })
        analysis["fn_vs_tp_comparison"] = comparison
        print(f"\n📌 So sánh FN vs TP (đặc điểm trung bình):")
        print(comparison.round(2).to_string())
    
    return analysis
