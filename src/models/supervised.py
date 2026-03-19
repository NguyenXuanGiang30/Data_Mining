"""
supervised.py - Train/predict cho Classification

Module này chịu trách nhiệm:
- Train SVM, Random Forest, XGBoost
- Xử lý mất cân bằng (class_weight, SMOTE)
- Cross-validation
- So sánh mô hình
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import os


def get_models(random_state=42):
    """
    Khởi tạo các mô hình phân lớp.
    Gồm 2 baseline (Dummy, Logistic Regression) + 4 mô hình cải tiến.
    
    Returns
    -------
    dict
        {name: model}
    """
    models = {
        # === BASELINE ===
        "Baseline (Dummy)": DummyClassifier(
            strategy="most_frequent", random_state=random_state
        ),
        "Baseline (Logistic)": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state
        ),
        # === IMPROVED MODELS ===
        "SVM (linear)": SVC(
            kernel="linear", probability=True,
            class_weight="balanced", random_state=random_state
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", probability=True,
            class_weight="balanced", random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight="balanced", random_state=random_state
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            scale_pos_weight=1, random_state=random_state,
            eval_metric="logloss", use_label_encoder=False
        ),
    }
    return models


def apply_smote(X_train, y_train, random_state=42):
    """
    Áp dụng SMOTE để cân bằng lớp.
    
    Returns
    -------
    X_resampled, y_resampled
    """
    print(f"   Before SMOTE: {dict(pd.Series(y_train).value_counts())}")
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"   After SMOTE:  {dict(pd.Series(y_res).value_counts())}")
    return X_res, y_res


def train_and_evaluate(X_train, X_test, y_train, y_test,
                       models=None, use_smote=False, random_state=42):
    """
    Train và đánh giá nhiều mô hình.
    
    Parameters
    ----------
    X_train, X_test : pd.DataFrame
    y_train, y_test : pd.Series
    models : dict, optional
    use_smote : bool
    random_state : int
    
    Returns
    -------
    results : dict
        {name: {metrics, confusion_matrix, model, curves...}}
    results_df : pd.DataFrame
        Bảng so sánh metrics.
    """
    if models is None:
        models = get_models(random_state)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SMOTE (nếu dùng)
    if use_smote:
        print("\n--- Applying SMOTE ---")
        X_train_scaled, y_train = apply_smote(X_train_scaled, y_train, random_state)
    
    results = {}
    
    print("\n" + "=" * 60)
    print("🏋️ TRAINING & EVALUATION")
    print("=" * 60)
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        
        # Curves
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        
        # Report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results[name] = {
            "model": model,
            "scaler": scaler,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": cm,
            "report": report,
            "fpr": fpr,
            "tpr": tpr,
            "precision": precision,
            "recall": recall,
            # False Negative analysis
            "fn_count": cm[1, 0],  # Actual=1, Predicted=0
            "fn_rate": cm[1, 0] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0,
        }
        
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   ROC-AUC:  {roc_auc:.4f}")
        print(f"   PR-AUC:   {pr_auc:.4f}")
        print(f"   FN Count: {cm[1, 0]} (Miss Rate: {results[name]['fn_rate']:.2%})")
        print(f"   Confusion Matrix:\n{cm}")
    
    # Bảng so sánh
    results_df = pd.DataFrame({
        name: {
            "F1-Score": r["f1"],
            "ROC-AUC": r["roc_auc"],
            "PR-AUC": r["pr_auc"],
            "FN Count": r["fn_count"],
            "FN Rate": r["fn_rate"],
        }
        for name, r in results.items()
    }).T
    
    results_df = results_df.sort_values("PR-AUC", ascending=False)
    
    print(f"\n{'=' * 60}")
    print("📊 MODEL COMPARISON")
    print(results_df.round(4).to_string())
    print(f"{'=' * 60}")
    
    return results, results_df


def cross_validate_models(X, y, models=None, cv=5, random_state=42):
    """
    Cross-validation cho nhiều mô hình.
    
    Returns
    -------
    cv_results : pd.DataFrame
    """
    if models is None:
        models = get_models(random_state)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    cv_results = {}
    print(f"\n📊 {cv}-Fold Cross Validation")
    print("-" * 50)
    
    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=skf, scoring="f1")
        cv_results[name] = {
            "mean_f1": scores.mean(),
            "std_f1": scores.std(),
            "scores": scores,
        }
        print(f"   {name}: F1 = {scores.mean():.4f} ± {scores.std():.4f}")
    
    return pd.DataFrame({
        name: {"Mean F1": r["mean_f1"], "Std F1": r["std_f1"]}
        for name, r in cv_results.items()
    }).T


def save_model(model, scaler, name, output_dir="outputs/models"):
    """Lưu model và scaler."""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{name.replace(' ', '_').lower()}.joblib")
    joblib.dump({"model": model, "scaler": scaler}, model_path)
    print(f"💾 Saved model: {model_path}")
    return model_path
