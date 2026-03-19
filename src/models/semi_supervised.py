"""
semi_supervised.py - Self-training / Label Spreading

Module này chịu trách nhiệm:
- Giả lập kịch bản thiếu nhãn (10-30% labeled)
- Self-training classifier
- Label Spreading
- So sánh supervised vs semi-supervised
- Learning curve theo % nhãn
"""

import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, average_precision_score, classification_report
from sklearn.model_selection import train_test_split


def create_partial_labels(y, label_ratio=0.1, random_state=42):
    """
    Giả lập kịch bản thiếu nhãn: giữ label_ratio% nhãn, còn lại đặt -1.
    
    Parameters
    ----------
    y : array-like
        Nhãn gốc.
    label_ratio : float
        Tỷ lệ nhãn giữ lại (0.05 = 5%, 0.1 = 10%, 0.2 = 20%).
    random_state : int
    
    Returns
    -------
    y_partial : np.ndarray
        Nhãn với phần unlabeled = -1.
    labeled_mask : np.ndarray
        Boolean mask cho dữ liệu có nhãn.
    """
    rng = np.random.RandomState(random_state)
    n = len(y)
    n_labeled = max(int(n * label_ratio), 2)  # ít nhất 2 mẫu có nhãn
    
    labeled_idx = rng.choice(n, size=n_labeled, replace=False)
    labeled_mask = np.zeros(n, dtype=bool)
    labeled_mask[labeled_idx] = True
    
    y_partial = np.full(n, -1)  # -1 = unlabeled
    y_partial[labeled_mask] = np.array(y)[labeled_mask]
    
    print(f"   Labeled: {labeled_mask.sum()} ({label_ratio*100:.0f}%), "
          f"Unlabeled: {(~labeled_mask).sum()} ({(1-label_ratio)*100:.0f}%)")
    
    return y_partial, labeled_mask


def run_self_training(X_train, y_partial, X_test, y_test, random_state=42):
    """
    Chạy Self-Training Classifier.
    
    Parameters
    ----------
    X_train : array-like (scaled)
    y_partial : array-like (với -1 cho unlabeled)
    X_test : array-like
    y_test : array-like
    
    Returns
    -------
    dict
        Kết quả gồm f1, pr_auc, model, pseudo_labels.
    """
    base_model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=random_state
    )
    self_trainer = SelfTrainingClassifier(base_model, threshold=0.75)
    self_trainer.fit(X_train, y_partial)
    
    y_pred = self_trainer.predict(X_test)
    y_prob = self_trainer.predict_proba(X_test)[:, 1]
    
    f1 = f1_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_prob)
    
    # Phân tích pseudo-labels
    pseudo_labels = self_trainer.transduction_
    n_pseudo = (y_partial == -1).sum()
    
    return {
        "model": self_trainer,
        "f1": f1,
        "pr_auc": pr_auc,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "pseudo_labels": pseudo_labels,
        "n_pseudo_labeled": n_pseudo,
    }


def run_label_spreading(X_train, y_partial, X_test, y_test):
    """
    Chạy Label Spreading.
    
    Parameters
    ----------
    X_train : array-like (scaled)
    y_partial : array-like (với -1 cho unlabeled)
    X_test : array-like
    y_test : array-like
    
    Returns
    -------
    dict
    """
    model = LabelSpreading(kernel="rbf", alpha=0.2, max_iter=200)
    model.fit(X_train, y_partial)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    f1 = f1_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_prob)
    
    return {
        "model": model,
        "f1": f1,
        "pr_auc": pr_auc,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "transduction": model.transduction_,
    }


def run_supervised_only(X_train, y_train, X_test, y_test, labeled_mask, random_state=42):
    """
    Chạy supervised chỉ với dữ liệu có nhãn (baseline so sánh).
    
    Parameters
    ----------
    X_train : array-like
    y_train : array-like (nhãn gốc, đầy đủ)
    X_test : array-like
    y_test : array-like
    labeled_mask : array-like (boolean)
    
    Returns
    -------
    dict
    """
    X_labeled = X_train[labeled_mask]
    y_labeled = np.array(y_train)[labeled_mask]
    
    model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=random_state
    )
    model.fit(X_labeled, y_labeled)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    f1 = f1_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_prob)
    
    return {
        "model": model,
        "f1": f1,
        "pr_auc": pr_auc,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def learning_curve_by_label_ratio(X_train, y_train, X_test, y_test,
                                   label_ratios=None, random_state=42):
    """
    Learning curve: so sánh supervised vs semi-supervised theo % nhãn.
    
    Parameters
    ----------
    X_train, y_train, X_test, y_test : array-like
    label_ratios : list of float
    random_state : int
    
    Returns
    -------
    pd.DataFrame
        Bảng kết quả với columns: label_ratio, supervised_f1, self_training_f1,
        label_spreading_f1, supervised_pr_auc, self_training_pr_auc, label_spreading_pr_auc
    """
    if label_ratios is None:
        label_ratios = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    
    print("=" * 60)
    print("📈 LEARNING CURVE BY LABEL RATIO")
    print("=" * 60)
    
    results = []
    
    for ratio in label_ratios:
        print(f"\n--- Label ratio: {ratio*100:.0f}% ---")
        
        # Tạo partial labels
        y_partial, labeled_mask = create_partial_labels(y_train, ratio, random_state)
        
        # 1. Supervised only
        sup = run_supervised_only(X_train, y_train, X_test, y_test, labeled_mask, random_state)
        print(f"   Supervised-only: F1={sup['f1']:.4f}, PR-AUC={sup['pr_auc']:.4f}")
        
        # 2. Self-training
        st = run_self_training(X_train, y_partial, X_test, y_test, random_state)
        print(f"   Self-training:   F1={st['f1']:.4f}, PR-AUC={st['pr_auc']:.4f}")
        
        # 3. Label Spreading
        ls = run_label_spreading(X_train, y_partial, X_test, y_test)
        print(f"   Label Spreading: F1={ls['f1']:.4f}, PR-AUC={ls['pr_auc']:.4f}")
        
        results.append({
            "label_ratio": ratio,
            "supervised_f1": sup["f1"],
            "self_training_f1": st["f1"],
            "label_spreading_f1": ls["f1"],
            "supervised_pr_auc": sup["pr_auc"],
            "self_training_pr_auc": st["pr_auc"],
            "label_spreading_pr_auc": ls["pr_auc"],
        })
    
    results_df = pd.DataFrame(results)
    
    print(f"\n{'=' * 60}")
    print("📊 LEARNING CURVE RESULTS")
    print(results_df.round(4).to_string(index=False))
    print(f"{'=' * 60}")
    
    return results_df


def analyze_pseudo_labels(y_true, pseudo_labels, labeled_mask):
    """
    Phân tích chất lượng pseudo-labels.
    
    Parameters
    ----------
    y_true : array-like (nhãn thật)
    pseudo_labels : array-like (nhãn do semi-supervised gán)
    labeled_mask : array-like (boolean)
    
    Returns
    -------
    dict
    """
    unlabeled_mask = ~labeled_mask
    y_true_unlabeled = np.array(y_true)[unlabeled_mask]
    pseudo_unlabeled = pseudo_labels[unlabeled_mask]
    
    correct = (y_true_unlabeled == pseudo_unlabeled).sum()
    total = unlabeled_mask.sum()
    accuracy = correct / total if total > 0 else 0
    
    # Phân tích sai theo lớp
    false_positive = ((y_true_unlabeled == 0) & (pseudo_unlabeled == 1)).sum()
    false_negative = ((y_true_unlabeled == 1) & (pseudo_unlabeled == 0)).sum()
    
    analysis = {
        "total_unlabeled": total,
        "correct_pseudo": correct,
        "pseudo_accuracy": accuracy,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }
    
    print(f"\n📋 PSEUDO-LABEL ANALYSIS:")
    print(f"   Total unlabeled: {total}")
    print(f"   Correct pseudo-labels: {correct} ({accuracy*100:.1f}%)")
    print(f"   False Positive (gán nhầm bệnh): {false_positive}")
    print(f"   False Negative (bỏ sót bệnh): {false_negative}")
    
    return analysis
