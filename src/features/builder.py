"""
builder.py - Feature engineering

Module này chịu trách nhiệm:
- Rời rạc hoá features cho Apriori
- Tạo đặc trưng mới
- Chọn features quan trọng
"""

import pandas as pd
import numpy as np


def discretize_for_apriori(df):
    """
    Rời rạc hoá các biến liên tục cho thuật toán Apriori.
    Chuyển tất cả features thành dạng categorical (True/False hoặc nhóm).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đã qua tiền xử lý (chưa scale).
    
    Returns
    -------
    pd.DataFrame
        DataFrame với tất cả cột là binary/categorical.
    """
    df_disc = pd.DataFrame()
    
    # Age groups
    if "age" in df.columns:
        df_disc["age_young"] = (df["age"] < 40).astype(int)
        df_disc["age_middle"] = ((df["age"] >= 40) & (df["age"] < 55)).astype(int)
        df_disc["age_senior"] = ((df["age"] >= 55) & (df["age"] < 65)).astype(int)
        df_disc["age_elderly"] = (df["age"] >= 65).astype(int)
    
    # Sex
    if "sex" in df.columns:
        if df["sex"].dtype == "object":
            df_disc["is_male"] = (df["sex"] == "Male").astype(int)
        else:
            df_disc["is_male"] = df["sex"].astype(int)
    
    # Chest pain type
    if "cp" in df.columns:
        if df["cp"].dtype == "object":
            for val in df["cp"].unique():
                col_name = f"cp_{val}".replace(" ", "_").lower()
                df_disc[col_name] = (df["cp"] == val).astype(int)
        else:
            df_disc["cp_asymptomatic"] = (df["cp"] == 0).astype(int)
    
    # Blood pressure
    if "trestbps" in df.columns:
        df_disc["bp_normal"] = (df["trestbps"] < 120).astype(int)
        df_disc["bp_elevated"] = ((df["trestbps"] >= 120) & (df["trestbps"] < 140)).astype(int)
        df_disc["bp_high"] = (df["trestbps"] >= 140).astype(int)
    
    # Cholesterol
    if "chol" in df.columns:
        df_disc["chol_normal"] = (df["chol"] < 200).astype(int)
        df_disc["chol_borderline"] = ((df["chol"] >= 200) & (df["chol"] < 240)).astype(int)
        df_disc["chol_high"] = (df["chol"] >= 240).astype(int)
    
    # Fasting blood sugar
    if "fbs" in df.columns:
        if df["fbs"].dtype == "object":
            df_disc["fbs_high"] = (df["fbs"] == "True").astype(int)
        else:
            df_disc["fbs_high"] = df["fbs"].astype(int)
    
    # Resting ECG
    if "restecg" in df.columns:
        if df["restecg"].dtype == "object":
            df_disc["ecg_normal"] = (df["restecg"] == "normal").astype(int)
            df_disc["ecg_abnormal"] = (df["restecg"] != "normal").astype(int)
        else:
            df_disc["ecg_normal"] = (df["restecg"] == 0).astype(int)
            df_disc["ecg_abnormal"] = (df["restecg"] != 0).astype(int)
    
    # Max heart rate
    if "thalch" in df.columns:
        df_disc["hr_low"] = (df["thalch"] < 120).astype(int)
        df_disc["hr_normal"] = ((df["thalch"] >= 120) & (df["thalch"] < 160)).astype(int)
        df_disc["hr_high"] = (df["thalch"] >= 160).astype(int)
    
    # Exercise induced angina
    if "exang" in df.columns:
        if df["exang"].dtype == "object":
            df_disc["exercise_angina"] = (df["exang"] == "True").astype(int)
        else:
            df_disc["exercise_angina"] = df["exang"].astype(int)
    
    # ST depression (oldpeak)
    if "oldpeak" in df.columns:
        df_disc["oldpeak_zero"] = (df["oldpeak"] == 0).astype(int)
        df_disc["oldpeak_low"] = ((df["oldpeak"] > 0) & (df["oldpeak"] < 2)).astype(int)
        df_disc["oldpeak_high"] = (df["oldpeak"] >= 2).astype(int)
    
    # Slope
    if "slope" in df.columns:
        if df["slope"].dtype == "object":
            for val in df["slope"].dropna().unique():
                col_name = f"slope_{val}".replace(" ", "_").lower()
                df_disc[col_name] = (df["slope"] == val).astype(int)
        else:
            df_disc["slope_flat"] = (df["slope"] == 1).astype(int)
    
    # Number of major vessels (ca)
    if "ca" in df.columns:
        df_disc["ca_zero"] = (df["ca"] == 0).astype(int)
        df_disc["ca_positive"] = (df["ca"] > 0).astype(int)
    
    # Thalassemia
    if "thal" in df.columns:
        if df["thal"].dtype == "object":
            df_disc["thal_normal"] = (df["thal"] == "normal").astype(int)
            df_disc["thal_fixed_defect"] = (df["thal"] == "fixed defect").astype(int)
            df_disc["thal_reversable"] = (df["thal"] == "reversable defect").astype(int)
        else:
            df_disc["thal_normal"] = (df["thal"] == 1).astype(int)
    
    # Target
    if "target" in df.columns:
        df_disc["heart_disease"] = df["target"].astype(int)
    elif "num" in df.columns:
        df_disc["heart_disease"] = (df["num"] > 0).astype(int)
    
    print(f"✅ Discretized {len(df_disc.columns)} binary features for Apriori")
    return df_disc


def select_features_for_modeling(df, target_col="target", drop_cols=None):
    """
    Chuẩn bị X, y cho modeling.
    
    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    drop_cols : list
    
    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    if drop_cols is None:
        drop_cols = ["id", "dataset", "num"]
    
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    cols_to_drop.append(target_col)
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    X = df.drop(columns=cols_to_drop)
    y = df[target_col]
    
    print(f"✅ X shape: {X.shape}, y shape: {y.shape}")
    print(f"   Features: {list(X.columns)}")
    print(f"   Target distribution: {dict(y.value_counts())}")
    
    return X, y
