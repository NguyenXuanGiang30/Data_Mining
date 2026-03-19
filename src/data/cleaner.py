"""
cleaner.py - Xử lý thiếu, outlier, encoding cơ bản

Module này chịu trách nhiệm:
- Xử lý missing values
- Phát hiện và xử lý outlier
- Encoding biến phân loại
- Chuẩn hoá / scaling
- Tạo biến target nhị phân
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def create_binary_target(df, target_col="num"):
    """
    Chuyển target từ multi-class (0-4) sang binary (0/1).
    0 = Không bệnh tim, 1 = Có bệnh tim (gộp mức 1-4).
    
    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    
    Returns
    -------
    pd.DataFrame
        DataFrame với cột 'target' mới (binary).
    """
    df = df.copy()
    df["target"] = (df[target_col] > 0).astype(int)
    print(f"✅ Created binary target: 0={(df['target']==0).sum()}, 1={(df['target']==1).sum()}")
    return df


def handle_missing_values(df, strategy="drop_high_missing", threshold=0.4):
    """
    Xử lý missing values.
    
    Parameters
    ----------
    df : pd.DataFrame
    strategy : str
        - 'drop_high_missing': drop cột missing > threshold, fill còn lại
        - 'fill_all': fill tất cả (median cho số, mode cho phân loại)
        - 'drop_rows': drop tất cả dòng có missing
    threshold : float
        Ngưỡng % missing để drop cột (dùng cho strategy 'drop_high_missing')
    
    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    missing_pct = df.isnull().sum() / len(df)
    
    print(f"📊 Missing values before cleaning:")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            print(f"   {col}: {df[col].isnull().sum()} ({missing_pct[col]*100:.1f}%)")
    
    if strategy == "drop_high_missing":
        # Drop cột missing quá nhiều
        high_missing_cols = missing_pct[missing_pct > threshold].index.tolist()
        if high_missing_cols:
            print(f"🗑️  Dropping columns with >{threshold*100}% missing: {high_missing_cols}")
            df = df.drop(columns=high_missing_cols)
        
        # Fill remaining missing
        df = _fill_missing(df)
    
    elif strategy == "fill_all":
        df = _fill_missing(df)
    
    elif strategy == "drop_rows":
        before = len(df)
        df = df.dropna()
        print(f"🗑️  Dropped {before - len(df)} rows with missing values")
    
    print(f"✅ After cleaning: {df.isnull().sum().sum()} missing values remain")
    return df


def _fill_missing(df):
    """Fill missing: median cho số, mode cho phân loại."""
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ["float64", "int64"]:
                fill_val = df[col].median()
                df[col] = df[col].fillna(fill_val)
                print(f"   📝 {col}: filled with median = {fill_val}")
            else:
                fill_val = df[col].mode()[0]
                df[col] = df[col].fillna(fill_val)
                print(f"   📝 {col}: filled with mode = {fill_val}")
    return df


def encode_categorical(df, method="label"):
    """
    Encoding biến phân loại.
    
    Parameters
    ----------
    df : pd.DataFrame
    method : str
        - 'label': Label encoding
        - 'onehot': One-hot encoding
    
    Returns
    -------
    pd.DataFrame, dict
        DataFrame đã encode, dict mapping encode.
    """
    df = df.copy()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    
    # Loại bỏ cột 'dataset' (không dùng cho modeling)
    if "dataset" in cat_cols:
        cat_cols.remove("dataset")
    
    encoding_map = {}
    
    if method == "label":
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoding_map[col] = dict(zip(le.classes_, le.transform(le.classes_)))
            print(f"   🔢 {col}: {encoding_map[col]}")
    
    elif method == "onehot":
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        print(f"   🔢 One-hot encoded: {cat_cols}")
        print(f"   New shape: {df.shape}")
    
    print(f"✅ Encoded {len(cat_cols)} categorical columns")
    return df, encoding_map


def remove_outliers(df, numeric_cols=None, method="iqr", factor=1.5):
    """
    Phát hiện và xử lý outlier bằng IQR.
    
    Parameters
    ----------
    df : pd.DataFrame
    numeric_cols : list, optional
        Danh sách cột số cần kiểm tra. Nếu None, dùng tất cả cột số.
    method : str
        'iqr': dùng IQR
    factor : float
        Hệ số IQR (mặc định 1.5)
    
    Returns
    -------
    pd.DataFrame
        DataFrame đã xử lý outlier.
    """
    df = df.copy()
    
    # Các cột đã encode hoặc binary, không nên clip outlier
    EXCLUDE_FROM_OUTLIER = ["target", "num", "id", "sex", "cp", "fbs", "restecg",
                            "exang", "slope", "ca", "thal"]
    
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in EXCLUDE_FROM_OUTLIER]
    
    total_outliers = 0
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if outliers > 0:
            # Clip thay vì drop
            df[col] = df[col].clip(lower, upper)
            total_outliers += outliers
            print(f"   📌 {col}: {outliers} outliers clipped to [{lower:.1f}, {upper:.1f}]")
    
    print(f"✅ Total outliers handled: {total_outliers}")
    return df


def scale_features(df, cols=None, method="standard"):
    """
    Chuẩn hoá features.
    
    Parameters
    ----------
    df : pd.DataFrame
    cols : list, optional
    method : str
        'standard': StandardScaler (mean=0, std=1)
    
    Returns
    -------
    pd.DataFrame, scaler
    """
    df = df.copy()
    
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()
        cols = [c for c in cols if c not in ["target", "num", "id"]]
    
    if method == "standard":
        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols])
        print(f"✅ StandardScaler applied to {len(cols)} columns")
    
    return df, scaler


def full_cleaning_pipeline(df, drop_cols=None, missing_strategy="drop_high_missing",
                           missing_threshold=0.4, encode_method="label",
                           handle_outliers=True, scale=False):
    """
    Pipeline tiền xử lý đầy đủ.
    
    Parameters
    ----------
    df : pd.DataFrame
    drop_cols : list
        Cột cần loại bỏ (vd: ['id', 'dataset'])
    missing_strategy : str
    missing_threshold : float
    encode_method : str
    handle_outliers : bool
    scale : bool
    
    Returns
    -------
    pd.DataFrame
    """
    print("=" * 60)
    print("🔧 FULL CLEANING PIPELINE")
    print("=" * 60)
    
    # 1. Drop cột không cần
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        print(f"\n--- Step 1: Dropped columns {drop_cols}")
    
    # 2. Tạo binary target
    if "num" in df.columns:
        print(f"\n--- Step 2: Create binary target")
        df = create_binary_target(df)
        df = df.drop(columns=["num"])
    
    # 3. Xử lý missing
    print(f"\n--- Step 3: Handle missing values ({missing_strategy})")
    df = handle_missing_values(df, strategy=missing_strategy, threshold=missing_threshold)
    
    # 4. Encode categorical
    print(f"\n--- Step 4: Encode categorical ({encode_method})")
    df, encoding_map = encode_categorical(df, method=encode_method)
    
    # 5. Handle outliers
    if handle_outliers:
        print(f"\n--- Step 5: Handle outliers (IQR)")
        df = remove_outliers(df)
    
    # 6. Scale
    scaler = None
    if scale:
        print(f"\n--- Step 6: Scale features")
        df, scaler = scale_features(df)
    
    print(f"\n{'=' * 60}")
    print(f"✅ Pipeline complete! Final shape: {df.shape}")
    print(f"{'=' * 60}")
    
    return df
