"""
loader.py - Đọc dữ liệu và kiểm tra schema

Module này chịu trách nhiệm:
- Load dữ liệu từ file CSV
- Kiểm tra schema (tên cột, kiểu dữ liệu)
- Hiển thị thông tin tổng quan về dataset
"""

import pandas as pd
import yaml
import os


def get_project_root():
    """Trỏ về thư mục gốc của project (Data_Mining)."""
    from pathlib import Path
    return Path(__file__).resolve().parents[2]


def load_params(config_path="configs/params.yaml"):
    """Đọc tham số cấu hình từ file YAML."""
    full_path = get_project_root() / config_path
    with open(full_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params

def load_raw_data(file_path=None):
    """
    Đọc dữ liệu thô từ file CSV.
    
    Parameters
    ----------
    file_path : str, optional
        Đường dẫn tới file CSV. Nếu None, lấy từ params.yaml.
    
    Returns
    -------
    pd.DataFrame
        DataFrame chứa dữ liệu thô.
    """
    if file_path is None:
        params = load_params()
        file_path = get_project_root() / params["paths"]["raw_data"]
    elif not str(file_path).startswith(str(get_project_root())) and not os.path.isabs(file_path):
        file_path = get_project_root() / file_path
    
    df = pd.read_csv(file_path)
    print(f"✅ Loaded data from: {file_path}")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def load_processed_data(file_path=None):
    """Đọc dữ liệu đã qua tiền xử lý."""
    if file_path is None:
        params = load_params()
        file_path = get_project_root() / params["paths"]["processed_data"]
    elif not str(file_path).startswith(str(get_project_root())) and not os.path.isabs(file_path):
        file_path = get_project_root() / file_path
    
    df = pd.read_csv(file_path)
    print(f"✅ Loaded processed data from: {file_path}")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def check_schema(df):
    """
    Kiểm tra schema của dataset Heart Disease.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame cần kiểm tra.
    
    Returns
    -------
    dict
        Thông tin schema.
    """
    expected_columns = [
        'id', 'age', 'sex', 'dataset', 'cp', 'trestbps', 'chol',
        'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope',
        'ca', 'thal', 'num'
    ]
    
    schema_info = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_count": df.isnull().sum().to_dict(),
        "missing_percent": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
    }
    
    # Kiểm tra cột có đúng không
    missing_cols = set(expected_columns) - set(df.columns)
    extra_cols = set(df.columns) - set(expected_columns)
    
    if missing_cols:
        print(f"⚠️  Missing columns: {missing_cols}")
    if extra_cols:
        print(f"⚠️  Extra columns: {extra_cols}")
    if not missing_cols and not extra_cols:
        print("✅ Schema check passed - all expected columns present")
    
    return schema_info


def get_data_summary(df):
    """
    Tạo bảng tổng hợp thông tin dữ liệu.
    
    Returns
    -------
    pd.DataFrame
        Bảng summary gồm: dtype, missing, unique, min, max, mean.
    """
    summary = pd.DataFrame({
        "dtype": df.dtypes,
        "missing": df.isnull().sum(),
        "missing_%": (df.isnull().sum() / len(df) * 100).round(2),
        "unique": df.nunique(),
        "sample": df.iloc[0],
    })
    
    # Thêm thống kê cho cột số
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        summary.loc[col, "min"] = df[col].min()
        summary.loc[col, "max"] = df[col].max()
        summary.loc[col, "mean"] = df[col].mean().round(2)
    
    return summary


# Data dictionary cho Heart Disease UCI dataset
DATA_DICTIONARY = {
    "id": "ID bệnh nhân",
    "age": "Tuổi (năm)",
    "sex": "Giới tính (Male/Female)",
    "dataset": "Nguồn dữ liệu (Cleveland, Hungary, Switzerland, VA Long Beach)",
    "cp": "Loại đau ngực (typical angina, atypical angina, non-anginal, asymptomatic)",
    "trestbps": "Huyết áp lúc nghỉ (mm Hg)",
    "chol": "Cholesterol huyết thanh (mg/dl)",
    "fbs": "Đường huyết lúc đói > 120 mg/dl (True/False)",
    "restecg": "Kết quả điện tâm đồ lúc nghỉ (normal, st-t abnormality, lv hypertrophy)",
    "thalch": "Nhịp tim tối đa đạt được (bpm)",
    "exang": "Đau thắt ngực khi vận động (True/False)",
    "oldpeak": "ST depression do vận động so với nghỉ",
    "slope": "Độ dốc đoạn ST đỉnh (upsloping, flat, downsloping)",
    "ca": "Số mạch máu chính được nhuộm bằng fluoroscopy (0-3)",
    "thal": "Thalassemia (normal, fixed defect, reversable defect)",
    "num": "Chẩn đoán bệnh tim (0: không bệnh, 1-4: mức độ bệnh)",
}


def print_data_dictionary():
    """In data dictionary."""
    print("=" * 70)
    print("DATA DICTIONARY - Heart Disease UCI Dataset")
    print("=" * 70)
    for col, desc in DATA_DICTIONARY.items():
        print(f"  {col:12s} : {desc}")
    print("=" * 70)
