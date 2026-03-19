# 2.4.1 Cấu trúc repo mẫu

```
DATA_MINING_PROJECT/
│
├── README.md
├── requirements.txt          # hoặc environment.yml
├── .gitignore
│
├── configs/
│   └── params.yaml           # tham số: seed, split, paths, hyperparams...
│
├── data/
│   ├── raw/                  # dữ liệu gốc (không commit nếu quá lớn)
│   └── processed/            # dữ liệu sau tiền xử lý (ưu tiên parquet/csv)
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocess_feature.ipynb
│   ├── 03_mining_or_clustering.ipynb
│   ├── 04_modeling.ipynb
│   ├── 04b_semi_supervised.ipynb    # CHỈ áp dụng cho đề tài có bán giám sát
│   └── 05_evaluation_report.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py         # đọc dữ liệu, kiểm tra schema
│   │   └── cleaner.py        # xử lý thiếu, outlier, encoding cơ bản
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── builder.py        # feature engineering (RFM, TF-IDF, lag, ...)
│   │
│   ├── mining/
│   │   ├── __init__.py
│   │   ├── association.py    # (nếu có) luật kết hợp / pattern
│   │   ├── clustering.py     # KMeans/HAC/DBSCAN + profiling
│   │   └── anomaly.py        # (nếu có) outlier/anomaly
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── supervised.py     # train/predict cho classification/regression
│   │   ├── semi_supervised.py # CHỈ áp dụng cho đề tài có bán giám sát
│   │   └── forecasting.py    # (nếu có) time series
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py        # accuracy, f1, auc, rmse, mae, ...
│   │   └── report.py         # tổng hợp bảng/biểu đồ kết quả
│   │
│   └── visualization/
│       ├── __init__.py
│       └── plots.py          # hàm vẽ dùng chung
│
├── scripts/
│   ├── run_pipeline.py       # chạy toàn bộ pipeline (khuyến khích)
│   └── run_papermill.py      # (khuyến khích) chạy notebook bằng papermill
│
├── outputs/
│   ├── figures/              # biểu đồ xuất ra
│   ├── tables/               # bảng kết quả
│   └── models/               # model đã train (pickle, joblib...)
│
└── reports/
    └── final_report.pdf      # báo cáo cuối cùng
```

---

## Giải thích vai trò từng thành phần

### 📁 configs/
| File | Vai trò |
|---|---|
| `params.yaml` | Lưu tất cả tham số: random seed, tỷ lệ train/test, đường dẫn file, hyperparams |

### 📁 data/
| Folder | Vai trò |
|---|---|
| `raw/` | Dữ liệu gốc từ Kaggle/UCI, không chỉnh sửa |
| `processed/` | Dữ liệu đã qua tiền xử lý, sẵn sàng cho mô hình |

### 📁 notebooks/ (theo thứ tự chạy)
| Notebook | Vai trò |
|---|---|
| `01_eda.ipynb` | Khám phá dữ liệu, thống kê mô tả, visualization |
| `02_preprocess_feature.ipynb` | Tiền xử lý + tạo đặc trưng |
| `03_mining_or_clustering.ipynb` | Apriori (luật kết hợp) + KMeans/Hierarchical (phân cụm) |
| `04_modeling.ipynb` | Train SVM/RF/XGBoost, so sánh kết quả |
| `04b_semi_supervised.ipynb` | Self-training / Label Spreading (tùy chọn) |
| `05_evaluation_report.ipynb` | Tổng hợp kết quả, vẽ biểu đồ so sánh |

### 📁 src/ (module Python tái sử dụng)
| Module | Class/File | Vai trò |
|---|---|---|
| `data/` | `loader.py` | Đọc dữ liệu, kiểm tra schema |
| `data/` | `cleaner.py` | Xử lý thiếu, outlier, encoding |
| `features/` | `builder.py` | Feature engineering |
| `mining/` | `association.py` | Apriori, luật kết hợp |
| `mining/` | `clustering.py` | KMeans, HAC, DBSCAN + profiling |
| `mining/` | `anomaly.py` | Phát hiện outlier/anomaly |
| `models/` | `supervised.py` | Train/predict classification/regression |
| `models/` | `semi_supervised.py` | Bán giám sát (tùy chọn) |
| `evaluation/` | `metrics.py` | Tính accuracy, f1, auc, rmse, mae |
| `evaluation/` | `report.py` | Tổng hợp bảng/biểu đồ kết quả |
| `visualization/` | `plots.py` | Hàm vẽ biểu đồ dùng chung |

### 📁 scripts/
| File | Vai trò |
|---|---|
| `run_pipeline.py` | Chạy toàn bộ pipeline end-to-end |
| `run_papermill.py` | Chạy notebook tự động bằng papermill |

### 📁 outputs/
| Folder | Vai trò |
|---|---|
| `figures/` | Lưu biểu đồ (PNG, SVG) |
| `tables/` | Lưu bảng kết quả (CSV) |
| `models/` | Lưu model đã train (pickle, joblib) |

### 📁 reports/
| File | Vai trò |
|---|---|
| `final_report.pdf` | Báo cáo cuối cùng nộp bài |
