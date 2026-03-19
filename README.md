# 🫀 Dự đoán Bệnh Tim (Heart Disease Prediction)

## Mô tả đề tài

Dự án **khai phá dữ liệu** (Data Mining) dự đoán nguy cơ bệnh tim mạch, sử dụng UCI Heart Disease dataset (920 bệnh nhân từ 4 trung tâm: Cleveland, Hungary, Switzerland, VA Long Beach).

**Mục tiêu:**
- Phát hiện pattern/rule liên quan đến bệnh tim (Apriori)
- Phân nhóm bệnh nhân theo nguy cơ (KMeans, Hierarchical)
- Xây dựng mô hình dự đoán chính xác (SVM, RF, XGBoost)
- Đánh giá hiệu quả semi-supervised khi thiếu nhãn

**Tiêu chí thành công:** F1-Score > 0.80, PR-AUC > 0.85 trên tập test.

## Dataset

| Thông tin | Chi tiết |
|---|---|
| **Nguồn** | [UCI/Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) |
| **Kích thước** | 920 dòng × 16 cột |
| **Target** | `num`: 0 = không bệnh, 1-4 = mức độ bệnh → binary (0/1) |
| **Đặc trưng** | age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal |

### Data Dictionary

| Cột | Mô tả | Kiểu |
|---|---|---|
| `id` | ID bệnh nhân | int |
| `age` | Tuổi (năm) | float |
| `sex` | Giới tính (Male/Female) | object |
| `dataset` | Nguồn dữ liệu | object |
| `cp` | Loại đau ngực (4 loại) | object |
| `trestbps` | Huyết áp lúc nghỉ (mmHg) | float |
| `chol` | Cholesterol huyết thanh (mg/dl) | float |
| `fbs` | Đường huyết > 120 mg/dl | object |
| `restecg` | Điện tâm đồ lúc nghỉ | object |
| `thalch` | Nhịp tim tối đa (bpm) | float |
| `exang` | Đau thắt ngực khi vận động | object |
| `oldpeak` | ST depression | float |
| `slope` | Độ dốc đoạn ST | object |
| `ca` | Số mạch màu fluoroscopy (0-3) | float |
| `thal` | Thalassemia | object |
| `num` | Chẩn đoán bệnh tim (0-4) | int |

### Rủi ro & Thách thức

- **Mất cân bằng lớp:** ~55% bệnh vs 45% không bệnh → dùng SMOTE + class_weight
- **Missing values:** `ca` ~65%, `thal` ~53%, `slope` ~33% thiếu → fill median/mode hoặc drop
- **Data leakage tiềm ẩn:** Phải scale SAU khi split train/test, SMOTE chỉ trên train
- **Dữ liệu đa nguồn:** 4 trung tâm y tế khác nhau, có thể gây bias

## Cài đặt

```bash
# Clone repo
git clone <repo_url>
cd Data_Mining

# Cài dependencies
pip install -r requirements.txt

# (Optional) Chạy Streamlit demo
pip install streamlit
```

## Cách chạy

### Option 1: Pipeline tự động
```bash
python scripts/run_pipeline.py
```

### Option 2: Jupyter Notebooks (từng bước)
```bash
jupyter notebook notebooks/
```
Chạy theo thứ tự: `01_eda` → `02_preprocess` → `03_mining` → `04_modeling` → `04b_semi_supervised` → `05_evaluation`

### Option 3: Chạy tất cả notebook bằng Papermill
```bash
python scripts/run_papermill.py
```

### Option 4: Streamlit demo (điểm thưởng)
```bash
streamlit run app.py
```

## Deploy nhanh

### A) Render (khuyên dùng)
Project đã có sẵn `render.yaml`.

1. Push code lên GitHub
2. Vào Render → **New +** → **Blueprint**
3. Chọn repo `Data_Mining`
4. Render tự đọc `render.yaml` và deploy

App sẽ chạy bằng lệnh:
```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

### B) Streamlit Community Cloud
1. Push code lên GitHub
2. Vào share.streamlit.io → **New app**
3. Chọn repo, branch, main file: `app.py`
4. Deploy

Cấu hình giao diện/serve đã đặt trong `.streamlit/config.toml`.

## Cấu trúc dự án

```
Data_Mining/
├── configs/
│   └── params.yaml              # Tham số cấu hình (seed, paths, hyperparams)
├── data/
│   ├── raw/heart.csv            # Dữ liệu gốc
│   └── processed/heart_clean.csv # Dữ liệu đã xử lý
├── doc/
│   ├── bang_danh_gia_diem.md    # Bảng tiêu chí chấm điểm
│   └── noi_dung_bao_cao.md     # Cấu trúc báo cáo
├── notebooks/
│   ├── 01_eda.ipynb             # Khám phá dữ liệu + Data Dictionary
│   ├── 02_preprocess_feature.ipynb # Tiền xử lý + Feature Engineering
│   ├── 03_mining_or_clustering.ipynb # Apriori + Clustering
│   ├── 04_modeling.ipynb        # Baseline + SVM/RF/XGBoost
│   ├── 04b_semi_supervised.ipynb # Self-training, Label Spreading
│   └── 05_evaluation_report.ipynb # Đánh giá + Insights
├── src/                         # Code Python tái sử dụng
│   ├── data/                    # loader.py, cleaner.py
│   ├── features/                # builder.py
│   ├── mining/                  # association.py, clustering.py
│   ├── models/                  # supervised.py, semi_supervised.py
│   ├── evaluation/              # metrics.py, report.py
│   └── visualization/           # plots.py
├── scripts/
│   ├── run_pipeline.py          # Chạy end-to-end pipeline
│   └── run_papermill.py         # Chạy notebooks tự động
├── outputs/
│   ├── figures/                 # Biểu đồ
│   ├── tables/                  # Bảng kết quả CSV
│   └── models/                  # Model đã train (.joblib)
├── app.py                       # Streamlit demo app
├── requirements.txt
└── README.md
```

## Kết quả mẫu

| Mô hình | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|
| Baseline (Dummy) | ~0.00 | 0.50 | ~0.55 |
| Baseline (Logistic) | ~0.82 | ~0.88 | ~0.87 |
| SVM (RBF) | ~0.85 | ~0.90 | ~0.90 |
| Random Forest | ~0.85 | ~0.90 | ~0.90 |
| XGBoost | ~0.87 | ~0.90 | ~0.89 |

> Tất cả kết quả đều reproducible với `seed=42`.

## Thành viên & Liên hệ

- Nguyễn Xuân Giang
- Phạm Đức Duy Tiến 
- Vương Đức Tuấn 
- Dương Văn Việt

