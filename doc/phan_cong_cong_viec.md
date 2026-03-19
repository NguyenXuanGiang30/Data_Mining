# Phân Công Công Việc — Heart Care AI (Data Mining)

> **Nguyên tắc:** Ai cũng có code · Phần báo cáo không tính · Tỷ lệ: **35 – 25 – 20 – 20**

---

## Thành viên 1 — 35%

**Phụ trách chính:** Data Mining core + Mô hình hoá (phần nặng nhất)

| Tiêu chí | Công việc | Code liên quan |
|---|---|---|
| **C. Data Mining core (2.0đ)** | Clustering (KMeans, DBSCAN), Association Rules, Anomaly Detection; đánh giá Silhouette/DBI; rút insight | `src/mining/clustering.py`, `src/mining/association.py`, `src/mining/anomaly.py`, `notebooks/03_mining_or_clustering.ipynb` |
| **D. Mô hình hoá (2.0đ)** | Huấn luyện ≥2 mô hình cải tiến + 1 baseline; so sánh kết quả; export model | `src/models/supervised.py`, `notebooks/04_modeling.ipynb` |
| **H. Repo (phần pipeline)** | Script chạy toàn bộ pipeline tự động, tái lập kết quả | `scripts/run_pipeline.py`, `scripts/run_papermill.py` |

---

## Thành viên 2 — 25%

**Phụ trách chính:** Semi-supervised + Đánh giá & Phân tích lỗi

| Tiêu chí | Công việc | Code liên quan |
|---|---|---|
| **F. Bán giám sát (1.0đ)** | Self-training / Label Spreading; learning curve theo % nhãn; phân tích pseudo-label sai | `src/models/semi_supervised.py`, `notebooks/04b_semi_supervised.ipynb` |
| **G. Đánh giá + insight (1.5đ)** | Confusion matrix, phân tích lỗi FN/FP; ≥5 actionable insights; residual analysis | `src/evaluation/metrics.py`, `src/evaluation/report.py`, `notebooks/05_evaluation_report.ipynb` |
| **E. Thiết kế thực nghiệm (1.0đ)** | Seed, train/test split, CV strategy; chọn & giải thích metric (F1, ROC-AUC, PR-AUC) | `src/models/supervised.py` (phần split/CV), `notebooks/04_modeling.ipynb` (phần config) |

---

## Thành viên 3 — 20%

**Phụ trách chính:** EDA & Tiền xử lý + Visualization

| Tiêu chí | Công việc | Code liên quan |
|---|---|---|
| **B. EDA & tiền xử lý (1.5đ)** | ≥3 biểu đồ + diễn giải; xử lý missing/outlier/duplicate; encoding/scaling; thống kê trước–sau | `src/data/cleaner.py`, `src/features/builder.py`, `notebooks/01_eda.ipynb`, `notebooks/02_preprocess_feature.ipynb` |
| **Visualization** | Tất cả biểu đồ: correlation matrix, distributions, model comparison, ROC/PR curves | `src/visualization/plots.py` |
| **H. Repo (phần data)** | Quản lý data pipeline, đảm bảo dữ liệu load đúng | `src/data/loader.py` |

---

## Thành viên 4 — 20%

**Phụ trách chính:** Mô tả bài toán + GUI Demo + Repo chuẩn

| Tiêu chí | Công việc | Code liên quan |
|---|---|---|
| **A. Bài toán + mô tả dữ liệu (1.0đ)** | Mục tiêu, nguồn dữ liệu, data dictionary, target/label, rủi ro (imbalance, leakage) | `README.md`, `doc/` |
| **H. Repo chuẩn (1.0đ)** | Cấu trúc repo, README, requirements.txt, configs, hướng dẫn chạy lại | `README.md`, `requirements.txt`, `configs/`, `.gitignore`, `render.yaml` |
| **Bonus: GUI Demo** | Streamlit app: form nhập liệu, predict, hiển thị kết quả, biểu đồ model performance | `app.py`, `.streamlit/` |

---

## Tổng hợp phân bổ điểm

| Thành viên | Tiêu chí chính | Tổng điểm phụ trách | Tỷ lệ |
|---|---|---|---|
| **TV1** | C (2.0) + D (2.0) + H pipeline | ~4.0đ | **35%** |
| **TV2** | F (1.0) + G (1.5) + E (1.0) | ~3.5đ | **25%** |
| **TV3** | B (1.5) + Visualization + H data | ~2.0đ | **20%** |
| **TV4** | A (1.0) + H repo (1.0) + GUI bonus | ~2.0đ + bonus | **20%** |

> **Ghi chú:** Mỗi thành viên đều có code contribution rõ ràng trong `src/` hoặc `app.py`.

---

## Hướng dẫn đổi tài khoản Git & Push qua nhánh cho từng người

### Quy trình chung cho mỗi thành viên

```
1. Đổi tài khoản Git → user.name + user.email
2. Tạo nhánh riêng (feature branch)
3. Add + Commit trên nhánh đó
4. Push nhánh lên GitHub
5. Quay về main → Merge nhánh vào main → Push main
```

> ⚠️ **Không dùng `--global`** để chỉ thay đổi trong repo này, không ảnh hưởng máy.

---

#### 👤 Thành viên 1 (35%) — Mining + Modeling + Pipeline

```powershell
# 1. Đổi tài khoản
git config user.name "TV1_TenGithub"
git config user.email "tv1@email.com"

# 2. Tạo nhánh riêng
git checkout -b feature/mining-modeling

# 3. Add + Commit
git add src/mining/clustering.py
git add src/mining/association.py
git add src/mining/anomaly.py
git add src/models/supervised.py
git add notebooks/03_mining_or_clustering.ipynb
git add notebooks/04_modeling.ipynb
git add scripts/run_pipeline.py
git add scripts/run_papermill.py
git commit -m "feat: Data Mining core (clustering, association rules) + Supervised models + Pipeline"

# 4. Push nhánh lên GitHub
git push origin feature/mining-modeling

# 5. Merge vào main
git checkout main
git merge feature/mining-modeling
git push --force origin main
```

---

#### 👤 Thành viên 2 (25%) — Semi-supervised + Evaluation

```powershell
# 1. Đổi tài khoản
git config user.name "TV2_TenGithub"
git config user.email "tv2@email.com"

# 2. Tạo nhánh riêng
git checkout -b feature/semi-supervised-eval

# 3. Add + Commit
git add src/models/semi_supervised.py
git add src/evaluation/metrics.py
git add src/evaluation/report.py
git add notebooks/04b_semi_supervised.ipynb
git add notebooks/05_evaluation_report.ipynb
git commit -m "feat: Semi-supervised learning + Evaluation metrics + Error analysis & insights"

# 4. Push nhánh lên GitHub
git push origin feature/semi-supervised-eval

# 5. Merge vào main
git checkout main
git merge feature/semi-supervised-eval
git push origin main
```

---

#### 👤 Thành viên 3 (20%) — EDA + Preprocessing + Visualization

```powershell
# 1. Đổi tài khoản
git config user.name "TV3_TenGithub"
git config user.email "tv3@email.com"

# 2. Tạo nhánh riêng
git checkout -b feature/eda-preprocessing

# 3. Add + Commit
git add src/data/cleaner.py
git add src/data/loader.py
git add src/features/builder.py
git add src/visualization/plots.py
git add notebooks/01_eda.ipynb
git add notebooks/02_preprocess_feature.ipynb
git commit -m "feat: EDA, data preprocessing, feature engineering & visualization"

# 4. Push nhánh lên GitHub
git push origin feature/eda-preprocessing

# 5. Merge vào main
git checkout main
git merge feature/eda-preprocessing
git push origin main
```

---

#### 👤 Thành viên 4 (20%) — Bài toán + Repo + GUI

```powershell
# 1. Đổi tài khoản
git config user.name "TV4_TenGithub"
git config user.email "tv4@email.com"

# 2. Tạo nhánh riêng
git checkout -b feature/setup-gui

# 3. Add + Commit
git add app.py
git add .streamlit/
git add README.md
git add requirements.txt
git add configs/
git add render.yaml
git add .gitignore
git add doc/
git commit -m "feat: Project setup, README, Streamlit GUI demo & documentation"

# 4. Push nhánh lên GitHub
git push origin feature/setup-gui

# 5. Merge vào main
git checkout main
git merge feature/setup-gui
git push origin main
```

---

### 💡 Mẹo quan trọng

1. **Thứ tự:** Làm lần lượt từng người: TV1 → TV2 → TV3 → TV4. **Không làm song song.**
2. **Chỉ cần `--force` ở lần push main đầu tiên** (TV1), các lần sau push bình thường.
3. **Sau khi xong 1 người**, kiểm tra lại:
   ```powershell
   git log --oneline --all --graph -10
   ```
4. **Nếu muốn chia nhiều commit cho 1 người** (tự nhiên hơn):
   ```powershell
   git add src/mining/clustering.py src/mining/association.py
   git commit -m "feat: KMeans & DBSCAN clustering implementation"

   git add src/mining/anomaly.py
   git commit -m "feat: Anomaly detection module"
   ```
5. **Xem ai đã commit gì:**
   ```powershell
   git shortlog -s -n
   ```
6. **Xem tất cả nhánh:**
   ```powershell
   git branch -a
   ```
