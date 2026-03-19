# 🫀 Đề tài 2: Dự đoán Bệnh Tim

## Thông tin chung

| Hạng mục | Nội dung |
|---|---|
| **Dataset** | UCI / Kaggle – Heart Disease Dataset |
| **Mục tiêu** | Dự đoán bệnh nhân có bị bệnh tim hay không |

---

## Các phần cần triển khai

### 1. Luật kết hợp (Association Rule Mining)

| Yêu cầu | Chi tiết |
|---|---|
| Rời rạc hoá | Các chỉ số theo ngưỡng (tuổi, cholesterol, huyết áp…) |
| Thuật toán | **Apriori** tìm tổ hợp triệu chứng liên quan bệnh tim |
| Báo cáo | **support, confidence, lift** |
| Diễn giải | Diễn giải luật có ý nghĩa y học |

**Ví dụ luật:** Nếu `cholesterol_cao` VÀ `tuổi > 55` VÀ `huyết_áp_cao` → Nguy cơ bệnh tim (support=0.15, confidence=0.82, lift=2.1)

---

### 2. Phân cụm (Clustering)

| Yêu cầu | Chi tiết |
|---|---|
| Chuẩn hoá | **StandardScaler** |
| Thuật toán | **KMeans** / **Hierarchical** |
| Đánh giá | **Silhouette Score** để xác định số cụm tối ưu |
| Output | Mô tả đặc điểm từng cụm nguy cơ |
| Insight | Rút insight nhóm nguy cơ cao |

---

### 3. Phân lớp (Classification) ⭐ TRỌNG TÂM

| Yêu cầu | Chi tiết |
|---|---|
| Mô hình | **SVM** / **Random Forest** / **XGBoost** |
| Xử lý mất cân bằng | **class_weight** / **SMOTE** |
| Metric ưu tiên | **PR-AUC**, **F1-score** |
| Phân tích lỗi | **Confusion Matrix**, đặc biệt quan tâm **False Negative** |

> ⚠️ **Lưu ý quan trọng:** Trong y tế, False Negative (bỏ sót bệnh nhân thực sự bị bệnh) 
> nguy hiểm hơn False Positive (báo nhầm người khoẻ bị bệnh). 
> Cần tối ưu Recall cho lớp "bệnh".

---

### 4. Bán giám sát (Semi-supervised) — *Tùy chọn*

| Yêu cầu | Chi tiết |
|---|---|
| Giả lập | Chỉ giữ **10–30% dữ liệu có nhãn** |
| Thuật toán | **Self-training** hoặc **Label Spreading** |
| So sánh | Supervised vs Semi-supervised |
| Metric | So sánh **PR-AUC** theo % nhãn |
| Phân tích | Pseudo-label sai và nhóm khó |

---

### 5. Hồi quy (Regression) — *Tùy chọn*

| Yêu cầu | Chi tiết |
|---|---|
| Mục tiêu | Dự đoán chỉ số sức khoẻ liên tục (vd: huyết áp) theo yếu tố nguy cơ |
| Mô hình | **Linear Regression** / **Ridge** / **XGBoost Regressor** |
| Metric | **MAE**, **RMSE** |
| Kiểm tra | Outlier và **leakage** |
| Ghi chú | Không bắt buộc chuỗi thời gian |

---

## Mapping vào Quy trình Khai phá Dữ liệu

| Bước quy trình | Áp dụng cho đề tài Bệnh Tim |
|---|---|
| 1. Data Source | Tải Heart Disease Dataset từ UCI/Kaggle |
| 2. Preprocessing | Xử lý missing, chuẩn hoá, mã hoá biến phân loại, SMOTE |
| 3. Feature/Representation | Rời rạc hoá cho Apriori, StandardScaler cho Clustering/Classification |
| 4. Mining/Modeling | Apriori, KMeans, SVM/RF/XGBoost, Semi-supervised |
| 5. Evaluation | PR-AUC, F1, Silhouette, Confusion Matrix, MAE/RMSE |
| 6. Semi-supervised | Self-training / Label Spreading với 10-30% nhãn |

---

## Phân loại ưu tiên

- ✅ **Bắt buộc:** Luật kết hợp, Phân cụm, Phân lớp
- 🔶 **Tùy chọn (cộng điểm):** Bán giám sát, Hồi quy
