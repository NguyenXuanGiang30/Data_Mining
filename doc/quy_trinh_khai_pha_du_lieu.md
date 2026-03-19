# 2.1 Quy trình khai phá dữ liệu tổng quát

Quy trình tổng quát tuân theo logic:

**Nguồn dữ liệu → Tiền xử lý → Đặc trưng/biểu diễn → Mô hình → Đánh giá**

---

## 1. Data Source (Nguồn dữ liệu)

- Dữ liệu dạng bảng / văn bản / ảnh / chuỗi thời gian / đồ thị… tùy đề tài
- Diverse inputs: Databases, Cloud, IoT, Logs

---

## 2. Preprocessing (Tiền xử lý)

- Làm sạch, xử lý thiếu, chuẩn hoá
- Mã hoá biến phân loại
- Cân bằng lớp (nếu cần)
- Tạo session (log), resample (time series)…

---

## 3. Feature / Representation (Đặc trưng/biểu diễn)

- TF-IDF / embeddings (text)
- Đặc trưng ảnh
- Đặc trưng hành vi (RFM)
- Lag features (chuỗi thời gian)
- Graph features
- Vector hoá giỏ hàng…

---

## 4. Mining / Modeling (Khai phá, mô hình hoá)

### 4a. Khai phá tri thức (Bắt buộc)
- Pattern mining
- Clustering
- Anomaly detection
- Rule extraction…

### 4b. Mô hình dự đoán (nếu đề tài có supervised)
- Classification / Regression / Forecasting
- Ghi rõ hyperparams
- Thời gian train
- Thiết lập thực nghiệm

---

## 5. Evaluation and Results (Đánh giá & kết quả)

- Dùng metric phù hợp
- Lập bảng/biểu đồ so sánh mô hình
- Kèm **insight** và **khuyến nghị hành động**

---

## 6. Nhánh bổ sung: Bán giám sát (Semi-supervised)

> Chỉ áp dụng khi đề tài là **phân lớp**

- Với đề tài phân lớp, nhóm phải thêm một nhánh thực nghiệm **"thiếu nhãn"**:
  - Giữ lại **p% nhãn** (p = 5/10/20), phần còn lại coi là unlabeled
  - So sánh **Supervised-only** (ít nhãn) vs **Semi-supervised** (self-training/pseudo-label hoặc label spreading/propagation)
  - Báo cáo **learning curve** theo % nhãn và phân tích rủi ro pseudo-label

---

## Sơ đồ quy trình

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  1. Data      │    │  2. Pre-     │    │  3. Feature/ │    │  4. Mining/  │    │  5. Eval &   │
│  Source       │───▶│  processing  │───▶│  Represent.  │───▶│  Modeling    │───▶│  Results     │
│  (Nguồn DL)  │    │  (Tiền xử lý)│    │  (Đặc trưng) │    │  (Khai phá)  │    │  (Đánh giá)  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
  Databases,         Cleaning,           Extraction,         Machine learning,   Validation,
  Cloud, IoT,        integration,        selection,          pattern recognition, interpretation,
  Logs               transformation,     construction,       predictive analysis  visualization,
                     reduction           embedding                                deployment
```
