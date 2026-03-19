# 2.2 Nội dung báo cáo (Bắt buộc, theo đúng thứ tự)

---

## Chương 1: Đặt vấn đề và phân tích yêu cầu

- Bối cảnh của đề tài
- Mục tiêu cần đạt được
- Tiêu chí thành công (metric nào? ngưỡng bao nhiêu?)
- Mô tả dữ liệu (dataset description)
- **EDA** (Exploratory Data Analysis): thống kê mô tả, phân phối, tương quan, missing values…

---

## Chương 2: Thiết kế giải pháp và quy trình khai phá

- Mô tả **pipeline** tổng thể (Data → Preprocessing → Feature → Mining → Evaluation)
- Chi tiết bước **tiền xử lý** (xử lý missing, chuẩn hoá, mã hoá, cân bằng lớp…)
- Chi tiết bước **đặc trưng** (chọn/tạo feature, rời rạc hoá cho Apriori…)
- **Lý do chọn kỹ thuật**: tại sao chọn SVM/RF/XGBoost? Tại sao KMeans? Tại sao Apriori?

---

## Chương 3: Phân tích mã nguồn và chức năng

- Mô tả **kiến trúc repo** (cấu trúc thư mục, cách tổ chức code)
- Các **module/class chính**:
  - `DataCleaner` – làm sạch và tiền xử lý dữ liệu
  - `FeatureBuilder` – tạo và chọn đặc trưng
  - `Miner` – khai phá tri thức (Apriori, Clustering)
  - `Trainer` – huấn luyện mô hình (Classification, Regression)
  - `Evaluator` – đánh giá và so sánh kết quả
- Giải thích chức năng và vai trò từng module

---

## Chương 4: Thử nghiệm và kết quả

- Dùng **metric phù hợp** cho từng phần:
  - Association: support, confidence, lift
  - Clustering: Silhouette Score
  - Classification: PR-AUC, F1-score, Confusion Matrix
  - Regression: MAE, RMSE (nếu làm)
- **Bảng/biểu đồ** so sánh các phương án (SVM vs RF vs XGBoost…)
- Phân tích False Negative

> ⚠️ **Nếu có Semi-supervised:** Phải thêm mục **"Thiếu nhãn: Supervised vs Semi-supervised"**
> - So sánh kết quả khi có 100% nhãn vs 10/20/30% nhãn
> - Learning curve theo % nhãn

---

## Chương 5: Thảo luận và so sánh

- So sánh **ưu/nhược điểm** từng phương án
- Giải thích **vì sao phương án A tốt hơn B**
- Nêu **thách thức** gặp phải trong quá trình thực hiện
- Bài học kinh nghiệm

> ⚠️ **Nếu có Semi-supervised:** Phải thêm mục thảo luận **Supervised vs Semi-supervised**
> - Khi nào semi-supervised hiệu quả?
> - Rủi ro pseudo-label sai

---

## Chương 6: Tổng kết và hướng phát triển

- **Tóm tắt kết quả** đạt được
- Đối chiếu với tiêu chí thành công ban đầu
- **Đề xuất cải tiến**:
  - Thêm dữ liệu
  - Thử mô hình khác (Deep Learning…)
  - Triển khai thực tế (web app, API…)
  - Kết hợp thêm dữ liệu y tế khác

---

## Checklist báo cáo

- [ ] Chương 1: Đặt vấn đề + EDA
- [ ] Chương 2: Pipeline + lý do chọn kỹ thuật
- [ ] Chương 3: Kiến trúc code + mô tả module
- [ ] Chương 4: Kết quả + bảng/biểu đồ
- [ ] Chương 5: Thảo luận + so sánh
- [ ] Chương 6: Tổng kết + hướng phát triển
- [ ] (Nếu có) Mục semi-supervised trong chương 4 & 5
