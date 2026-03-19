# Bảng Đánh Giá Điểm Đề Tài Data Mining

**Tổng điểm tối đa: 10.0 điểm**

---

## A. Bài toán + mô tả dữ liệu + data dictionary (1.0đ)

| Mức | Mô tả |
|---|---|
| **Đạt** | Mục tiêu rõ ràng; mô tả nguồn dữ liệu; giải thích cột/nhãn/target; có data dictionary; nêu rủi ro như mất cân bằng lớp, thiếu dữ liệu, data leakage (nếu có) |
| **Trung bình** | Có mô tả nhưng thiếu 1–2 ý quan trọng (vd thiếu data dictionary hoặc chưa nói về leakage/imbalance) |
| **Chưa đạt** | Mơ hồ; thiếu nguồn dữ liệu; không nêu target/label hoặc không giải thích dữ liệu |

---

## B. EDA & tiền xử lý (1.5đ)

| Mức | Mô tả |
|---|---|
| **Đạt** | EDA có ít nhất 3 biểu đồ kèm diễn giải; xử lý missing/outlier/duplicate; encoding/scaling hợp lý; có thống kê trước–sau hoặc pipeline hoá |
| **Trung bình** | Có EDA và tiền xử lý nhưng phân tích còn nông; thiếu kiểm soát tham số hoặc thiếu thống kê trước–sau |
| **Chưa đạt** | EDA sơ sài hoặc chỉ chụp hình; tiền xử lý tùy tiện hoặc sai |

---

## C. Data Mining core — pattern/cluster/anomaly/rule/graph (2.0đ) ⭐

| Mức | Mô tả |
|---|---|
| **Đạt** | Có "khai phá tri thức" đúng chất (phân cụm/pattern/anomaly/rule/graph); trình bày tham số; có đánh giá (silhouette/DBI/coverage/runtime…); rút insight rõ ràng |
| **Trung bình** | Có mining nhưng hời hợt; thiếu đánh giá hoặc thiếu diễn giải kết quả |
| **Chưa đạt** | Không có phần mining; chỉ huấn luyện mô hình dự đoán |

---

## D. Mô hình hoá + baseline so sánh (≥ 2 baseline) (2.0đ) ⭐

| Mức | Mô tả |
|---|---|
| **Đạt** | Có ít nhất 2 mô hình cải tiến và 1 mô hình baseline; giải thích lựa chọn; có so sánh rõ ràng |
| **Trung bình** | Có baseline nhưng chưa rõ vai trò/thiết lập; thiếu so sánh hoặc mô hình cải tiến chưa thuyết phục |
| **Chưa đạt** | Chỉ 1 mô hình hoặc không có so sánh baseline |

---

## E. Thiết kế thực nghiệm + metric đúng (1.0đ)

| Mức | Mô tả |
|---|---|
| **Đạt** | Split/CV hợp lý; đặt seed; tránh leakage; chọn metric phù hợp (F1/PR-AUC/ROC-AUC; RMSE/MAE/sMAPE; silhouette/DBI…) |
| **Trung bình** | Có thực nghiệm nhưng thiếu kiểm soát (seed/leakage); metric đúng nhưng giải thích chưa rõ |
| **Chưa đạt** | Thiết kế thực nghiệm sai; metric không phù hợp hoặc không nêu rõ |

---

## F. Bán giám sát hoặc nhánh thay thế tương đương (1.0đ)

| Mức | Mô tả |
|---|---|
| **Đạt** | Nếu đề tài phân lớp: có kịch bản thiếu nhãn (10–30% labeled); so sánh supervised-only vs semi-supervised (self-training/label spreading); có learning curve theo % nhãn và phân tích pseudo-label sai |
| **Trung bình** | Có triển khai nhưng thiếu một phần (thiếu learning curve/thiếu phân tích lỗi/thiếu so sánh) |
| **Chưa đạt** | Không thực hiện semi-supervised (khi bắt buộc) hoặc không có nhánh thay thế (khi không áp dụng) |

---

## G. Đánh giá, phân tích lỗi & insight hành động (1.5đ)

| Mức | Mô tả |
|---|---|
| **Đạt** | Có phân tích lỗi (confusion matrix/residual); nếu dạng sai phổ biến; có ít nhất 5 insight "có hành động" (actionable) gắn với kết quả |
| **Trung bình** | Có insight nhưng chung chung; phân tích lỗi còn nông |
| **Chưa đạt** | Không phân tích lỗi; insight mơ hồ hoặc không có khuyến nghị |

---

## H. Repo GitHub chuẩn + chạy lại được (reproducible) (1.0đ)

| Mức | Mô tả |
|---|---|
| **Đạt** | Repo đúng cấu trúc; có README, requirements/environment; có configs/outputs; chạy lại tạo ra kết quả; notebook "sạch" (gọi code từ src) |
| **Trung bình** | Repo tương đối ổn nhưng thiếu 1–2 phần (vd thiếu script chạy pipeline hoặc thiếu hướng dẫn); vẫn còn nhiều code trong notebook |
| **Chưa đạt** | Repo lộn xộn; thiếu hướng dẫn; không chạy lại được |

---

## Bảng tổng hợp điểm

| Tiêu chí | Điểm tối đa | Ghi chú |
|---|---|---|
| A. Bài toán + mô tả dữ liệu | **1.0** | |
| B. EDA & tiền xử lý | **1.5** | ≥ 3 biểu đồ |
| C. Data Mining core | **2.0** | ⭐ Trọng tâm |
| D. Mô hình hoá + baseline | **2.0** | ⭐ ≥ 2 baseline |
| E. Thiết kế thực nghiệm | **1.0** | seed, CV, metric |
| F. Bán giám sát / nhánh thay thế | **1.0** | learning curve |
| G. Đánh giá + insight | **1.5** | ≥ 5 actionable insights |
| H. Repo chuẩn + reproducible | **1.0** | chạy lại được |
| **TỔNG** | **10.0** | + điểm thưởng GUI demo |
