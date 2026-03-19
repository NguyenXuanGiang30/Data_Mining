# Quy ước và Quy định Đề tài

---

## 2.4.2 Quy ước đặt tên luồng pipeline

1. **Notebook đặt theo thứ tự 01 → 05** để người chấm chạy/đọc theo pipeline
2. **`src/` chứa toàn bộ logic chính**; notebook chỉ gọi hàm/lớp và trình bày kết quả
3. **Tất cả đường dẫn và tham số quan trọng** đặt trong `configs/params.yaml`

### Ví dụ thứ tự notebook:
```
01_eda.ipynb                  → Khám phá dữ liệu
02_preprocess_feature.ipynb   → Tiền xử lý + Feature
03_mining_or_clustering.ipynb → Khai phá / Phân cụm
04_modeling.ipynb             → Mô hình phân lớp
04b_semi_supervised.ipynb     → Bán giám sát (tùy chọn)
05_evaluation_report.ipynb    → Đánh giá & báo cáo
```

### Nguyên tắc quan trọng:
- **Notebook** = nơi trình bày, gọi hàm, hiển thị kết quả
- **src/** = nơi viết logic, class, function tái sử dụng
- **configs/params.yaml** = nơi quản lý tập trung tham số (seed, split ratio, hyperparams, paths…)

---

## 2.4.3 Quy định về dữ liệu và lưu trữ

1. **Không commit dữ liệu lớn vào GitHub**
2. Nếu dataset nặng, nhóm phải cung cấp:
   - **Link dataset + hướng dẫn tải** trong `README.md`, hoặc
   - **Script tải dữ liệu** trong `scripts/` (nếu có thể)

### Gợi ý .gitignore cho data:
```gitignore
# Không commit dữ liệu gốc lớn
data/raw/*.csv
data/raw/*.zip
data/raw/*.xlsx

# Không commit model nặng
outputs/models/*.pkl
outputs/models/*.joblib
```

---

## Tóm tắt quy tắc cần nhớ

| Quy tắc | Mô tả |
|---|---|
| Đặt tên notebook | Theo thứ tự `01` → `05` |
| Logic code | Viết trong `src/`, notebook chỉ gọi hàm |
| Tham số | Tập trung trong `configs/params.yaml` |
| Dữ liệu lớn | Không commit, cung cấp link/script tải |
| README.md | Phải có hướng dẫn tải dataset |

---

## 2.4.4 Yêu cầu tái lập (Reproducibility)

Repo được coi là **đạt** khi người khác có thể thực hiện đúng 4 bước sau và thu được kết quả tương tự:

### 4 bước tái lập:

```bash
# Bước 1: Cài đặt dependencies
pip install -r requirements.txt

# Bước 2: Cập nhật đường dẫn dữ liệu
# Sửa file configs/params.yaml → trỏ đến data/raw/

# Bước 3: Chạy toàn bộ pipeline
python scripts/run_papermill.py

# Bước 4: Kiểm tra kết quả
# → outputs/figures/, outputs/tables/ phải khớp với báo cáo
```

### Checklist tái lập:

| # | Yêu cầu | Kiểm tra |
|---|---|---|
| 1 | `requirements.txt` đầy đủ | `pip install` không lỗi |
| 2 | `configs/params.yaml` rõ ràng | Chỉ cần sửa path dữ liệu |
| 3 | `run_papermill.py` chạy được | Tất cả notebook chạy xong không lỗi |
| 4 | Kết quả khớp báo cáo | Hình/bảng trong `outputs/` giống báo cáo |

> ⚠️ **Lưu ý:** Phải set **random seed** cố định trong `params.yaml` để đảm bảo kết quả tái lập được chính xác.

---

## 2.5 Điểm thưởng

Có **GUI / Demo app** → cộng điểm theo mức hoàn thiện.

### Công cụ gợi ý:
| Công cụ | Đặc điểm |
|---|---|
| **Streamlit** | Đơn giản, nhanh, phù hợp demo Data Science |
| **Gradio** | Dễ tạo interface cho ML model |
| **Web nhỏ** (Flask/FastAPI) | Linh hoạt hơn, phù hợp nếu muốn custom UI |

### Ý tưởng demo cho đề tài Bệnh Tim:
- Người dùng nhập các chỉ số sức khoẻ (tuổi, cholesterol, huyết áp…)
- Hệ thống dự đoán nguy cơ bệnh tim (có/không + xác suất)
- Hiển thị biểu đồ phân tích, luật kết hợp liên quan
- Cho thấy bệnh nhân thuộc cụm nguy cơ nào

> 💡 **Tip:** Streamlit là lựa chọn nhanh nhất, chỉ cần 1 file `app.py` là đủ demo.
