# 📁 Hướng Dẫn Tải & Quản Lý Dữ Liệu (Data Guide)

Tài liệu này hướng dẫn cách tải và quản lý các bộ dữ liệu được sử dụng trong dự án **Alternative Credit Scoring & Churn Analysis**.

---

## 📌 Các Bộ Dữ Liệu Trong Dự Án

Dự án sử dụng 2 bộ dữ liệu chính từ Kaggle:

1. **Home Credit Default Risk** (Cuộc thi Kaggle):
   - **Nguồn**: [Home Credit Default Risk - Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk)
   - **Mục đích**: Đánh giá rủi ro tín dụng dựa trên dữ liệu giao dịch, lịch sử tín dụng ngoài hệ thống (Bureau), đơn vay trước đây và thanh toán trả góp.
   - **Thư mục lưu**: `data/raw/home-credit-default-risk/`

2. **Vietnam Bank Churn Dataset 2025**:
   - **Nguồn**: [tranhuunhan/vietnam-bank-churn-dataset-2025](https://www.kaggle.com/datasets/tranhuunhan/vietnam-bank-churn-dataset-2025)
   - **Mục đích**: Nghiên cứu hành vi số, phân khúc khách hàng và nguy cơ rời bỏ ngân hàng tại Việt Nam.
   - **Thư mục lưu**: `data/raw/vietnam-bank-churn-dataset-2025/`

---

## 🛠️ Cấu Trúc Thư Mục Dữ Liệu

Sau khi tải thành công, cấu trúc thư mục dữ liệu thô (`data/raw/`) sẽ như sau:

```text
data/
└── raw/
    ├── home-credit-default-risk/
    │   ├── application_train.csv
    │   ├── application_test.csv
    │   ├── bureau.csv
    │   ├── bureau_balance.csv
    │   ├── POS_CASH_balance.csv
    │   ├── credit_card_balance.csv
    │   ├── previous_application.csv
    │   ├── installments_payments.csv
    │   ├── HomeCredit_columns_description.csv
    │   └── sample_submission.csv
    └── vietnam-bank-churn-dataset-2025/
        └── bank_churn_dataset_80k.csv
```

> ⚠️ **Lưu ý**: Tất cả dữ liệu trong thư mục `data/` đều đã được thêm vào `.gitignore` để **không commit file dữ liệu lớn lên GitHub**.

---

## 🚀 Hướng Dẫn Tải Dữ Liệu Tự Động

### Bước 1: Cấu hình Kaggle API Token

1. Tạo file `.env` từ file mẫu `.env.example` nếu chưa có:
   ```bash
   cp .env.example .env
   ```
2. Lấy **Kaggle API Token**:
   - Truy cập: [Kaggle Account Settings](https://www.kaggle.com/settings)
   - Chọn **Create Legacy API Key** (hoặc **Generate New Token**).
3. Điền Token vào file `.env`:
   ```env
   KAGGLE_API_TOKEN=KGAT_your_token_here
   ```

---

### Bước 2: Chấp nhận điều khoản cuộc thi Home Credit (Chỉ làm 1 lần)

Do `home-credit-default-risk` là bộ dữ liệu cuộc thi Kaggle:
1. Truy cập liên kết: 👉 **[Home Credit Default Risk Rules](https://www.kaggle.com/competitions/home-credit-default-risk/rules)**
2. Đăng nhập tài khoản Kaggle của bạn và nhấn nút **"I Understand and Accept"** (Chấp nhận điều khoản cuộc thi).

---

### Bước 3: Chạy Script Tải Dữ Liệu

Chạy lệnh sau tại thư mục gốc của dự án:

```bash
# Sử dụng uv (khuyên dùng)
uv run python utils/load_data.py

# Hoặc dùng python trực tiếp
python utils/load_data.py
```

### ⚡ Cơ chế Tối Ưu (Skip Download):
Script `utils/load_data.py` được tích hợp sẵn cơ chế kiểm tra sự tồn tại của dữ liệu. Nếu dữ liệu đã có sẵn trong `data/raw/{tên_dataset}`, script sẽ tự động bỏ qua bước tải lại để tiết kiệm thời gian và tài nguyên mạng.

---

## 🔄 Tải Lại Dữ Liệu Khi Cần (Force Re-download)

Nếu bạn muốn ép buộc tải lại toàn bộ dữ liệu mới nhất từ Kaggle, mở file `utils/load_data.py` hoặc gọi hàm Python:

```python
from utils.load_data import load_all_datasets

# Đặt force_download=True để tải lại toàn bộ
load_all_datasets(force_download=True)
```
