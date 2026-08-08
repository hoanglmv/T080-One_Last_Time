# 📁 Hướng Dẫn Tải & Quản Lý Dữ Liệu (Data Guide)

Tài liệu này hướng dẫn cách tải và quản lý các bộ dữ liệu được sử dụng trong dự án **Alternative Credit Scoring Engine**.

---

## 📌 Bộ Dữ Liệu Trong Dự Án

Dự án sử dụng bộ dữ liệu chính từ Kaggle:

1. **Home Credit Default Risk** (Cuộc thi Kaggle):
   - **Nguồn**: [Home Credit Default Risk - Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk)
   - **Mục đích**: Đánh giá rủi ro tín dụng dựa trên dữ liệu giao dịch, lịch sử tín dụng ngoài hệ thống (Bureau), đơn vay trước đây và thanh toán trả góp.
   - **Thư mục lưu**: `data/raw/home-credit-default-risk/`

---

## 🛠️ Cấu Trúc Thư Mục Dữ Liệu

Sau khi tải thành công, cấu trúc thư mục dữ liệu thô (`data/raw/`) sẽ như sau:

```text
data/
└── raw/
    └── home-credit-default-risk/
        ├── application_train.csv
        ├── application_test.csv
        ├── bureau.csv
        ├── bureau_balance.csv
        ├── POS_CASH_balance.csv
        ├── credit_card_balance.csv
        ├── previous_application.csv
        ├── installments_payments.csv
        ├── HomeCredit_columns_description.csv
        └── sample_submission.csv
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

### Bước 3: Chạy script tải dữ liệu tự động

Chạy script Python được cung cấp sẵn để tải tự động bộ dữ liệu về thư mục `data/raw/`:

```bash
uv run python utils/load_data.py
```
