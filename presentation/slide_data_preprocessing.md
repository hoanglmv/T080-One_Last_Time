# 🎨 Thiết Kế Nội Dung Slide: 3. TIỀN XỬ LÝ DỮ LIỆU (Alternative Credit Scoring)

Tài liệu này hướng dẫn thiết kế nội dung chi tiết và visual cho **Phần 3: Tiền Xử Lý Dữ Liệu** trong Slide thuyết trình dự án Alternative Credit Scoring (Home Credit Default Risk).

---

## 📌 Bố Cục Tổng Quan Cho Phần "3. TIỀN XỬ LÝ DỮ LIỆU"

Dựa trên khung slide bạn đang dựng với 4 nhánh chính:
1. **Lọc Scope** (Phạm vi dữ liệu & Làm sạch mẫu)
2. **Xác Định Label** (Định nghĩa biến mục tiêu & Xử lý mất cân bằng)
3. **Kỹ Thuật Xử Lý & Tạo Biến (Preprocessing & Feature Engineering)** *(Điểm bổ sung cho bullet còn trống)*
4. **Lọc Feature** (Sàng lọc đặc trưng theo IV/WoE & Multicollinearity)

---

## 🖼️ SLIDE TỔNG QUAN: 3. TIỀN XỬ LÝ DỮ LIỆU (Overview Layout)

### Gợi ý Visual Layout (4 Block Cards trên Google Slides / PowerPoint):
- Sử dụng bố cục **4 ô chữ nhật (4-Card Grid Layout)** hoặc **Quy trình 4 bước (Horizontal 4-Step Pipeline)**.
- Màu chủ đạo: Dark Navy (`#1E293B`) làm nền, Accent Teal (`#0EA5E9`) cho highlight, Coral Red (`#F43F5E`) cho rủi ro.

```
+-----------------------------------------------------------------------------------+
|                            3. TIỀN XỬ LÝ DỮ LIỆU                                 |
+------------------------------------+----------------------------------------------+
| 📍 1. LỌC SCOPE                    | 🎯 2. XÁC ĐỊNH LABEL                         |
| • Giới hạn khách hàng Unbanked     | • TARGET = 1 (Vỡ nợ / Quá hạn 90+ ngày)      |
| • Tách mẫu application_train.csv   | • Tỷ lệ mất cân bằng: 8.07% vs 91.93%        |
| • Xử lý anomaly DAYS_EMPLOYED      | • Xử lý: class_weight / scale_pos_weight     |
+------------------------------------+----------------------------------------------+
| ⚙️ 3. PREPROCESSING & FEATURE ENG  | 🔍 4. LỌC FEATURE                            |
| • Encoded & Impute (Stratified)    | • Missing Rate filter (< 75%)                |
| • Tạo chỉ số ACS phái sinh         | • Sàng lọc bằng WoE / Information Value (IV) |
| • Aggregations từ bureau & prev    | • Lọc đa cộng tuyến (|r| < 0.8)              |
+------------------------------------+----------------------------------------------+
```

---

## 📝 CHI TIẾT NỘI DUNG TỪNG SLIDE (Slide Content & Text Script)

### SLIDE 3.1: LỌC SCOPE (Data Scope & Population Definition)
**Tiêu đề Slide**: `3.1 Lọc Scope - Xác Định Phạm Vi Mẫu Dữ Liệu`

* **Khung mẫu nghiên cứu (Target Population)**:
  - Tập trung vào khách hàng **Thin-file / Unbanked** xin vay tiêu dùng cá nhân.
  - Quy mô dữ liệu thô: **307,511 bản ghi** từ `application_train.csv`.
* **Quy tắc làm sạch Scope (Data Cleaning Rules)**:
  - **Phát hiện Anomaly**: Biến `DAYS_EMPLOYED = 365243` (giá trị dị biệt tương đương ~1000 năm làm việc).
  - **Xử lý**: Chuyển `365243` thành `NaN` và bổ sung cờ chỉ báo `DAYS_EMPLOYED_ANOM = 1` để giữ nguyên thông tin hành vi.
* **Liên kết dữ liệu lịch sử (Relational Scope)**:
  - Ghép nối dữ liệu ngoài ngân hàng (`bureau.csv`), lịch sử xin vay (`previous_application.csv`), và nhật ký trả góp (`installments_payments.csv`) theo mã khách hàng `SK_ID_CURR`.

---

### SLIDE 3.2: XÁC ĐỊNH LABEL (Target Label Definition & Imbalance Strategy)
**Tiêu đề Slide**: `3.2 Xác Định Label - Định Nghĩa Rủi Ro & Xử Lý Mất Cân Bằng`

* **Định nghĩa Biến Mục Tiêu (`TARGET`)**:
  - `TARGET = 1`: Khách hàng vỡ nợ hoặc quá hạn thanh toán nghiêm trọng (Default / Late Payment).
  - `TARGET = 0`: Khách hàng hoàn trả khoản vay đúng hạn (Non-default).
* **Đặc điểm Mất Cân Bằng Dữ Liệu (Class Imbalance)**:
  - Tỷ lệ vỡ nợ thực tế: **8.07%** (24,825 hồ sơ vỡ nợ / 282,686 hồ sơ trả nợ tốt).
  - *Thách thức*: Mô hình dễ bị lệch về phân loại đa số (Class 0).
* **Chiến lược Xử lý & Đánh giá**:
  - **Không dùng Undersampling thô bạo** để tránh tổn thất thông tin khách hàng.
  - Áp dụng trọng số điều chỉnh: `class_weight='balanced'` (Logistic Regression), `is_unbalance=True` (LightGBM), `scale_pos_weight` (XGBoost).
  - Đánh giá bằng bộ thước đo chuẩn ngân hàng: **ROC-AUC**, **PR-AUC**, **KS Statistic** (mục tiêu $KS > 40\%$), **Gini Coefficient**.

---

### SLIDE 3.3: TIỀN XỬ LÝ & KỸ THUẬT TẠO BIẾN (Preprocessing & Feature Engineering)
**Tiêu đề Slide**: `3.3 Preprocessing & Feature Engineering Cho Alternative Data`

* **Ngăn chặn Data Leakage**:
  - Thực hiện chia Stratified K-Fold (80% Train / 20% Val) **trước khi** thực hiện Imputation và Scaler.
* **Tạo Chỉ Số Tín Dụng Thay Thế (Alternative Credit Ratios)**:
  - `CREDIT_TO_INCOME_RATIO = AMT_CREDIT / AMT_INCOME_TOTAL`: Tỷ lệ nợ trên thu nhập.
  - `ANNUITY_TO_INCOME_RATIO = AMT_ANNUITY / AMT_INCOME_TOTAL`: Tỷ lệ nghĩa vụ trả nợ định kỳ.
  - `PAYMENT_RATE = AMT_ANNUITY / AMT_CREDIT`: Tỷ lệ hoàn trả hàng kỳ.
  - `PHONE_CHANGE_YEARS = |DAYS_LAST_PHONE_CHANGE| / 365.25`: Thâm niên sử dụng số điện thoại.
* **Tổng hợp Đặc trưng Đa bảng (Relational Aggregations)**:
  - `BUREAU_CREDIT_DAY_OVERDUE_MAX`: Số ngày quá hạn tối đa tại các tổ chức tín dụng ngoài.
  - `INST_DPD_MEAN`: Số ngày chậm trả trung bình trong lịch sử trả góp.

---

### SLIDE 3.4: LỌC FEATURE (Feature Selection & Reduction)
**Tiêu đề Slide**: `3.4 Lọc Feature - Sàng Lọc Đặc Trưng Phân Tách Rủi Ro`

* **Quy trình Sàng lọc 3 Lớp (3-Tier Feature Selection)**:

| Lớp (Tier) | Phương Pháp Lọc | Tiêu Chí Loại Bỏ | Ý Nghĩa / Mục Đích |
| :--- | :--- | :--- | :--- |
| **Lớp 1: Missing Filter** | Tỷ lệ khuyết dữ liệu | Missing Rate $> 75\%$ | Loại bỏ biến thiếu thông tin nghiêm trọng |
| **Lọc WoE & IV** | Information Value (IV) | $IV < 0.02$ (Uninformative) | Giữ lại biến có sức mạnh phân tách ($IV \ge 0.02$) |
| **Lọc Đa Cộng Tuyến** | Correlation Matrix | Tương quan Pearson $|r| > 0.8$ | Loại bỏ biến dư thừa, tránh làm méo mô hình Scorecard |

* **Kết quả Lọc Feature**:
  - Giữ lại các đặc trưng thay thế quan trọng hàng đầu: `EXT_SOURCE_1,2,3`, `CREDIT_TO_INCOME_RATIO`, `DAYS_LAST_PHONE_CHANGE`, `NAME_HOUSING_TYPE`.

---

## 🗣️ LỜI THOẠI GỢI Ý (Presenter Notes / Speaker Script)

> *"Kính thưa thầy cô và các bạn, phần 3 Tiền xử lý dữ liệu là bước xương sống trong bài toán Alternative Credit Scoring. Chúng tôi thực hiện qua 4 bước chính:*
> 1. **Lọc Scope**: Định hình đúng tập khách hàng Unbanked và làm sạch các điểm anomaly như biến thâm niên làm việc dị biệt.
> 2. **Xác định Label**: Gán nhãn nợ xấu TARGET (tỷ lệ 8.07%) và sử dụng chiến lược điều chỉnh trọng số lớp thay vì cắt giảm dữ liệu.
> 3. **Preprocessing & Feature Engineering**: Thiết kế các chỉ số khả năng chi trả thay thế như tỷ lệ nợ/thu nhập, thâm niên số điện thoại, và tổng hợp lịch sử trả góp.
> 4. **Lọc Feature**: Áp dụng quy trình lọc 3 lớp dựa trên tỷ lệ missing, chỉ số Information Value (IV), và kiểm tra đa cộng tuyến để thu được tập biến tối ưu nhất cho mô hình."*
