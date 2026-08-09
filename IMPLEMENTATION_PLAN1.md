# 📄 Implementation Plan: Xây Dựng Mô Hình Credit Scoring Chỉ Sử Dụng Dữ Liệu Thay Thế (Alternative-Data-Only Model)

> **Mã kế hoạch**: `IMPLEMENTATION_PLAN1.md`  
> **Dự án**: Alternative Credit Scoring (`P080-One_Last_Time`)  
> **Mục tiêu**: Xây dựng, huấn luyện và đánh giá mô hình rủi ro tín dụng chuyên biệt **chỉ sử dụng Dữ Liệu Thay Thế (Alternative Data Only)** dành cho nhóm khách hàng **Thin-file** và **Unbanked/Underbanked**.

---

## 📌 1. Đặt Vấn Đề & Mục Tiêu Kỹ Thuật

### 1.1 Bối cảnh
Trong hệ thống đánh giá tín dụng hiện tại của dự án, mô hình Ensemble Champion đang sử dụng kết hợp giữa dữ liệu truyền thống (Credit Bureau, lịch sử nợ `bureau.csv`, điểm tín dụng `EXT_SOURCE_1/2/3`) và dữ liệu thay thế. 

Tuy nhiên, đối với nhóm khách hàng **Thin-file** (người trẻ, sinh viên, người làm tự do) hoặc nhóm **Unbanked** (chưa từng mở tài khoản hoặc vay vốn ngân hàng), dữ liệu tín dụng truyền thống hoàn toàn không tồn tại (Missing/NaN). Do đó, việc xây dựng một mô hình độc lập **chỉ dựa trên Dữ liệu Thay thế (Alternative-Data-Only Model)** là yêu cầu cấp thiết để:
1. Đảm bảo tính bao phủ tài chính (Financial Inclusion) cho 100% hồ sơ không có dữ liệu CIC/Bureau.
2. Đóng vai trò là **Mô hình Dự phòng (Fallback Model)** trong hệ thống Serving khi các nguồn dữ liệu truyền thống bị gián đoạn.
3. Đo lường chỉ số **Incremental Lift** (giá trị đóng góp thuần) của dữ liệu thay thế khi so sánh đối đầu với Mô hình Truyền thống (Traditional-Only) và Mô hình Kết hợp (Hybrid Champion).

---

## 📚 2. Nguồn Tham Khảo Học Thuật (Academic References)

Toàn bộ thiết kế và lựa chọn biến số trong kế hoạch này tuân thủ các nghiên cứu học thuật peer-reviewed và tài liệu chính thức từ các tổ chức tài chính quốc tế:

1. **World Bank Group & CGAP (2017)** — *Alternative Data Assessing Credit Risk for Financial Inclusion*:
   - Nghiên cứu chứng minh rằng việc khai thác dữ liệu hành vi di động, thâm niên thuê bao telco và dữ liệu thiết bị giúp cải thiện **20–30%** độ chính xác phân loại rủi ro tín dụng cho nhóm khách hàng lần đầu xin cấp tín dụng (Thin-file).
   - Nguồn: [World Bank / CGAP Report 2017](https://www.cgap.org/publications/alternative-data-assessing-credit-risk-financial-inclusion)

2. **Óskarsdóttir et al. (2019)** — *The value of big data for credit scoring: Enhancing financial inclusion using mobile phone data* (IEEE Transactions on Knowledge and Data Engineering / ScienceDirect):
   - Đánh giá khả năng dự báo vỡ nợ của các biến di động và hành vi số thuần túy bằng thuật toán Gradient Boosting (LightGBM/XGBoost), đạt chỉ số **ROC-AUC > 0.78** trên tập khách hàng không có dữ liệu ngân hàng.
   - DOI: [10.1016/j.eswa.2019.02.029](https://doi.org/10.1016/j.eswa.2019.02.029)

3. **Bazarbash, M. (2019)** — *Fintech in Financial Inclusion: Machine Learning Applications in Credit Scoring* (IMF Working Paper):
   - Phân tích tính hiệu quả của các mô hình học máy phi tuyến (Tree-based Ensembles) trong việc khai thác mối quan hệ phi tuyến giữa đặc trưng hành vi số và xác suất vỡ nợ.
   - Nguồn: [IMF Working Paper WP/19/109](https://www.imf.org/en/Publications/WP/Issues/2019/05/17/Fintech-in-Financial-Inclusion-Machine-Learning-Applications-in-Credit-Scoring-46862)

4. **Kaggle Home Credit Default Risk Competition (2018)**:
   - Bộ dữ liệu mở tiêu chuẩn cho bài toán dự đoán rủi ro vỡ nợ (`TARGET`) đối với nhóm khách hàng bị từ chối bởi các ngân hàng truyền thống.
   - URL: [Kaggle Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)

---

## 💻 3. Nguồn Tham Khảo Code Cụ Thể Trong Codebase Dự Án

Kế hoạch này kế thừa và mở rộng trực tiếp từ các module cốt lõi hiện có trong thư mục `src/credit_scoring/`:

1. **[src/credit_scoring/features.py](file:///home/myvh07/Project/P080-One_Last_Time/src/credit_scoring/features.py)**:
   - Quản lý danh sách đặc trưng `SERVING_RAW_FEATURES` (dòng 23–49), `ENGINEERED_FEATURES` (dòng 51–71).
   - Hàm biến đổi sinh đặc trưng `engineer_application_features()` (dòng 119–168).
   - Hàm tải và ghép nối dữ liệu `build_home_credit_features()` (dòng 422–467).
2. **[src/credit_scoring/ensemble_pipeline.py](file:///home/myvh07/Project/P080-One_Last_Time/src/credit_scoring/ensemble_pipeline.py)**:
   - Tiền xử lý dữ liệu `build_preprocessors()` với Imputation, One-Hot Encoding và StandardScaler (dòng 35–79).
   - Tối ưu hóa trọng số SLSQP Blending Ensemble `optimize_blending_weights()` (dòng 82–104).
   - Pipeline huấn luyện `train_ensemble_pipeline()` (dòng 106–268).
3. **[src/credit_scoring/metrics.py](file:///home/myvh07/Project/P080-One_Last_Time/src/credit_scoring/metrics.py)**:
   - Đánh giá khả năng phân tách: `roc_auc`, `gini`, `ks`, `pr_auc` (dòng 50–69).
   - Đánh giá hiệu chỉnh xác suất: `ece_10`, `brier`, `log_loss` (dòng 24–35, 63–65).
   - Đánh giá độ ổn định & công bằng: `population_stability_index()`, `fairness_report()` (dòng 72–191).
4. **[src/credit_scoring/training.py](file:///home/myvh07/Project/P080-One_Last_Time/src/credit_scoring/training.py)**:
   - Chiến lược chia tập dữ liệu stratified split `make_split()` và lưu vết cấu hình metadata.
5. **[scripts/train_ensemble.py](file:///home/myvh07/Project/P080-One_Last_Time/scripts/train_ensemble.py)**:
   - Script chạy offline và xuất kết quả huấn luyện ra file artifact `artifacts/models/ensemble_model.joblib`.

---

## 🔍 4. Thiết Kế Tập Đặc Trưng Dữ Liệu Thay Thế (Alternative Feature Taxonomy)

Để đảm bảo mô hình không bị rò rỉ bất kỳ dữ liệu tín dụng truyền thống nào, chúng ta tiến hành phân loại và lọc bỏ nghiêm ngặt các biến số:

### 4.1 Danh sách các biến BỊ LOẠI BỎ (Excluded Traditional Features)
- **Điểm tín dụng bên thứ 3**: `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`, `FE_EXT_SOURCE_*`.
- **Lịch sử credit bureau & dư nợ cũ**: Tất cả các bảng `bureau.csv`, `bureau_balance.csv`, `previous_application.csv`, `installments_payments.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv`.
- **Thông tin hạn mức nợ & tài chính truyền thống**: `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`, `FE_CREDIT_TO_INCOME`, `FE_ANNUITY_TO_INCOME`, `FE_PAYMENT_RATE`, `FE_CREDIT_TO_GOODS`.

### 4.2 Bảng Chi Tiết Tập Biến Thay Thế Được Sử Dụng (Alternative Feature Matrix)

| Tên biến (Feature Name) | Kiểu dữ liệu (Data Type) | Ví dụ mẫu (Sample Value) | Mô tả & Ý nghĩa tín dụng (Credit Rationale) | Phân loại |
| :--- | :--- | :--- | :--- | :--- |
| `DAYS_LAST_PHONE_CHANGE` | `Numeric (Int)` | `-1200` | Số ngày từ lần đổi số điện thoại gần nhất. Thể hiện độ ổn định cư trú và liên lạc. | Telco & Device |
| `FE_PHONE_CHANGE_YEARS` | `Numeric (Float)` | `3.28` | Thâm niên dùng số điện thoại quy đổi theo năm. | Telco & Device |
| `FE_PHONE_TO_AGE` | `Numeric (Float)` | `0.09` | Tỷ lệ thâm niên số điện thoại trên tổng tuổi đời. | Telco & Device |
| `FLAG_EMP_PHONE` | `Binary (Int)` | `1` | Khách hàng có cung cấp số điện thoại cơ quan/nơi làm việc hay không. | Telco & Device |
| `FLAG_WORK_PHONE` | `Binary (Int)` | `1` | Khách hàng có cung cấp số điện thoại bàn làm việc hay không. | Telco & Device |
| `FLAG_EMAIL` | `Binary (Int)` | `0` | Khách hàng có khai báo email cá nhân hay không (chỉ số số hóa). | Telco & Device |
| `NAME_HOUSING_TYPE` | `Categorical` | `"Rented apartment"` | Loại hình cư trú (ở nhà thuê, ở với cha mẹ, nhà riêng) thay thế cho tài sản thế chấp. | Demographics |
| `NAME_EDUCATION_TYPE` | `Categorical` | `"Higher education"` | Trình độ học vấn của khách hàng. | Demographics |
| `OCCUPATION_TYPE` | `Categorical` | `"Laborers"` | Loại hình nghề nghiệp chi tiết. | Demographics |
| `ORGANIZATION_TYPE` | `Categorical` | `"Business Entity Type 3"` | Loại hình tổ chức/lĩnh vực công tác. | Demographics |
| `FLAG_OWN_CAR` | `Categorical` | `"Y"` | Tỷ lệ sở hữu phương tiện cá nhân (ô tô). | Assets |
| `FLAG_OWN_REALTY` | `Categorical` | `"Y"` | Tỷ lệ sở hữu bất động sản/nhà ở. | Assets |
| `CNT_CHILDREN` | `Numeric (Int)` | `1` | Số lượng con cái phụ thuộc. | Demographics |
| `CNT_FAM_MEMBERS` | `Numeric (Float)` | `3.0` | Tổng số thành viên trong gia đình. | Demographics |
| `FE_INCOME_PER_PERSON` | `Numeric (Float)` | `60000.0` | Thu nhập bình quân chia theo đầu người trong gia đình. | Socio-Economic |
| `FE_INCOME_PER_CHILD` | `Numeric (Float)` | `90000.0` | Thu nhập bình quân chia theo số con phụ thuộc. | Socio-Economic |
| `FE_AGE_YEARS` | `Numeric (Float)` | `35.0` | Tuổi quy đổi theo năm. | Demographics |
| `FE_EMPLOYMENT_YEARS` | `Numeric (Float)` | `5.0` | Thâm niên làm việc quy đổi theo năm. | Demographics |
| `FE_EMPLOYMENT_TO_AGE` | `Numeric (Float)` | `0.14` | Tỷ lệ thời gian làm việc trên tổng tuổi đời. | Demographics |
| `OBS_30_CNT_SOCIAL_CIRCLE` | `Numeric (Float)` | `2.0` | Số lượng người trong mạng lưới quan hệ 30 ngày được quan sát. | Social Circle |
| `DEF_30_CNT_SOCIAL_CIRCLE` | `Numeric (Float)` | `0.0` | Số lượng người trong mạng lưới quan hệ 30 ngày bị nợ quá hạn. | Social Circle |
| `OBS_60_CNT_SOCIAL_CIRCLE` | `Numeric (Float)` | `2.0` | Số lượng người trong mạng lưới quan hệ 60 ngày được quan sát. | Social Circle |
| `DEF_60_CNT_SOCIAL_CIRCLE` | `Numeric (Float)` | `0.0` | Số lượng người trong mạng lưới quan hệ 60 ngày bị nợ quá hạn. | Social Circle |
| `digital_behavior` | `Categorical` | `"High_Mobile_User"` | Hành vi tương tác và thói quen sử dụng nền tảng di động số. | Digital Behavior |
| `engagement_score` | `Numeric (Float)` | `0.85` | Điểm số mức độ gắn kết với dịch vụ số. | Digital Behavior |

---

## 🛠️ 5. Các Bước Triển Khai Cụ Thể (Proposed Implementation Plan)

### Component 1: Định nghĩa Đặc trưng & Data Pipeline

#### [MODIFY] [features.py](file:///home/myvh07/Project/P080-One_Last_Time/src/credit_scoring/features.py)
- Thêm biến hằng số `ALTERNATIVE_ONLY_RAW_FEATURES` chứa danh sách 100% đặc trưng thay thế thuần túy.
- Thêm biến hằng số `ALTERNATIVE_ONLY_ENGINEERED_FEATURES` (loại bỏ các biến tỷ lệ tài chính như `FE_CREDIT_TO_INCOME`).
- Tạo hàm `build_alternative_only_features(data_dir, sample_size=None)` để xuất DataFrame chỉ gồm các đặc trưng thay thế + `SK_ID_CURR` + `TARGET`.

### Component 2: Pipeline Huấn Luyện & Mô Hình hóa

#### [NEW] [alternative_pipeline.py](file:///home/myvh07/Project/P080-One_Last_Time/src/credit_scoring/alternative_pipeline.py)
- Triển khai hàm `train_alternative_pipeline()` huấn luyện 4 mô hình độc lập:
  1. `LogisticRegression` (Baseline Scorecard với Imputer + OneHot + Scaler).
  2. `LightGBM` (Tree Champion).
  3. `XGBoost`.
  4. `CatBoost`.
- Tích hợp hàm `optimize_blending_weights()` để tìm trọng số tối ưu tạo ra **Alternative Ensemble Model**.
- Áp dụng `PlattCalibrator` để hiệu chỉnh xác suất rủi ro về đúng phân phối thực tế.

### Component 3: Scripts Chạy Offline & Benchmark So Sánh

#### [NEW] [train_alternative_model.py](file:///home/myvh07/Project/P080-One_Last_Time/scripts/train_alternative_model.py)
- Script chạy huấn luyện mô hình thay thế từ dữ liệu raw, xuất file artifact mô hình tại `artifacts/models/alternative_only_model.joblib`.

#### [NEW] [compare_models.py](file:///home/myvh07/Project/P080-One_Last_Time/scripts/compare_models.py)
- Script so sánh đối đầu 3 mô hình trên tập Test:
  1. **Traditional-Only Model** (Chỉ dùng Bureau + `EXT_SOURCE` + Khoản vay).
  2. **Alternative-Only Model** (Chỉ dùng Dữ liệu Thay thế).
  3. **Hybrid Champion Model** (Mô hình kết hợp hiện tại).
- Báo cáo chỉ số **Incremental Lift** về ROC-AUC, Gini, KS, PR-AUC, ECE.

### Component 4: Kiểm Thử Độc Lập & Kiểm Soát Rò Rỉ Dữ Liệu (Unit Tests)

#### [NEW] [test_alternative_model.py](file:///home/myvh07/Project/P080-One_Last_Time/tests/test_alternative_model.py)
- Viết test kiểm tra đảm bảo `build_alternative_only_features()` không chứa bất kỳ cột rò rỉ nào (`EXT_SOURCE_*`, `BUREAU_*`, `PREV_*`, `AMT_CREDIT`, `AMT_ANNUITY`).
- Kiểm tra tính toàn vẹn của mô hình khi chạy `predict_proba()`.

---

## 📊 6. Kế Hoạch Kiểm Thử & Đánh Giá (Verification & Audit Plan)

### 6.1 Kiểm thử tự động (Automated Verification)
Chạy bộ kiểm thử pytest để đảm bảo code hoạt động chính xác và không bị đứt gãy:
```bash
uv run pytest tests/test_alternative_model.py
uv run ruff check src scripts tests
```

### 6.2 Đánh giá hiệu năng và chỉ số Lift kỳ vọng

Mô hình sẽ được đánh giá qua bộ 8 chỉ số chuẩn mực trong [metrics.py](file:///home/myvh07/Project/P080-One_Last_Time/src/credit_scoring/metrics.py):

| Thước đo (Metric) | Mô hình Truyền thống (Dự kiến) | Mô hình Thay thế (Mục tiêu) | Mô hình Kết hợp Hybrid | Ý nghĩa Đánh giá |
| :--- | :--- | :--- | :--- | :--- |
| **ROC-AUC** | $\sim 0.740$ | $\mathbf{\ge 0.680}$ | $\sim 0.770$ | Đánh giá khả năng phân biệt rủi ro tổng quát trên nhóm Thin-file. |
| **Gini Index** | $\sim 48.0\%$ | $\mathbf{\ge 36.0\%}$ | $\sim 54.0\%$ | Khả năng phân tách theo tiêu chuẩn Scorecard ngân hàng ($2 \times AUC - 1$). |
| **KS Statistic** | $\sim 36.0\%$ | $\mathbf{\ge 28.0\%}$ | $\sim 42.0\%$ | Khoảng cách phân tách tối đa giữa khách hàng tốt và nợ xấu. |
| **PR-AUC** | $\sim 0.220$ | $\mathbf{\ge 0.160}$ | $\sim 0.260$ | Độ chính xác nhận diện nợ xấu trên dữ liệu mất cân bằng ($8\%$). |
| **ECE (Expected Calibration)**| $< 0.020$ | $\mathbf{< 0.020}$ | $< 0.015$ | Mức độ tin cậy của xác suất rủi ro dự báo. |
| **PSI (Stability Index)** | $< 0.10$ | $\mathbf{< 0.10}$ | $< 0.10$ | Độ ổn định phân phối xác suất rủi ro. |

### 6.3 Chẩn đoán Tính Công Bằng (Fairness Audit)
Thực hiện chạy `fairness_report()` đối với biến giới tính `CODE_GENDER` trên mô hình Alternative-Only để đảm bảo các đặc trưng cư trú, nghề nghiệp và di động không gây ra sự phân biệt đối xử bất hợp lý giữa Nam và Nữ ($FPR\_gap < 0.05$).

---

## ⚠️ 7. Giới Hạn & Giả Định Nghiên Cứu (Limitations & Assumptions)

1. **Giới hạn về dữ liệu**: Dữ liệu thay thế có độ nhiễu cao hơn và tỷ lệ biến đổi theo thời gian nhanh hơn dữ liệu tín dụng CIC truyền thống.
2. **Giả định về bài toán Serving**: Đối với khách hàng Thin-file, nếu `EXT_SOURCE_2` bị missing, hệ thống Serving sẽ tự động chuyển hướng qua dự báo bằng `Alternative-Only Model`.
