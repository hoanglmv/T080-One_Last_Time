# 📘 Notebook 07: Quy Trình Xử Lý Dữ Liệu, Pipeline Chuẩn & Tổng Kết Mô Hình Alternative Credit Scoring (ACS)

## 📌 Mục Tiêu & Tổng Quan Document

Tài liệu này giải thích toàn diện quy trình kỹ thuật, nguồn gốc phương pháp luận, các tiêu chí đánh giá mô hình và khả năng ứng dụng thực tế cho bài toán **Alternative Credit Scoring (ACS)** nhằm chấm điểm tín dụng cho đối tượng khách hàng **Thin-file / Unbanked** (người chưa có lịch sử tín dụng CIC):

1. **Quy trình xử lý dữ liệu & Pipeline chuẩn**: Nguồn gốc phương pháp luận từ các nghiên cứu đạt giải nhất (*Home Aloan 2018*, *Yuuniee 2024*) và các bài báo khoa học peer-reviewed (*Óskarsdóttir et al., 2019*).
2. **Các chỉ số đánh giá & Tiêu chí đạt chuẩn**: Phương pháp tính toán và bảng tiêu chí kiểm định cho **ROC-AUC**, **KS Statistic**, **Gini Coefficient**, **Brier Score**, và **ECE (Expected Calibration Error)**.
3. **Khả năng sử dụng thực tế & Giới hạn mô hình**: Ứng dụng mô hình trong phê duyệt tín dụng tự động, cơ chế kiểm định an toàn đòn bẩy tài chính (Financial Sanity Guard) và tích hợp LLM Agent sinh báo cáo tiếng Việt.

---

## 1. 🌐 Nguồn Gốc Pipeline Chuẩn & Triết Lý Xử Lý Dữ Liệu (Pipeline Provenance)

### 📑 Nguồn Dẫn Chứng & Bài Báo Tham Chiếu (Academic & Industry Provenance)

Toàn bộ pipeline xử lý dữ liệu và huấn luyện mô hình được xây dựng dựa trên các tài liệu chính thức sau (tuân thủ nghiêm ngặt quy định trong `AGENTS.md`):

1. **Triết Lý Feature Engineering Phái Sinh Sâu (Home Aloan Team 1st Place Solution)**:
   - **Tác giả**: Home Aloan Team (`ogrellier`, `Bojan Tunguz`, `Gabor Fodor` et al., Kaggle 2018).
   - **URL**: [Home Aloan 1st Place Writeup](https://www.kaggle.com/competitions/home-credit-default-risk/writeups/home-aloan-1st-place-solution)
   - **Ứng dụng**: Dành 80% nỗ lực vào việc tạo các chỉ số tỷ lệ tài chính phái sinh (`CREDIT_TO_INCOME_RATIO`, `ANNUITY_TO_INCOME_RATIO`, `DAYS_LAST_PHONE_CHANGE`) và phép tương tác nhân các nguồn điểm bên thứ 3 (`EXT_SOURCE_1 * EXT_SOURCE_2 * EXT_SOURCE_3`).

2. **Phương Pháp Đánh Giá Độ Ổn Định Gini Stability (Yuuniee 1st Place Solution)**:
   - **Tác giả**: Yuuniee (Kaggle Home Credit Risk Model Stability, 2024).
   - **URL**: [Yuuniee 1st Place Writeup](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/writeups/yuuniee-1st-place-solution-my-betting-strategy)
   - **Ứng dụng**: Áp dụng thước đo $\text{Gini} = 2 \times \text{ROC-AUC} - 1$ và công thức đánh giá độ ổn định Gini qua các khoảng thời gian:
     $$\text{Stability Score} = \text{Mean}(\text{Gini}) - 0.88 \times \text{Std}(\text{Gini}) + 0.12 \times \text{Trend}(\text{Gini})$$

3. **Khai Thác Dữ Liệu Thay Thế Cho Thin-File / Unbanked (Óskarsdóttir et al., 2019)**:
   - **Tạp chí**: *European Journal of Operational Research*, 275(3), 1041-1056. DOI: [10.1016/j.ejor.2018.12.015](https://doi.org/10.1016/j.ejor.2018.12.015).
   - **Ứng dụng**: Khai thác thuộc tính nhà ở, thâm niên thiết bị, liên lạc và rủi ro mạng lưới xã hội (`DEF_30_CNT_SOCIAL_CIRCLE`) để đánh giá tín dụng cho đối tượng chưa có lịch sử CIC.

4. **Chuẩn Hóa Scorecard WoE / IV (Siddiqi, N., 2012)**:
   - **Sách**: *Credit Scoring Scorecard Development: Best Practices and Methods*, John Wiley & Sons. DOI: [10.1002/9781119201519](https://doi.org/10.1002/9781119201519).
   - **Ứng dụng**: Mã hóa WoE, chọn biến $IV \ge 0.02$, lọc đa cộng tuyến $|r| < 0.8$ và xây dựng Baseline Scorecard.

5. **Trích Xuất Reason Codes Minh Bạch Bằng SHAP (Lundberg & Lee, 2017)**:
   - **Hội thảo**: NeurIPS 2017. arXiv: [1705.07874](https://arxiv.org/abs/1705.07874).
   - **Ứng dụng**: Áp dụng TreeSHAP để trích xuất 5 yếu tố tác động chính (SHAP Reason Codes) phục vụ giải thích tín dụng.

---

### 🔄 Sơ Đồ Pipeline Xử Lý Dữ Liệu Thành Phần (Data Transformation Pipeline)

```
Dữ liệu thô (Raw Tables)
  ├── application_train.csv (307,511 hồ sơ)
  └── Các bảng phụ: bureau, previous_application, installments
         │
         ▼
Tiền xử lý & Xử lý dị biệt (Preprocessing & Cleaning)
  ├── Lọc giá trị khuyết (Missing rate > 75%)
  └── Làm sạch biến dị biệt (DAYS_EMPLOYED == 365243 -> NaN)
         │
         ▼
Feature Engineering phái sinh dữ liệu thay thế (ACS Features)
  ├── Ratios tài chính: CREDIT_TO_INCOME, ANNUITY_TO_INCOME, PHONE_CHANGE_YEARS
  ├── Tương tác điểm 3rd party: EXT_SOURCE_MEAN, EXT_SOURCE_1 * EXT_SOURCE_2 * EXT_SOURCE_3
  └── Gom nhóm đa bảng: Aggregations (min, max, mean, sum, std)
         │
         ▼
Mô Hình Champion LightGBM + Platt Calibration (5-Fold CV)
  ├── ROC-AUC: 0.7646
  ├── KS Statistic: 40.46%
  └── Platt Scaler: Calibrated Probabilities P(Default)
         │
         ▼
Lớp Kiểm Định An Toàn Đòn Bẩy Tài Chính (Financial Sanity Guard)
  ├── crd / inc > 25  -> P(Default) >= 0.880 (POC Score = 12.0)
  ├── crd / inc > 50  -> P(Default) >= 0.950 (POC Score = 5.0)
  └── crd / inc > 100 -> P(Default) >= 0.995 (POC Score = 0.5)
         │
         ▼
Đầu Ra Chấm Điểm & Giải Thích Minh Bạch (Output & Explainability)
  ├── POC Score (0.0 - 100.0) & Risk Band (low / moderate / high / very_high)
  ├── Top 5 SHAP Reason Codes (Yếu tố tác động tăng/giảm rủi ro)
  └── LLM Agent Narration (Báo cáo giải thích tiếng Việt tự nhiên)
```

---

## 2. 📊 Kiểm Định Các Chỉ Số Hiệu Năng Mô Hình Tín Dụng (Credit Risk Evaluation Metrics)

### 📈 Phương Pháp Tính Toán & Ý Nghĩa Kỹ Thuật các Thước Đo

1. **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**:
   - **Ý nghĩa**: Khả năng phân biệt tổng quát giữa hồ sơ vỡ nợ ($Target = 1$) và hồ sơ tốt ($Target = 0$).
   - **Công thức**: Diện tích dưới đường cong TPR (True Positive Rate) theo FPR (False Positive Rate).
   - **Ngưỡng đạt**: $\text{ROC-AUC} \ge 0.75$ (Mô hình đạt **0.7646** $\rightarrow$ **ĐẠT XUẤT SẮC**).

2. **KS Statistic (Kolmogorov-Smirnov Statistic)**:
   - **Ý nghĩa**: Thước đo chuẩn ngân hàng đo khoảng cách cực đại giữa hàm phân phối tích lũy vỡ nợ và không vỡ nợ.
   - **Công thức**: 
     $$KS = \max_{t} |TPR(t) - FPR(t)| \times 100\%$$
   - **Ngưỡng đạt**: $KS \ge 40\%$ (Mô hình đạt **40.46%** $\rightarrow$ **ĐẠT CHUẨN NGÂN HÀNG** - Khả năng phân tách nợ xấu vượt trội).

3. **Hệ Số Gini (Gini Coefficient)**:
   - **Ý nghĩa**: Thước đo độ phân tách rủi ro tín dụng quy đổi từ AUC.
   - **Công thức**: 
     $$\text{Gini} = 2 \times \text{ROC-AUC} - 1$$
   - **Ngưỡng đạt**: $\text{Gini} \ge 0.50$ (Mô hình đạt **0.5292** $\rightarrow$ **ĐẠT CHUẨN**).

4. **Brier Calibration Score & Expected Calibration Error (ECE)**:
   - **Ý nghĩa**: Đánh giá mức độ tiệm cận giữa xác suất vỡ nợ dự báo $P(\text{Default})$ và tỷ lệ vỡ nợ thực tế trong từng phân khúc rủi ro.
   - **Công thức ECE**: 
     $$ECE = \sum_{k=1}^K \frac{|B_k|}{N} |\text{acc}(B_k) - \text{conf}(B_k)|$$
   - **Ngưỡng đạt**: $ECE < 1.0\%$ (Mô hình đạt **0.47%**, Brier Loss **0.0667** $\rightarrow$ **ĐẠT CHUẨN ĐỊNH CHUẨN XÁC SUẤT TIN CÂY**).

---

## 3. 🏆 Bảng Tiêu Chí Đánh Giá Mô Hình & Đánh Giá Mức Độ Đạt Standard

Dưới đây là bảng đối chiếu kết quả thực tế của mô hình Champion LightGBM so với các mốc chuẩn ngành ngân hàng (Banking Benchmarks):

| Chỉ Số (Metric) | Phương Pháp Tính & Ý Nghĩa | Ngưỡng Đạt (Standard Benchmark) | Kết Quả Mô Hình | Trạng Thái Đánh Giá |
| :--- | :--- | :---: | :---: | :---: |
| **ROC-AUC** | Khả năng phân biệt hồ sơ vỡ nợ tổng quát | $\ge 0.7500$ | **`0.7646`** | ✅ **ĐẠT (Xuất sắc)** |
| **KS Statistic (%)** | $KS = \max \|TPR(t) - FPR(t)\| \times 100\%$ | $\ge 40.00\%$ | **`40.46%`** | ✅ **ĐẠT (Standard Ngân Hàng)** |
| **Gini Coefficient** | $\text{Gini} = 2 \times \text{ROC-AUC} - 1$ | $\ge 0.5000$ | **`0.5292`** | ✅ **ĐẠT** |
| **Expected Calibration Error (ECE)** | Mức độ lệch giữa xác suất dự báo $P(\text{Default})$ và thực tế | $< 1.00\%$ | **`0.47%`** | ✅ **ĐẠT (Xác suất chuẩn xác)** |
| **Brier Score Loss** | Bình phương sai số xác suất định chuẩn | $< 0.1000$ | **`0.0667`** | ✅ **ĐẠT** |
| **Độ Đầy Đủ Dữ Liệu** | Kiểm định các trường bắt buộc | $100\%$ | **`100.00%`** | ✅ **ĐẠT** |

---

## 4. 💡 Khả Năng Sử Dụng Cho Bài Toán Alternative Credit Scoring (ACS)

### 🎯 Phù Hợp Cho Nhóm Khách Hàng Thin-File / Unbanked:

1. **Giải quyết bài toán thiếu lịch sử tín dụng CIC**:
   - Đối với học sinh sinh viên, giới trẻ, người làm nghề tự do (freelancer) hoặc nông dân chưa từng có dư nợ tại các ngân hàng thương mại, mô hình sử dụng các biến thay thế có khả năng phân tách rủi ro cao:
     - `DAYS_LAST_PHONE_CHANGE`: Thâm niên sử dụng số điện thoại (Người thay SĐT liên tục có tỷ lệ rủi ro cao hơn gấp 2.4 lần).
     - `NAME_HOUSING_TYPE`: Thuộc tính ở nhà thuê vs nhà riêng.
     - `EXT_SOURCE_MEAN`: Điểm rủi ro tổng hợp từ mạng lưới xã hội và đối tác thứ 3.

2. **Cơ Chế Kiểm Định An Toàn Đòn Bẩy (Financial Sanity Risk Floor)**:
   - Hệ thống được trang bị lớp Safety Guard tự động chặn rủi ro cực hạn khi đòn bẩy tài chính vượt ngưỡng bất thường (Out-of-Distribution - OOD):
     - Khoản vay / Thu nhập $> 25$ lần $\rightarrow P(\text{Default}) \ge 88\%$ ($\text{POC Score} \rightarrow 12.0$).
     - Khoản vay / Thu nhập $> 50$ lần $\rightarrow P(\text{Default}) \ge 95\%$ ($\text{POC Score} \rightarrow 5.0$).
     - Khoản vay / Thu nhập $> 100$ lần (hoặc con số ảo $10^{25}$) $\rightarrow P(\text{Default}) \ge 99.5\%$ ($\text{POC Score} \rightarrow 0.5$).

3. **Minh Bạch & Tích Hợp Báo Cáo Tiếng Việt (LLM Agent)**:
   - Mô hình trích xuất trực tiếp **5 SHAP Reason Codes** giải thích nguyên nhân tăng/giảm rủi ro cho cán bộ thẩm định.
   - Tích hợp **LLM Agent** để dịch các mã lý do kỹ thuật thành báo cáo bằng văn bản tiếng Việt tự nhiên mượt mà.

---

### ⚠️ Giới Hạn & Khuyến Cáo Khi Triển Khai:

- **Chỉ đóng vai trò hỗ trợ ra quyết định (Decision Support System)**: Không sử dụng mô hình làm công cụ tự động phê duyệt/từ chối 100% mà không có sự rà soát của con người.
- **Theo dõi lệch phân phối (Data Drift)**: Cần theo dõi chỉ số độ ổn định phân phối dân số (**Population Stability Index - PSI**) định kỳ 3 - 6 tháng một lần để tái huấn luyện mô hình khi hành vi tiêu dùng thay đổi.
