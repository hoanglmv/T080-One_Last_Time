# 🏦 Quy Định Khung Cấp Tín Dụng & Thẩm Định Rủi Ro Nội Bộ Ngân Hàng

> **Mã tài liệu**: `bank_internal_policy.md`  
> **Cơ quan ban hành**: Khối Quản trị Rủi ro Tín dụng & Ủy ban Thẩm định tín dụng.  
> **Phạm vi áp dụng**: Đánh giá và phê duyệt các khoản vay tiêu dùng, cho vay tín chấp cá nhân (bao gồm luồng thẩm định tự động qua mô hình AI/Alternative Data).

---

## 📊 1. Ma Trận Phê Duyệt Tín Dụng Theo Phân Nhóm Rủi Ro (Risk Band Approval Matrix)

Mô hình Credit Scoring phân loại khách hàng thành 4 nhóm rủi ro (Risk Band). Mọi đề xuất cấp tín dụng do hệ thống/LLM tạo ra phải tuân thủ nghiêm ngặt các hạn mức và mức lãi suất quy định dưới đây:

| Phân nhóm rủi ro (Risk Band) | Điểm tín dụng (POC Score) | Hạn mức vay tối đa (Max Loan Amount) | Trần lãi suất trong hạn (%/năm) | Tỷ lệ DTI tối đa (Max DTI) | Yêu cầu hồ sơ chứng minh |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Low Risk** | 700 - 1000 | **200.000.000 VNĐ** | 12.0% - 14.5%/năm | 45% | eKYC + Dữ liệu di động/định danh |
| **Moderate Risk** | 600 - 699 | **100.000.000 VNĐ** | 15.0% - 17.5%/năm | 45% | eKYC + Thâm niên số di động >= 12 tháng |
| **High Risk** | 500 - 599 | **30.000.000 VNĐ** | 18.0% - 20.0%/năm | 40% | eKYC + Xác thực nơi ở & công việc |
| **Very High Risk** | 0 - 499 | **0 VNĐ (Từ chối)** | Không áp dụng | 0% | **Từ chối cấp tín dụng (REJECT)** |

---

## 🛑 2. Điều Kiện Cứng Về Độ Tuổi & Khả Năng Nợ (Hard Eligibility Criteria)

1. **Độ tuổi người vay (Age Limits)**:
   - Độ tuổi tối thiểu: Từ **đủ 18 tuổi** tại thời điểm nộp hồ sơ.
   - Độ tuổi tối đa: **Không quá 60 tuổi** tính đến thời điểm đáo hạn khoản vay.
   - *Quy tắc kiểm tra*: Nếu `age < 18` hoặc `age > 60` -> **Hard Decline (Từ chối tự động)**.

2. **Chỉ số trả nợ trên thu nhập (Debt-to-Income Ratio - DTI)**:
   - Công thức tính: `DTI = (Tổng nghĩa vụ trả nợ hàng tháng hiện tại + Tiền gốc & lãi khoản vay mới) / Thu nhập hàng tháng`.
   - Ngưỡng tối đa cho phép: `DTI <= 45%`.
   - *Quy tắc kiểm tra*: Nếu `DTI > 45%` -> **Tự động giảm số tiền đề xuất hoặc điều chỉnh tăng thời gian vay để hạ DTI về <= 45%**.

3. **Hạn mức vay nhỏ qua phương thức điện tử (e-KYC Small Loan Cap)**:
   - Theo quy định pháp luật và chính sách nội bộ ngân hàng, các khoản vay duyệt tự động hoàn toàn qua e-KYC không thế chấp tài sản có hạn mức tối đa **100.000.000 VNĐ**.
   - *Quy tắc kiểm tra*: Nếu `is_ekyc == True` và `requested_amount > 100_000_000` -> **Giảm số tiền đề xuất về tối đa 100.000.000 VNĐ**.

---

## 🚫 3. Danh Mục Mục Đích Cho Vay Bị Cấm Nội Bộ (Prohibited Purpose Checklist)

Hệ thống LLM tuyệt đối **KHÔNG** đề xuất cấp tín dụng đối với các mục đích vay thuộc danh mục sau:

1. **Kinh doanh tài chính rủi ro cao**: Đầu tư chứng khoán, giao dịch chứng khoán phái sinh, giao dịch tiền ảo (Cryptocurrency/Bitcoin), ngoại hối (Forex) tự do.
2. **Cờ bạc, cá cược**: Tham gia các hình thức cá độ, cờ bạc, trò chơi có thưởng trực tuyến hoặc trực tiếp.
3. **Đảo nợ trái phép**: Đảo nợ khoản vay cũ tại ngân hàng hoặc thanh toán nợ xấu tín dụng đen.
4. **Mua bất động sản rủi ro cao / Đất nông nghiệp chưa đủ pháp lý**.
5. **Kinh doanh hàng cấm, vũ khí, chất cấm** theo Luật Đầu tư Việt Nam.

---

## 🧾 4. Quy Định Về Tính Minh Bạch & Cảnh Báo Rủi Ro (Transparency & Disclosure)

Mọi bản đề xuất khoản vay (Loan Proposal) do LLM hoặc nhân viên tư vấn đưa ra cho khách hàng **bắt buộc phải bao gồm**:
1. **Lãi suất rõ ràng**: Lãi suất theo năm (%/năm) và phương pháp tính lãi (theo dư nợ giảm dần).
2. **Lịch trả nợ ước tính**: Số tiền gốc + lãi trả hàng tháng.
3. **Tổng chi phí khoản vay**: Tổng số tiền phải trả đến khi đáo hạn.
4. **Khuyên đọc kỹ hợp đồng**: Cảnh báo khách hàng cân nhắc khả năng tài chính cá nhân trước khi ký kết.
