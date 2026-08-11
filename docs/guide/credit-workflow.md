# Workflow xử lý vay tích hợp ML, XAI và Policy RAG

Workflow hiện thực hóa hai luồng hồ sơ có lịch sử tín dụng và chưa có lịch sử
tín dụng. Kết quả của hệ thống luôn là `system_recommendation`; cán bộ tín dụng
phải xác nhận quyết định cuối cùng qua API review.

## API

- `POST /api/v1/credit/workflow/applications`: tiếp nhận và xử lý hồ sơ.
- `POST /api/v1/credit/workflow/applications/{application_id}/review`: cán bộ
  tín dụng ra quyết định cuối cùng.

## Dữ liệu đầu vào

| Field | Type | Example | Vai trò trong workflow |
| --- | --- | --- | --- |
| `application_id` | `string \| null` | `APP-2026-001` | ID hồ sơ; hệ thống tự sinh nếu bỏ trống |
| `customer_reference` | `string` | `CRM-0001` | Đối soát khách hàng; audit chỉ lưu SHA-256 hash |
| `has_credit_history` | `boolean` | `false` | Chọn `traditional_credit` hoặc `alternative_credit` |
| `documents` | `list[string]` | `["identity", "income_proof"]` | Kiểm tra hồ sơ tối thiểu |
| `application` | `object` | `{"AMT_INCOME_TOTAL": 25000000}` | Feature input cho model đã huấn luyện |
| `applicant_age` | `integer` | `30` | Kiểm tra eligibility/policy |
| `monthly_income` | `number` | `25000000` | Kiểm tra DTI và khả năng trả nợ |
| `current_monthly_debt` | `number` | `1000000` | Tính tổng nghĩa vụ nợ hàng tháng |
| `requested_loan_amount` | `number` | `50000000` | Hạn mức đề xuất và policy check |
| `loan_term_months` | `integer` | `24` | Tính payment/DTI |
| `proposed_interest_rate` | `number` | `12.0` | Kiểm tra trần lãi suất/policy |
| `loan_purpose` | `string` | `Mua thiết bị gia đình` | RAG query và prohibited-purpose rules |
| `is_ekyc` | `boolean` | `true` | Áp dụng giới hạn tương ứng của policy engine |
| `top_k` | `integer` | `6` | Số reason codes trả về |

## Ví dụ nộp hồ sơ chưa có lịch sử tín dụng

```bash
curl -X POST http://localhost:8000/api/v1/credit/workflow/applications \
  -H 'Content-Type: application/json' \
  -d '{
    "customer_reference": "CRM-0001",
    "has_credit_history": false,
    "documents": ["identity", "income_proof"],
    "application": {
      "AMT_INCOME_TOTAL": 25000000,
      "AMT_CREDIT": 50000000,
      "NAME_INCOME_TYPE": "Working",
      "NAME_EDUCATION_TYPE": "Higher education"
    },
    "applicant_age": 30,
    "monthly_income": 25000000,
    "current_monthly_debt": 1000000,
    "requested_loan_amount": 50000000,
    "loan_term_months": 24,
    "proposed_interest_rate": 12.0,
    "loan_purpose": "Mua thiết bị gia đình",
    "is_ekyc": true
  }'
```

Với `has_credit_history=false`, các trường `EXT_SOURCE_1/2/3` bị loại trước khi
chấm điểm để tránh vô tình đưa bureau data vào luồng alternative-only.

## Ví dụ quyết định của cán bộ tín dụng

```bash
curl -X POST \
  http://localhost:8000/api/v1/credit/workflow/applications/APP-2026-001/review \
  -H 'Content-Type: application/json' \
  -d '{
    "decision": "APPROVE",
    "officer_id": "officer-01",
    "notes": "Đã đối chiếu hồ sơ gốc và policy evidence."
  }'
```

`REVIEW` chuyển hồ sơ về `NEEDS_INFORMATION` và bắt buộc có `notes`. Một hard
rule `REJECT` không thể bị API đổi trực tiếp thành `APPROVE`.

## Mapping 12 giai đoạn

1. Intake: nhận hồ sơ và tạo `application_id`.
2. Validation: kiểm tra `identity` và `income_proof`.
3. Routing: chọn `traditional_credit` hoặc `alternative_credit`.
4. Risk scoring: gọi model artifact và trả PD, score, risk band.
5. XAI: dùng reason codes/contribution do model bundle tạo.
6. Policy RAG: truy xuất chunk từ `docs/policies`.
7. Decision Engine: tổng hợp ML, data quality và hard rules.
8. Recommendation explanation: giải thích kèm nguồn policy đã truy xuất.
9. Officer review: giữ trạng thái `PENDING_REVIEW` đến khi có quyết định người.
10. Audit: lưu event JSONL và current-state projection trong `data/logs`.
11. Loan monitoring: sử dụng API feedback và portfolio hiện có sau giải ngân.
12. Feedback/retraining: sử dụng `FeedbackLearningService`; không tự deploy model
    chỉ dựa trên số lượng mẫu.

## Giới hạn

- `identity` và `income_proof` mới là document codes, chưa có OCR, eKYC provider
  hoặc chữ ký số.
- Policy RAG hiện là lexical retrieval trên tài liệu Markdown, chưa có vector DB,
  effective-date filtering hoặc quy trình phê duyệt văn bản pháp lý.
- `docs/policies/bank_internal_policy.md` là policy nội bộ của POC, không được
  xem là quy định chính thức của một ngân hàng thực tế.
- Workflow không tự động phê duyệt tín dụng. Production cần authentication,
  authorization, encryption, immutable audit storage và legal/model validation.
