# Model Card — Home Credit Payment Difficulty POC

## Phạm vi

- Model: `logistic_regression`
- Version: `hc-poc-20260808T111014Z-52e96b89`
- Feature set: `alternative_only`
- Split: `stratified_random_70_15_15_no_temporal_claim`
- Số mẫu: `50000`

## Kết quả trên test holdout

| Metric | Giá trị |
|---|---:|
| ROC-AUC | 0.665773 |
| PR-AUC | 0.145995 |
| KS | 0.249145 |
| Gini | 0.331546 |
| Brier | 0.072022 |
| ECE (10 bins) | 0.004620 |

## Giới hạn bắt buộc

1. `TARGET` là payment difficulty theo định nghĩa ẩn danh của cuộc thi, không phải định nghĩa PD pháp lý phổ quát.
2. Home Credit 2018 không có application timestamp; split hiện tại không chứng minh temporal stability.
3. Artifact này là POC nghiên cứu, không phải policy phê duyệt/từ chối khoản vay.
4. LLM, nếu bật, chỉ diễn đạt reason codes; không được thay đổi xác suất, risk band hoặc kết luận model.
