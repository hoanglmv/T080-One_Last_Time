# Model Card — Home Credit Payment Difficulty POC

## Phạm vi

- Model: `lightgbm`
- Version: `hc-poc-20260805T162444Z-52e96b89`
- Feature set: `full`
- Split: `stratified_random_70_15_15_no_temporal_claim`
- Số mẫu: `50000`

## Kết quả trên test holdout

| Metric | Giá trị |
|---|---:|
| ROC-AUC | 0.764619 |
| PR-AUC | 0.264726 |
| KS | 0.404595 |
| Gini | 0.529237 |
| Brier | 0.066668 |
| ECE (10 bins) | 0.004716 |

## Giới hạn bắt buộc

1. `TARGET` là payment difficulty theo định nghĩa ẩn danh của cuộc thi, không phải định nghĩa PD pháp lý phổ quát.
2. Home Credit 2018 không có application timestamp; split hiện tại không chứng minh temporal stability.
3. Artifact này là POC nghiên cứu, không phải policy phê duyệt/từ chối khoản vay.
4. LLM, nếu bật, chỉ diễn đạt reason codes; không được thay đổi xác suất, risk band hoặc kết luận model.
