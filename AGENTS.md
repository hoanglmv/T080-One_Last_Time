# AGENTS.md

## Tổng quan dự án

Dự án nghiên cứu và phát triển mô hình Alternative Credit Scoring nhằm đánh giá khả năng tín dụng của khách hàng dựa trên dữ liệu thay thế.

## Mục tiêu

- Nghiên cứu các phương pháp Alternative Credit Scoring.
- Khảo sát bài báo khoa học và dự án thực tế.
- Xây dựng mô hình, dữ liệu, và pipeline phù hợp cho thị trường Việt Nam.
- Huấn luyện mô hình dự đoán rủi ro tín dụng.
- Đánh giá khả năng giải thích, độ công bằng và hiệu quả của mô hình.

## Nguồn dữ liệu đang nghiên cứu
- Home Credit Default Risk
- Vietnam Bank Churn Dataset
- Dữ liệu giao dịch, viễn thông, hành vi số và nhân khẩu học

## Nguyên tắc làm việc

- Không tự suy đoán nguồn gốc hoặc ý nghĩa của dữ liệu.
- Khi đưa ra nhận định học thuật, phải kèm nguồn dẫn chứng.
- Ưu tiên bài báo khoa học, tài liệu chính thức và dự án có mã nguồn.
- Phân biệt rõ dữ liệu credit scoring, fraud detection và churn.
- Ghi rõ giới hạn khi sử dụng dataset thay thế.
- Không xem dữ liệu fraud detection là dữ liệu credit risk nếu chưa giải thích phương pháp chuyển đổi.

## Yêu cầu về nguồn tham khảo

- Ưu tiên bài báo peer-reviewed, arXiv, IEEE, Springer và ScienceDirect.
- Với dataset, ưu tiên nguồn phát hành chính thức hoặc trang cuộc thi gốc.
- Mọi đường dẫn phải được kiểm tra còn truy cập được.
- Ghi rõ tên bài báo, tác giả, năm xuất bản và DOI hoặc URL.

## Quy trình Machine Learning

1. Xác định biến mục tiêu.
2. Kiểm tra chất lượng và mức độ phù hợp của dữ liệu.
3. Ngăn chặn data leakage.
4. Chia dữ liệu theo thời gian nếu dữ liệu có timestamp.
5. Xây dựng baseline bằng Logistic Regression.
6. So sánh với Random Forest, XGBoost hoặc LightGBM.
7. Đánh giá bằng ROC-AUC, PR-AUC, KS, Gini và calibration.
8. Giải thích mô hình bằng SHAP hoặc phương pháp tương đương.
9. Phân tích fairness giữa các nhóm khách hàng.

## Quy ước đầu ra

- Giải thích bằng tiếng Việt.
- Tên biến, mã nguồn và thuật ngữ kỹ thuật giữ bằng tiếng Anh.
- Bảng dữ liệu phải có mô tả trường, kiểu dữ liệu, ví dụ và vai trò trong mô hình.
- Không chỉnh sửa hoặc xóa dữ liệu gốc.
- Các bước xử lý dữ liệu phải có khả năng tái lập.

## Điều kiện hoàn thành

Một công việc chỉ được xem là hoàn thành khi:

- Có kết quả đầu ra rõ ràng.
- Có nguồn dẫn chứng cho các kết luận quan trọng.
- Mã nguồn chạy được.
- Các kiểm thử hoặc bước kiểm tra liên quan đã được thực hiện.
- Giới hạn và giả định được nêu rõ.