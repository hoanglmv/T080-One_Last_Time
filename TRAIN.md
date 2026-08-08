# 📖 Hướng Dẫn Huấn Luyện & So Sánh Mô Hình (Training & Evaluation Guide)

Tài liệu này hướng dẫn chi tiết cách chạy script huấn luyện linh hoạt `run/train.py`, xem so sánh các mô hình bằng `run/compare.ipynb` và xuất tệp nộp bài cho Kaggle.

---

## 🚀 1. Hướng Dẫn Nhanh (Quick Start)

### 🔹 Train Mô Hình LightGBM Mặc Định:
```bash
uv run python run/train.py --model lightgbm
```

### 🔹 Train Mô Hình Ensemble 3 Mô Hình (LightGBM + XGBoost + CatBoost):
```bash
uv run python run/train.py --model ensemble3
```

### 🔹 Train Mô Hình Ensemble 4 Mô Hình (LightGBM + XGBoost + CatBoost + Logistic Regression):
```bash
uv run python run/train.py --model ensemble4
```

---

## 🛠️ 2. Các Cờ Lệnh Tùy Chọn Trong `run/train.py`

| Cờ Lệnh (Argument) | Tùy Chọn (Choices) | Mặc Định | Mô Tả |
| :--- | :--- | :---: | :--- |
| `--model` | `lightgbm`, `xgboost`, `catboost`, `logistic_regression`, `ensemble3`, `ensemble4` | `lightgbm` | Chọn kiến trúc mô hình đơn lẻ hoặc mô hình Ensemble. |
| `--feature-set` | `serving`, `full` | `serving` | `serving`: Bộ 22 biến core (Tối ưu tốc độ API <50ms).<br>`full`: Bộ biến mở rộng đầy đủ. |
| `--sample-size` | `NUMBER` (VD: `50000`) | `None` | Giới hạn số bản ghi train để thử nghiệm nhanh. |
| `--seed` | `INTEGER` | `42` | Random seed đảm bảo tính tái lập (Reproducibility). |
| `--output-dir` | `PATH` | `artifacts/models` | Thư mục lưu tệp mô hình `.joblib`. |

---

## 💡 3. Các Ví Dụ Chạy Huấn Luyện Thực Tế

### Ví Dụ 1: Huấn luyện mô hình XGBoost đơn lẻ với bộ biến Full
```bash
uv run python run/train.py --model xgboost --feature-set full
```

### Ví Dụ 2: Huấn luyện mô hình CatBoost đơn lẻ
```bash
uv run python run/train.py --model catboost
```

### Ví Dụ 3: Huấn luyện Baseline Logistic Regression
```bash
uv run python run/train.py --model logistic_regression
```

### Ví Dụ 4: Huấn luyện Ensemble 3 mô hình trên 50,000 bản ghi mẫu
```bash
uv run python run/train.py --model ensemble3 --sample-size 50000
```

---

## 📊 4. Xem Bảng So Sánh Mô Hình Với `run/compare.ipynb`

Sau khi huấn luyện các mô hình, bạn mở notebook **[`run/compare.ipynb`](file:///home/myvh07/Project/P080-One_Last_Time/run/compare.ipynb)** để:
1. Tự động nạp toàn bộ các tệp mô hình `.joblib` đã lưu trong `artifacts/models/`.
2. Xem bảng so sánh 5 chỉ số chính: **ROC-AUC, PR-AUC, Gini Index, KS Statistic, ECE Calibration**.
3. Xem biểu đồ cột so sánh trực quan hiệu năng các mô hình.

---

## 📄 5. Xuất Tệp Nộp Bài Cho Kaggle (Kaggle Submission)

### Xuất tệp submission từ Mô Hình Ensemble:
```bash
uv run python scripts/generate_ensemble_submission.py
```
*(Kết quả sẽ được lưu tại `artifacts/submission_ensemble.csv`)*.

### Xuất tệp submission từ Mô Hình LightGBM Đơn Lẻ:
```bash
uv run python scripts/generate_kaggle_submission.py
```
*(Kết quả sẽ được lưu tại `artifacts/submission.csv`)*.
