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

| Cờ Lệnh (Argument) | Tùy Chọn / Kiểu | Mặc Định | Mô Tả |
| :--- | :--- | :---: | :--- |
| `--model` | `lightgbm`, `xgboost`, `catboost`, `logistic_regression`, `ensemble3`, `ensemble4` | `lightgbm` | Chọn kiến trúc mô hình đơn lẻ hoặc mô hình Ensemble. |
| `--feature-set` | `serving`, `full` | `serving` | `serving`: Bộ 22 biến core (Tối ưu tốc độ API <50ms).<br>`full`: Bộ biến mở rộng đầy đủ. |
| `--sample-size` | `INTEGER` (VD: `50000`) | `None` | Giới hạn số bản ghi train để thử nghiệm nhanh. |
| `--seed` | `INTEGER` | `42` | Random seed đảm bảo tính tái lập (Reproducibility). |
| `--output-dir` | `PATH` | `artifacts/models` | Thư mục lưu tệp mô hình `.joblib`. |
| `--n-estimators` | `INTEGER` | `300 / 250` | Số lượng cây / vòng lặp boosting. |
| `--learning-rate` | `FLOAT` (VD: `0.05`) | `0.03 / 0.04` | Tốc độ học (Learning rate). |
| `--max-depth` | `INTEGER` (VD: `8`) | `5 / 6` | Độ sâu tối đa của cây (XGBoost / CatBoost). |
| `--num-leaves` | `INTEGER` (VD: `63`) | `31` | Số lá tối đa của cây (LightGBM). |
| `--subsample` | `FLOAT` (VD: `0.85`) | `0.8` | Tỷ lệ lấy mẫu hàng (Row subsample ratio). |
| `--colsample-bytree` | `FLOAT` (VD: `0.85`) | `0.8` | Tỷ lệ lấy mẫu cột đặc trưng (Column subsample ratio). |
| `--c-reg` | `FLOAT` (VD: `1.0`) | `0.1` | Hệ số nghịch đảo C phạt cho Logistic Regression. |

---

## 💡 3. Các Ví Dụ Chạy Huấn Luyện Tùy Chỉnh Siêu Tham Số (Hyperparameter Customization)

### Ví Dụ 1: Tùy chỉnh Learning Rate & Số lượng cây cho XGBoost
```bash
uv run python run/train.py --model xgboost --n-estimators 500 --learning-rate 0.02 --max-depth 6
```

### Ví Dụ 2: Tùy chỉnh Num Leaves & Subsample cho LightGBM
```bash
uv run python run/train.py --model lightgbm --num-leaves 63 --subsample 0.85 --learning-rate 0.05
```

### Ví Dụ 3: Huấn luyện Ensemble 4 mô hình với Learning Rate tùy chỉnh
```bash
uv run python run/train.py --model ensemble4 --n-estimators 400 --learning-rate 0.025
```

### Ví Dụ 4: Tùy chỉnh hệ số phạt C cho Logistic Regression
```bash
uv run python run/train.py --model logistic_regression --c-reg 1.0
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
