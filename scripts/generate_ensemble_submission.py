"""Generate Kaggle submission file using Multi-Model Blending Ensemble."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import joblib
import numpy as np
import pandas as pd


def main():
    print("⏳ Đang khởi tạo quá trình tạo file submission Ensemble cho Kaggle...")
    start_time = time.time()

    model_path = Path("artifacts/models/ensemble_model.joblib")
    test_path = Path("data/raw/home-credit-default-risk/application_test.csv")
    output_path = Path("artifacts/submission_ensemble.csv")

    if not model_path.exists():
        print(f"⚠️ Chưa thấy tệp {model_path}. Vui lòng chạy 'uv run python scripts/train_ensemble.py' trước!")
        return

    if not test_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tập dữ liệu test tại {test_path}")

    print(f"📦 Loading Ensemble artifact từ {model_path}...")
    res = joblib.load(model_path)
    models = res["models"]
    lin_prep = res["linear_preprocessor"]
    tree_prep = res["tree_preprocessor"]
    weights = res["optimal_weights"]
    calibrator = res["calibrator"]
    feature_cols = res["feature_cols"]

    print(f"📄 Đang đọc dữ liệu application_test.csv ({test_path})...")
    test_df = pd.read_csv(test_path)
    num_rows = len(test_df)
    print(f"   Tổng số hồ sơ test: {num_rows:,} bản ghi.")

    # Prepare features
    X_test_raw = test_df.copy()
    for col in feature_cols:
        if col not in X_test_raw:
            X_test_raw[col] = np.nan
    X_test_raw = X_test_raw[feature_cols].copy()

    # Normalize numeric and categorical columns
    for col in X_test_raw.columns:
        if pd.api.types.is_numeric_dtype(X_test_raw[col]):
            X_test_raw[col] = pd.to_numeric(X_test_raw[col], errors="coerce")
        else:
            X_test_raw[col] = X_test_raw[col].map(lambda v: str(v) if pd.notna(v) else np.nan)

    print("⚡ Preprocessing test feature matrices...")
    X_test_lin = lin_prep.transform(X_test_raw)
    X_test_tree = tree_prep.transform(X_test_raw)

    print("🤖 Đang dự đoán xác suất từ 4 mô hình thành phần...")
    p_logreg = models["LogisticRegression"].predict_proba(X_test_lin)[:, 1]
    p_lgbm = models["LightGBM"].predict_proba(X_test_tree)[:, 1]
    p_xgb = models["XGBoost"].predict_proba(X_test_tree)[:, 1]
    p_cat = models["CatBoost"].predict_proba(X_test_tree)[:, 1]

    # Weighted Blending
    print(f"🎯 Kết hợp trọng số Blending: { {k: round(v, 4) for k, v in weights.items()} }...")
    p_ensemble_raw = (
        weights.get("LogisticRegression", 0.0) * p_logreg +
        weights.get("LightGBM", 0.0) * p_lgbm +
        weights.get("XGBoost", 0.0) * p_xgb +
        weights.get("CatBoost", 0.0) * p_cat
    )

    # Calibration
    p_ensemble_calibrated = calibrator.predict(p_ensemble_raw)

    # Export Kaggle Submission
    submission = pd.DataFrame({
        "SK_ID_CURR": test_df["SK_ID_CURR"],
        "TARGET": p_ensemble_calibrated
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    elapsed = time.time() - start_time
    print(f"\n================================================================================")
    print(f"✅ ĐÃ XUẤT TỆP ENSEMBLE SUBMISSION THÀNH CÔNG!")
    print(f"📌 Đường dẫn tệp: {output_path.resolve()}")
    print(f"📊 Kích thước tệp: {output_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"⏱️ Tổng thời gian xử lý: {elapsed:.2f} giây")
    print(f"👀 5 bản ghi đầu tiên:")
    print(submission.head())
    print("================================================================================")


if __name__ == "__main__":
    main()
