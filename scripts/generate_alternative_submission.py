"""Generate Kaggle Submission file for Alternative-Only Model and Hybrid Model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import joblib
import numpy as np
import pandas as pd
from src.credit_scoring.features import engineer_application_features


def generate_submission(model_path: Path, output_path: Path, model_label: str):
    print(f"\n================================================================================")
    print(f"🚀 TẠO FILE SUBMISSION KAGGLE CHO: {model_label}")
    print(f"================================================================================")

    if not model_path.exists():
        print(f"⚠️ Chưa thấy tệp model tại {model_path}. Đang tiến hành huấn luyện...")
        from src.credit_scoring.ensemble_pipeline import train_ensemble_pipeline
        feature_set = "alternative_only" if "alternative" in str(model_path) else "serving"
        res = train_ensemble_pipeline(feature_set=feature_set, sample_size=30000, seed=42)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(res, model_path, compress=3)
    else:
        print(f"📦 Loading model artifact từ: {model_path}")
        res = joblib.load(model_path)

    models = res["models"]
    lin_prep = res["linear_preprocessor"]
    tree_prep = res["tree_preprocessor"]
    weights = res["optimal_weights"]
    calibrator = res["calibrator"]
    feature_cols = res["feature_cols"]

    test_csv_path = Path("data/raw/home-credit-default-risk/application_test.csv")
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file application_test.csv tại {test_csv_path}")

    print(f"📄 Đang đọc và biến đổi đặc trưng dữ liệu test: {test_csv_path}...")
    test_df_raw = pd.read_csv(test_csv_path)
    num_rows = len(test_df_raw)
    print(f"   Tổng số hồ sơ test: {num_rows:,} bản ghi.")

    # Apply row-local feature engineering
    test_df_engineered = engineer_application_features(test_df_raw)

    # Ensure all feature columns exist
    X_test_raw = test_df_engineered.copy()
    for col in feature_cols:
        if col not in X_test_raw:
            X_test_raw[col] = np.nan
    X_test_raw = X_test_raw[feature_cols].copy()

    # Preprocessing feature matrices
    print("⚡ Preprocessing feature matrices...")
    X_test_lin = lin_prep.transform(X_test_raw)
    X_test_tree = tree_prep.transform(X_test_raw)

    # Predict proba for all sub-models
    print("🤖 Đang dự đoán xác suất rủi ro từ bộ mô hình Ensemble...")
    val_preds = {}
    test_preds = {}

    p_logreg = models["LogisticRegression"].predict_proba(X_test_lin)[:, 1]
    p_lgbm = models["LightGBM"].predict_proba(X_test_tree)[:, 1]
    p_xgb = models["XGBoost"].predict_proba(X_test_tree)[:, 1]
    p_cat = models["CatBoost"].predict_proba(X_test_tree)[:, 1]

    p_ensemble_raw = (
        weights.get("LogisticRegression", 0.0) * p_logreg +
        weights.get("LightGBM", 0.0) * p_lgbm +
        weights.get("XGBoost", 0.0) * p_xgb +
        weights.get("CatBoost", 0.0) * p_cat
    )

    # Calibrate predictions
    p_ensemble_calibrated = calibrator.predict(p_ensemble_raw)

    # Create submission dataframe
    submission = pd.DataFrame({
        "SK_ID_CURR": test_df_raw["SK_ID_CURR"].astype(int),
        "TARGET": np.clip(p_ensemble_calibrated, 0.0, 1.0)
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    print(f"✅ ĐÃ TẠO THÀNH CÔNG TỆP SUBMISSION KAGGLE!")
    print(f"📌 Đường dẫn tệp: {output_path.resolve()}")
    print(f"📊 Kích thước tệp: {output_path.stat().st_size / (1024*1024):.2f} MB ({len(submission):,} dòng)")
    print(f"👀 Xem 5 bản ghi đầu tiên:")
    print(submission.head(5))


def main():
    start_time = time.time()

    # 1. Generate Alternative-Only Submission
    alt_model_path = Path("artifacts/models/alternative_only_model.joblib")
    alt_sub_path = Path("artifacts/submission_alternative_only.csv")
    generate_submission(alt_model_path, alt_sub_path, "Alternative-Only Model (Chỉ Dữ Liệu Thay Thế)")

    # 2. Generate Hybrid Ensemble Submission
    hybrid_model_path = Path("artifacts/models/ensemble_model.joblib")
    hybrid_sub_path = Path("artifacts/submission_hybrid_ensemble.csv")
    generate_submission(hybrid_model_path, hybrid_sub_path, "Hybrid Ensemble Model (Dữ Liệu Truyền Thống + Thay Thế)")

    elapsed = time.time() - start_time
    print(f"\n================================================================================")
    print(f"🎉 HOÀN THÀNH TẠO 2 FILE SUBMISSION KAGGLE TRONG {elapsed:.2f} GIÂY!")
    print(f"1️⃣ File 1 (Alternative-Only): {alt_sub_path.resolve()}")
    print(f"2️⃣ File 2 (Hybrid Ensemble):  {hybrid_sub_path.resolve()}")
    print(f"================================================================================")


if __name__ == "__main__":
    main()
