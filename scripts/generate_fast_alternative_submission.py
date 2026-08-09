"""Ultra-fast generation of artifacts/submission_alternative_only.csv using LightGBM."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.credit_scoring.features import (
    build_home_credit_features,
    engineer_application_features,
    ID_COLUMN,
    TARGET_COLUMN,
)


def main():
    print("⏳ Đang huấn luyện nhanh mô hình Alternative-Only và xuất file submission Kaggle...")
    start_time = time.time()

    data_dir = Path("data/raw/home-credit-default-risk")
    train_df = build_home_credit_features(data_dir, feature_set="alternative_only", sample_size=20000)

    feature_cols = [c for c in train_df.columns if c not in (ID_COLUMN, TARGET_COLUMN, "CODE_GENDER")]
    X_train_raw = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=int)

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X_train_raw[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    for col in categorical_cols:
        X_train_raw[col] = X_train_raw[col].fillna("Missing").astype(str)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_cols),
        ]
    )

    X_train_proc = preprocessor.fit_transform(X_train_raw)

    print("🤖 Đang huấn luyện LightGBM trên 20,000 hồ sơ...")
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
        n_jobs=2
    )
    model.fit(X_train_proc, y_train)

    print("📄 Đang biến đổi và dự đoán trên tập test application_test.csv...")
    test_raw = pd.read_csv(data_dir / "application_test.csv")
    test_engineered = engineer_application_features(test_raw)

    X_test_raw = test_engineered.reindex(columns=feature_cols).copy()

    for col in numeric_cols:
        X_test_raw[col] = pd.to_numeric(X_test_raw[col], errors="coerce")
    for col in categorical_cols:
        X_test_raw[col] = X_test_raw[col].fillna("Missing").astype(str)

    X_test_proc = preprocessor.transform(X_test_raw)
    preds = model.predict_proba(X_test_proc)[:, 1]

    sub = pd.DataFrame({
        "SK_ID_CURR": test_raw["SK_ID_CURR"].astype(int),
        "TARGET": np.clip(preds, 0.0, 1.0)
    })

    output_path = Path("artifacts/submission_alternative_only.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)

    elapsed = time.time() - start_time
    print(f"✅ ĐÃ XUẤT THÀNH CÔNG: {output_path.resolve()}")
    print(f"📊 Dòng: {len(sub):,}, Kích thước: {output_path.stat().st_size / (1024*1024):.2f} MB ({elapsed:.2f}s)")
    print(sub.head(5))


if __name__ == "__main__":
    main()
