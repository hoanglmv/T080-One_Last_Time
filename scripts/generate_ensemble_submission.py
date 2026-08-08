"""Generate a Home Credit Kaggle submission from an ensemble artifact."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.credit_scoring.ensemble_pipeline import (  # noqa: E402
    _prepare_catboost_frame,
    rank_averaging_transform,
)
from src.credit_scoring.features import engineer_application_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Home Credit ensemble submission")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/models/ensemble3_model.joblib"),
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("data/raw/home-credit-default-risk/application_test.csv"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/submission_ensemble.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()

    if not args.model_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy ensemble artifact: {args.model_path}")
    if not args.test_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy application test: {args.test_path}")

    artifact = joblib.load(args.model_path)
    models = artifact["models"]
    weights = artifact["optimal_weights"]
    feature_cols = artifact["feature_cols"]

    test_df = pd.read_csv(args.test_path)
    x_test_raw = engineer_application_features(test_df)
    for column in feature_cols:
        if column not in x_test_raw:
            x_test_raw[column] = np.nan
    x_test_raw = x_test_raw[feature_cols].copy()

    component_predictions: dict[str, np.ndarray] = {}
    weighted_names = set(weights)
    if "LogisticRegression" in weighted_names:
        transformed = artifact["linear_preprocessor"].transform(x_test_raw)
        component_predictions["LogisticRegression"] = models["LogisticRegression"].predict_proba(
            transformed
        )[:, 1]

    tree_names = weighted_names & {"LightGBM", "XGBoost"}
    if tree_names:
        transformed = artifact["tree_preprocessor"].transform(x_test_raw)
        for name in sorted(tree_names):
            component_predictions[name] = models[name].predict_proba(transformed)[:, 1]

    if "CatBoost" in weighted_names:
        (catboost_frame,) = _prepare_catboost_frame(
            x_test_raw,
            numeric_cols=artifact["catboost_numeric_cols"],
            categorical_cols=artifact["catboost_categorical_cols"],
            numeric_fill=artifact["catboost_numeric_fill"],
        )
        component_predictions["CatBoost"] = models["CatBoost"].predict_proba(catboost_frame)[:, 1]

    if set(component_predictions) != weighted_names:
        missing = sorted(weighted_names - set(component_predictions))
        raise ValueError(f"Không hỗ trợ prediction cho models: {missing}")
    if artifact.get("use_rank_blending", False):
        component_predictions = rank_averaging_transform(component_predictions)

    raw_probability = sum(weights[name] * component_predictions[name] for name in weights)
    probability = artifact["calibrator"].predict(raw_probability)
    submission = pd.DataFrame({"SK_ID_CURR": test_df["SK_ID_CURR"], "TARGET": probability})

    if submission["SK_ID_CURR"].duplicated().any():
        raise ValueError("Submission chứa SK_ID_CURR trùng lặp")
    if submission["TARGET"].isna().any() or not submission["TARGET"].between(0.0, 1.0).all():
        raise ValueError("TARGET phải là xác suất hữu hạn trong [0, 1]")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_path, index=False)
    print(f"Models: {list(weights)}")
    print(f"Weights: { {name: round(value, 4) for name, value in weights.items()} }")
    print(f"Rows: {len(submission):,}")
    print(f"TARGET range: [{submission['TARGET'].min():.6f}, {submission['TARGET'].max():.6f}]")
    print(f"Saved: {args.output_path.resolve()}")
    print(f"Elapsed: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
