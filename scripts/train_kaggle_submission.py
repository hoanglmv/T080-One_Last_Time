"""Train the full relational LightGBM model and create a Kaggle submission."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.credit_scoring.features import (  # noqa: E402
    ID_COLUMN,
    TARGET_COLUMN,
    build_home_credit_features,
)
from src.credit_scoring.metrics import credit_metrics  # noqa: E402
from src.credit_scoring.model import PlattCalibrator  # noqa: E402
from src.credit_scoring.training import (  # noqa: E402
    _candidate_models,
    _fit_candidate,
    make_split,
    select_model_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train full Home Credit model and write Kaggle submission")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/home-credit-default-risk"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/submission_full_lightgbm.csv"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/submission_full_lightgbm_report.json"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def prepare_lightgbm_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Use LightGBM-native missing/category support with a compact float32 matrix."""
    train_out = train[feature_columns].copy()
    test_out = test[feature_columns].copy()
    categorical = [column for column in feature_columns if not pd.api.types.is_numeric_dtype(train_out[column])]
    for column in categorical:
        categories = pd.Index(train_out[column].dropna().astype(str).unique()).union(
            pd.Index(test_out[column].dropna().astype(str).unique())
        )
        train_out[column] = pd.Categorical(train_out[column].astype("string"), categories=categories)
        test_out[column] = pd.Categorical(test_out[column].astype("string"), categories=categories)
    for column in set(feature_columns) - set(categorical):
        train_out[column] = pd.to_numeric(train_out[column], errors="coerce").astype("float32")
        test_out[column] = pd.to_numeric(test_out[column], errors="coerce").astype("float32")
    return train_out, test_out, categorical


def main() -> None:
    args = parse_args()
    print("Building full relational train features...")
    train = build_home_credit_features(args.data_dir, feature_set="full")
    print("Building full relational test features...")
    test = build_home_credit_features(
        args.data_dir,
        feature_set="full",
        application_file="application_test.csv",
    )

    feature_columns, dropped = select_model_features(train, feature_set="full")
    for column in feature_columns:
        if column not in test:
            test[column] = np.nan
    x_all, x_test, categorical = prepare_lightgbm_frames(train, test, feature_columns)

    split = make_split(train, seed=args.seed)
    x_fit = x_all.iloc[split.train]
    x_validation = x_all.iloc[split.validation]
    x_holdout = x_all.iloc[split.test]
    y_fit = train.iloc[split.train][TARGET_COLUMN].to_numpy(dtype=int)
    y_validation = train.iloc[split.validation][TARGET_COLUMN].to_numpy(dtype=int)
    y_holdout = train.iloc[split.test][TARGET_COLUMN].to_numpy(dtype=int)

    model = _candidate_models(args.seed, include_logistic=False)["lightgbm"]
    model = _fit_candidate("lightgbm", model, x_fit, y_fit, x_validation, y_validation)
    validation_raw = model.predict_proba(x_validation)[:, 1]
    calibrator = PlattCalibrator().fit(validation_raw, y_validation)
    holdout_probability = calibrator.predict(model.predict_proba(x_holdout)[:, 1])

    best_iteration = int(model.best_iteration_ or model.n_estimators)
    final_model = lgb.LGBMClassifier(**model.get_params())
    final_model.set_params(n_estimators=best_iteration)
    final_model.fit(x_all, train[TARGET_COLUMN].to_numpy(dtype=int))
    probability = calibrator.predict(final_model.predict_proba(x_test)[:, 1])

    submission = pd.DataFrame({ID_COLUMN: test[ID_COLUMN].astype(int), TARGET_COLUMN: probability})
    if len(submission) != 48_744:
        raise ValueError(f"Expected 48,744 rows, received {len(submission):,}")
    if submission[ID_COLUMN].duplicated().any():
        raise ValueError("Submission contains duplicate SK_ID_CURR")
    if submission[TARGET_COLUMN].isna().any() or not submission[TARGET_COLUMN].between(0, 1).all():
        raise ValueError("TARGET must contain finite probabilities in [0, 1]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    report = {
        "rows": len(submission),
        "features": len(feature_columns),
        "best_iteration": best_iteration,
        "categorical_features": categorical,
        "holdout_metrics": credit_metrics(y_holdout, holdout_probability),
        "target_min": float(probability.min()),
        "target_max": float(probability.max()),
        "target_mean": float(probability.mean()),
        "dropped_features": dropped,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"submission": str(args.output.resolve()), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
