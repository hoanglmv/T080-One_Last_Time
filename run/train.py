"""Flexible Model Training CLI script.

Supports single models (LightGBM, XGBoost, CatBoost, Logistic Regression)
and multi-model Ensembles (Ensemble-3, Ensemble-4).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from src.credit_scoring.ensemble_pipeline import (  # noqa: E402
    optimize_blending_weights,
    rank_averaging_transform,
    train_ensemble_pipeline,
)
from src.credit_scoring.metrics import credit_metrics  # noqa: E402
from src.credit_scoring.model import PlattCalibrator  # noqa: E402
from src.credit_scoring.training import train_credit_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alternative Credit Scoring Model Training CLI")
    parser.add_argument(
        "--model",
        type=str,
        default="lightgbm",
        choices=["lightgbm", "xgboost", "catboost", "logistic_regression", "ensemble3", "ensemble4"],
        help="Select model architecture or ensemble configuration to train.",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="serving",
        choices=["serving", "full"],
        help="Feature engineering set (serving=22 core features, full=extended features).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional row limit for fast testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/home-credit-default-risk"),
        help="Path to raw Home Credit data directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/models"),
        help="Directory to save model artifacts and reports.",
    )
    # Hyperparameters (Optional - defaults used if not set)
    parser.add_argument("--n-estimators", type=int, default=None, help="Number of boosting trees/iterations.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate.")
    parser.add_argument("--max-depth", type=int, default=None, help="Maximum tree depth for XGBoost/CatBoost.")
    parser.add_argument("--num-leaves", type=int, default=None, help="Number of leaves for LightGBM.")
    parser.add_argument("--subsample", type=float, default=None, help="Row subsample ratio.")
    parser.add_argument("--colsample-bytree", type=float, default=None, help="Column subsample ratio.")
    parser.add_argument(
        "--c-reg", type=float, default=None, help="Inverse regularization strength for Logistic Regression."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print(f"🚀 RUNNING TRAINING PIPELINE FOR MODEL: [{args.model.upper()}]")
    print(f"📌 Feature Set: {args.feature_set} | Seed: {args.seed} | Sample Size: {args.sample_size}")
    if args.learning_rate is not None or args.n_estimators is not None:
        print(
            f"⚙️ Custom Hyperparams: n_estimators={args.n_estimators}, learning_rate={args.learning_rate}, max_depth={args.max_depth}, num_leaves={args.num_leaves}"
        )
    print("================================================================================")

    if args.model in ("ensemble3", "ensemble4"):
        # Train Ensemble Pipeline
        res = train_ensemble_pipeline(
            data_dir=args.data_dir,
            feature_set=args.feature_set,
            sample_size=args.sample_size,
            seed=args.seed,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            num_leaves=args.num_leaves,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            c_reg=args.c_reg,
        )

        all_models = res["models"]
        val_preds = res["val_preds"]
        test_preds = res["test_preds"]
        y_test_true = res["y_test"]

        if args.model == "ensemble3":
            # Keep only LightGBM, XGBoost, CatBoost
            selected_names = ["LightGBM", "XGBoost", "CatBoost"]
            filtered_models = {k: v for k, v in all_models.items() if k in selected_names}
            filtered_val_preds = rank_averaging_transform({k: v for k, v in val_preds.items() if k in selected_names})
            filtered_test_preds = rank_averaging_transform({k: v for k, v in test_preds.items() if k in selected_names})

            weights = optimize_blending_weights(filtered_val_preds, res["y_val"], seed=args.seed)

            # Compute ensemble prediction on test set
            val_ensemble_pred = np.zeros(len(res["y_val"]))
            test_ensemble_pred = np.zeros(len(y_test_true))
            for name, w in weights.items():
                val_ensemble_pred += w * filtered_val_preds[name]
                test_ensemble_pred += w * filtered_test_preds[name]

            calibrator = PlattCalibrator().fit(val_ensemble_pred, res["y_val"])
            test_ensemble_calibrated = calibrator.predict(test_ensemble_pred)
            test_metric = credit_metrics(y_test_true, test_ensemble_calibrated)

            artifact_data = {
                "model_type": "ensemble3",
                "models": filtered_models,
                "optimal_weights": weights,
                "linear_preprocessor": res["linear_preprocessor"],
                "tree_preprocessor": res["tree_preprocessor"],
                "calibrator": calibrator,
                "feature_cols": res["feature_cols"],
                "catboost_numeric_cols": res["catboost_numeric_cols"],
                "catboost_categorical_cols": res["catboost_categorical_cols"],
                "catboost_numeric_fill": res["catboost_numeric_fill"],
                "use_rank_blending": True,
                "test_metrics": test_metric,
            }
            output_file = args.output_dir / "ensemble3_model.joblib"
        else:
            artifact_data = {
                "model_type": "ensemble4",
                "models": res["models"],
                "optimal_weights": res["optimal_weights"],
                "linear_preprocessor": res["linear_preprocessor"],
                "tree_preprocessor": res["tree_preprocessor"],
                "calibrator": res["calibrator"],
                "feature_cols": res["feature_cols"],
                "catboost_numeric_cols": res["catboost_numeric_cols"],
                "catboost_categorical_cols": res["catboost_categorical_cols"],
                "catboost_numeric_fill": res["catboost_numeric_fill"],
                "use_rank_blending": res["use_rank_blending"],
                "test_metrics": res["test_metrics"]["Ensemble_Calibrated"],
            }
            output_file = args.output_dir / "ensemble4_model.joblib"

        joblib.dump(artifact_data, output_file, compress=3)
        metrics = artifact_data["test_metrics"]

        print("\n================================================================================")
        print(f"✅ TRAINED AND SAVED [{args.model.upper()}] TO: {output_file.resolve()}")
        print(
            f"📊 TEST ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | Gini: {metrics['gini']:.4f} | KS: {metrics['ks']:.4f}"
        )
        print("================================================================================")

    else:
        # Train Single Model
        if args.model == "lightgbm":
            bundle, report = train_credit_model(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                feature_set=args.feature_set,
                sample_size=args.sample_size,
                seed=args.seed,
            )
            metrics = report["champion_test_metrics"]
            output_file = args.output_dir / "credit_model.joblib"
        else:
            # Train Single XGBoost / CatBoost / Logistic Regression via Ensemble pipeline base models
            res = train_ensemble_pipeline(
                data_dir=args.data_dir,
                feature_set=args.feature_set,
                sample_size=args.sample_size,
                seed=args.seed,
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                max_depth=args.max_depth,
                num_leaves=args.num_leaves,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                c_reg=args.c_reg,
            )
            model_key_map = {
                "xgboost": "XGBoost",
                "catboost": "CatBoost",
                "logistic_regression": "LogisticRegression",
            }
            target_key = model_key_map[args.model]
            single_model = res["models"][target_key]
            metrics = res["test_metrics"][target_key]

            prep = res["linear_preprocessor"] if args.model == "logistic_regression" else res["tree_preprocessor"]

            single_artifact = {
                "model_type": args.model,
                "model": single_model,
                "preprocessor": prep,
                "feature_cols": res["feature_cols"],
                "test_metrics": metrics,
            }
            if args.model == "catboost":
                single_artifact.update(
                    {
                        "preprocessor": None,
                        "catboost_numeric_cols": res["catboost_numeric_cols"],
                        "catboost_categorical_cols": res["catboost_categorical_cols"],
                        "catboost_numeric_fill": res["catboost_numeric_fill"],
                    }
                )
            output_file = args.output_dir / f"{args.model}_model.joblib"
            joblib.dump(single_artifact, output_file, compress=3)

        print("\n================================================================================")
        print(f"✅ TRAINED AND SAVED [{args.model.upper()}] TO: {output_file.resolve()}")
        print(
            f"📊 TEST ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | Gini: {metrics['gini']:.4f} | KS: {metrics['ks']:.4f}"
        )
        print("================================================================================")


if __name__ == "__main__":
    main()
