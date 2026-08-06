"""Multi-model Training and Weighted Blending Ensemble Pipeline.

Integrates Logistic Regression, LightGBM, XGBoost, and CatBoost into a unified
blending ensemble with automated weight optimization for maximum ROC-AUC.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import catboost as cb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

from src.credit_scoring.features import (
    ENGINEERED_FEATURES,
    ID_COLUMN,
    SERVING_RAW_FEATURES,
    TARGET_COLUMN,
    build_home_credit_features,
)
from src.credit_scoring.metrics import credit_metrics
from src.credit_scoring.model import PlattCalibrator
from src.credit_scoring.training import make_split


def build_preprocessors(
    frame: pd.DataFrame, feature_columns: list[str]
) -> tuple[ColumnTransformer, ColumnTransformer, list[str], list[str]]:
    numeric_cols = [c for c in feature_columns if pd.api.types.is_numeric_dtype(frame[c])]
    categorical_cols = [c for c in feature_columns if c not in numeric_cols]

    # Preprocessor for Linear & Tree models (Imputation + One-Hot + Scaler)
    linear_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    # Preprocessor for GBDT Tree models (Native Categorical / Ordinal Imputer)
    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    return linear_preprocessor, tree_preprocessor, numeric_cols, categorical_cols


def optimize_blending_weights(predictions: dict[str, np.ndarray], y_true: np.ndarray) -> dict[str, float]:
    """Find non-negative weights summing to 1 that maximize ROC-AUC on validation split."""
    model_names = list(predictions.keys())
    pred_matrix = np.column_stack([predictions[name] for name in model_names])
    num_models = len(model_names)

    from sklearn.metrics import roc_auc_score

    def loss_func(weights: np.ndarray) -> float:
        # We minimize -ROC_AUC
        blended = np.dot(pred_matrix, weights)
        return -float(roc_auc_score(y_true, blended))

    # Initial equal weights
    init_weights = np.ones(num_models) / num_models
    bounds = [(0.0, 1.0) for _ in range(num_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)

    weights = res.x / np.sum(res.x)
    return {name: float(w) for name, w in zip(model_names, weights)}


def train_ensemble_pipeline(
    *,
    data_dir: Path | str = Path("data/raw/home-credit-default-risk"),
    feature_set: str = "serving",
    sample_size: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    data_path = Path(data_dir)
    print("⏳ Building Home Credit feature set...")
    frame = build_home_credit_features(data_path, feature_set=feature_set, sample_size=sample_size)
    
    if feature_set == "serving":
        raw_features = SERVING_RAW_FEATURES
    else:
        raw_features = [c for c in frame.columns if c not in (ID_COLUMN, TARGET_COLUMN)]
        
    feature_cols = [c for c in frame.columns if c not in (ID_COLUMN, TARGET_COLUMN)]
    
    split = make_split(frame, seed=seed)
    train_df = frame.iloc[split.train].copy()
    val_df = frame.iloc[split.validation].copy()
    test_df = frame.iloc[split.test].copy()

    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=int)
    y_val = val_df[TARGET_COLUMN].to_numpy(dtype=int)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=int)

    X_train_raw = train_df[feature_cols]
    X_val_raw = val_df[feature_cols]
    X_test_raw = test_df[feature_cols]

    linear_prep, tree_prep, num_cols, cat_cols = build_preprocessors(frame, feature_cols)

    print("⚡ Preprocessing feature matrices...")
    X_train_lin = linear_prep.fit_transform(X_train_raw)
    X_val_lin = linear_prep.transform(X_val_raw)
    X_test_lin = linear_prep.transform(X_test_raw)

    X_train_tree = tree_prep.fit_transform(X_train_raw)
    X_val_tree = tree_prep.transform(X_val_raw)
    X_test_tree = tree_prep.transform(X_test_raw)

    models: dict[str, Any] = {}
    val_preds: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}
    test_metrics: dict[str, dict[str, float]] = {}

    print("🤖 [1/4] Training Baseline Logistic Regression...")
    logreg = LogisticRegression(C=0.1, max_iter=500, random_state=seed)
    logreg.fit(X_train_lin, y_train)
    models["LogisticRegression"] = logreg
    val_preds["LogisticRegression"] = logreg.predict_proba(X_val_lin)[:, 1]
    test_preds["LogisticRegression"] = logreg.predict_proba(X_test_lin)[:, 1]
    test_metrics["LogisticRegression"] = credit_metrics(y_test, test_preds["LogisticRegression"])

    print("🤖 [2/4] Training LightGBM Champion...")
    lgbm = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbosity=-1,
        n_jobs=-1,
    )
    lgbm.fit(X_train_tree, y_train)
    models["LightGBM"] = lgbm
    val_preds["LightGBM"] = lgbm.predict_proba(X_val_tree)[:, 1]
    test_preds["LightGBM"] = lgbm.predict_proba(X_test_tree)[:, 1]
    test_metrics["LightGBM"] = credit_metrics(y_test, test_preds["LightGBM"])

    print("🤖 [3/4] Training XGBoost...")
    xgboost_model = xgb.XGBClassifier(
        n_estimators=250,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        eval_metric="auc",
    )
    xgboost_model.fit(X_train_tree, y_train)
    models["XGBoost"] = xgboost_model
    val_preds["XGBoost"] = xgboost_model.predict_proba(X_val_tree)[:, 1]
    test_preds["XGBoost"] = xgboost_model.predict_proba(X_test_tree)[:, 1]
    test_metrics["XGBoost"] = credit_metrics(y_test, test_preds["XGBoost"])

    print("🤖 [4/4] Training CatBoost...")
    catboost_model = cb.CatBoostClassifier(
        iterations=300,
        learning_rate=0.04,
        depth=6,
        random_seed=seed,
        verbose=0,
        thread_count=-1,
    )
    catboost_model.fit(X_train_tree, y_train)
    models["CatBoost"] = catboost_model
    val_preds["CatBoost"] = catboost_model.predict_proba(X_val_tree)[:, 1]
    test_preds["CatBoost"] = catboost_model.predict_proba(X_test_tree)[:, 1]
    test_metrics["CatBoost"] = credit_metrics(y_test, test_preds["CatBoost"])

    print("🎯 Optimizing Blending Weights on Validation Split...")
    optimal_weights = optimize_blending_weights(val_preds, y_val)
    print("   Optimal Weights:", {k: round(v, 4) for k, v in optimal_weights.items()})

    # Compute Ensemble predictions on test set
    val_ensemble_pred = np.zeros(len(y_val))
    test_ensemble_pred = np.zeros(len(y_test))

    for name, w in optimal_weights.items():
        val_ensemble_pred += w * val_preds[name]
        test_ensemble_pred += w * test_preds[name]

    # Fit Platt Calibrator on Ensemble Validation Predictions
    calibrator = PlattCalibrator().fit(val_ensemble_pred, y_val)
    test_ensemble_calibrated = calibrator.predict(test_ensemble_pred)

    test_metrics["Ensemble_Uncalibrated"] = credit_metrics(y_test, test_ensemble_pred)
    test_metrics["Ensemble_Calibrated"] = credit_metrics(y_test, test_ensemble_calibrated)

    return {
        "models": models,
        "linear_preprocessor": linear_prep,
        "tree_preprocessor": tree_prep,
        "optimal_weights": optimal_weights,
        "calibrator": calibrator,
        "test_metrics": test_metrics,
        "feature_cols": feature_cols,
        "raw_features": raw_features,
        "val_preds": val_preds,
        "test_preds": test_preds,
        "test_ensemble_pred": test_ensemble_pred,
        "test_ensemble_calibrated": test_ensemble_calibrated,
        "y_test": y_test,
    }
