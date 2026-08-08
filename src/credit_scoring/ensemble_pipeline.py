"""Multi-model Training and Weighted Blending Ensemble Pipeline.

Integrates Logistic Regression, LightGBM, XGBoost, and CatBoost into a unified
blending ensemble with automated weight optimization for maximum ROC-AUC.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import differential_evolution
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.credit_scoring.features import (
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


def rank_averaging_transform(predictions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convert each model's scores to percentile ranks for scale-free blending."""
    ranked: dict[str, np.ndarray] = {}
    for name, values in predictions.items():
        scores = np.asarray(values, dtype=float)
        if scores.ndim != 1 or len(scores) == 0:
            raise ValueError(f"Predictions for {name} must be a non-empty 1D array")
        ranked[name] = rankdata(scores, method="average") / len(scores)
    return ranked


def optimize_blending_weights(
    predictions: dict[str, np.ndarray], y_true: np.ndarray, *, seed: int = 42
) -> dict[str, float]:
    """Maximize ROC-AUC with a deterministic, gradient-free optimizer.

    A softmax maps unconstrained optimizer variables to the probability simplex.
    Single-model vertices are also evaluated, so the blend cannot score below the
    best component on the optimization split.
    """
    if not predictions:
        raise ValueError("At least one prediction vector is required")
    model_names = list(predictions.keys())
    pred_matrix = np.column_stack([predictions[name] for name in model_names])
    num_models = len(model_names)

    from sklearn.metrics import roc_auc_score

    def softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        return exp / exp.sum()

    def loss_func(logits: np.ndarray) -> float:
        weights = softmax(logits)
        blended = np.dot(pred_matrix, weights)
        return -float(roc_auc_score(y_true, blended))

    result = differential_evolution(
        loss_func,
        bounds=[(-6.0, 6.0)] * num_models,
        seed=seed,
        maxiter=80,
        popsize=12,
        polish=False,
        workers=1,
    )
    weights = softmax(result.x)

    # A rank objective is piecewise constant. Explicitly checking vertices makes
    # this routine robust when the evolutionary search lands on a plateau.
    candidate_weights = [weights, np.ones(num_models) / num_models]
    candidate_weights.extend(np.eye(num_models))
    weights = min(candidate_weights, key=lambda w: loss_func(np.log(np.clip(w, 1e-12, 1.0))))
    return {name: float(w) for name, w in zip(model_names, weights)}


def _prepare_catboost_frame(
    train: pd.DataFrame,
    *others: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    numeric_fill: pd.Series | dict[str, float] | None = None,
) -> tuple[pd.DataFrame, ...]:
    """Impute while preserving categorical columns for CatBoost native handling."""
    train_out = train.copy()
    fill_values = pd.Series(numeric_fill) if numeric_fill is not None else train_out[numeric_cols].median()
    for column in numeric_cols:
        train_out[column] = pd.to_numeric(train_out[column], errors="coerce").fillna(fill_values[column])
    for column in categorical_cols:
        train_out[column] = train_out[column].fillna("__MISSING__").astype(str)

    outputs = [train_out]
    for frame in others:
        output = frame.copy()
        for column in numeric_cols:
            output[column] = pd.to_numeric(output[column], errors="coerce").fillna(fill_values[column])
        for column in categorical_cols:
            output[column] = output[column].fillna("__MISSING__").astype(str)
        outputs.append(output)
    return tuple(outputs)


def train_ensemble_pipeline(
    *,
    data_dir: Path | str = Path("data/raw/home-credit-default-risk"),
    feature_set: str = "serving",
    sample_size: int | None = None,
    seed: int = 42,
    n_estimators: int | None = None,
    learning_rate: float | None = None,
    max_depth: int | None = None,
    num_leaves: int | None = None,
    subsample: float | None = None,
    colsample_bytree: float | None = None,
    c_reg: float | None = None,
    use_rank_blending: bool = True,
    drop_weak_logreg: bool = True,
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

    x_train_raw = train_df[feature_cols]
    x_val_raw = val_df[feature_cols]
    x_test_raw = test_df[feature_cols]

    linear_prep, tree_prep, num_cols, cat_cols = build_preprocessors(frame, feature_cols)

    print("⚡ Preprocessing feature matrices...")
    x_train_lin = linear_prep.fit_transform(x_train_raw)
    x_val_lin = linear_prep.transform(x_val_raw)
    x_test_lin = linear_prep.transform(x_test_raw)

    x_train_tree = tree_prep.fit_transform(x_train_raw)
    x_val_tree = tree_prep.transform(x_val_raw)
    x_test_tree = tree_prep.transform(x_test_raw)
    x_train_cat, x_val_cat, x_test_cat = _prepare_catboost_frame(
        x_train_raw, x_val_raw, x_test_raw, numeric_cols=num_cols, categorical_cols=cat_cols
    )

    models: dict[str, Any] = {}
    val_preds: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}
    test_metrics: dict[str, dict[str, float]] = {}

    # Defaults
    lr_lgbm = learning_rate if learning_rate is not None else 0.03
    lr_xgb = learning_rate if learning_rate is not None else 0.03
    lr_cb = learning_rate if learning_rate is not None else 0.04

    n_est_lgbm = n_estimators if n_estimators is not None else 300
    n_est_xgb = n_estimators if n_estimators is not None else 250
    n_est_cb = n_estimators if n_estimators is not None else 300

    depth_xgb = max_depth if max_depth is not None else 5
    depth_cb = max_depth if max_depth is not None else 6
    leaves_lgbm = num_leaves if num_leaves is not None else 31

    sub_sample = subsample if subsample is not None else 0.8
    col_sample = colsample_bytree if colsample_bytree is not None else 0.8
    c_val = c_reg if c_reg is not None else 0.1

    print("🤖 [1/4] Training Baseline Logistic Regression...")
    logreg = LogisticRegression(C=c_val, max_iter=500, random_state=seed)
    logreg.fit(x_train_lin, y_train)
    models["LogisticRegression"] = logreg
    val_preds["LogisticRegression"] = logreg.predict_proba(x_val_lin)[:, 1]
    test_preds["LogisticRegression"] = logreg.predict_proba(x_test_lin)[:, 1]
    test_metrics["LogisticRegression"] = credit_metrics(y_test, test_preds["LogisticRegression"])

    print("🤖 [2/4] Training LightGBM Champion...")
    lgbm = lgb.LGBMClassifier(
        n_estimators=n_est_lgbm,
        learning_rate=lr_lgbm,
        num_leaves=leaves_lgbm,
        subsample=sub_sample,
        colsample_bytree=col_sample,
        random_state=seed,
        verbosity=-1,
        n_jobs=-1,
    )
    lgbm.fit(x_train_tree, y_train)
    models["LightGBM"] = lgbm
    val_preds["LightGBM"] = lgbm.predict_proba(x_val_tree)[:, 1]
    test_preds["LightGBM"] = lgbm.predict_proba(x_test_tree)[:, 1]
    test_metrics["LightGBM"] = credit_metrics(y_test, test_preds["LightGBM"])

    print("🤖 [3/4] Training XGBoost...")
    xgboost_model = xgb.XGBClassifier(
        n_estimators=n_est_xgb,
        learning_rate=lr_xgb,
        max_depth=depth_xgb,
        subsample=sub_sample,
        colsample_bytree=col_sample,
        random_state=seed,
        n_jobs=-1,
        eval_metric="auc",
    )
    xgboost_model.fit(x_train_tree, y_train)
    models["XGBoost"] = xgboost_model
    val_preds["XGBoost"] = xgboost_model.predict_proba(x_val_tree)[:, 1]
    test_preds["XGBoost"] = xgboost_model.predict_proba(x_test_tree)[:, 1]
    test_metrics["XGBoost"] = credit_metrics(y_test, test_preds["XGBoost"])

    print("🤖 [4/4] Training CatBoost...")
    catboost_model = cb.CatBoostClassifier(
        iterations=n_est_cb,
        learning_rate=lr_cb,
        depth=depth_cb,
        random_seed=seed,
        verbose=0,
        thread_count=-1,
    )
    catboost_model.fit(x_train_cat, y_train, cat_features=cat_cols)
    models["CatBoost"] = catboost_model
    val_preds["CatBoost"] = catboost_model.predict_proba(x_val_cat)[:, 1]
    test_preds["CatBoost"] = catboost_model.predict_proba(x_test_cat)[:, 1]
    test_metrics["CatBoost"] = credit_metrics(y_test, test_preds["CatBoost"])

    print("🎯 Optimizing Blending Weights on Validation Split...")
    blend_val_preds = rank_averaging_transform(val_preds) if use_rank_blending else val_preds
    blend_test_preds = rank_averaging_transform(test_preds) if use_rank_blending else test_preds

    candidate_names = list(blend_val_preds)
    if drop_weak_logreg and "LogisticRegression" in candidate_names and len(candidate_names) > 1:
        from sklearn.metrics import roc_auc_score

        without_lr = {k: v for k, v in blend_val_preds.items() if k != "LogisticRegression"}
        weights_all = optimize_blending_weights(blend_val_preds, y_val, seed=seed)
        weights_without = optimize_blending_weights(without_lr, y_val, seed=seed)
        auc_all = roc_auc_score(
            y_val, sum(weights_all[k] * blend_val_preds[k] for k in weights_all)
        )
        auc_without = roc_auc_score(
            y_val, sum(weights_without[k] * without_lr[k] for k in weights_without)
        )
        # Prefer the simpler tree-only blend on a tie; retain LogReg only when it
        # demonstrates a measurable validation gain.
        optimal_weights = weights_all if auc_all > auc_without + 1e-5 else weights_without
    else:
        optimal_weights = optimize_blending_weights(blend_val_preds, y_val, seed=seed)
    print("   Optimal Weights:", {k: round(v, 4) for k, v in optimal_weights.items()})

    # Compute Ensemble predictions on test set
    val_ensemble_pred = np.zeros(len(y_val))
    test_ensemble_pred = np.zeros(len(y_test))

    for name, w in optimal_weights.items():
        val_ensemble_pred += w * blend_val_preds[name]
        test_ensemble_pred += w * blend_test_preds[name]

    # Fit Platt Calibrator on Ensemble Validation Predictions
    calibrator = PlattCalibrator().fit(val_ensemble_pred, y_val)
    test_ensemble_calibrated = calibrator.predict(test_ensemble_pred)

    test_metrics["Ensemble_Uncalibrated"] = credit_metrics(y_test, test_ensemble_pred)
    test_metrics["Ensemble_Calibrated"] = credit_metrics(y_test, test_ensemble_calibrated)

    return {
        "models": models,
        "linear_preprocessor": linear_prep,
        "tree_preprocessor": tree_prep,
        "catboost_numeric_cols": num_cols,
        "catboost_categorical_cols": cat_cols,
        "catboost_numeric_fill": x_train_raw[num_cols].median().to_dict(),
        "use_rank_blending": use_rank_blending,
        "optimal_weights": optimal_weights,
        "calibrator": calibrator,
        "test_metrics": test_metrics,
        "feature_cols": feature_cols,
        "raw_features": raw_features,
        "val_preds": val_preds,
        "test_preds": test_preds,
        "test_ensemble_pred": test_ensemble_pred,
        "test_ensemble_calibrated": test_ensemble_calibrated,
        "y_val": y_val,
        "y_test": y_test,
    }
