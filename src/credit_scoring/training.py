"""Reproducible training, model selection and report generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.credit_scoring.features import (
    ENGINEERED_FEATURES,
    ID_COLUMN,
    PROTECTED_COLUMNS,
    SERVING_RAW_FEATURES,
    TARGET_COLUMN,
    TIME_COLUMN_CANDIDATES,
    build_home_credit_features,
)
from src.credit_scoring.metrics import (
    credit_metrics,
    fairness_report,
    gini_stability_by_period,
    population_stability_index,
    select_ks_threshold,
)
from src.credit_scoring.model import CreditModelBundle, PlattCalibrator


@dataclass(frozen=True)
class SplitResult:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    strategy: str
    time_column: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_training_data(frame: pd.DataFrame, *, source_path: Path) -> dict[str, Any]:
    target = frame[TARGET_COLUMN]
    missing = frame.isna().mean().sort_values(ascending=False)
    numeric = frame.select_dtypes(include=[np.number])
    infinite_count = int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum()) if not numeric.empty else 0
    return {
        "source": str(source_path),
        "source_sha256": sha256_file(source_path),
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "target_semantics": (
            "TARGET=1 biểu thị payment difficulties theo định nghĩa ẩn danh của cuộc thi; "
            "không đồng nhất mặc định với Basel 90+ DPD hoặc mọi định nghĩa 'default'."
        ),
        "target_counts": {str(key): int(value) for key, value in target.value_counts(dropna=False).items()},
        "target_rate": float(target.mean()),
        "duplicate_applicant_ids": int(frame[ID_COLUMN].duplicated().sum()) if ID_COLUMN in frame else None,
        "infinite_numeric_values": infinite_count,
        "days_employed_sentinel_remaining": int((frame.get("DAYS_EMPLOYED") == 365243).sum())
        if "DAYS_EMPLOYED" in frame
        else None,
        "top_missing_columns": [
            {"field": str(column), "missing_rate": float(rate)} for column, rate in missing.head(20).items()
        ],
        "time_columns_present": [column for column in TIME_COLUMN_CANDIDATES if column in frame],
    }


def make_split(frame: pd.DataFrame, *, seed: int) -> SplitResult:
    target = frame[TARGET_COLUMN].to_numpy(dtype=int)
    time_column = next((column for column in TIME_COLUMN_CANDIDATES if column in frame), None)
    indices = np.arange(len(frame))

    if time_column is not None:
        temporal = pd.Series(frame[time_column])
        if "date" in time_column.lower():
            temporal = pd.to_datetime(temporal, errors="coerce")
        valid_time = temporal.notna()
        if valid_time.all() and temporal.nunique() >= 10:
            ordered_periods = np.sort(temporal.unique())
            train_cut = ordered_periods[max(0, int(len(ordered_periods) * 0.70) - 1)]
            validation_cut = ordered_periods[max(1, int(len(ordered_periods) * 0.85) - 1)]
            train_index = indices[(temporal <= train_cut).to_numpy()]
            validation_index = indices[((temporal > train_cut) & (temporal <= validation_cut)).to_numpy()]
            test_index = indices[(temporal > validation_cut).to_numpy()]
            if all(len(np.unique(target[split])) == 2 for split in (train_index, validation_index, test_index)):
                return SplitResult(
                    train=train_index,
                    validation=validation_index,
                    test=test_index,
                    strategy="chronological_70_15_15",
                    time_column=time_column,
                )

    train_validation, test = train_test_split(indices, test_size=0.15, random_state=seed, stratify=target)
    train, validation = train_test_split(
        train_validation,
        test_size=0.1764705882,
        random_state=seed,
        stratify=target[train_validation],
    )
    return SplitResult(
        train=np.sort(train),
        validation=np.sort(validation),
        test=np.sort(test),
        strategy="stratified_random_70_15_15_no_temporal_claim",
        time_column=None,
    )


def select_model_features(train: pd.DataFrame, *, feature_set: str) -> tuple[list[str], dict[str, list[str]]]:
    excluded = {
        TARGET_COLUMN,
        ID_COLUMN,
        *PROTECTED_COLUMNS,
        *TIME_COLUMN_CANDIDATES,
        "SK_ID_BUREAU",
        "SK_ID_PREV",
    }
    if feature_set == "serving":
        candidates = [
            *SERVING_RAW_FEATURES,
            *ENGINEERED_FEATURES,
            "DAYS_EMPLOYED_ANOMALY",
        ]
        candidates = [column for column in candidates if column in train and column not in excluded]
    else:
        candidates = [column for column in train if column not in excluded and not column.startswith("SK_ID_")]

    selected: list[str] = []
    dropped: dict[str, list[str]] = {"missing_gt_95pct": [], "constant": [], "high_cardinality_category": []}
    for column in candidates:
        series = train[column]
        if float(series.isna().mean()) > 0.95:
            dropped["missing_gt_95pct"].append(column)
        elif series.nunique(dropna=True) <= 1:
            dropped["constant"].append(column)
        elif not pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 100:
            dropped["high_cardinality_category"].append(column)
        else:
            selected.append(column)
    return selected, dropped


def build_preprocessor(frame: pd.DataFrame, columns: list[str]) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric = [column for column in columns if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in columns if column not in numeric]
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]
    )
    preprocessor = ColumnTransformer(
        [("num", numeric_pipeline, numeric), ("cat", categorical_pipeline, categorical)],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )
    return preprocessor, numeric, categorical


def _candidate_models(seed: int, *, include_logistic: bool = True) -> dict[str, Any]:
    models: dict[str, Any] = {
        "lightgbm": lgb.LGBMClassifier(
            objective="binary",
            n_estimators=5000,
            learning_rate=0.02,
            num_leaves=63,
            max_depth=8,
            min_child_samples=90,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        ),
    }
    if include_logistic:
        models["logistic_regression"] = LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=600,
            solver="liblinear",
            random_state=seed,
        )
    return models


def _fit_candidate(
    name: str,
    estimator: Any,
    x_train: Any,
    y_train: np.ndarray,
    x_validation: Any,
    y_validation: np.ndarray,
) -> Any:
    if name == "lightgbm":
        estimator.fit(
            x_train,
            y_train,
            eval_X=x_validation,
            eval_y=y_validation,
            eval_metric="auc",
            callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
        )
    else:
        estimator.fit(x_train, y_train)
    return estimator


def _risk_thresholds(probability: np.ndarray) -> list[float]:
    values = [float(value) for value in np.quantile(probability, [0.50, 0.80, 0.95])]
    for index in range(1, len(values)):
        values[index] = max(values[index], values[index - 1] + 1e-6)
    return [float(np.clip(value, 0.0, 1.0)) for value in values]


def _write_model_card(path: Path, report: dict[str, Any]) -> None:
    test = report["champion_test_metrics"]
    content = f"""# Model Card — Home Credit Payment Difficulty POC

## Phạm vi

- Model: `{report["champion"]}`
- Version: `{report["model_version"]}`
- Feature set: `{report["feature_set"]}`
- Split: `{report["split"]["strategy"]}`
- Số mẫu: `{report["data_profile"]["rows"]}`

## Kết quả trên test holdout

| Metric | Giá trị |
|---|---:|
| ROC-AUC | {test["roc_auc"]:.6f} |
| PR-AUC | {test["pr_auc"]:.6f} |
| KS | {test["ks"]:.6f} |
| Gini | {test["gini"]:.6f} |
| Brier | {test["brier"]:.6f} |
| ECE (10 bins) | {test["ece_10"]:.6f} |

## Giới hạn bắt buộc

1. `TARGET` là payment difficulty theo định nghĩa ẩn danh của cuộc thi, không phải định nghĩa PD pháp lý phổ quát.
2. Home Credit 2018 không có application timestamp; split hiện tại không chứng minh temporal stability.
3. Artifact này là POC nghiên cứu, không phải policy phê duyệt/từ chối khoản vay.
4. LLM, nếu bật, chỉ diễn đạt reason codes; không được thay đổi xác suất, risk band hoặc kết luận model.
"""
    path.write_text(content, encoding="utf-8")


def train_credit_model(
    *,
    data_dir: Path,
    output_dir: Path,
    feature_set: str = "serving",
    sample_size: int | None = None,
    seed: int = 42,
) -> tuple[CreditModelBundle, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = build_home_credit_features(data_dir, feature_set=feature_set, sample_size=sample_size)
    if frame[TARGET_COLUMN].isna().any() or frame[TARGET_COLUMN].nunique() != 2:
        raise ValueError("TARGET must be a complete binary column")
    if ID_COLUMN in frame and frame[ID_COLUMN].duplicated().any():
        raise ValueError("SK_ID_CURR must be unique at applicant level")

    split = make_split(frame, seed=seed)
    train_frame = frame.iloc[split.train]
    validation_frame = frame.iloc[split.validation]
    test_frame = frame.iloc[split.test]
    feature_columns, dropped = select_model_features(train_frame, feature_set=feature_set)
    preprocessor, numeric, categorical = build_preprocessor(train_frame, feature_columns)

    x_train = preprocessor.fit_transform(train_frame[feature_columns])
    x_validation = preprocessor.transform(validation_frame[feature_columns])
    x_test = preprocessor.transform(test_frame[feature_columns])
    y_train = train_frame[TARGET_COLUMN].to_numpy(dtype=int)
    y_validation = validation_frame[TARGET_COLUMN].to_numpy(dtype=int)
    y_test = test_frame[TARGET_COLUMN].to_numpy(dtype=int)

    candidates: dict[str, Any] = {}
    validation_reports: dict[str, dict[str, Any]] = {}
    selection_scores: dict[str, float] = {}
    for name, estimator in _candidate_models(seed, include_logistic=feature_set != "full").items():
        estimator = _fit_candidate(name, estimator, x_train, y_train, x_validation, y_validation)
        probability = estimator.predict_proba(x_validation)[:, 1]
        metrics = credit_metrics(y_validation, probability)
        stability: dict[str, Any] | None = None
        selection_score = metrics["roc_auc"]
        if split.time_column is not None:
            stability = gini_stability_by_period(
                y_validation,
                probability,
                validation_frame[split.time_column],
            )
            if stability.get("available"):
                selection_score = float(stability["gini_stability"])
        candidates[name] = estimator
        validation_reports[name] = {"metrics": metrics, "stability": stability, "selection_score": selection_score}
        selection_scores[name] = selection_score

    champion_name = max(selection_scores, key=selection_scores.get)
    champion = candidates[champion_name]
    validation_raw_probability = champion.predict_proba(x_validation)[:, 1]
    calibrator = PlattCalibrator().fit(validation_raw_probability, y_validation)
    validation_probability = calibrator.predict(validation_raw_probability)
    test_raw_probability = champion.predict_proba(x_test)[:, 1]
    test_probability = calibrator.predict(test_raw_probability)
    train_probability = calibrator.predict(champion.predict_proba(x_train)[:, 1])
    decision_threshold = select_ks_threshold(y_validation, validation_probability)

    model_time = datetime.now(UTC).replace(microsecond=0)
    source_path = data_dir / "application_train.csv"
    profile = profile_training_data(frame, source_path=source_path)
    version = f"hc-poc-{model_time:%Y%m%dT%H%M%SZ}-{profile['source_sha256'][:8]}"

    raw_input_features = [
        column for column in feature_columns if column not in ENGINEERED_FEATURES and column != "DAYS_EMPLOYED_ANOMALY"
    ]
    required = [
        column
        for column in (
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
        )
        if column in raw_input_features
    ]
    bundle = CreditModelBundle(
        preprocessor=preprocessor,
        estimator=champion,
        calibrator=calibrator,
        feature_columns=feature_columns,
        numeric_columns=numeric,
        categorical_columns=categorical,
        raw_input_features=raw_input_features,
        required_input_features=required,
        model_name=champion_name,
        model_version=version,
        risk_thresholds=_risk_thresholds(validation_probability),
        decision_threshold=decision_threshold,
        metadata={
            "trained_at": model_time.isoformat(),
            "feature_set": feature_set,
            "sample_size": sample_size,
            "split_strategy": split.strategy,
            "source_sha256": profile["source_sha256"],
            "target_semantics": profile["target_semantics"],
        },
    )

    stability_report = (
        gini_stability_by_period(y_test, test_probability, test_frame[split.time_column])
        if split.time_column is not None
        else {
            "available": False,
            "reason": (
                "Home Credit 2018 không có WEEK_NUM/date_decision. Không dùng SK_ID hoặc relative DAYS làm timestamp giả."
            ),
        }
    )
    fairness: dict[str, Any] | None = None
    if "CODE_GENDER" in test_frame:
        fairness = fairness_report(
            y_test,
            test_probability,
            test_frame["CODE_GENDER"],
            threshold=decision_threshold,
            min_group_size=max(25, min(100, len(test_frame) // 20)),
        )

    report: dict[str, Any] = {
        "model_version": version,
        "feature_set": feature_set,
        "seed": seed,
        "data_profile": profile,
        "split": {
            "strategy": split.strategy,
            "time_column": split.time_column,
            "train_rows": int(len(split.train)),
            "validation_rows": int(len(split.validation)),
            "test_rows": int(len(split.test)),
        },
        "selected_feature_count": len(feature_columns),
        "dropped_features": dropped,
        "candidate_validation": validation_reports,
        "champion": champion_name,
        "calibration_validation_raw": credit_metrics(y_validation, validation_raw_probability),
        "calibration_validation_platt": credit_metrics(y_validation, validation_probability),
        "champion_test_metrics": credit_metrics(y_test, test_probability),
        "score_psi_train_vs_test": population_stability_index(train_probability, test_probability),
        "temporal_stability": stability_report,
        "fairness_by_gender": fairness,
        "decision_threshold_for_diagnostics_only": decision_threshold,
        "risk_band_quantile_thresholds": bundle.risk_thresholds,
        "limitations": [
            "Không có temporal key trong Home Credit 2018 nên chưa có out-of-time validation.",
            "Risk band dựa trên quantile validation, không phải lending policy hay risk appetite đã phê duyệt.",
            "Fairness report là chẩn đoán dataset, không phải chứng nhận tuân thủ hoặc fairness nhân quả.",
            "POC form dùng application-level features; full relational history cần JSON/pipeline nâng cao.",
        ],
    }

    artifact_path = output_dir / "credit_model.joblib"
    bundle.save(artifact_path)
    (output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    _write_model_card(output_dir / "MODEL_CARD.md", report)
    return bundle, report
