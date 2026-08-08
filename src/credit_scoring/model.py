"""Serializable model bundle and deterministic local explanations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.credit_scoring.features import FEATURE_DESCRIPTIONS, engineer_application_features

BLOCKED_INPUT_FIELDS = {
    "TARGET",
    "SK_ID_CURR",
    "FULL_NAME",
    "NAME",
    "ADDRESS",
    "PHONE",
    "EMAIL",
    "CODE_GENDER",
}


@dataclass
class PlattCalibrator:
    """One-dimensional logistic calibration fitted on a dedicated holdout."""

    estimator: LogisticRegression | None = None

    def fit(self, probability: np.ndarray, target: np.ndarray) -> PlattCalibrator:
        clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        self.estimator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
        self.estimator.fit(logits, np.asarray(target, dtype=int))
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        if self.estimator is None:
            return np.asarray(probability, dtype=float)
        clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        return self.estimator.predict_proba(logits)[:, 1]


@dataclass
class CreditModelBundle:
    """All state required to reproduce training-time transformations online."""

    preprocessor: Any
    estimator: Any
    calibrator: PlattCalibrator
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    raw_input_features: list[str]
    required_input_features: list[str]
    model_name: str
    model_version: str
    risk_thresholds: list[float]
    decision_threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str) -> CreditModelBundle:
        bundle = joblib.load(path)
        if not isinstance(bundle, cls):
            raise TypeError(f"Artifact at {path} is not a CreditModelBundle")
        return bundle

    def save(self, path: Path | str) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output, compress=3)

    @staticmethod
    def _normalize_records(records: list[dict[str, Any]]) -> pd.DataFrame:
        normalized: list[dict[str, Any]] = []
        for record in records:
            normalized.append({str(key).upper(): value for key, value in record.items()})
        return pd.DataFrame(normalized)

    def prepare(self, records: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        raw = self._normalize_records(records)
        quality: list[dict[str, Any]] = []
        prepared_rows: list[dict[str, Any]] = []

        for _, row in raw.iterrows():
            supplied = {column for column in raw.columns if pd.notna(row.get(column))}
            blocked = sorted(supplied & BLOCKED_INPUT_FIELDS)
            accepted = supplied & set(self.raw_input_features)
            unknown = sorted(supplied - set(self.raw_input_features) - set(BLOCKED_INPUT_FIELDS))
            missing_required = sorted(set(self.required_input_features) - accepted)
            completeness = 1.0 - len(missing_required) / max(1, len(self.required_input_features))
            warnings: list[str] = []
            if blocked:
                warnings.append("Các trường định danh/target/protected đã bị loại trước khi chấm điểm.")
            if completeness < 0.7:
                warnings.append("Hồ sơ thiếu nhiều trường quan trọng; kết quả có độ tin cậy thấp hơn.")
            if unknown:
                warnings.append("Một số trường không thuộc schema của model và đã bị bỏ qua.")
            # Financial sanity & extreme leverage check
            income_val = float(row.get("AMT_INCOME_TOTAL", np.nan) or np.nan)
            credit_val = float(row.get("AMT_CREDIT", np.nan) or np.nan)
            annuity_val = float(row.get("AMT_ANNUITY", np.nan) or np.nan)
            if pd.notna(income_val) and income_val > 0:
                if pd.notna(credit_val) and (credit_val / income_val > 25.0 or credit_val > 1e12):
                    warnings.append(
                        "Cảnh báo đòn bẩy tài chính cực hạn: Khoản vay vượt quá 25 lần thu nhập hoặc là bất thường OOD."
                    )
                if pd.notna(annuity_val) and annuity_val / income_val > 1.5:
                    warnings.append(
                        "Cảnh báo đòn bẩy tài chính cực hạn: Nghĩa vụ trả hàng tháng vượt quá 150% thu nhập."
                    )
            quality.append(
                {
                    "completeness": round(float(completeness), 4),
                    "missing_required_fields": missing_required,
                    "ignored_fields": sorted(set(blocked + unknown)),
                    "warnings": warnings,
                }
            )
            prepared_rows.append({column: row.get(column, np.nan) for column in self.raw_input_features})

        frame = engineer_application_features(pd.DataFrame(prepared_rows))
        for column in self.feature_columns:
            if column not in frame:
                frame[column] = np.nan
        frame = frame[self.feature_columns].copy()
        for column in self.numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in self.categorical_columns:
            frame[column] = frame[column].map(lambda value: str(value) if pd.notna(value) else np.nan)
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        return frame, quality

    def _raw_probability(self, transformed: Any) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(transformed)[:, 1], dtype=float)

    def predict_proba(self, records: list[dict[str, Any]]) -> tuple[np.ndarray, pd.DataFrame, list[dict[str, Any]]]:
        frame, quality = self.prepare(records)
        transformed = self.preprocessor.transform(frame)
        raw_probability = self._raw_probability(transformed)
        probability = np.clip(self.calibrator.predict(raw_probability), 0.0, 1.0)

        # Apply Financial Sanity Risk Floor scaling dynamically for extreme Out-of-Distribution leverage
        for idx, row in enumerate(records):
            norm_row = {str(k).upper(): v for k, v in row.items()}
            try:
                inc = float(norm_row.get("AMT_INCOME_TOTAL", 0) or 0)
                crd = float(norm_row.get("AMT_CREDIT", 0) or 0)
                ann = float(norm_row.get("AMT_ANNUITY", 0) or 0)
                if inc > 0:
                    lev_credit = crd / inc
                    lev_ann = ann / inc
                    if lev_credit > 100.0 or crd > 1e12 or lev_ann > 5.0:
                        probability[idx] = max(probability[idx], 0.995)
                    elif lev_credit > 50.0 or lev_ann > 3.0:
                        probability[idx] = max(probability[idx], 0.95)
                    elif lev_credit > 25.0 or lev_ann > 1.8:
                        probability[idx] = max(probability[idx], 0.88)
            except Exception:
                pass

        return probability, frame, quality

    def risk_band(self, probability: float) -> str:
        low, high, very_high = self.risk_thresholds
        if probability < low:
            return "low"
        if probability < high:
            return "moderate"
        if probability < very_high:
            return "high"
        return "very_high"

    @staticmethod
    def _json_value(value: Any) -> Any:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _base_feature_name(self, transformed_name: str) -> str:
        name = transformed_name.split("__", 1)[-1]
        if name.startswith("missingindicator_"):
            return name.removeprefix("missingindicator_")
        # Longest match avoids confusing e.g. AMT_CREDIT with AMT_CREDIT_SUM.
        for column in sorted(self.categorical_columns, key=len, reverse=True):
            if name == column or name.startswith(f"{column}_"):
                return column
        return name

    def _contributions(self, transformed: Any) -> np.ndarray:
        if self.model_name == "lightgbm" and hasattr(self.estimator, "booster_"):
            contribution = self.estimator.booster_.predict(transformed, pred_contrib=True)
            if hasattr(contribution, "toarray"):
                arr = contribution.toarray()
            else:
                arr = np.asarray(contribution)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr[:, :-1]
        if self.model_name == "logistic_regression" and hasattr(self.estimator, "coef_"):
            dense = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
            if dense.ndim == 1:
                dense = dense.reshape(1, -1)
            return dense * np.asarray(self.estimator.coef_[0])
        raise RuntimeError(f"Local contribution is not implemented for {self.model_name}")

    def explain(self, records: list[dict[str, Any]], top_k: int = 6) -> list[list[dict[str, Any]]]:
        frame, _ = self.prepare(records)
        transformed = self.preprocessor.transform(frame)
        contributions = self._contributions(transformed)
        transformed_names = list(self.preprocessor.get_feature_names_out())
        explanations: list[list[dict[str, Any]]] = []

        for row_index in range(len(frame)):
            grouped: dict[str, float] = {}
            for name, contribution in zip(transformed_names, contributions[row_index], strict=True):
                base_name = self._base_feature_name(name)
                grouped[base_name] = grouped.get(base_name, 0.0) + float(contribution)
            ranked = sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)[:top_k]
            row_explanations: list[dict[str, Any]] = []
            for feature, contribution in ranked:
                direction = "increase_risk" if contribution > 0 else "decrease_risk"
                description = FEATURE_DESCRIPTIONS.get(feature, feature.replace("_", " ").lower())
                effect = "làm tăng" if contribution > 0 else "làm giảm"
                row_explanations.append(
                    {
                        "feature": feature,
                        "description": description,
                        "value": self._json_value(frame.iloc[row_index].get(feature)),
                        "contribution": round(contribution, 6),
                        "direction": direction,
                        "reason": f"{description} đang {effect} mức rủi ro dự đoán so với mức nền của model.",
                    }
                )
            explanations.append(row_explanations)
        return explanations

    def score(self, records: list[dict[str, Any]], *, top_k: int = 6) -> list[dict[str, Any]]:
        probability, _, quality = self.predict_proba(records)
        explanations = self.explain(records, top_k=top_k)
        results: list[dict[str, Any]] = []
        for index, value in enumerate(probability):
            pd_value = float(value)
            results.append(
                {
                    "model_version": self.model_version,
                    "model_name": self.model_name,
                    "payment_difficulty_probability": round(pd_value, 6),
                    "poc_score": round(100.0 * (1.0 - pd_value), 2),
                    "poc_score_definition": "100 × (1 - calibrated probability); đây không phải điểm CIC/credit bureau.",
                    "risk_band": self.risk_band(pd_value),
                    "top_factors": explanations[index],
                    "data_quality": quality[index],
                    "disclaimer": (
                        "POC nghiên cứu trên dữ liệu Home Credit 2018; không dùng như quyết định phê duyệt/từ chối tự động."
                    ),
                }
            )
        return results
