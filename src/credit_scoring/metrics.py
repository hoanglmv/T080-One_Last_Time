"""Credit-risk discrimination, calibration, stability and fairness metrics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _as_float_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def expected_calibration_error(y_true: Iterable[int], probability: Iterable[float], bins: int = 10) -> float:
    y = np.asarray(list(y_true), dtype=int)
    probability_array = np.clip(_as_float_array(probability), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.digitize(probability_array, edges[1:-1], right=True)
    error = 0.0
    for index in range(bins):
        mask = indices == index
        if not mask.any():
            continue
        error += mask.mean() * abs(float(y[mask].mean()) - float(probability_array[mask].mean()))
    return float(error)


def select_ks_threshold(y_true: Iterable[int], probability: Iterable[float]) -> float:
    y = np.asarray(list(y_true), dtype=int)
    probability_array = _as_float_array(probability)
    fpr, tpr, thresholds = roc_curve(y, probability_array)
    valid = np.isfinite(thresholds)
    if not valid.any():
        return 0.5
    valid_indices = np.flatnonzero(valid)
    best = valid_indices[int(np.argmax((tpr - fpr)[valid]))]
    return float(np.clip(thresholds[best], 0.0, 1.0))


def credit_metrics(y_true: Iterable[int], probability: Iterable[float]) -> dict[str, float]:
    """Return ranking and probability metrics without choosing a business policy."""
    y = np.asarray(list(y_true), dtype=int)
    probability_array = np.clip(_as_float_array(probability), 1e-7, 1 - 1e-7)
    auc = float(roc_auc_score(y, probability_array))
    fpr, tpr, _ = roc_curve(y, probability_array)
    precision, recall, _ = precision_recall_curve(y, probability_array)
    return {
        "roc_auc": auc,
        "pr_auc": float(np.trapezoid(precision[::-1], recall[::-1])),
        "average_precision": float(average_precision_score(y, probability_array)),
        "ks": float(np.max(tpr - fpr)),
        "gini": float(2 * auc - 1),
        "brier": float(brier_score_loss(y, probability_array)),
        "log_loss": float(log_loss(y, probability_array, labels=[0, 1])),
        "ece_10": expected_calibration_error(y, probability_array, bins=10),
        "default_rate": float(y.mean()),
        "mean_prediction": float(probability_array.mean()),
        "n": float(len(y)),
    }


def population_stability_index(
    reference: Iterable[float],
    current: Iterable[float],
    *,
    bins: int = 10,
) -> float:
    """Calculate PSI using bins fixed exclusively from the reference sample."""
    reference_array = _as_float_array(reference)
    current_array = _as_float_array(current)
    reference_array = reference_array[np.isfinite(reference_array)]
    current_array = current_array[np.isfinite(current_array)]
    if not len(reference_array) or not len(current_array):
        return float("nan")
    edges = np.unique(np.quantile(reference_array, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(reference_array, bins=edges)[0].astype(float)
    cur_counts = np.histogram(current_array, bins=edges)[0].astype(float)
    epsilon = 1e-6
    ref_rate = np.clip(ref_counts / ref_counts.sum(), epsilon, None)
    cur_rate = np.clip(cur_counts / cur_counts.sum(), epsilon, None)
    return float(np.sum((cur_rate - ref_rate) * np.log(cur_rate / ref_rate)))


def gini_stability_by_period(
    y_true: Iterable[int],
    probability: Iterable[float],
    periods: Iterable[Any],
) -> dict[str, Any]:
    """Compute the official competition-style weekly Gini stability components."""
    frame = pd.DataFrame(
        {
            "target": list(y_true),
            "probability": list(probability),
            "period": list(periods),
        }
    ).dropna(subset=["period"])
    rows: list[dict[str, Any]] = []
    for period, group in frame.groupby("period", sort=True):
        if len(group) < 2 or group["target"].nunique() < 2:
            continue
        auc = float(roc_auc_score(group["target"], group["probability"]))
        rows.append({"period": period, "n": len(group), "auc": auc, "gini": 2 * auc - 1})
    if len(rows) < 2:
        return {
            "available": False,
            "reason": "Cần ít nhất hai giai đoạn có đủ cả hai lớp target.",
            "period_metrics": rows,
        }

    x = np.arange(len(rows), dtype=float)
    gini = np.asarray([row["gini"] for row in rows], dtype=float)
    slope, intercept = np.polyfit(x, gini, 1)
    residuals = gini - (slope * x + intercept)
    residual_std = float(np.std(residuals))
    score = float(gini.mean() + 88.0 * min(0.0, float(slope)) - 0.5 * residual_std)
    return {
        "available": True,
        "period_metrics": rows,
        "mean_gini": float(gini.mean()),
        "slope": float(slope),
        "residual_std": residual_std,
        "gini_stability": score,
    }


def fairness_report(
    y_true: Iterable[int],
    probability: Iterable[float],
    groups: Iterable[Any],
    *,
    threshold: float,
    min_group_size: int = 100,
) -> dict[str, Any]:
    """Audit group metrics on an untouched validation/test sample.

    This is a diagnostic report, not a proof of legal or substantive fairness.
    """
    frame = pd.DataFrame(
        {
            "target": list(y_true),
            "probability": list(probability),
            "group": list(groups),
        }
    ).dropna(subset=["group"])
    metrics: list[dict[str, Any]] = []
    for name, group in frame.groupby("group", sort=True):
        if len(group) < min_group_size:
            continue
        target = group["target"].to_numpy(dtype=int)
        prediction = group["probability"].to_numpy(dtype=float) >= threshold
        positive = target == 1
        negative = ~positive
        row: dict[str, Any] = {
            "group": str(name),
            "n": int(len(group)),
            "default_rate": float(target.mean()),
            "predicted_high_risk_rate": float(prediction.mean()),
            "mean_prediction": float(group["probability"].mean()),
            "tpr": float(prediction[positive].mean()) if positive.any() else None,
            "fpr": float(prediction[negative].mean()) if negative.any() else None,
            "brier": float(brier_score_loss(target, group["probability"])),
            "roc_auc": float(roc_auc_score(target, group["probability"])) if len(np.unique(target)) == 2 else None,
        }
        metrics.append(row)

    numeric_keys = ("predicted_high_risk_rate", "tpr", "fpr", "brier", "roc_auc")
    gaps: dict[str, float] = {}
    for key in numeric_keys:
        values = [float(row[key]) for row in metrics if row[key] is not None]
        if values:
            gaps[f"max_min_{key}_gap"] = max(values) - min(values)
    return {
        "available": len(metrics) >= 2,
        "threshold": threshold,
        "groups": metrics,
        "gaps": gaps,
        "limitation": "Chỉ là kiểm tra chẩn đoán trên dataset; không xác lập tuân thủ pháp lý hoặc fairness nhân quả.",
    }
