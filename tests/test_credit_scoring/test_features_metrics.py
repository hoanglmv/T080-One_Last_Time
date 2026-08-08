import numpy as np
import pandas as pd
import pytest

from src.credit_scoring.features import engineer_application_features, safe_divide
from src.credit_scoring.metrics import (
    credit_metrics,
    gini_stability_by_period,
    population_stability_index,
)


def test_safe_divide_does_not_add_arbitrary_offset():
    numerator = pd.Series([10.0, 10.0, 5.0])
    denominator = pd.Series([2.0, 0.0, np.nan])
    result = safe_divide(numerator, denominator)
    assert result.iloc[0] == 5.0
    assert result.iloc[1:].isna().all()


def test_application_features_handle_days_employed_sentinel():
    frame = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [100_000.0],
            "AMT_CREDIT": [250_000.0],
            "AMT_ANNUITY": [25_000.0],
            "AMT_GOODS_PRICE": [200_000.0],
            "DAYS_BIRTH": [-3652.5],
            "DAYS_EMPLOYED": [365243],
            "CNT_FAM_MEMBERS": [2],
            "CNT_CHILDREN": [0],
            "EXT_SOURCE_1": [0.4],
            "EXT_SOURCE_2": [0.5],
            "EXT_SOURCE_3": [0.6],
        }
    )
    result = engineer_application_features(frame)
    assert result.loc[0, "DAYS_EMPLOYED_ANOMALY"] == 1
    assert pd.isna(result.loc[0, "DAYS_EMPLOYED"])
    assert result.loc[0, "FE_CREDIT_TO_INCOME"] == pytest.approx(2.5)
    assert result.loc[0, "FE_EXT_SOURCE_MEAN"] == pytest.approx(0.5)


def test_credit_and_stability_metrics_are_consistent():
    target = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    probability = np.array([0.1, 0.9, 0.2, 0.8, 0.15, 0.85, 0.3, 0.7])
    report = credit_metrics(target, probability)
    assert report["roc_auc"] == pytest.approx(1.0)
    assert report["gini"] == pytest.approx(1.0)
    assert report["ks"] == pytest.approx(1.0)

    stability = gini_stability_by_period(target, probability, [1, 1, 1, 1, 2, 2, 2, 2])
    assert stability["available"] is True
    assert stability["gini_stability"] == pytest.approx(1.0)


def test_psi_is_zero_for_identical_distributions():
    reference = np.linspace(0.01, 0.99, 1000)
    assert population_stability_index(reference, reference) == pytest.approx(0.0)
