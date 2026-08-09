import numpy as np
import pandas as pd
import pytest

from src.credit_scoring.features import (
    aggregate_credit_card,
    aggregate_pos_cash,
    engineer_application_features,
    grouped_linear_trend,
    safe_divide,
)
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


def test_grouped_linear_trend_uses_relative_time_direction():
    frame = pd.DataFrame(
        {
            "loan": [1, 1, 1, 2, 2],
            "month": [-3, -2, -1, -2, -1],
            "dpd": [0.0, 2.0, 4.0, 3.0, 3.0],
        }
    )
    trend = grouped_linear_trend(
        frame,
        group_column="loan",
        time_column="month",
        value_column="dpd",
        output_column="trend",
    ).set_index("loan")
    assert trend.loc[1, "trend"] == pytest.approx(2.0)
    assert trend.loc[2, "trend"] == pytest.approx(0.0)


def test_balance_tables_are_aggregated_via_contract_level(tmp_path):
    pos = pd.DataFrame(
        {
            "SK_ID_CURR": [10, 10, 10, 10],
            "SK_ID_PREV": [101, 101, 102, 102],
            "MONTHS_BALANCE": [-2, -1, -2, -1],
            "CNT_INSTALMENT": [4, 4, 6, 6],
            "CNT_INSTALMENT_FUTURE": [2, 1, 5, 4],
            "SK_DPD": [0, 3, 0, 0],
            "SK_DPD_DEF": [0, 1, 0, 0],
        }
    )
    card = pd.DataFrame(
        {
            "SK_ID_CURR": [10, 10],
            "SK_ID_PREV": [201, 201],
            "MONTHS_BALANCE": [-2, -1],
            "AMT_BALANCE": [50.0, 80.0],
            "AMT_CREDIT_LIMIT_ACTUAL": [100.0, 100.0],
            "AMT_DRAWINGS_CURRENT": [20.0, 30.0],
            "AMT_PAYMENT_CURRENT": [10.0, 5.0],
            "AMT_PAYMENT_TOTAL_CURRENT": [10.0, 5.0],
            "AMT_INST_MIN_REGULARITY": [8.0, 10.0],
            "AMT_TOTAL_RECEIVABLE": [50.0, 80.0],
            "SK_DPD": [0, 2],
            "SK_DPD_DEF": [0, 1],
        }
    )
    pos.to_csv(tmp_path / "POS_CASH_balance.csv", index=False)
    card.to_csv(tmp_path / "credit_card_balance.csv", index=False)

    pos_features = aggregate_pos_cash(tmp_path).set_index("SK_ID_CURR")
    card_features = aggregate_credit_card(tmp_path).set_index("SK_ID_CURR")

    assert pos_features.loc[10, "POS_LOAN_COUNT"] == 2
    assert pos_features.loc[10, "POS_FE_POS_DPD_TREND_MEAN"] > 0
    assert card_features.loc[10, "CC_CARD_COUNT"] == 1
    assert card_features.loc[10, "CC_CC_LOAN_FE_CC_UNDERPAID_MIN_SUM_MEAN"] == pytest.approx(1.0)


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


def test_alternative_only_feature_set_has_no_traditional_features(tmp_path):
    from src.credit_scoring.features import build_home_credit_features

    # Create dummy application_train.csv
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [100001],
            "TARGET": [0],
            "CODE_GENDER": ["M"],
            "NAME_CONTRACT_TYPE": ["Cash loans"],
            "AMT_INCOME_TOTAL": [150000.0],
            "AMT_CREDIT": [300000.0],
            "AMT_ANNUITY": [15000.0],
            "DAYS_BIRTH": [-10000],
            "DAYS_EMPLOYED": [-1000],
            "DAYS_LAST_PHONE_CHANGE": [-100],
            "EXT_SOURCE_1": [0.5],
            "EXT_SOURCE_2": [0.6],
            "EXT_SOURCE_3": [0.7],
        }
    )
    df.to_csv(tmp_path / "application_train.csv", index=False)

    frame = build_home_credit_features(tmp_path, feature_set="alternative_only")

    # Verify no traditional features present
    excluded = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_CREDIT", "AMT_ANNUITY", "FE_CREDIT_TO_INCOME"]
    for col in excluded:
        assert col not in frame.columns

    # Verify alternative features present
    assert "DAYS_LAST_PHONE_CHANGE" in frame.columns
    assert "AMT_INCOME_TOTAL" in frame.columns

