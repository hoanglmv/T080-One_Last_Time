"""Feature engineering shared by offline training and online scoring.

The implementation adapts public ideas from the two Home Credit winning
solutions without copying leaderboard-only post-processing. All joins are
target-free and all event-based features are built from observations at or
before the application reference date (relative day/month <= 0).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"
PROTECTED_COLUMNS = ("CODE_GENDER",)
TIME_COLUMN_CANDIDATES = ("WEEK_NUM", "date_decision", "DATE_DECISION")

# A compact field set that a user can reasonably provide in the POC form.
SERVING_RAW_FEATURES = (
    "NAME_CONTRACT_TYPE",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
)

ENGINEERED_FEATURES = (
    "FE_CREDIT_TO_INCOME",
    "FE_ANNUITY_TO_INCOME",
    "FE_PAYMENT_RATE",
    "FE_CREDIT_TO_GOODS",
    "FE_INCOME_PER_PERSON",
    "FE_INCOME_PER_CHILD",
    "FE_EMPLOYMENT_TO_AGE",
    "FE_AGE_YEARS",
    "FE_EMPLOYMENT_YEARS",
    "FE_REGISTRATION_YEARS",
    "FE_ID_PUBLISH_YEARS",
    "FE_PHONE_CHANGE_YEARS",
    "FE_PHONE_TO_AGE",
    "FE_EXT_SOURCE_MEAN",
    "FE_EXT_SOURCE_STD",
    "FE_EXT_SOURCE_MIN",
    "FE_EXT_SOURCE_MAX",
    "FE_EXT_SOURCE_PRODUCT",
    "FE_EXT_SOURCE_MISSING_COUNT",
)

FEATURE_DESCRIPTIONS = {
    "AMT_INCOME_TOTAL": "Thu nhập khai báo của khách hàng",
    "AMT_CREDIT": "Giá trị khoản tín dụng đề nghị",
    "AMT_ANNUITY": "Khoản thanh toán định kỳ",
    "AMT_GOODS_PRICE": "Giá trị hàng hóa được tài trợ",
    "DAYS_BIRTH": "Tuổi của khách hàng tại thời điểm hồ sơ",
    "DAYS_EMPLOYED": "Thời gian làm việc tại thời điểm hồ sơ",
    "EXT_SOURCE_1": "Điểm rủi ro chuẩn hóa từ nguồn dữ liệu ngoài số 1",
    "EXT_SOURCE_2": "Điểm rủi ro chuẩn hóa từ nguồn dữ liệu ngoài số 2",
    "EXT_SOURCE_3": "Điểm rủi ro chuẩn hóa từ nguồn dữ liệu ngoài số 3",
    "FE_CREDIT_TO_INCOME": "Tỷ lệ khoản vay trên thu nhập",
    "FE_ANNUITY_TO_INCOME": "Tỷ lệ nghĩa vụ trả định kỳ trên thu nhập",
    "FE_PAYMENT_RATE": "Tỷ lệ thanh toán định kỳ trên giá trị khoản vay",
    "FE_CREDIT_TO_GOODS": "Tỷ lệ khoản vay trên giá trị hàng hóa",
    "FE_INCOME_PER_PERSON": "Thu nhập bình quân theo thành viên gia đình",
    "FE_INCOME_PER_CHILD": "Thu nhập bình quân theo số con",
    "FE_EMPLOYMENT_TO_AGE": "Tỷ lệ thời gian làm việc trên tuổi",
    "FE_AGE_YEARS": "Tuổi quy đổi theo năm",
    "FE_EMPLOYMENT_YEARS": "Thời gian làm việc quy đổi theo năm",
    "FE_PHONE_CHANGE_YEARS": "Số năm kể từ lần đổi điện thoại gần nhất",
    "FE_EXT_SOURCE_MEAN": "Trung bình các điểm rủi ro bên ngoài",
    "FE_EXT_SOURCE_STD": "Mức chênh lệch giữa các điểm rủi ro bên ngoài",
    "FE_EXT_SOURCE_MIN": "Điểm rủi ro bên ngoài thấp nhất",
    "FE_EXT_SOURCE_MAX": "Điểm rủi ro bên ngoài cao nhất",
    "FE_EXT_SOURCE_PRODUCT": "Tương tác nhân giữa các điểm rủi ro bên ngoài",
    "FE_EXT_SOURCE_MISSING_COUNT": "Số nguồn điểm rủi ro bên ngoài còn thiếu",
    "BUREAU_CREDIT_COUNT": "Số khoản tín dụng được ghi nhận tại credit bureau",
    "PREV_APPLICATION_COUNT": "Số hồ sơ vay trước đây tại Home Credit",
    "INSTAL_PAYMENT_COUNT": "Số lần thanh toán trả góp quan sát được",
}


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric series or an aligned all-missing series."""
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide without arbitrary offsets; zero/invalid denominators become missing."""
    denominator = denominator.where(denominator.abs() > 1e-12)
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def engineer_application_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create row-local application features used in both training and serving."""
    result = frame.copy()
    if "DAYS_EMPLOYED" in result:
        result["DAYS_EMPLOYED_ANOMALY"] = (result["DAYS_EMPLOYED"] == 365243).astype("int8")
        result["DAYS_EMPLOYED"] = result["DAYS_EMPLOYED"].replace(365243, np.nan)

    income = _series(result, "AMT_INCOME_TOTAL")
    credit = _series(result, "AMT_CREDIT")
    annuity = _series(result, "AMT_ANNUITY")
    goods = _series(result, "AMT_GOODS_PRICE")
    family = _series(result, "CNT_FAM_MEMBERS")
    children = _series(result, "CNT_CHILDREN")
    birth_days = _series(result, "DAYS_BIRTH").abs()
    employment_days = _series(result, "DAYS_EMPLOYED").abs()

    result["FE_CREDIT_TO_INCOME"] = safe_divide(credit, income)
    result["FE_ANNUITY_TO_INCOME"] = safe_divide(annuity, income)
    result["FE_PAYMENT_RATE"] = safe_divide(annuity, credit)
    result["FE_CREDIT_TO_GOODS"] = safe_divide(credit, goods)
    result["FE_INCOME_PER_PERSON"] = safe_divide(income, family)
    result["FE_INCOME_PER_CHILD"] = safe_divide(income, children + 1.0)
    result["FE_EMPLOYMENT_TO_AGE"] = safe_divide(employment_days, birth_days)
    result["FE_AGE_YEARS"] = birth_days / 365.25
    result["FE_EMPLOYMENT_YEARS"] = employment_days / 365.25
    result["FE_REGISTRATION_YEARS"] = _series(result, "DAYS_REGISTRATION").abs() / 365.25
    result["FE_ID_PUBLISH_YEARS"] = _series(result, "DAYS_ID_PUBLISH").abs() / 365.25
    result["FE_PHONE_CHANGE_YEARS"] = _series(result, "DAYS_LAST_PHONE_CHANGE").abs() / 365.25
    result["FE_PHONE_TO_AGE"] = safe_divide(_series(result, "DAYS_LAST_PHONE_CHANGE").abs(), birth_days)

    ext_columns = [column for column in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if column in result]
    if ext_columns:
        ext = result[ext_columns].apply(pd.to_numeric, errors="coerce")
        result["FE_EXT_SOURCE_MEAN"] = ext.mean(axis=1)
        result["FE_EXT_SOURCE_STD"] = ext.std(axis=1)
        result["FE_EXT_SOURCE_MIN"] = ext.min(axis=1)
        result["FE_EXT_SOURCE_MAX"] = ext.max(axis=1)
        result["FE_EXT_SOURCE_PRODUCT"] = ext.product(axis=1, min_count=len(ext_columns))
        result["FE_EXT_SOURCE_MISSING_COUNT"] = ext.isna().sum(axis=1)
    else:
        for column in ENGINEERED_FEATURES[-6:]:
            result[column] = np.nan

    document_columns = [column for column in result if column.startswith("FLAG_DOCUMENT_")]
    if document_columns:
        result["FE_DOCUMENT_COUNT"] = result[document_columns].sum(axis=1)

    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result


def _flatten_aggregation_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    frame.columns = [f"{prefix}_{column}_{stat}".upper() for column, stat in frame.columns]
    return frame.reset_index()


def _filter_ids(frame: pd.DataFrame, applicant_ids: set[int] | None) -> pd.DataFrame:
    if applicant_ids is None:
        return frame
    return frame[frame[ID_COLUMN].isin(applicant_ids)]


def aggregate_bureau(data_dir: Path, applicant_ids: set[int] | None = None) -> pd.DataFrame:
    """Aggregate bureau records, including active/closed status and debt ratios."""
    columns = [
        ID_COLUMN,
        "SK_ID_BUREAU",
        "CREDIT_ACTIVE",
        "CREDIT_TYPE",
        "DAYS_CREDIT",
        "DAYS_CREDIT_ENDDATE",
        "DAYS_CREDIT_UPDATE",
        "CREDIT_DAY_OVERDUE",
        "AMT_CREDIT_MAX_OVERDUE",
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_OVERDUE",
        "AMT_CREDIT_SUM_LIMIT",
        "AMT_ANNUITY",
        "CNT_CREDIT_PROLONG",
    ]
    bureau = pd.read_csv(data_dir / "bureau.csv", usecols=columns)
    bureau = _filter_ids(bureau, applicant_ids)
    # Positive relative days would be future information and are excluded.
    bureau = bureau[bureau["DAYS_CREDIT"].isna() | (bureau["DAYS_CREDIT"] <= 0)].copy()
    bureau["FE_BUREAU_DEBT_TO_CREDIT"] = safe_divide(
        _series(bureau, "AMT_CREDIT_SUM_DEBT"), _series(bureau, "AMT_CREDIT_SUM")
    )
    bureau["FE_BUREAU_IS_ACTIVE"] = bureau["CREDIT_ACTIVE"].eq("Active").astype("int8")
    bureau["FE_BUREAU_IS_CLOSED"] = bureau["CREDIT_ACTIVE"].eq("Closed").astype("int8")

    numeric = [
        "DAYS_CREDIT",
        "DAYS_CREDIT_ENDDATE",
        "DAYS_CREDIT_UPDATE",
        "CREDIT_DAY_OVERDUE",
        "AMT_CREDIT_MAX_OVERDUE",
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_OVERDUE",
        "AMT_CREDIT_SUM_LIMIT",
        "AMT_ANNUITY",
        "CNT_CREDIT_PROLONG",
        "FE_BUREAU_DEBT_TO_CREDIT",
        "FE_BUREAU_IS_ACTIVE",
        "FE_BUREAU_IS_CLOSED",
    ]
    spec = {column: ["min", "max", "mean", "sum"] for column in numeric}
    overall = _flatten_aggregation_columns(bureau.groupby(ID_COLUMN).agg(spec), "BUREAU")
    counts = bureau.groupby(ID_COLUMN).size().rename("BUREAU_CREDIT_COUNT").reset_index()
    overall = overall.merge(counts, on=ID_COLUMN, how="left", validate="one_to_one")

    for status, prefix in (("Active", "BUREAU_ACTIVE"), ("Closed", "BUREAU_CLOSED")):
        subset = bureau[bureau["CREDIT_ACTIVE"].eq(status)]
        if subset.empty:
            continue
        status_spec = {
            "DAYS_CREDIT": ["min", "max", "mean"],
            "AMT_CREDIT_SUM": ["mean", "sum"],
            "AMT_CREDIT_SUM_DEBT": ["mean", "sum"],
            "CREDIT_DAY_OVERDUE": ["max", "mean"],
        }
        status_agg = _flatten_aggregation_columns(subset.groupby(ID_COLUMN).agg(status_spec), prefix)
        overall = overall.merge(status_agg, on=ID_COLUMN, how="left", validate="one_to_one")

    return overall


def _aggregate_recent_records(
    frame: pd.DataFrame,
    *,
    sort_column: str,
    value_columns: Sequence[str],
    prefix: str,
    windows: Iterable[int] = (3, 5),
) -> list[pd.DataFrame]:
    outputs: list[pd.DataFrame] = []
    ordered = frame.sort_values([ID_COLUMN, sort_column], ascending=[True, False])
    for window in windows:
        recent = ordered.groupby(ID_COLUMN, sort=False).head(window)
        spec = {column: ["mean", "max", "min"] for column in value_columns}
        outputs.append(_flatten_aggregation_columns(recent.groupby(ID_COLUMN).agg(spec), f"{prefix}_RECENT_{window}"))
    return outputs


def aggregate_previous_applications(data_dir: Path, applicant_ids: set[int] | None = None) -> pd.DataFrame:
    """Aggregate all and recent 3/5 previous applications."""
    columns = [
        ID_COLUMN,
        "SK_ID_PREV",
        "NAME_CONTRACT_STATUS",
        "DAYS_DECISION",
        "AMT_ANNUITY",
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "AMT_DOWN_PAYMENT",
        "AMT_GOODS_PRICE",
        "RATE_DOWN_PAYMENT",
        "CNT_PAYMENT",
    ]
    previous = pd.read_csv(data_dir / "previous_application.csv", usecols=columns)
    previous = _filter_ids(previous, applicant_ids)
    previous = previous[previous["DAYS_DECISION"].isna() | (previous["DAYS_DECISION"] <= 0)].copy()
    previous["FE_PREV_APPLICATION_TO_CREDIT"] = safe_divide(
        _series(previous, "AMT_APPLICATION"), _series(previous, "AMT_CREDIT")
    )
    previous["FE_PREV_APPROVED"] = previous["NAME_CONTRACT_STATUS"].eq("Approved").astype("int8")
    previous["FE_PREV_REFUSED"] = previous["NAME_CONTRACT_STATUS"].eq("Refused").astype("int8")

    values = [
        "DAYS_DECISION",
        "AMT_ANNUITY",
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "AMT_DOWN_PAYMENT",
        "AMT_GOODS_PRICE",
        "RATE_DOWN_PAYMENT",
        "CNT_PAYMENT",
        "FE_PREV_APPLICATION_TO_CREDIT",
        "FE_PREV_APPROVED",
        "FE_PREV_REFUSED",
    ]
    spec = {column: ["min", "max", "mean", "sum"] for column in values}
    output = _flatten_aggregation_columns(previous.groupby(ID_COLUMN).agg(spec), "PREV")
    counts = previous.groupby(ID_COLUMN).size().rename("PREV_APPLICATION_COUNT").reset_index()
    output = output.merge(counts, on=ID_COLUMN, how="left", validate="one_to_one")
    for recent in _aggregate_recent_records(
        previous,
        sort_column="DAYS_DECISION",
        value_columns=values[:9],
        prefix="PREV",
    ):
        output = output.merge(recent, on=ID_COLUMN, how="left", validate="one_to_one")
    return output


def aggregate_installments(data_dir: Path, applicant_ids: set[int] | None = None) -> pd.DataFrame:
    """Aggregate payment behaviour while excluding payments after the reference date."""
    columns = [
        ID_COLUMN,
        "SK_ID_PREV",
        "NUM_INSTALMENT_VERSION",
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
    ]
    installments = pd.read_csv(data_dir / "installments_payments.csv", usecols=columns)
    installments = _filter_ids(installments, applicant_ids)
    installments = installments[
        installments["DAYS_ENTRY_PAYMENT"].isna() | (installments["DAYS_ENTRY_PAYMENT"] <= 0)
    ].copy()
    installments["FE_INSTAL_PAYMENT_RATIO"] = safe_divide(
        _series(installments, "AMT_PAYMENT"), _series(installments, "AMT_INSTALMENT")
    )
    installments["FE_INSTAL_PAYMENT_DIFF"] = _series(installments, "AMT_INSTALMENT") - _series(
        installments, "AMT_PAYMENT"
    )
    installments["FE_INSTAL_DPD"] = (
        _series(installments, "DAYS_ENTRY_PAYMENT") - _series(installments, "DAYS_INSTALMENT")
    ).clip(lower=0)
    installments["FE_INSTAL_DBD"] = (
        _series(installments, "DAYS_INSTALMENT") - _series(installments, "DAYS_ENTRY_PAYMENT")
    ).clip(lower=0)
    installments["FE_INSTAL_LATE"] = installments["FE_INSTAL_DPD"].gt(0).astype("int8")

    values = [
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
        "FE_INSTAL_PAYMENT_RATIO",
        "FE_INSTAL_PAYMENT_DIFF",
        "FE_INSTAL_DPD",
        "FE_INSTAL_DBD",
        "FE_INSTAL_LATE",
    ]
    spec = {column: ["min", "max", "mean", "sum"] for column in values}
    output = _flatten_aggregation_columns(installments.groupby(ID_COLUMN).agg(spec), "INSTAL")
    counts = installments.groupby(ID_COLUMN).size().rename("INSTAL_PAYMENT_COUNT").reset_index()
    output = output.merge(counts, on=ID_COLUMN, how="left", validate="one_to_one")

    recent_year = installments[installments["DAYS_INSTALMENT"] >= -365]
    if not recent_year.empty:
        recent_spec = {
            "FE_INSTAL_DPD": ["max", "mean", "sum"],
            "FE_INSTAL_PAYMENT_RATIO": ["mean", "min"],
            "FE_INSTAL_LATE": ["mean", "sum"],
        }
        recent = _flatten_aggregation_columns(recent_year.groupby(ID_COLUMN).agg(recent_spec), "INSTAL_RECENT_12M")
        output = output.merge(recent, on=ID_COLUMN, how="left", validate="one_to_one")
    return output


def aggregate_pos_cash(data_dir: Path, applicant_ids: set[int] | None = None) -> pd.DataFrame:
    columns = [ID_COLUMN, "MONTHS_BALANCE", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE", "SK_DPD", "SK_DPD_DEF"]
    pos = pd.read_csv(data_dir / "POS_CASH_balance.csv", usecols=columns)
    pos = _filter_ids(pos, applicant_ids)
    pos = pos[pos["MONTHS_BALANCE"].isna() | (pos["MONTHS_BALANCE"] <= 0)]
    spec = {
        "MONTHS_BALANCE": ["min", "max", "mean"],
        "CNT_INSTALMENT": ["min", "max", "mean"],
        "CNT_INSTALMENT_FUTURE": ["min", "max", "mean"],
        "SK_DPD": ["max", "mean", "sum"],
        "SK_DPD_DEF": ["max", "mean", "sum"],
    }
    output = _flatten_aggregation_columns(pos.groupby(ID_COLUMN).agg(spec), "POS")
    return output.merge(
        pos.groupby(ID_COLUMN).size().rename("POS_RECORD_COUNT").reset_index(),
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )


def aggregate_credit_card(data_dir: Path, applicant_ids: set[int] | None = None) -> pd.DataFrame:
    columns = [
        ID_COLUMN,
        "MONTHS_BALANCE",
        "AMT_BALANCE",
        "AMT_CREDIT_LIMIT_ACTUAL",
        "AMT_DRAWINGS_CURRENT",
        "AMT_PAYMENT_CURRENT",
        "AMT_PAYMENT_TOTAL_CURRENT",
        "AMT_TOTAL_RECEIVABLE",
        "SK_DPD",
        "SK_DPD_DEF",
    ]
    card = pd.read_csv(data_dir / "credit_card_balance.csv", usecols=columns)
    card = _filter_ids(card, applicant_ids)
    card = card[card["MONTHS_BALANCE"].isna() | (card["MONTHS_BALANCE"] <= 0)].copy()
    card["FE_CC_UTILIZATION"] = safe_divide(_series(card, "AMT_BALANCE"), _series(card, "AMT_CREDIT_LIMIT_ACTUAL"))
    values = [column for column in card.columns if column != ID_COLUMN]
    spec = {column: ["min", "max", "mean", "sum"] for column in values}
    output = _flatten_aggregation_columns(card.groupby(ID_COLUMN).agg(spec), "CC")
    return output.merge(
        card.groupby(ID_COLUMN).size().rename("CC_RECORD_COUNT").reset_index(),
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )


def build_home_credit_features(
    data_dir: Path,
    *,
    feature_set: str = "serving",
    sample_size: int | None = None,
) -> pd.DataFrame:
    """Build an applicant-level matrix from immutable raw CSV files.

    ``serving`` uses fields available in the POC form. ``application`` keeps all
    application columns. ``full`` additionally aggregates the five relational
    tables. Full mode is intentionally explicit because it is memory intensive.
    """
    if feature_set not in {"serving", "application", "full"}:
        raise ValueError("feature_set must be one of: serving, application, full")
    application = pd.read_csv(data_dir / "application_train.csv", nrows=sample_size)
    application = engineer_application_features(application)

    if feature_set == "serving":
        # Protected columns are retained only for holdout fairness diagnostics;
        # ``select_model_features`` excludes them from the model matrix and the
        # online scorer rejects them as inputs.
        selected = [
            ID_COLUMN,
            TARGET_COLUMN,
            *PROTECTED_COLUMNS,
            *SERVING_RAW_FEATURES,
            *ENGINEERED_FEATURES,
            "DAYS_EMPLOYED_ANOMALY",
        ]
        return application[[column for column in selected if column in application]].copy()
    if feature_set == "application":
        return application

    applicant_ids = set(application[ID_COLUMN].astype(int)) if sample_size is not None else None
    aggregators = (
        aggregate_bureau,
        aggregate_previous_applications,
        aggregate_installments,
        aggregate_pos_cash,
        aggregate_credit_card,
    )
    for aggregate in aggregators:
        child = aggregate(data_dir, applicant_ids)
        application = application.merge(child, on=ID_COLUMN, how="left", validate="one_to_one")
    return application
