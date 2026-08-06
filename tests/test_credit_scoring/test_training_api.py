from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import get_settings
from src.credit_scoring.training import train_credit_model
from src.services.credit_scoring import load_credit_model


def _synthetic_application(path: Path, rows: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    income = rng.lognormal(mean=11.8, sigma=0.45, size=rows)
    credit = rng.lognormal(mean=13.0, sigma=0.55, size=rows)
    annuity = credit * rng.uniform(0.025, 0.08, size=rows)
    ext2 = rng.uniform(0.05, 0.95, size=rows)
    ext3 = rng.uniform(0.05, 0.95, size=rows)
    risk_logit = -2.2 + 0.7 * (credit / income > 4) - 2.0 * (ext2 - 0.5) - 1.4 * (ext3 - 0.5)
    probability = 1 / (1 + np.exp(-risk_logit))
    target = rng.binomial(1, probability)
    frame = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100_000, 100_000 + rows),
            "TARGET": target,
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], size=rows),
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity,
            "AMT_GOODS_PRICE": credit * rng.uniform(0.8, 1.1, size=rows),
            "DAYS_BIRTH": -rng.integers(20 * 365, 70 * 365, size=rows),
            "DAYS_EMPLOYED": -rng.integers(30, 25 * 365, size=rows),
            "DAYS_REGISTRATION": -rng.integers(10, 20 * 365, size=rows),
            "DAYS_ID_PUBLISH": -rng.integers(10, 10 * 365, size=rows),
            "DAYS_LAST_PHONE_CHANGE": -rng.integers(1, 8 * 365, size=rows),
            "CNT_CHILDREN": rng.integers(0, 4, size=rows),
            "CNT_FAM_MEMBERS": rng.integers(1, 6, size=rows),
            "EXT_SOURCE_1": rng.uniform(0.05, 0.95, size=rows),
            "EXT_SOURCE_2": ext2,
            "EXT_SOURCE_3": ext3,
            "REGION_RATING_CLIENT": rng.integers(1, 4, size=rows),
            "REGION_RATING_CLIENT_W_CITY": rng.integers(1, 4, size=rows),
            "FLAG_OWN_CAR": rng.choice(["Y", "N"], size=rows),
            "FLAG_OWN_REALTY": rng.choice(["Y", "N"], size=rows),
            "NAME_INCOME_TYPE": rng.choice(["Working", "Commercial associate", "Pensioner"], size=rows),
            "NAME_EDUCATION_TYPE": rng.choice(["Secondary", "Higher education"], size=rows),
            "NAME_FAMILY_STATUS": rng.choice(["Married", "Single"], size=rows),
            "NAME_HOUSING_TYPE": rng.choice(["House / apartment", "With parents"], size=rows),
            "OCCUPATION_TYPE": rng.choice(["Laborers", "Managers", None], size=rows),
            "ORGANIZATION_TYPE": rng.choice(["Business", "School", "Government"], size=rows),
            "CODE_GENDER": rng.choice(["F", "M"], size=rows),
        }
    )
    frame.to_csv(path / "application_train.csv", index=False)
    return frame


@pytest.fixture(scope="module")
def trained_artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("credit_model")
    data_dir = root / "raw"
    output_dir = root / "model"
    data_dir.mkdir()
    frame = _synthetic_application(data_dir)
    bundle, report = train_credit_model(data_dir=data_dir, output_dir=output_dir, feature_set="serving", seed=7)
    assert report["split"]["strategy"] == "stratified_random_70_15_15_no_temporal_claim"
    assert report["champion_test_metrics"]["roc_auc"] >= 0.5
    return output_dir / "credit_model.joblib", frame, bundle


def test_bundle_scores_and_explains_without_protected_fields(trained_artifact):
    artifact, frame, bundle = trained_artifact
    row = frame.iloc[0].drop(labels=["TARGET", "SK_ID_CURR"]).to_dict()
    row["CODE_GENDER"] = "F"
    result = bundle.score([row], top_k=4)[0]
    assert 0 <= result["payment_difficulty_probability"] <= 1
    assert 0 <= result["poc_score"] <= 100
    assert len(result["top_factors"]) == 4
    assert "CODE_GENDER" in result["data_quality"]["ignored_fields"]
    assert artifact.is_file()


@pytest.mark.asyncio
async def test_credit_score_api(client, trained_artifact):
    artifact, frame, _ = trained_artifact
    settings = get_settings()
    original_path = settings.credit_model_path
    settings.credit_model_path = str(artifact)
    load_credit_model.cache_clear()
    try:
        application = frame.iloc[1].drop(labels=["TARGET", "SK_ID_CURR", "CODE_GENDER"]).to_dict()
        application = {key: (None if pd.isna(value) else value) for key, value in application.items()}
        response = await client.post(
            "/api/v1/credit/score",
            json={"application": application, "explain_with_llm": False, "top_k": 5},
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["llm_used"] is False
        assert len(payload["top_factors"]) == 5
        assert "không phải điểm CIC" in payload["poc_score_definition"]

        info = await client.get("/api/v1/credit/model")
        assert info.status_code == 200
        assert info.json()["model_version"] == payload["model_version"]
    finally:
        settings.credit_model_path = original_path
        load_credit_model.cache_clear()
