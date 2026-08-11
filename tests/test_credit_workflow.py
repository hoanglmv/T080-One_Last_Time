from pathlib import Path
from types import SimpleNamespace

import pytest

from src.services.credit_workflow import (
    CreditApplicationWorkflow,
    InvalidWorkflowTransitionError,
    WorkflowAuditRepository,
)
from src.services.policy_rag import PolicyRAGService


def make_request(**overrides):
    values = {
        "application_id": "APP-TEST-001",
        "customer_reference": "customer-sensitive-reference",
        "has_credit_history": True,
        "documents": ["identity", "income_proof"],
        "application": {
            "AMT_INCOME_TOTAL": 25_000_000,
            "AMT_CREDIT": 50_000_000,
            "EXT_SOURCE_2": 0.75,
        },
        "applicant_age": 30,
        "monthly_income": 25_000_000,
        "current_monthly_debt": 1_000_000,
        "requested_loan_amount": 50_000_000,
        "loan_term_months": 24,
        "proposed_interest_rate": 12.0,
        "loan_purpose": "Mua thiết bị gia đình",
        "is_ekyc": True,
        "top_k": 6,
    }
    values.update(overrides)

    class Request(SimpleNamespace):
        def model_dump(self):
            return dict(self.__dict__)

    return Request(**values)


def fake_score(risk_band="low", completeness=1.0, captured=None):
    async def scorer(application, *, explain_with_llm, top_k, history_mode):
        if captured is not None:
            captured.update(application)
        return {
            "model_version": "test-v1",
            "model_name": "test-model",
            "auto_detected_model": "test-model",
            "model_routing_reason": "test",
            "payment_difficulty_probability": 0.1,
            "poc_score": 90.0,
            "poc_score_definition": "test",
            "risk_band": risk_band,
            "top_factors": [{"feature": "income"}],
            "data_quality": {
                "completeness": completeness,
                "missing_required_fields": [],
                "ignored_fields": [],
                "warnings": [],
            },
            "friendly_explanation": "test",
            "llm_used": False,
            "disclaimer": "test",
        }

    return scorer


def build_workflow(tmp_path: Path, scorer):
    policy_dir = tmp_path / "policies"
    policy_dir.mkdir()
    (policy_dir / "bank_policy.md").write_text(
        "# Chính sách tín dụng\n## Vay tiêu dùng\nHạn mức và lãi suất phải được thẩm định.",
        encoding="utf-8",
    )
    return CreditApplicationWorkflow(
        repository=WorkflowAuditRepository(tmp_path / "audit"),
        rag_service=PolicyRAGService(policy_dir),
        scorer=scorer,
    )


@pytest.mark.asyncio
async def test_missing_documents_stops_before_scoring(tmp_path):
    called = False

    async def scorer(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("scorer must not run")

    workflow = build_workflow(tmp_path, scorer)
    result = await workflow.submit(make_request(documents=["identity"]))

    assert result["status"] == "NEEDS_INFORMATION"
    assert result["missing_documents"] == ["income_proof"]
    assert called is False


@pytest.mark.asyncio
async def test_traditional_route_requires_human_review_and_can_approve(tmp_path):
    workflow = build_workflow(tmp_path, fake_score())
    result = await workflow.submit(make_request())

    assert result["route"] == "traditional_credit"
    assert result["system_recommendation"] == "APPROVE"
    assert result["status"] == "PENDING_REVIEW"
    assert result["policy_evidence"]

    reviewed = workflow.review(result["application_id"], "APPROVE", "officer-01", "Đã kiểm tra.")
    assert reviewed["status"] == "APPROVED"
    assert reviewed["final_decision"] == "APPROVE"


@pytest.mark.asyncio
async def test_alternative_route_strips_bureau_fields(tmp_path):
    captured = {}
    workflow = build_workflow(tmp_path, fake_score(captured=captured))
    result = await workflow.submit(make_request(has_credit_history=False))

    assert result["route"] == "alternative_credit"
    assert "EXT_SOURCE_2" not in captured
    assert "AMT_INCOME_TOTAL" in captured


@pytest.mark.asyncio
async def test_hard_reject_cannot_be_overridden_to_approve(tmp_path):
    workflow = build_workflow(tmp_path, fake_score(risk_band="very_high"))
    result = await workflow.submit(make_request())
    assert result["system_recommendation"] == "REJECT"

    with pytest.raises(InvalidWorkflowTransitionError):
        workflow.review(result["application_id"], "APPROVE", "officer-01", "Override")


@pytest.mark.asyncio
async def test_audit_hashes_customer_reference(tmp_path):
    workflow = build_workflow(tmp_path, fake_score())
    await workflow.submit(make_request())

    audit = workflow.repository.events_file.read_text(encoding="utf-8")
    assert "customer-sensitive-reference" not in audit
    assert "customer_reference_hash" in audit


@pytest.mark.asyncio
async def test_workflow_api_returns_missing_documents(client, tmp_path, monkeypatch):
    from src.api import routes

    workflow = build_workflow(tmp_path, fake_score())
    monkeypatch.setattr(routes, "credit_application_workflow", workflow)
    payload = make_request(documents=["identity"]).model_dump()

    response = await client.post("/api/v1/credit/workflow/applications", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "NEEDS_INFORMATION"
    assert body["missing_documents"] == ["income_proof"]
