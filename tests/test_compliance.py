"""
Tests for Legal & Bank Compliance Policy Engine & RAG Retrieval.
"""

import pytest

from src.credit_scoring.compliance_policy import (
    LoanProposal,
    evaluate_compliance,
    verify_hard_rules,
)
from src.services.policy_rag import PolicyRAGService, policy_rag_service


def test_approved_compliant_proposal():
    proposal = LoanProposal(
        applicant_age=30,
        monthly_income=25_000_000.0,
        current_monthly_debt=2_000_000.0,
        loan_amount=50_000_000.0,
        loan_term_months=12,
        interest_rate_annual=14.0,
        loan_purpose="Vay mua sắm thiết bị gia đình",
        risk_band="low",
        is_ekyc=True,
    )
    result = evaluate_compliance(proposal)
    assert result.is_compliant is True
    assert result.decision == "APPROVED"
    assert len(result.violations) == 0
    assert result.modified_proposal.loan_amount == 50_000_000.0


def test_legal_interest_rate_cap_violation():
    """Lãi suất 24.0%/năm vượt trần Dân sự 20.0%/năm -> Phải điều chỉnh hạ về <= 20.0%/năm."""
    proposal = LoanProposal(
        applicant_age=28,
        monthly_income=20_000_000.0,
        current_monthly_debt=0.0,
        loan_amount=20_000_000.0,
        loan_term_months=12,
        interest_rate_annual=24.0,
        loan_purpose="Vay tiêu dùng cá nhân",
        risk_band="high",
        is_ekyc=True,
    )
    result = evaluate_compliance(proposal)
    assert result.is_compliant is False
    assert result.decision == "MODIFIED_WITH_WARNINGS"
    assert any(v.rule_id == "RULE_003_LEGAL_INTEREST_CAP" for v in result.violations)
    assert result.modified_proposal.interest_rate_annual <= 20.0


def test_prohibited_loan_purpose():
    """Vay mua tiền ảo / Bitcoin -> Từ chối cấp tín dụng."""
    proposal = LoanProposal(
        applicant_age=25,
        monthly_income=30_000_000.0,
        current_monthly_debt=0.0,
        loan_amount=30_000_000.0,
        loan_term_months=12,
        interest_rate_annual=15.0,
        loan_purpose="Vay đầu tư Bitcoin và Forex",
        risk_band="low",
        is_ekyc=True,
    )
    result = evaluate_compliance(proposal)
    assert result.decision == "REJECTED"
    assert any(v.rule_id == "RULE_007_PROHIBITED_LOAN_PURPOSE" for v in result.violations)


def test_applicant_age_violation():
    """Người vay 17 tuổi hoặc 62 tuổi -> Từ chối do vi phạm độ tuổi quy định."""
    proposal_underage = LoanProposal(
        applicant_age=17,
        monthly_income=15_000_000.0,
        current_monthly_debt=0.0,
        loan_amount=10_000_000.0,
        loan_term_months=12,
        interest_rate_annual=15.0,
        loan_purpose="Vay mua laptop học tập",
        risk_band="moderate",
        is_ekyc=True,
    )
    res_underage = evaluate_compliance(proposal_underage)
    assert res_underage.decision == "REJECTED"
    assert any(v.rule_id == "RULE_002_AGE_LIMIT" for v in res_underage.violations)

    proposal_overage = LoanProposal(
        applicant_age=65,
        monthly_income=15_000_000.0,
        current_monthly_debt=0.0,
        loan_amount=10_000_000.0,
        loan_term_months=12,
        interest_rate_annual=15.0,
        loan_purpose="Vay sửa chữa nhà",
        risk_band="moderate",
        is_ekyc=True,
    )
    res_overage = evaluate_compliance(proposal_overage)
    assert res_overage.decision == "REJECTED"
    assert any(v.rule_id == "RULE_002_AGE_LIMIT" for v in res_overage.violations)


def test_ekyc_small_loan_limit():
    """Vay e-KYC 150 triệu -> Điều chỉnh hạn mức về tối đa 100 triệu VNĐ."""
    proposal = LoanProposal(
        applicant_age=35,
        monthly_income=40_000_000.0,
        current_monthly_debt=0.0,
        loan_amount=150_000_000.0,
        loan_term_months=24,
        interest_rate_annual=14.0,
        loan_purpose="Vay tiêu dùng",
        risk_band="low",
        is_ekyc=True,
    )
    result = evaluate_compliance(proposal)
    assert result.decision == "MODIFIED_WITH_WARNINGS"
    assert any(v.rule_id == "RULE_005_EKYC_LOAN_CAP" for v in result.violations)
    assert result.modified_proposal.loan_amount <= 100_000_000.0


def test_very_high_risk_band_rejection():
    """Phân nhóm Very High Risk -> Từ chối cấp tín dụng."""
    proposal = LoanProposal(
        applicant_age=30,
        monthly_income=15_000_000.0,
        current_monthly_debt=0.0,
        loan_amount=10_000_000.0,
        loan_term_months=12,
        interest_rate_annual=18.0,
        loan_purpose="Vay tiêu dùng",
        risk_band="very_high",
        is_ekyc=True,
    )
    result = evaluate_compliance(proposal)
    assert result.decision == "REJECTED"
    assert any(v.rule_id == "RULE_001_VERY_HIGH_RISK" for v in result.violations)


def test_policy_rag_service_retrieval():
    """Kiểm tra RAG Service nạp tài liệu và truy vấn ngữ cảnh chính sách."""
    rag_service = PolicyRAGService()
    chunks = rag_service.retrieve_relevant_policies(query="lãi suất trần bộ luật dân sự", top_k=2)
    assert len(chunks) > 0
    formatted = rag_service.format_retrieved_context_for_llm(query="hạn mức e-KYC 100 triệu", top_k=2)
    assert "QUY ĐỊNH PHÁP LUẬT" in formatted
    assert len(formatted) > 50
