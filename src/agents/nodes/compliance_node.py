"""
LangGraph Nodes for Policy RAG Pre-Retrieval and Post-Generation Compliance Checking.
"""

from typing import Any

from src.agents.state import AgentState
from src.credit_scoring.compliance_policy import (
    ComplianceCheckResult,
    LoanProposal,
    evaluate_compliance,
)
from src.services.policy_rag import policy_rag_service


async def policy_retrieval_node(state: AgentState) -> dict[str, Any]:
    """Node RAG Phase 1: Truy vấn ngữ cảnh chính sách và tiêm vào state."""
    query = state.get("query", "")
    metadata = state.get("metadata", {})
    loan_purpose = metadata.get("loan_purpose", query)

    # Truy vấn đoạn chính sách liên quan nhất
    rag_context = policy_rag_service.format_retrieved_context_for_llm(query=f"{query} {loan_purpose}", top_k=3)

    return {
        "context": rag_context,
        "metadata": {**metadata, "policy_rag_retrieved": True},
    }


async def compliance_checker_node(state: AgentState) -> dict[str, Any]:
    """Node RAG Phase 2: Đối soát đề xuất vay qua Compliance Engine."""
    metadata = state.get("metadata", {})
    proposal_dict = metadata.get("proposal", {})

    if not proposal_dict:
        # Nếu chưa có proposal trong metadata, lấy từ state mặc định
        proposal_dict = {
            "applicant_age": metadata.get("applicant_age", 30),
            "monthly_income": metadata.get("monthly_income", 15_000_000.0),
            "current_monthly_debt": metadata.get("current_monthly_debt", 0.0),
            "loan_amount": metadata.get("requested_loan_amount", 50_000_000.0),
            "loan_term_months": metadata.get("loan_term_months", 12),
            "interest_rate_annual": metadata.get("proposed_interest_rate", 16.0),
            "loan_purpose": metadata.get("loan_purpose", "Vay tiêu dùng cá nhân"),
            "risk_band": metadata.get("risk_band", "moderate"),
            "is_ekyc": metadata.get("is_ekyc", True),
        }

    try:
        loan_prop = LoanProposal(**proposal_dict)
        compliance_result: ComplianceCheckResult = evaluate_compliance(loan_prop)

        # Định dạng câu giải thích thân thiện cho khách hàng
        explanation_parts = [compliance_result.explanation]
        if compliance_result.violations:
            explanation_parts.append("\nChi tiết kiểm tra quy định:")
            for v in compliance_result.violations:
                explanation_parts.append(f"- [{v.severity}] {v.rule_name}: {v.description} -> Gợi ý: {v.suggested_remediation}")

        friendly_explanation = "\n".join(explanation_parts)

        return {
            "analysis": compliance_result.model_dump_json(),
            "response": friendly_explanation,
            "metadata": {
                **metadata,
                "compliance_result": compliance_result.model_dump(),
                "decision": compliance_result.decision,
            },
        }

    except Exception as exc:
        return {
            "error": f"Lỗi khi đánh giá tuân thủ chính sách: {exc!s}",
            "metadata": metadata,
        }
