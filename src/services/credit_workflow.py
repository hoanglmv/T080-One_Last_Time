"""End-to-end orchestration for the credit application activity diagram.

The workflow deliberately separates a model recommendation from the final
credit decision. A credit officer must finalize every scored application.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from src.credit_scoring.compliance_policy import LoanProposal, evaluate_compliance
from src.services.credit_scoring import score_application
from src.services.policy_rag import PolicyRAGService, policy_rag_service

WorkflowStatus = Literal["NEEDS_INFORMATION", "PENDING_REVIEW", "REJECTED", "APPROVED"]
Recommendation = Literal["REJECT", "REVIEW", "APPROVE"]


@dataclass(frozen=True)
class WorkflowStage:
    stage: str
    status: Literal["completed", "blocked", "pending"]
    detail: str


class WorkflowNotFoundError(LookupError):
    pass


class InvalidWorkflowTransitionError(ValueError):
    pass


class WorkflowAuditRepository:
    """Append-only JSONL audit trail plus a small current-state projection."""

    def __init__(self, directory: Path | None = None) -> None:
        root = directory or Path(__file__).resolve().parents[2] / "data" / "logs"
        self.directory = root
        self.events_file = root / "credit_workflow_audit.jsonl"
        self.state_file = root / "credit_workflow_state.json"

    def append(self, application_id: str, event: str, payload: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "application_id": application_id,
            "event": event,
            "payload": payload,
        }
        with self.events_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def save_state(self, application_id: str, state: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        states = self._load_states()
        states[application_id] = state
        temporary = self.state_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(states, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        temporary.replace(self.state_file)

    def get_state(self, application_id: str) -> dict[str, Any]:
        state = self._load_states().get(application_id)
        if state is None:
            raise WorkflowNotFoundError(application_id)
        return state

    def _load_states(self) -> dict[str, Any]:
        if not self.state_file.exists():
            return {}
        try:
            value = json.loads(self.state_file.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}


Scorer = Callable[..., Awaitable[dict[str, Any]]]


class CreditApplicationWorkflow:
    REQUIRED_DOCUMENTS = {"identity", "income_proof"}
    SENSITIVE_FIELDS = {"FULL_NAME", "NAME", "ADDRESS", "PHONE", "EMAIL", "SK_ID_CURR"}

    def __init__(
        self,
        repository: WorkflowAuditRepository | None = None,
        rag_service: PolicyRAGService | None = None,
        scorer: Scorer = score_application,
    ) -> None:
        self.repository = repository or WorkflowAuditRepository()
        self.rag_service = rag_service or policy_rag_service
        self.scorer = scorer

    async def submit(self, request: Any) -> dict[str, Any]:
        application_id = request.application_id or f"APP-{uuid.uuid4().hex[:12].upper()}"
        stages = [
            WorkflowStage("intake", "completed", "Đã tiếp nhận hồ sơ."),
            WorkflowStage("validation", "pending", "Đang kiểm tra tính đầy đủ và hợp lệ."),
        ]
        supplied_documents = {item.strip().lower() for item in request.documents}
        missing_documents = sorted(self.REQUIRED_DOCUMENTS - supplied_documents)
        if missing_documents:
            stages[-1] = WorkflowStage("validation", "blocked", "Hồ sơ thiếu tài liệu bắt buộc.")
            result = {
                "application_id": application_id,
                "status": "NEEDS_INFORMATION",
                "route": None,
                "missing_documents": missing_documents,
                "system_recommendation": None,
                "score": None,
                "policy_evidence": [],
                "compliance_report": None,
                "proposal": None,
                "explanation": "Vui lòng bổ sung tài liệu bắt buộc trước khi chấm điểm.",
                "stages": [asdict(stage) for stage in stages],
                "final_decision": None,
                "officer_notes": None,
            }
            self._persist(application_id, "APPLICATION_NEEDS_INFORMATION", request, result)
            return result

        route = "traditional_credit" if request.has_credit_history else "alternative_credit"
        stages[-1] = WorkflowStage("validation", "completed", "Hồ sơ đủ tài liệu bắt buộc.")
        stages.append(WorkflowStage("routing", "completed", f"Đã định tuyến sang {route}."))

        model_input = dict(request.application)
        if not request.has_credit_history:
            for field in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
                model_input.pop(field, None)
        scoring = await self.scorer(
            model_input,
            explain_with_llm=False,
            top_k=request.top_k,
            history_mode="traditional" if request.has_credit_history else "alternative",
        )
        stages.extend(
            [
                WorkflowStage("risk_scoring", "completed", scoring["auto_detected_model"]),
                WorkflowStage("xai", "completed", f"Đã tạo {len(scoring['top_factors'])} reason codes."),
            ]
        )

        query = (
            f"Khoản vay {request.loan_purpose}; hạn mức {request.requested_loan_amount}; "
            f"lãi suất {request.proposed_interest_rate}; eKYC={request.is_ekyc}"
        )
        chunks = self.rag_service.retrieve_relevant_policies(query, top_k=3)
        evidence = [
            {"chunk_id": chunk.chunk_id, "source_document": chunk.source_document, "title": chunk.title}
            for chunk in chunks
        ]
        stages.append(
            WorkflowStage(
                "policy_rag",
                "completed" if evidence else "blocked",
                f"Đã truy xuất {len(evidence)} policy chunks." if evidence else "Kho policy không có bằng chứng phù hợp.",
            )
        )

        proposal = LoanProposal(
            applicant_age=request.applicant_age,
            monthly_income=request.monthly_income,
            current_monthly_debt=request.current_monthly_debt,
            loan_amount=request.requested_loan_amount,
            loan_term_months=request.loan_term_months,
            interest_rate_annual=request.proposed_interest_rate,
            loan_purpose=request.loan_purpose,
            risk_band=scoring["risk_band"],
            is_ekyc=request.is_ekyc,
        )
        compliance = evaluate_compliance(proposal)
        recommendation = self._recommend(scoring, compliance.decision, bool(evidence))
        stages.extend(
            [
                WorkflowStage("decision_engine", "completed", f"System recommendation: {recommendation}."),
                WorkflowStage("recommendation_explanation", "completed", "Đã tổng hợp ML, XAI và policy evidence."),
                WorkflowStage("officer_review", "pending", "Chờ cán bộ tín dụng ra quyết định cuối cùng."),
            ]
        )
        explanation = self._explain(recommendation, scoring, compliance.decision, evidence)
        result = {
            "application_id": application_id,
            "status": "PENDING_REVIEW",
            "route": route,
            "missing_documents": [],
            "system_recommendation": recommendation,
            "score": scoring,
            "policy_evidence": evidence,
            "compliance_report": compliance.model_dump(),
            "proposal": (compliance.modified_proposal or proposal).model_dump(),
            "explanation": explanation,
            "stages": [asdict(stage) for stage in stages],
            "final_decision": None,
            "officer_notes": None,
        }
        self._persist(application_id, "APPLICATION_SCORED", request, result)
        return result

    def review(
        self,
        application_id: str,
        decision: Literal["REJECT", "REVIEW", "APPROVE"],
        officer_id: str,
        notes: str,
    ) -> dict[str, Any]:
        state = self.repository.get_state(application_id)
        if state["status"] != "PENDING_REVIEW":
            raise InvalidWorkflowTransitionError("Chỉ hồ sơ PENDING_REVIEW mới được ra quyết định.")
        if decision == "APPROVE" and state["system_recommendation"] == "REJECT":
            raise InvalidWorkflowTransitionError("Không thể override hard-rule REJECT thành APPROVE.")
        if decision == "REVIEW" and not notes.strip():
            raise InvalidWorkflowTransitionError("Yêu cầu bổ sung phải có ghi chú.")

        state["status"] = "APPROVED" if decision == "APPROVE" else "REJECTED" if decision == "REJECT" else "NEEDS_INFORMATION"
        state["final_decision"] = decision
        state["officer_notes"] = notes
        state["stages"] = [
            *[stage for stage in state["stages"] if stage["stage"] != "officer_review"],
            asdict(WorkflowStage("officer_review", "completed", f"Quyết định bởi cán bộ {officer_id}.")),
            asdict(WorkflowStage("audit_log", "completed", "Đã lưu quyết định cuối cùng.")),
        ]
        self.repository.save_state(application_id, state)
        self.repository.append(
            application_id,
            "OFFICER_DECISION",
            {"officer_id": officer_id, "decision": decision, "notes": notes},
        )
        return state

    @staticmethod
    def _recommend(scoring: dict[str, Any], compliance_decision: str, has_evidence: bool) -> Recommendation:
        if compliance_decision == "REJECTED" or scoring["risk_band"] == "very_high":
            return "REJECT"
        completeness = float(scoring["data_quality"]["completeness"])
        if compliance_decision == "MODIFIED_WITH_WARNINGS" or scoring["risk_band"] == "high":
            return "REVIEW"
        if completeness < 0.7 or not has_evidence:
            return "REVIEW"
        return "APPROVE"

    @staticmethod
    def _explain(
        recommendation: Recommendation,
        scoring: dict[str, Any],
        compliance_decision: str,
        evidence: list[dict[str, str]],
    ) -> str:
        sources = ", ".join(item["source_document"] for item in evidence) or "không có policy evidence"
        return (
            f"Đề xuất {recommendation}: PD={scoring['payment_difficulty_probability']:.4f}, "
            f"risk_band={scoring['risk_band']}, compliance={compliance_decision}. "
            f"Nguồn chính sách truy xuất: {sources}. Đây là đề xuất hỗ trợ; cán bộ tín dụng quyết định cuối cùng."
        )

    def _persist(self, application_id: str, event: str, request: Any, result: dict[str, Any]) -> None:
        raw = request.model_dump()
        raw["customer_reference_hash"] = hashlib.sha256(request.customer_reference.encode("utf-8")).hexdigest()
        raw.pop("customer_reference", None)
        raw["application"] = {
            key: value for key, value in raw.get("application", {}).items() if key.upper() not in self.SENSITIVE_FIELDS
        }
        self.repository.save_state(application_id, result)
        self.repository.append(application_id, event, {"input": raw, "result": result})


credit_application_workflow = CreditApplicationWorkflow()
