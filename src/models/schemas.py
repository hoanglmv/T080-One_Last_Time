import math
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000, description="Tin nhắn từ user")


class ChatResponse(BaseModel):
    response: str = Field(..., description="Phản hồi từ agent")
    analysis: str = Field(default="", description="Phân tích nội bộ")


class TextExtractRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=10000, description="Văn bản mô tả hồ sơ vay tiếng Việt")


class TextExtractResponse(BaseModel):
    extracted: dict[str, Any]
    llm_used: bool
    summary: str


class CreditScoreRequest(BaseModel):
    application: dict[str, Any] = Field(
        ...,
        description="Một hồ sơ application theo tên cột Home Credit, ví dụ AMT_CREDIT.",
    )
    explain_with_llm: bool = Field(default=False, description="Chỉ diễn đạt reason codes; không đổi điểm.")
    top_k: int = Field(default=6, ge=1, le=10)

    @field_validator("application")
    @classmethod
    def validate_application(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("application must not be empty")
        if len(value) > 500:
            raise ValueError("application has too many fields")
        for key, item in value.items():
            if not key or len(key) > 128:
                raise ValueError("application contains an invalid field name")
            if item is not None and not isinstance(item, (str, int, float, bool)):
                raise ValueError(f"{key} must be a scalar value")
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError(f"{key} must be finite")
        return value


class CreditFactor(BaseModel):
    feature: str
    description: str
    value: Any = None
    contribution: float
    direction: Literal["increase_risk", "decrease_risk"]
    reason: str


class CreditDataQuality(BaseModel):
    completeness: float
    missing_required_fields: list[str]
    ignored_fields: list[str]
    warnings: list[str]


class CreditScoreResponse(BaseModel):
    model_version: str
    model_name: str
    auto_detected_model: str = Field(default="", description="Mô hình được hệ thống tự động nhận biết")
    model_routing_reason: str = Field(default="", description="Lý do hệ thống tự động nhận biết mô hình")
    payment_difficulty_probability: float
    poc_score: float
    poc_score_definition: str
    risk_band: Literal["low", "moderate", "high", "very_high"]
    top_factors: list[CreditFactor]
    data_quality: CreditDataQuality
    friendly_explanation: str
    llm_used: bool
    disclaimer: str


class ComplianceViolationSchema(BaseModel):
    rule_id: str
    rule_name: str
    severity: str
    law_or_policy_source: str
    description: str
    suggested_remediation: str


class ComplianceReportSchema(BaseModel):
    is_compliant: bool
    decision: Literal["APPROVED", "REJECTED", "MODIFIED_WITH_WARNINGS"]
    violations: list[ComplianceViolationSchema] = Field(default_factory=list)
    citation_references: list[str] = Field(default_factory=list)
    explanation: str = ""


class LoanRecommendationRequest(BaseModel):
    applicant_age: int = Field(..., ge=0, le=120, description="Tuổi khách hàng")
    monthly_income: float = Field(..., ge=0.0, description="Thu nhập hàng tháng")
    current_monthly_debt: float = Field(default=0.0, ge=0.0, description="Nợ hàng tháng hiện tại")
    requested_loan_amount: float = Field(..., ge=0.0, description="Số tiền muốn vay")
    loan_term_months: int = Field(default=12, ge=1, le=120, description="Thời hạn vay (tháng)")
    proposed_interest_rate: float = Field(..., ge=0.0, le=100.0, description="Lãi suất vay (%/năm)")
    loan_purpose: str = Field(..., min_length=1, description="Mục đích sử dụng vốn")
    risk_band: Literal["low", "moderate", "high", "very_high"] = Field(..., description="Phân nhóm rủi ro tín dụng")
    is_ekyc: bool = Field(default=True, description="Vay e-KYC không thế chấp")


class LoanRecommendationResponse(BaseModel):
    decision: Literal["APPROVED", "REJECTED", "MODIFIED_WITH_WARNINGS"]
    original_proposal: dict[str, Any]
    approved_proposal: dict[str, Any]
    compliance_report: ComplianceReportSchema
    friendly_explanation: str
    shap_explanation: dict[str, Any] | None = None


class SHAPFactorSchema(BaseModel):
    feature: str
    feature_name_vi: str
    value: Any = None
    shap_value: float
    contribution_percentage: float
    direction: Literal["increase_risk", "decrease_risk"]
    reason: str
    law_citation: str | None = None


class SHAPExplanationSchema(BaseModel):
    base_value: float
    prediction_score: float
    top_positive_factors: list[SHAPFactorSchema] = Field(default_factory=list)
    top_negative_factors: list[SHAPFactorSchema] = Field(default_factory=list)
    summary_vi: str = ""


class FeedbackOutcomeRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    actual_target: int = Field(..., ge=0, le=1, description="0: Trả đúng hạn, 1: Nợ quá hạn/nợ xấu")
    actual_dpd_days: int = Field(default=0, ge=0, description="Số ngày quá hạn (DPD)")
    loan_status: str = Field(default="CLOSED", description="LOAN_DISBURSED, PAID_OFF, DEFAULTED, CLOSED")


class FeedbackOutcomeResponse(BaseModel):
    success: bool
    session_id: str
    message: str
    retraining_trigger_status: dict[str, Any]


class PortfolioDelinquencyResponse(BaseModel):
    total_disbursed_loans: int
    total_disbursed_amount: float
    delinquency_rate_dpd30_pct: float
    delinquency_rate_dpd90_pct: float
    group_distribution: dict[str, int]
    roll_rate_matrix: dict[str, float]
    summary_vi: str


class ConversionFunnelResponse(BaseModel):
    total_sessions: int
    conversion_rate_pct: float
    disbursed_count: int
    stage_counts: dict[str, int]
    summary_vi: str


class CreditWorkflowRequest(BaseModel):
    application_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{3,64}$")
    customer_reference: str = Field(..., min_length=1, max_length=256, description="Mã tham chiếu; chỉ hash được audit.")
    has_credit_history: bool
    documents: list[str] = Field(default_factory=list, max_length=30)
    application: dict[str, Any]
    applicant_age: int = Field(..., ge=18, le=120)
    monthly_income: float = Field(..., gt=0)
    current_monthly_debt: float = Field(default=0.0, ge=0)
    requested_loan_amount: float = Field(..., gt=0)
    loan_term_months: int = Field(default=12, ge=1, le=120)
    proposed_interest_rate: float = Field(..., ge=0, le=100)
    loan_purpose: str = Field(..., min_length=1, max_length=500)
    is_ekyc: bool = True
    top_k: int = Field(default=6, ge=1, le=10)

    @field_validator("application")
    @classmethod
    def validate_workflow_application(cls, value: dict[str, Any]) -> dict[str, Any]:
        return CreditScoreRequest.validate_application(value)


class WorkflowStageSchema(BaseModel):
    stage: str
    status: Literal["completed", "blocked", "pending"]
    detail: str


class CreditWorkflowResponse(BaseModel):
    application_id: str
    status: Literal["NEEDS_INFORMATION", "PENDING_REVIEW", "REJECTED", "APPROVED"]
    route: Literal["traditional_credit", "alternative_credit"] | None = None
    missing_documents: list[str] = Field(default_factory=list)
    system_recommendation: Literal["REJECT", "REVIEW", "APPROVE"] | None = None
    score: dict[str, Any] | None = None
    policy_evidence: list[dict[str, str]] = Field(default_factory=list)
    compliance_report: dict[str, Any] | None = None
    proposal: dict[str, Any] | None = None
    explanation: str
    stages: list[WorkflowStageSchema]
    final_decision: Literal["REJECT", "REVIEW", "APPROVE"] | None = None
    officer_notes: str | None = None


class CreditOfficerReviewRequest(BaseModel):
    decision: Literal["REJECT", "REVIEW", "APPROVE"]
    officer_id: str = Field(..., min_length=1, max_length=128)
    notes: str = Field(default="", max_length=2000)

