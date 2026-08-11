"""
Compliance Policy Engine for LLM Credit Recommendations.
Strictly enforces Vietnamese Legal Policies and Commercial Bank Underwriting Guidelines.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class LoanProposal(BaseModel):
    """Đề xuất khoản vay do LLM hoặc hệ thống tư vấn tạo ra."""
    applicant_age: int = Field(..., ge=0, le=120, description="Tuổi của khách hàng")
    monthly_income: float = Field(..., ge=0.0, description="Thu nhập hàng tháng (VNĐ)")
    current_monthly_debt: float = Field(default=0.0, ge=0.0, description="Nợ phải trả hàng tháng hiện tại (VNĐ)")
    loan_amount: float = Field(..., ge=0.0, description="Số tiền vay đề xuất (VNĐ)")
    loan_term_months: int = Field(default=12, ge=1, le=120, description="Thời hạn vay (tháng)")
    interest_rate_annual: float = Field(..., ge=0.0, le=100.0, description="Lãi suất vay theo năm (%)")
    loan_purpose: str = Field(..., min_length=1, description="Mục đích sử dụng vốn vay")
    risk_band: Literal["low", "moderate", "high", "very_high"] = Field(..., description="Phân nhóm rủi ro tín dụng")
    is_ekyc: bool = Field(default=True, description="Vay trực tuyến e-KYC không thế chấp")


class ViolationItem(BaseModel):
    """Mô tả chi tiết vi phạm chính sách."""
    rule_id: str
    rule_name: str
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    law_or_policy_source: str
    description: str
    suggested_remediation: str


class ComplianceCheckResult(BaseModel):
    """Kết quả kiểm định tuân thủ chính sách."""
    is_compliant: bool
    decision: Literal["APPROVED", "REJECTED", "MODIFIED_WITH_WARNINGS"]
    violations: list[ViolationItem] = Field(default_factory=list)
    original_proposal: LoanProposal
    modified_proposal: Optional[LoanProposal] = None
    citation_references: list[str] = Field(default_factory=list)
    explanation: str = ""


# ---------------------------------------------------------------------------
# Risk Band Limits
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskBandPolicy:
    max_amount: float
    max_rate: float
    max_dti: float
    allowed: bool


RISK_BAND_POLICIES: dict[str, RiskBandPolicy] = {
    "low": RiskBandPolicy(max_amount=200_000_000.0, max_rate=14.5, max_dti=0.45, allowed=True),
    "moderate": RiskBandPolicy(max_amount=100_000_000.0, max_rate=17.5, max_dti=0.45, allowed=True),
    "high": RiskBandPolicy(max_amount=30_000_000.0, max_rate=20.0, max_dti=0.40, allowed=True),
    "very_high": RiskBandPolicy(max_amount=0.0, max_rate=0.0, max_dti=0.0, allowed=False),
}

PROHIBITED_PURPOSE_KEYWORDS: list[tuple[str, str]] = [
    (r"(chứng khoán|cổ phiếu|phái sinh|stock|equity)", "Đầu tư / Kinh doanh chứng khoán"),
    (r"(bitcoin|crypto|tiền ảo|forex|ngoại hối)", "Giao dịch tiền ảo / Forex"),
    (r"(cá cược|cờ bạc|gambling|bóng đá|lô đề)", "Cá cược / Cờ bạc"),
    (r"(đảo nợ|trả nợ ngân hàng khác|tín dụng đen|bốc bát họ)", "Đảo nợ / Trả nợ tín dụng đen"),
    (r"(vàng miếng|sJC)", "Mua vàng miếng"),
    (r"(gửi tiết kiệm|gửi tiền)", "Gửi tiền tiết kiệm"),
    (r"(ma túy|vũ khí|hàng cấm)", "Kinh doanh hàng cấm / Vũ khí / Chất cấm"),
]

LEGAL_RATE_CAP_ANNUAL = 20.0  # Điều 468 Bộ luật Dân sự 2015
EKYC_SMALL_LOAN_CAP = 100_000_000.0  # Thông tư 43/2016 & 18/2019/TT-NHNN
MIN_APPLICANT_AGE = 18
MAX_APPLICANT_AGE = 60


# ---------------------------------------------------------------------------
# Compliance Checking Functions
# ---------------------------------------------------------------------------

def calculate_monthly_payment(loan_amount: float, annual_rate: float, term_months: int) -> float:
    """Tính khoản phải trả hàng tháng (gốc + lãi dư nợ giảm dần hoặc chia đều)."""
    if loan_amount <= 0 or term_months <= 0:
        return 0.0
    monthly_rate = (annual_rate / 100.0) / 12.0
    if monthly_rate == 0:
        return loan_amount / term_months
    # Công thức trả góp đều (Annuity payment)
    payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** term_months) / ((1 + monthly_rate) ** term_months - 1)
    return payment


def verify_hard_rules(proposal: LoanProposal) -> tuple[list[ViolationItem], Optional[LoanProposal]]:
    """Kiểm tra danh mục quy tắc cứng (Deterministic Hard Rules)."""
    violations: list[ViolationItem] = []
    mod = proposal.model_copy(deep=True)
    is_modified = False

    # Rule 1: Very High Risk -> Immediate REJECT
    if proposal.risk_band == "very_high":
        violations.append(
            ViolationItem(
                rule_id="RULE_001_VERY_HIGH_RISK",
                rule_name="Từ chối tín dụng cho phân nhóm rủi ro rất cao",
                severity="CRITICAL",
                law_or_policy_source="Bank Internal Credit Policy - Section 1",
                description=f"Khách hàng thuộc phân nhóm rủi ro {proposal.risk_band.upper()} (Very High Risk) không đủ điều kiện cấp tín dụng.",
                suggested_remediation="Từ chối cấp tín dụng và thông báo chính sách rủi ro.",
            )
        )
        return violations, None

    # Rule 2: Tuổi người vay (18 - 60)
    if proposal.applicant_age < MIN_APPLICANT_AGE or proposal.applicant_age > MAX_APPLICANT_AGE:
        violations.append(
            ViolationItem(
                rule_id="RULE_002_AGE_LIMIT",
                rule_name="Giới hạn độ tuổi người vay",
                severity="CRITICAL",
                law_or_policy_source="Luật TCTD 2024 & Bank Credit Policy - Section 2",
                description=f"Độ tuổi người vay ({proposal.applicant_age} tuổi) nằm ngoài phạm vi cho phép ({MIN_APPLICANT_AGE} - {MAX_APPLICANT_AGE} tuổi).",
                suggested_remediation="Từ chối cấp tín dụng do không đủ điều kiện độ tuổi.",
            )
        )

    # Rule 3: Trần lãi suất Dân sự (20.0%/năm)
    if proposal.interest_rate_annual > LEGAL_RATE_CAP_ANNUAL:
        violations.append(
            ViolationItem(
                rule_id="RULE_003_LEGAL_INTEREST_CAP",
                rule_name="Trần lãi suất theo Bộ luật Dân sự",
                severity="HIGH",
                law_or_policy_source="Điều 468 Bộ luật Dân sự 2015 (Luật số 91/2015/QH13)",
                description=f"Lãi suất đề xuất {proposal.interest_rate_annual}%/năm vượt quá trần quy định {LEGAL_RATE_CAP_ANNUAL}%/năm.",
                suggested_remediation=f"Điều chỉnh giảm lãi suất vay về mức tối đa cho phép: {LEGAL_RATE_CAP_ANNUAL}%/năm.",
            )
        )
        mod.interest_rate_annual = LEGAL_RATE_CAP_ANNUAL
        is_modified = True

    # Rule 4: Lãi suất theo Risk Band
    rb_policy = RISK_BAND_POLICIES.get(proposal.risk_band, RISK_BAND_POLICIES["high"])
    if mod.interest_rate_annual > rb_policy.max_rate:
        violations.append(
            ViolationItem(
                rule_id="RULE_004_RISK_BAND_RATE_CAP",
                rule_name="Trần lãi suất theo Risk Band",
                severity="HIGH",
                law_or_policy_source="Bank Credit Policy - Section 1 Matrix",
                description=f"Lãi suất {mod.interest_rate_annual}%/năm vượt trần quy định cho Risk Band {proposal.risk_band.upper()} ({rb_policy.max_rate}%/năm).",
                suggested_remediation=f"Điều chỉnh lãi suất về tối đa {rb_policy.max_rate}%/năm.",
            )
        )
        mod.interest_rate_annual = rb_policy.max_rate
        is_modified = True

    # Rule 5: Hạn mức vay nhỏ qua e-KYC (100tr)
    if proposal.is_ekyc and proposal.loan_amount > EKYC_SMALL_LOAN_CAP:
        violations.append(
            ViolationItem(
                rule_id="RULE_005_EKYC_LOAN_CAP",
                rule_name="Hạn mức dư nợ vay tiêu dùng e-KYC",
                severity="HIGH",
                law_or_policy_source="Thông tư 43/2016/TT-NHNN & Thông tư 18/2019/TT-NHNN",
                description=f"Số tiền vay đề xuất ({proposal.loan_amount:,.0f} VNĐ) vượt trần vay tiêu dùng trực tuyến e-KYC không thế chấp ({EKYC_SMALL_LOAN_CAP:,.0f} VNĐ).",
                suggested_remediation=f"Giảm số tiền vay e-KYC về tối đa {EKYC_SMALL_LOAN_CAP:,.0f} VNĐ.",
            )
        )
        mod.loan_amount = EKYC_SMALL_LOAN_CAP
        is_modified = True

    # Rule 6: Hạn mức vay theo Risk Band
    if mod.loan_amount > rb_policy.max_amount:
        violations.append(
            ViolationItem(
                rule_id="RULE_006_RISK_BAND_AMOUNT_CAP",
                rule_name="Hạn mức vay tối đa theo Risk Band",
                severity="HIGH",
                law_or_policy_source="Bank Credit Policy - Section 1 Matrix",
                description=f"Số tiền vay ({mod.loan_amount:,.0f} VNĐ) vượt hạn mức tối đa cho Risk Band {proposal.risk_band.upper()} ({rb_policy.max_amount:,.0f} VNĐ).",
                suggested_remediation=f"Giảm số tiền vay về mức tối đa {rb_policy.max_amount:,.0f} VNĐ.",
            )
        )
        mod.loan_amount = rb_policy.max_amount
        is_modified = True

    # Rule 7: Mục đích cho vay bị cấm
    purpose_lower = proposal.loan_purpose.lower()
    for pattern, cat in PROHIBITED_PURPOSE_KEYWORDS:
        if re.search(pattern, purpose_lower):
            violations.append(
                ViolationItem(
                    rule_id="RULE_007_PROHIBITED_LOAN_PURPOSE",
                    rule_name="Mục đích sử dụng vốn vay bị cấm",
                    severity="CRITICAL",
                    law_or_policy_source="Thông tư 39/2016/TT-NHNN, TT 06/2023/TT-NHNN & Bank Policy Section 3",
                    description=f"Mục đích vay '{proposal.loan_purpose}' thuộc danh mục bị cấm: {cat}.",
                    suggested_remediation="Từ chối cấp tín dụng do mục đích sử dụng vốn không hợp pháp.",
                )
            )

    # Rule 8: Kiểm tra tỷ lệ DTI (Debt-to-Income <= 45%)
    if proposal.monthly_income > 0:
        new_payment = calculate_monthly_payment(mod.loan_amount, mod.interest_rate_annual, mod.loan_term_months)
        total_monthly_debt = proposal.current_monthly_debt + new_payment
        dti = total_monthly_debt / proposal.monthly_income

        if dti > rb_policy.max_dti:
            violations.append(
                ViolationItem(
                    rule_id="RULE_008_DTI_EXCEEDED",
                    rule_name="Tỷ lệ trả nợ trên thu nhập (DTI) vượt ngưỡng",
                    severity="HIGH",
                    law_or_policy_source="Bank Credit Policy - Section 2 (Max DTI <= 45%)",
                    description=f"Tỷ lệ DTI tính toán ({dti * 100:.1f}%) vượt ngưỡng tối đa cho phép ({rb_policy.max_dti * 100:.0f}%).",
                    suggested_remediation="Giảm hạn mức vay hoặc kéo dài thời hạn vay để hạ chỉ số DTI.",
                )
            )
            # Tự động điều chỉnh hạn mức vay để DTI vừa đúng max_dti
            available_monthly = (proposal.monthly_income * rb_policy.max_dti) - proposal.current_monthly_debt
            if available_monthly > 0:
                # Ước tính hạn mức vay tối đa phù hợp DTI
                monthly_rate = (mod.interest_rate_annual / 100.0) / 12.0
                if monthly_rate > 0:
                    max_dti_loan = available_monthly * ((1 + monthly_rate) ** mod.loan_term_months - 1) / (monthly_rate * (1 + monthly_rate) ** mod.loan_term_months)
                    mod.loan_amount = min(mod.loan_amount, max_dti_loan)
                    is_modified = True
            else:
                mod.loan_amount = 0.0

    return violations, (mod if is_modified else proposal)


def evaluate_compliance(proposal: LoanProposal) -> ComplianceCheckResult:
    """Đánh giá toàn bộ tuân thủ chính sách cho bản đề xuất khoản vay."""
    violations, modified_proposal = verify_hard_rules(proposal)

    has_critical = any(v.severity == "CRITICAL" for v in violations)
    has_any_violation = len(violations) > 0

    citations = [
        "Bộ luật Dân sự 2015 (Luật số 91/2015/QH13) - Điều 468",
        "Luật Các tổ chức tín dụng 2024 (Luật số 32/2024/QH15) - Điều 102, 103, 104",
        "Thông tư 39/2016/TT-NHNN, TT 06/2023/TT-NHNN & TT 12/2024/TT-NHNN",
        "Thông tư 43/2016/TT-NHNN & TT 18/2019/TT-NHNN",
        "Nghị định 13/2023/NĐ-CP về Bảo vệ dữ liệu cá nhân",
        "Quy định Khung Cấp Tín Dụng & Thẩm Định Rủi Ro Nội Bộ Ngân Hàng",
    ]

    if has_critical:
        decision = "REJECTED"
        is_compliant = False
        explanation = "Đề xuất bị TỪ CHỐI do vi phạm các điều khoản chính sách pháp luật cứng hoặc tiêu chuẩn an toàn rủi ro tối thiểu."
    elif has_any_violation:
        decision = "MODIFIED_WITH_WARNINGS"
        is_compliant = False
        explanation = "Đề xuất đã được ĐIỀU CHỈNH TỰ ĐỘNG để tuân thủ các quy định về lãi suất, hạn mức và tỷ lệ DTI."
    else:
        decision = "APPROVED"
        is_compliant = True
        explanation = "Đề xuất khoản vay TUÂN THỦ HOÀN TOÀN các quy định pháp luật Việt Nam và chính sách tín dụng của Ngân hàng."

    return ComplianceCheckResult(
        is_compliant=is_compliant,
        decision=decision,
        violations=violations,
        original_proposal=proposal,
        modified_proposal=modified_proposal if decision != "APPROVED" else proposal,
        citation_references=citations,
        explanation=explanation,
    )
