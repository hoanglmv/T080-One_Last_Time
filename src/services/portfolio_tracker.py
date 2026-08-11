"""
Portfolio Delinquency & Conversion Funnel Tracking Service.
Tracks post-approval loan performance (DPD30+, DPD90+, Roll Rates) and customer service take-up funnel analytics.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Optional


@dataclass
class LoanRecord:
    loan_id: str
    session_id: str
    customer_name: str
    applicant_age: int
    monthly_income: float
    loan_amount: float
    risk_band: str
    stage: str  # RECOMMENDATION_GENERATED, OFFER_PRESENTED, OFFER_ACCEPTED, LOAN_DISBURSED, PAID_OFF, DEFAULTED
    created_at: float
    disbursed_at: Optional[float] = None
    dpd_days: int = 0  # Days Past Due
    credit_group: int = 1  # Group 1 (Good), Group 2 (Watchlist), Group 3-5 (Bad Debt)


class PortfolioTrackerService:
    """Service theo dõi danh mục dư nợ, tỷ lệ nhảy nợ và phễu chuyển đổi dịch vụ."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        if data_dir is None:
            base_dir = Path(__file__).resolve().parents[2]
            data_dir = base_dir / "data"
        self.data_dir = data_dir
        self.portfolio_file = self.data_dir / "portfolio_tracker.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if not self.portfolio_file.exists():
            self._initialize_seed_data()

    def _initialize_seed_data(self) -> None:
        """Khởi tạo dữ liệu mô phỏng danh mục tín dụng thực tế nếu chưa có file."""
        now = time.time()
        seed_loans: list[LoanRecord] = [
            # Hồ sơ Nhóm 1 (Tốt)
            LoanRecord("L001", "S001", "Nguyễn Văn A", 28, 20000000.0, 50000000.0, "low", "LOAN_DISBURSED", now - 86400 * 90, now - 86400 * 85, 0, 1),
            LoanRecord("L002", "S002", "Trần Thị B", 32, 25000000.0, 70000000.0, "low", "LOAN_DISBURSED", now - 86400 * 60, now - 86400 * 55, 0, 1),
            LoanRecord("L003", "S003", "Lê Văn C", 25, 15000000.0, 30000000.0, "moderate", "LOAN_DISBURSED", now - 86400 * 45, now - 86400 * 40, 0, 1),
            LoanRecord("L004", "S004", "Phạm Văn D", 40, 30000000.0, 100000000.0, "low", "PAID_OFF", now - 86400 * 180, now - 86400 * 175, 0, 1),
            
            # Hồ sơ Nhóm 2 (Cảnh báo DPD30+)
            LoanRecord("L005", "S005", "Hoàng Văn E", 29, 18000000.0, 40000000.0, "moderate", "LOAN_DISBURSED", now - 86400 * 60, now - 86400 * 50, 15, 2),
            LoanRecord("L006", "S006", "Đặng Thị F", 24, 14000000.0, 25000000.0, "high", "LOAN_DISBURSED", now - 86400 * 40, now - 86400 * 35, 35, 2),
            
            # Hồ sơ Nhóm 3-5 (Nợ xấu DPD90+)
            LoanRecord("L007", "S007", "Vũ Văn G", 22, 12000000.0, 20000000.0, "high", "DEFAULTED", now - 86400 * 120, now - 86400 * 110, 95, 3),

            # Hồ sơ các giai đoạn phễu chuyển đổi (Conversion Funnel)
            LoanRecord("L008", "S008", "Ngô Văn H", 35, 22000000.0, 60000000.0, "moderate", "OFFER_PRESENTED", now - 86400 * 5),
            LoanRecord("L009", "S009", "Bùi Thị I", 27, 19000000.0, 45000000.0, "low", "OFFER_ACCEPTED", now - 86400 * 2),
            LoanRecord("L010", "S010", "Đỗ Văn K", 31, 16000000.0, 35000000.0, "moderate", "RECOMMENDATION_GENERATED", now - 86400 * 1),
        ]
        self.save_loans(seed_loans)

    def load_loans(self) -> list[LoanRecord]:
        """Tải danh sách hồ sơ khoản vay."""
        if not self.portfolio_file.exists():
            return []
        with open(self.portfolio_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
            return [LoanRecord(**r) for r in raw]

    def save_loans(self, loans: list[LoanRecord]) -> None:
        """Lưu danh sách hồ sơ khoản vay."""
        with open(self.portfolio_file, "w", encoding="utf-8") as f:
            json.dump([asdict(l) for l in loans], f, ensure_ascii=False, indent=2)

    def record_loan_stage(
        self,
        session_id: str,
        stage: str,
        applicant_name: str = "Khách hàng",
        applicant_age: int = 30,
        monthly_income: float = 20000000.0,
        loan_amount: float = 50000000.0,
        risk_band: str = "moderate",
    ) -> LoanRecord:
        """Ghi nhận hoặc cập nhật giai đoạn phễu chuyển đổi cho khoản vay."""
        loans = self.load_loans()
        existing = next((l for l in loans if l.session_id == session_id), None)
        now = time.time()

        if existing:
            existing.stage = stage
            if stage == "LOAN_DISBURSED" and not existing.disbursed_at:
                existing.disbursed_at = now
            self.save_loans(loans)
            return existing

        loan_id = f"L{len(loans) + 1:03d}"
        new_loan = LoanRecord(
            loan_id=loan_id,
            session_id=session_id,
            customer_name=applicant_name,
            applicant_age=applicant_age,
            monthly_income=monthly_income,
            loan_amount=loan_amount,
            risk_band=risk_band,
            stage=stage,
            created_at=now,
            disbursed_at=now if stage == "LOAN_DISBURSED" else None,
        )
        loans.append(new_loan)
        self.save_loans(loans)
        return new_loan

    def update_dpd_and_group(self, loan_id: str, dpd_days: int) -> Optional[LoanRecord]:
        """Cập nhật số ngày quá hạn DPD và nhóm nợ."""
        loans = self.load_loans()
        loan = next((l for l in loans if l.loan_id == loan_id), None)
        if not loan:
            return None

        loan.dpd_days = dpd_days
        if dpd_days <= 10:
            loan.credit_group = 1  # Nợ đủ tiêu chuẩn
        elif dpd_days <= 30:
            loan.credit_group = 2  # Nợ cần chú ý (Group 2)
        elif dpd_days <= 90:
            loan.credit_group = 3  # Nợ dưới tiêu chuẩn (Group 3)
            loan.stage = "DEFAULTED"
        else:
            loan.credit_group = 5  # Nợ có khả năng mất vốn (Group 5 Bad Debt)
            loan.stage = "DEFAULTED"

        self.save_loans(loans)
        return loan

    def get_delinquency_stats(self) -> dict[str, Any]:
        """Tính toán các chỉ số nợ quá hạn và tỷ lệ nhảy nợ trong danh mục."""
        loans = self.load_loans()
        disbursed_loans = [l for l in loans if l.stage in {"LOAN_DISBURSED", "PAID_OFF", "DEFAULTED"}]
        total_disbursed = len(disbursed_loans)

        if total_disbursed == 0:
            return {
                "total_disbursed_loans": 0,
                "delinquency_rate_dpd30": 0.0,
                "delinquency_rate_dpd90": 0.0,
                "group_distribution": {},
                "roll_rate_matrix": {},
            }

        dpd30_count = sum(1 for l in disbursed_loans if l.dpd_days >= 30)
        dpd90_count = sum(1 for l in disbursed_loans if l.dpd_days >= 90)

        group_counts: dict[str, int] = {
            "Group 1 (Đủ tiêu chuẩn - DPD <= 10)": sum(1 for l in disbursed_loans if l.credit_group == 1),
            "Group 2 (Cần chú ý - DPD 11-30)": sum(1 for l in disbursed_loans if l.credit_group == 2),
            "Group 3-5 (Nợ xấu - DPD > 30)": sum(1 for l in disbursed_loans if l.credit_group >= 3),
        }

        # Ma trận chuyển nhóm nợ Roll Rate (Group 1 -> 2 -> 3-5)
        roll_rates = {
            "Group 1 to Group 2": round((group_counts["Group 2 (Cần chú ý - DPD 11-30)"] / max(1, total_disbursed)) * 100, 2),
            "Group 2 to Group 3-5": round((group_counts["Group 3-5 (Nợ xấu - DPD > 30)"] / max(1, total_disbursed)) * 100, 2),
        }

        return {
            "total_disbursed_loans": total_disbursed,
            "total_disbursed_amount": sum(l.loan_amount for l in disbursed_loans),
            "delinquency_rate_dpd30_pct": round((dpd30_count / total_disbursed) * 100.0, 2),
            "delinquency_rate_dpd90_pct": round((dpd90_count / total_disbursed) * 100.0, 2),
            "group_distribution": group_counts,
            "roll_rate_matrix": roll_rates,
            "summary_vi": f"Tổng dư nợ giải ngân: {total_disbursed} khoản vay. Tỷ lệ nợ quá hạn DPD30+: {(dpd30_count / total_disbursed) * 100:.1f}%, DPD90+: {(dpd90_count / total_disbursed) * 100:.1f}%.",
        }

    def get_conversion_funnel_stats(self) -> dict[str, Any]:
        """Tính toán phễu chuyển đổi tỷ lệ khách hàng sử dụng dịch vụ được tư vấn."""
        loans = self.load_loans()
        total_sessions = len(loans)

        if total_sessions == 0:
            return {"total_sessions": 0, "conversion_rate_pct": 0.0, "stages": {}}

        stage_counts = {
            "1. RECOMMENDATION_GENERATED": sum(1 for l in loans if l.stage == "RECOMMENDATION_GENERATED"),
            "2. OFFER_PRESENTED": sum(1 for l in loans if l.stage in {"OFFER_PRESENTED", "OFFER_ACCEPTED", "LOAN_DISBURSED", "PAID_OFF", "DEFAULTED"}),
            "3. OFFER_ACCEPTED": sum(1 for l in loans if l.stage in {"OFFER_ACCEPTED", "LOAN_DISBURSED", "PAID_OFF", "DEFAULTED"}),
            "4. LOAN_DISBURSED": sum(1 for l in loans if l.stage in {"LOAN_DISBURSED", "PAID_OFF", "DEFAULTED"}),
        }

        disbursed_count = stage_counts["4. LOAN_DISBURSED"]
        conversion_rate = (disbursed_count / max(1, total_sessions)) * 100.0

        return {
            "total_sessions": total_sessions,
            "conversion_rate_pct": round(conversion_rate, 2),
            "disbursed_count": disbursed_count,
            "stage_counts": stage_counts,
            "summary_vi": f"Tỷ lệ chuyển đổi khách hàng sử dụng dịch vụ sau tư vấn: {conversion_rate:.1f}% ({disbursed_count}/{total_sessions} hồ sơ đã giải ngân).",
        }


# Singleton Instance
portfolio_tracker_service = PortfolioTrackerService()
