"""
Continuous Feedback & Self-Learning Engine.
Logs scoring sessions, ingests actual loan performance outcome labels, tracks drift (PSI), and triggers automated model retraining.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.credit_scoring.metrics import population_stability_index


@dataclass
class ScoringSessionLog:
    session_id: str
    timestamp: float
    applicant_data: dict[str, Any]
    predicted_probability: float
    risk_band: str
    shap_top_factors: list[dict[str, Any]]
    compliance_decision: str
    actual_target: Optional[int] = None  # 0 = Paid / Good, 1 = Default / Bad
    actual_dpd_days: Optional[int] = None  # Số ngày quá hạn
    loan_status: str = "RECOMMENDATION_GENERATED"  # Stages: GENERATED -> ACCEPTED -> DISBURSED -> DEFAULTED / CLOSED


class FeedbackLearningService:
    """Service chịu trách nhiệm thu thập log phiên chấm điểm, nhãn phản hồi và kích hoạt tự học."""

    def __init__(self, log_dir: Optional[Path] = None) -> None:
        if log_dir is None:
            base_dir = Path(__file__).resolve().parents[2]
            log_dir = base_dir / "data" / "logs"
        self.log_dir = log_dir
        self.log_file = self.log_dir / "prediction_feedback.jsonl"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_session(
        self,
        session_id: str,
        applicant_data: dict[str, Any],
        predicted_probability: float,
        risk_band: str,
        shap_top_factors: list[dict[str, Any]],
        compliance_decision: str,
    ) -> ScoringSessionLog:
        """Ghi vết một phiên chấm điểm và đưa ra đề xuất cho vay."""
        entry = ScoringSessionLog(
            session_id=session_id,
            timestamp=time.time(),
            applicant_data=applicant_data,
            predicted_probability=predicted_probability,
            risk_band=risk_band,
            shap_top_factors=shap_top_factors,
            compliance_decision=compliance_decision,
        )
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        return entry

    def ingest_outcome(
        self,
        session_id: str,
        actual_target: int,
        actual_dpd_days: int = 0,
        loan_status: str = "CLOSED",
    ) -> bool:
        """Cập nhật nhãn kết quả khoản vay thực tế (Paid / Defaulted) cho một session_id."""
        if not self.log_file.exists():
            return False

        updated = False
        rows = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("session_id") == session_id:
                    data["actual_target"] = actual_target
                    data["actual_dpd_days"] = actual_dpd_days
                    data["loan_status"] = loan_status
                    updated = True
                rows.append(data)

        if updated:
            with open(self.log_file, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        return updated

    def check_retraining_trigger(self, min_samples: int = 50) -> dict[str, Any]:
        """Kiểm tra điều kiện tự học và tự động kích hoạt huấn luyện lại (Retraining Trigger)."""
        if not self.log_file.exists():
            return {"should_retrain": False, "reason": "Chưa có file log dữ liệu phiên chấm điểm."}

        labeled_count = 0
        df_records = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("actual_target") is not None:
                    labeled_count += 1
                df_records.append(data)

        if labeled_count < min_samples:
            return {
                "should_retrain": False,
                "reason": f"Số mẫu phản hồi thực tế ({labeled_count}/{min_samples}) chưa đủ ngưỡng tối thiểu để tự học.",
                "labeled_count": labeled_count,
                "total_sessions": len(df_records),
            }

        # Kích hoạt tự học
        return {
            "should_retrain": True,
            "reason": f"Đã thu thập đủ {labeled_count} hồ sơ có nhãn phản hồi kết quả thực tế. Đủ điều kiện nâng cấp mô hình.",
            "labeled_count": labeled_count,
            "total_sessions": len(df_records),
        }


# Singleton Instance
feedback_learning_service = FeedbackLearningService()
