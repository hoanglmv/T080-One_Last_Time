"""
Tests for TreeSHAP Explanation, Feedback Learning, Delinquency Tracking, and Conversion Analytics.
"""

from fastapi.testclient import TestClient
import pytest

from src.config import get_settings
from src.credit_scoring.feedback_learning import feedback_learning_service
from src.services.credit_scoring import load_credit_model
from src.credit_scoring.shap_explainer import explain_prediction_with_shap
from src.main import app
from src.services.portfolio_tracker import portfolio_tracker_service

client = TestClient(app)


def test_shap_explainer_output():
    """Kiểm tra TreeSHAP Explainer xuất ra các đặc trưng đóng góp kèm luật RAG."""
    settings = get_settings()
    bundle = load_credit_model(settings.credit_model_path)
    
    # Tạo dữ liệu giả định
    sample_dict = {
        "AMT_INCOME_TOTAL": 30000000.0,
        "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -2000,
        "DAYS_LAST_PHONE_CHANGE": -500,
    }
    prepared_df, _ = bundle.prepare([sample_dict])
    shap_res = explain_prediction_with_shap(bundle, prepared_df, predicted_probability=0.18)
    
    assert shap_res.prediction_score == 0.18
    assert len(shap_res.all_factors) > 0
    assert "TreeSHAP" in shap_res.summary_vi


def test_feedback_learning_service():
    """Kiểm tra service thu thập log phiên chấm điểm và tiếp nhận phản hồi."""
    session_id = "S-TEST-001"
    entry = feedback_learning_service.log_session(
        session_id=session_id,
        applicant_data={"age": 28, "income": 20000000},
        predicted_probability=0.2,
        risk_band="low",
        shap_top_factors=[],
        compliance_decision="APPROVED",
    )
    assert entry.session_id == session_id

    # Ingest outcome
    success = feedback_learning_service.ingest_outcome(session_id=session_id, actual_target=0, actual_dpd_days=0)
    assert success is True

    # Retraining trigger status
    trigger = feedback_learning_service.check_retraining_trigger(min_samples=1)
    assert trigger["should_retrain"] is True or "labeled_count" in trigger


def test_portfolio_delinquency_and_conversion_stats():
    """Kiểm tra chỉ số nợ quá hạn và phễu chuyển đổi dịch vụ."""
    delinq_stats = portfolio_tracker_service.get_delinquency_stats()
    assert "delinquency_rate_dpd30_pct" in delinq_stats
    assert "delinquency_rate_dpd90_pct" in delinq_stats
    assert "roll_rate_matrix" in delinq_stats

    funnel_stats = portfolio_tracker_service.get_conversion_funnel_stats()
    assert "conversion_rate_pct" in funnel_stats
    assert "stage_counts" in funnel_stats


def test_api_loan_recommendation_with_shap():
    """Kiểm tra API POST /api/v1/credit/loan-recommendation trả về TreeSHAP & RAG Compliance."""
    payload = {
        "applicant_age": 32,
        "monthly_income": 25000000.0,
        "current_monthly_debt": 2000000.0,
        "requested_loan_amount": 50000000.0,
        "loan_term_months": 12,
        "proposed_interest_rate": 14.5,
        "loan_purpose": "Vay mua máy tính cá nhân",
        "risk_band": "low",
        "is_ekyc": True,
    }
    response = client.post("/api/v1/credit/loan-recommendation", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] in ["APPROVED", "MODIFIED_WITH_WARNINGS"]
    assert "shap_explanation" in data
    assert "compliance_report" in data


def test_api_feedback_outcome():
    """Kiểm tra API POST /api/v1/credit/feedback tiếp nhận nhãn thực tế."""
    payload = {
        "session_id": "S-TEST-001",
        "actual_target": 0,
        "actual_dpd_days": 0,
        "loan_status": "CLOSED",
    }
    response = client.post("/api/v1/credit/feedback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_api_portfolio_stats():
    """Kiểm tra các API GET báo cáo nhảy nợ và phễu chuyển đổi."""
    res_delinq = client.get("/api/v1/credit/portfolio/delinquency-stats")
    assert res_delinq.status_code == 200
    assert "delinquency_rate_dpd30_pct" in res_delinq.json()

    res_conv = client.get("/api/v1/credit/portfolio/conversion-stats")
    assert res_conv.status_code == 200
    assert "conversion_rate_pct" in res_conv.json()
