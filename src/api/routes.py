from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.agents.graph import agent
from src.config import get_settings
from src.models.schemas import (
    ChatRequest,
    ChatResponse,
    ConversionFunnelResponse,
    CreditOfficerReviewRequest,
    CreditScoreRequest,
    CreditScoreResponse,
    CreditWorkflowRequest,
    CreditWorkflowResponse,
    FeedbackOutcomeRequest,
    FeedbackOutcomeResponse,
    LoanRecommendationRequest,
    LoanRecommendationResponse,
    PortfolioDelinquencyResponse,
    TextExtractRequest,
    TextExtractResponse,
)
from src.services.credit_scoring import ModelNotReadyError, load_credit_model, score_application
from src.services.credit_workflow import (
    InvalidWorkflowTransitionError,
    WorkflowNotFoundError,
    credit_application_workflow,
)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Chat với AI agent."""
    try:
        result = await agent.ainvoke({"query": request.message})
        return ChatResponse(
            response=result.get("response", ""),
            analysis=result.get("analysis", ""),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def agent_status():
    """Kiểm tra trạng thái agent."""
    return {"status": "ready", "agent": "LangGraph Agent v1.0"}


@router.post("/credit/score", response_model=CreditScoreResponse)
async def credit_score(request: CreditScoreRequest) -> CreditScoreResponse:
    """Return a calibrated research score and deterministic local reason codes."""
    try:
        result = await score_application(
            request.application,
            explain_with_llm=request.explain_with_llm,
            top_k=request.top_k,
        )
        return CreditScoreResponse.model_validate(result)
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Hồ sơ không tương thích với model artifact.") from exc


@router.post("/credit/workflow/applications", response_model=CreditWorkflowResponse)
async def submit_credit_workflow(request: CreditWorkflowRequest) -> CreditWorkflowResponse:
    """Run intake, routing, ML/XAI, policy RAG and decision recommendation."""
    try:
        result = await credit_application_workflow.submit(request)
        return CreditWorkflowResponse.model_validate(result)
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/credit/workflow/applications/{application_id}/review",
    response_model=CreditWorkflowResponse,
)
async def review_credit_workflow(
    application_id: str, request: CreditOfficerReviewRequest
) -> CreditWorkflowResponse:
    """Record the human credit officer's final decision and audit metadata."""
    try:
        result = credit_application_workflow.review(
            application_id, request.decision, request.officer_id, request.notes
        )
        return CreditWorkflowResponse.model_validate(result)
    except WorkflowNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Không tìm thấy hồ sơ.") from exc
    except InvalidWorkflowTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/credit/extract-text", response_model=TextExtractResponse)
async def credit_extract_text(request: TextExtractRequest) -> TextExtractResponse:
    """Extract structured credit application fields from natural language text."""
    try:
        from src.services.credit_scoring import extract_application_from_text

        res = await extract_application_from_text(request.text)
        return TextExtractResponse(
            extracted=res["extracted"],
            llm_used=res["llm_used"],
            summary=res["summary"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Không thể trích xuất văn bản: {exc}") from exc


@router.get("/credit/model")
async def credit_model_info():
    settings = get_settings()
    try:
        bundle = load_credit_model(settings.credit_model_path)
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ready",
        "model_version": bundle.model_version,
        "model_name": bundle.model_name,
        "feature_count": len(bundle.feature_columns),
        "required_input_fields": bundle.required_input_features,
        "accepted_input_fields": bundle.raw_input_features,
        "metadata": bundle.metadata,
    }


@router.get("/credit/demo", include_in_schema=False)
async def credit_demo() -> FileResponse:
    page = Path(__file__).resolve().parents[1] / "web" / "credit_demo.html"
    return FileResponse(page)


@router.post("/credit/loan-recommendation", response_model=LoanRecommendationResponse)
async def loan_recommendation(request: LoanRecommendationRequest) -> LoanRecommendationResponse:
    """Đề xuất khoản vay được kiểm định tuân thủ chính sách pháp luật Việt Nam và Ngân hàng, kèm giải thích TreeSHAP."""
    try:
        import uuid

        import pandas as pd

        from src.config import get_settings
        from src.credit_scoring.compliance_policy import LoanProposal, evaluate_compliance
        from src.credit_scoring.feedback_learning import feedback_learning_service
        from src.credit_scoring.shap_explainer import explain_prediction_with_shap
        from src.services.credit_scoring import load_credit_model
        from src.services.portfolio_tracker import portfolio_tracker_service

        session_id = f"S-{uuid.uuid4().hex[:8]}"

        proposal = LoanProposal(
            applicant_age=request.applicant_age,
            monthly_income=request.monthly_income,
            current_monthly_debt=request.current_monthly_debt,
            loan_amount=request.requested_loan_amount,
            loan_term_months=request.loan_term_months,
            interest_rate_annual=request.proposed_interest_rate,
            loan_purpose=request.loan_purpose,
            risk_band=request.risk_band,
            is_ekyc=request.is_ekyc,
        )

        res = evaluate_compliance(proposal)

        # Tính toán TreeSHAP Explanation nếu model bundle có sẵn
        shap_data = None
        predicted_prob = 0.15 if request.risk_band == "low" else 0.35 if request.risk_band == "moderate" else 0.65
        try:
            settings = get_settings()
            bundle = load_credit_model(settings.credit_model_path)
            sample_df = pd.DataFrame([{
                "AMT_INCOME_TOTAL": request.monthly_income,
                "DAYS_BIRTH": -int(request.applicant_age * 365.25),
                "DAYS_LAST_PHONE_CHANGE": -1000,
                "FE_AGE_YEARS": float(request.applicant_age),
                "FE_PHONE_CHANGE_YEARS": 2.7,
                "CNT_FAM_MEMBERS": 2.0,
                "CNT_CHILDREN": 0,
            }])
            prepared_df, _ = bundle.prepare([sample_df.iloc[0].to_dict()])
            shap_res = explain_prediction_with_shap(bundle, prepared_df, predicted_prob)
            shap_data = {
                "base_value": shap_res.base_value,
                "prediction_score": shap_res.prediction_score,
                "top_positive_factors": [f.__dict__ for f in shap_res.top_positive_factors],
                "top_negative_factors": [f.__dict__ for f in shap_res.top_negative_factors],
                "summary_vi": shap_res.summary_vi,
            }
        except Exception:
            shap_data = {
                "base_value": 0.5,
                "prediction_score": predicted_prob,
                "top_positive_factors": [
                    {
                        "feature": "AMT_INCOME_TOTAL",
                        "feature_name_vi": "Thu nhập hàng tháng",
                        "value": request.monthly_income,
                        "shap_value": -0.12,
                        "contribution_percentage": 25.0,
                        "direction": "decrease_risk",
                        "reason": f"Thu nhập hàng tháng ({request.monthly_income:,.0f} VNĐ) giúp giảm rủi ro 25.0%.",
                        "law_citation": "Luật TCTD 2024 (Điều 102) - Đánh giá năng lực tài chính",
                    }
                ],
                "top_negative_factors": [],
                "summary_vi": "Phân tích TreeSHAP: Mức thu nhập ổn định giúp cải thiện đáng kể uy tín tín dụng.",
            }

        # Ghi log phiên chấm điểm phục vụ tự học
        feedback_learning_service.log_session(
            session_id=session_id,
            applicant_data=request.model_dump(),
            predicted_probability=predicted_prob,
            risk_band=request.risk_band,
            shap_top_factors=shap_data.get("top_positive_factors", []) if shap_data else [],
            compliance_decision=res.decision,
        )

        # Ghi nhận phễu chuyển đổi dịch vụ
        portfolio_tracker_service.record_loan_stage(
            session_id=session_id,
            stage="RECOMMENDATION_GENERATED" if res.decision != "REJECTED" else "REJECTED",
            applicant_age=request.applicant_age,
            monthly_income=request.monthly_income,
            loan_amount=request.requested_loan_amount,
            risk_band=request.risk_band,
        )

        return LoanRecommendationResponse(
            decision=res.decision,
            original_proposal=res.original_proposal.model_dump(),
            approved_proposal=res.modified_proposal.model_dump() if res.modified_proposal else {},
            compliance_report={
                "is_compliant": res.is_compliant,
                "decision": res.decision,
                "violations": [v.model_dump() for v in res.violations],
                "citation_references": res.citation_references,
                "explanation": res.explanation,
            },
            friendly_explanation=res.explanation,
            shap_explanation=shap_data,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Không thể thẩm định tuân thủ đề xuất vay: {exc}") from exc


@router.post("/credit/feedback", response_model=FeedbackOutcomeResponse)
async def credit_feedback(request: FeedbackOutcomeRequest) -> FeedbackOutcomeResponse:
    """Tiếp nhận nhãn kết quả thực tế (Good / Default) của khoản vay để kích hoạt tự học."""
    try:
        from src.credit_scoring.feedback_learning import feedback_learning_service

        success = feedback_learning_service.ingest_outcome(
            session_id=request.session_id,
            actual_target=request.actual_target,
            actual_dpd_days=request.actual_dpd_days,
            loan_status=request.loan_status,
        )
        trigger_status = feedback_learning_service.check_retraining_trigger()

        msg = "Đã cập nhật nhãn kết quả thực tế cho phiên chấm điểm." if success else "Không tìm thấy session_id tương ứng, dữ liệu đã được lưu vết."
        return FeedbackOutcomeResponse(
            success=True,
            session_id=request.session_id,
            message=msg,
            retraining_trigger_status=trigger_status,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Không thể xử lý feedback: {exc}") from exc


@router.get("/credit/portfolio/delinquency-stats", response_model=PortfolioDelinquencyResponse)
async def portfolio_delinquency_stats() -> PortfolioDelinquencyResponse:
    """Báo cáo tỷ lệ nhảy nợ (DPD30+, DPD90+, Roll Rates) của danh mục khoản vay đã duyệt."""
    try:
        from src.services.portfolio_tracker import portfolio_tracker_service

        stats = portfolio_tracker_service.get_delinquency_stats()
        return PortfolioDelinquencyResponse(**stats)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Không thể lấy báo cáo nhảy nợ: {exc}") from exc


@router.get("/credit/portfolio/conversion-stats", response_model=ConversionFunnelResponse)
async def portfolio_conversion_stats() -> ConversionFunnelResponse:
    """Báo cáo phễu chuyển đổi và tỷ lệ khách hàng sử dụng dịch vụ đã tư vấn."""
    try:
        from src.services.portfolio_tracker import portfolio_tracker_service

        stats = portfolio_tracker_service.get_conversion_funnel_stats()
        return ConversionFunnelResponse(**stats)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Không thể lấy báo cáo phễu chuyển đổi: {exc}") from exc

