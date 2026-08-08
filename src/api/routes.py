from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.agents.graph import agent
from src.config import get_settings
from src.models.schemas import (
    ChatRequest,
    ChatResponse,
    CreditScoreRequest,
    CreditScoreResponse,
    TextExtractRequest,
    TextExtractResponse,
)
from src.services.credit_scoring import ModelNotReadyError, load_credit_model, score_application

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
