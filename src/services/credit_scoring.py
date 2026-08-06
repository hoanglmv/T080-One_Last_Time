"""Application service for deterministic scoring and optional LLM narration."""

from __future__ import annotations

import asyncio
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_settings
from src.credit_scoring.model import CreditModelBundle
from src.services.llm import get_llm


class ModelNotReadyError(RuntimeError):
    """Raised when the configured scoring artifact is unavailable."""


@lru_cache(maxsize=4)
def load_credit_model(path: str) -> CreditModelBundle:
    artifact = Path(path)
    if not artifact.is_file():
        raise ModelNotReadyError(
            f"Credit model artifact is not ready at {artifact}. Run: python -m src.credit_scoring.cli train"
        )
    return CreditModelBundle.load(artifact)


def deterministic_summary(result: dict[str, Any]) -> str:
    band_labels = {
        "low": "thấp",
        "moderate": "trung bình",
        "high": "cao",
        "very_high": "rất cao",
    }
    factors = result["top_factors"][:3]
    detail = " ".join(factor["reason"] for factor in factors)
    return (
        f"Xác suất payment difficulty ước tính là "
        f"{100 * result['payment_difficulty_probability']:.2f}%, thuộc nhóm rủi ro "
        f"{band_labels[result['risk_band']]}. {detail} Kết quả này chỉ hỗ trợ rà soát, không phải quyết định tín dụng."
    )


async def _llm_summary(result: dict[str, Any]) -> str:
    settings = get_settings()
    if not settings.enable_llm_explanations or not settings.openai_api_key:
        raise RuntimeError("LLM explanation is disabled or no API key is configured")

    # Only derived reason codes are sent. Identifiers, protected attributes and raw
    # application payload are intentionally excluded.
    safe_payload = {
        "probability_percent": round(100 * result["payment_difficulty_probability"], 2),
        "poc_score": result["poc_score"],
        "risk_band": result["risk_band"],
        "factors": [
            {
                "description": factor["description"],
                "direction": factor["direction"],
                "reason": factor["reason"],
            }
            for factor in result["top_factors"]
        ],
    }
    messages = [
        SystemMessage(
            content=(
                "Bạn diễn đạt reason codes của một POC credit-risk bằng tiếng Việt rõ ràng, thân thiện. "
                "Không thay đổi số, risk band hay hướng tác động. Không suy đoán nguyên nhân nhân quả, "
                "không đưa ra quyết định duyệt/từ chối, không khuyên lách hệ thống. Nêu 2-4 yếu tố chính, "
                "giới hạn dữ liệu và nhắc cần human review. Viết tối đa 180 từ."
            )
        ),
        HumanMessage(content=json.dumps(safe_payload, ensure_ascii=False)),
    ]
    response = await asyncio.wait_for(
        get_llm(temperature=0.1).ainvoke(messages),
        timeout=settings.llm_explanation_timeout_seconds,
    )
    return str(response.content)


async def score_application(
    application: dict[str, Any],
    *,
    explain_with_llm: bool,
    top_k: int,
) -> dict[str, Any]:
    settings = get_settings()
    bundle = load_credit_model(settings.credit_model_path)
    result = bundle.score([application], top_k=top_k)[0]
    result["friendly_explanation"] = deterministic_summary(result)
    result["llm_used"] = False
    if explain_with_llm:
        try:
            result["friendly_explanation"] = await _llm_summary(result)
            result["llm_used"] = True
        except Exception:
            # Scoring never depends on an external LLM. The deterministic reason
            # codes remain the source of truth and are always returned.
            result["data_quality"]["warnings"].append(
                "LLM không khả dụng; hệ thống đã dùng phần giải thích xác định tại chỗ."
            )
    return result
