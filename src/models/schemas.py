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
