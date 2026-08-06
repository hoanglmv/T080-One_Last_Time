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


def heuristic_text_extractor(text: str) -> dict[str, Any]:
    import re
    extracted: dict[str, Any] = {}
    lower_text = text.lower()

    # Tuổi
    age_match = re.search(r"(\d{1,2})\s*(?:tuổi|t)", lower_text)
    if age_match:
        age = int(age_match.group(1))
        if 18 <= age <= 90:
            extracted["AGE_YEARS_INPUT"] = age

    # Thâm niên làm việc
    emp_match = re.search(r"(?:thâm niên|làm việc|làm|kinh nghiệm)\s*(\d{1,2})\s*năm", lower_text)
    if emp_match:
        emp = int(emp_match.group(1))
        if 0 <= emp <= 60:
            extracted["EMPLOYMENT_YEARS_INPUT"] = emp

    # Helper function to parse monetary amounts in VND (triệu, tr, tỷ, nghìn)
    def parse_money(match_str: str, unit_str: str) -> int | None:
        try:
            num = float(match_str.replace(".", "").replace(",", "."))
            unit = unit_str.lower().strip()
            if "tỷ" in unit:
                return int(num * 1_000_000_000)
            elif "triệu" in unit or "tr" in unit:
                return int(num * 1_000_000)
            elif "nghìn" in unit or "k" in unit:
                return int(num * 1_000)
            elif num < 1000:
                return int(num * 1_000_000)
            return int(num)
        except Exception:
            return None

    # Thu nhập
    inc_match = re.search(r"(?:thu nhập|lương|kiếm được)\s*[\approx:]*\s*([\d\.,]+)\s*(triệu|tr|tỷ|nghìn|k|vnđ|đ)*", lower_text)
    if inc_match:
        val = parse_money(inc_match.group(1), inc_match.group(2) or "")
        if val and val > 0:
            extracted["AMT_INCOME_TOTAL"] = val

    # Khoản vay
    crd_match = re.search(r"(?:vay|cần vay|khoản vay)\s*[\approx:]*\s*([\d\.,]+)\s*(triệu|tr|tỷ|nghìn|k|vnđ|đ)*", lower_text)
    if crd_match:
        val = parse_money(crd_match.group(1), crd_match.group(2) or "")
        if val and val > 0:
            extracted["AMT_CREDIT"] = val

    # Khoản trả hàng tháng
    ann_match = re.search(r"(?:trả hàng tháng|trả định kỳ|góp hàng tháng|trả mỗi tháng)\s*[\approx:]*\s*([\d\.,]+)\s*(triệu|tr|tỷ|nghìn|k|vnđ|đ)*", lower_text)
    if ann_match:
        val = parse_money(ann_match.group(1), ann_match.group(2) or "")
        if val and val > 0:
            extracted["AMT_ANNUITY"] = val

    # Loại thu nhập
    if any(k in lower_text for k in ["công chức", "nhà nước", "viên chức"]):
        extracted["NAME_INCOME_TYPE"] = "State servant"
    elif any(k in lower_text for k in ["kinh doanh", "thương mại", "buôn bán", "hợp tác"]):
        extracted["NAME_INCOME_TYPE"] = "Commercial associate"
    elif any(k in lower_text for k in ["hưu trí", "lương hưu", "nghỉ hưu"]):
        extracted["NAME_INCOME_TYPE"] = "Pensioner"
    else:
        extracted["NAME_INCOME_TYPE"] = "Working"

    # Trình độ học vấn
    if any(k in lower_text for k in ["đại học", "sau đại học", "thạc sĩ", "tiến sĩ"]):
        extracted["NAME_EDUCATION_TYPE"] = "Higher education"
    elif any(k in lower_text for k in ["đang học đại học", "dở dang"]):
        extracted["NAME_EDUCATION_TYPE"] = "Incomplete higher"
    elif any(k in lower_text for k in ["trung học cơ sở", "cấp 2"]):
        extracted["NAME_EDUCATION_TYPE"] = "Lower secondary"
    else:
        extracted["NAME_EDUCATION_TYPE"] = "Secondary / secondary special"

    # Loại nhà ở
    if any(k in lower_text for k in ["nhà riêng", "căn hộ sở hữu", "sở hữu nhà"]):
        extracted["NAME_HOUSING_TYPE"] = "House / apartment"
    elif any(k in lower_text for k in ["thuê", "trọ"]):
        extracted["NAME_HOUSING_TYPE"] = "Rented apartment"
    elif any(k in lower_text for k in ["bố mẹ", "cha mẹ"]):
        extracted["NAME_HOUSING_TYPE"] = "With parents"

    # Tình trạng gia đình
    if any(k in lower_text for k in ["đã kết hôn", "lập gia đình", "có gia đình"]):
        extracted["NAME_FAMILY_STATUS"] = "Married"
    elif any(k in lower_text for k in ["độc thân", "chưa kết hôn"]):
        extracted["NAME_FAMILY_STATUS"] = "Single / not married"
    elif any(k in lower_text for k in ["ly hôn", "ly thân"]):
        extracted["NAME_FAMILY_STATUS"] = "Separated"

    return extracted


async def extract_application_from_text(text: str) -> dict[str, Any]:
    settings = get_settings()
    if settings.openai_api_key and settings.enable_llm_explanations:
        try:
            messages = [
                SystemMessage(
                    content=(
                        "Bạn là chuyên gia trích xuất dữ liệu hồ sơ tín dụng bằng tiếng Việt. "
                        "Hãy đọc đoạn văn bản mô tả khách hàng và trích xuất thành 1 JSON object hợp lệ chứa các trường:\n"
                        "- AMT_INCOME_TOTAL (số: thu nhập tổng VNĐ, vd 15000000 hoặc 180000)\n"
                        "- AMT_CREDIT (số: giá trị khoản vay VNĐ, vd 50000000)\n"
                        "- AMT_ANNUITY (số: trả hàng tháng VNĐ, vd 3000000)\n"
                        "- AMT_GOODS_PRICE (số: giá trị hàng hóa VNĐ, vd 45000000)\n"
                        "- AGE_YEARS_INPUT (số: số tuổi, vd 38)\n"
                        "- EMPLOYMENT_YEARS_INPUT (số: thâm niên năm, vd 8)\n"
                        "- CNT_FAM_MEMBERS (số: số người gia đình, vd 2)\n"
                        "- NAME_CONTRACT_TYPE ('Cash loans' hoặc 'Revolving loans')\n"
                        "- NAME_INCOME_TYPE ('Working', 'Commercial associate', 'Pensioner', 'State servant')\n"
                        "- NAME_EDUCATION_TYPE ('Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary')\n"
                        "- NAME_HOUSING_TYPE ('House / apartment', 'With parents', 'Municipal apartment', 'Rented apartment')\n"
                        "- NAME_FAMILY_STATUS ('Married', 'Single / not married', 'Civil marriage', 'Separated')\n"
                        "Chỉ trả về JSON object thuần túy, không có markdown formatting."
                    )
                ),
                HumanMessage(content=text),
            ]
            response = await asyncio.wait_for(
                get_llm(temperature=0.0).ainvoke(messages),
                timeout=10.0,
            )
            content_str = str(response.content).strip()
            if content_str.startswith("```"):
                content_str = content_str.split("```")[1]
                if content_str.startswith("json"):
                    content_str = content_str[4:].strip()
            parsed = json.loads(content_str)
            if isinstance(parsed, dict) and len(parsed) > 0:
                return {
                    "extracted": parsed,
                    "llm_used": True,
                    "summary": f"Đã trích xuất {len(parsed)} trường thông tin bằng LLM AI."
                }
        except Exception:
            pass

    heuristic = heuristic_text_extractor(text)
    return {
        "extracted": heuristic,
        "llm_used": False,
        "summary": f"Đã trích xuất {len(heuristic)} trường thông tin bằng Bộ quy tắc (Rule-based Fallback)."
    }
