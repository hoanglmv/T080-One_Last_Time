import os
from src.agents.state import AgentState
from src.config import get_settings


async def analyze_node(state: AgentState) -> dict:
    """Phân tích query từ user và chuẩn bị ngữ cảnh tín dụng."""
    query = state.get("query", "").strip()
    return {"analysis": f"Xử lý truy vấn tín dụng: {query}"}


async def respond_node(state: AgentState) -> dict:
    """Tạo phản hồi tư vấn tín dụng chuyên sâu từ LLM hoặc Knowledge Base."""
    query = state.get("query", "").strip()
    error = state.get("error")

    if error:
        return {"response": f"Lỗi hệ thống: {error}"}

    settings = get_settings()

    # Nếu có OPENAI_API_KEY -> Gọi LLM OpenAI
    if settings.openai_api_key:
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
            completion = await client.chat.completions.create(
                model=settings.model_name or "gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Bạn là Trợ Lý AI Chuyên Gia Tín Dụng Thay Thế (Alternative Credit Scoring Advisor) "
                            "thuộc hệ thống Home Credit POC. Hãy trả lời ngắn gọn, chuyên nghiệp, lịch sự bằng tiếng Việt. "
                            "Cung cấp kiến thức về điểm tín dụng POC (0-100), xác suất vỡ nợ P(Default), kiểm định đòn bẩy "
                            "Financial Sanity Guard (crd/inc > 25), mã SHAP Reason Codes và lời khuyên tài chính cá nhân."
                        )
                    },
                    {"role": "user", "content": query}
                ],
                temperature=0.3,
                max_tokens=500
            )
            response_text = completion.choices[0].message.content
            return {"response": response_text}
        except Exception:
            pass  # Chuyển sang fallback Knowledge Base nếu có lỗi API key

    # Tri thức Fallback chuyên sâu (Rule-based Credit Domain Expert)
    q_lower = query.lower()

    if any(k in q_lower for k in ["poc", "thang điểm", "cách tính", "điểm là gì"]):
        response = (
            "📌 **Quy trình Chấm điểm POC (Proof-of-Concept Score)**:\n\n"
            "• **Thang điểm**: 0.0 - 100.0 điểm (Điểm càng cao thể hiện năng lực trả nợ càng tốt).\n"
            "• **Công thức**: `POC Score = 100 × (1 - P(Default))`.\n"
            "• **Ý nghĩa**: Với tỷ lệ nợ xấu tự nhiên ~8.07%, một khách hàng bình thường sẽ có xác suất vỡ nợ "
            "P(Default) từ 2% - 8%, tương ứng điểm POC đạt từ **92 - 98 điểm**.\n"
            "• **Phân khúc rủi ro**: >85 (Thấp - Phê duyệt nhanh), 70-85 (Vừa - Thẩm định thêm), <70 (Cao)."
        )
    elif any(k in q_lower for k in ["đòn bẩy", "sanity", "kiểm định", "an toàn", "rủi ro"]):
        response = (
            "🛡️ **Cơ chế Kiểm định An toàn Đòn bẩy Tài chính (Financial Sanity Guard)**:\n\n"
            "• **Vấn đề**: Các mô hình Cây quyết định (LightGBM) dễ bỏ sót rủi ro khi gặp dữ liệu đòn bẩy cực đoan ngoài miền (OOD).\n"
            "• **Cơ chế kiểm định ép sàn**:\n"
            "  - Khoản vay / Thu nhập > 25 lần ➔ Ép P(Default) ≥ 88.0% (POC Score ≤ 12.0)\n"
            "  - Khoản vay / Thu nhập > 50 lần ➔ Ép P(Default) ≥ 95.0% (POC Score ≤ 5.0)\n"
            "  - Khoản vay / Thu nhập > 100 lần ➔ Ép P(Default) ≥ 99.5% (POC Score ≤ 0.5)\n"
            "• **Mục đích**: Bảo vệ ngân hàng khỏi các hồ sơ gian lận hoặc kiệt quệ tài chính cực đoan."
        )
    elif any(k in q_lower for k in ["tiền việt", "vnđ", "nước ngoài", "bộ dữ liệu"]):
        response = (
            "🌐 **Khả năng áp dụng Tiền Việt (VNĐ) trên Dữ liệu Quốc tế**:\n\n"
            "1. **Bất biến tỷ lệ (Scale Invariance)**: Mô hình học trên các chỉ số đòn bẩy phi kích thước "
            "(Vay/Thu nhập, Trả hàng tháng/Thu nhập). Khi nhân cả thu nhập và khoản vay với tỷ giá quy đổi, tỷ lệ này không đổi.\n"
            "2. **Domain Adaptation**: Hệ thống tự động chuẩn hóa quy mô đơn vị tiền VNĐ về miền giá trị của mô hình.\n"
            "3. **Tương đồng sản phẩm**: Tập dữ liệu do Home Credit Group phát hành có cấu trúc sản phẩm tín dụng tiêu dùng "
            "và hành vi khách hàng Thin-file/Unbanked tương đồng 100% tại Việt Nam."
        )
    elif any(k in q_lower for k in ["tăng điểm", "cải thiện", "lời khuyên", "nâng điểm"]):
        response = (
            "💡 **Lời khuyên Cải thiện Điểm Tín dụng POC**:\n\n"
            "1. **Giảm tỷ lệ đòn bẩy**: Đảm bảo tổng khoản vay không vượt quá 3 - 5 lần thu nhập năm.\n"
            "2. **Tối ưu nghĩa vụ hàng tháng (DSR)**: Giữ tiền trả gốc + lãi hàng tháng < 30% thu nhập.\n"
            "3. **Duy trì thông tin ổn định**: Giữ thâm niên sử dụng SĐT và thâm niên công tác lâu dài.\n"
            "4. **Cung cấp điểm 3rd Party tốt**: Duy trì điểm rủi ro bên thứ 3 (Telco/Mạng xã hội) uy tín."
        )
    else:
        response = (
            f"🤖 **Trợ Lý AI Thẩm Định Tín Dụng (ACS Advisor)**:\n\n"
            f"Tôi đã ghi nhận câu hỏi của bạn: *\"{query}\"*\n\n"
            f"Hệ thống Alternative Credit Scoring sử dụng mô hình **Champion LightGBM + Platt Calibration** "
            f"với chỉ số **ROC-AUC = 0.7646** và **KS Statistic = 40.46%**.\n\n"
            f"Bạn có thể hỏi tôi về:\n"
            f"1. *Thang điểm POC và phân khúc rủi ro là gì?*\n"
            f"2. *Cơ chế kiểm định đòn bẩy tài chính an toàn hoạt động ra sao?*\n"
            f"3. *Lý do mô hình dùng được cho đơn vị tiền Việt (VNĐ)?*\n"
            f"4. *Làm sao để cải thiện điểm tín dụng?*"
        )

    return {"response": response}

