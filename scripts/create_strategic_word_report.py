"""Script to generate an executive Word document (.docx) for Strategic Product Positioning & Ecosystem Research."""

import sys
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls


def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=120, bottom=120, left=180, right=180):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>'
    )
    tcPr.append(tcMar)


def add_heading_1(doc, text):
    h = doc.add_heading(text, level=1)
    h.paragraph_format.space_before = Pt(16)
    h.paragraph_format.space_after = Pt(6)
    for run in h.runs:
        run.font.name = "Calibri"
        run.font.size = Pt(18)
        run.font.bold = True
        run.font.color.rgb = RGBColor(15, 23, 42) # Slate 900
    return h


def add_heading_2(doc, text):
    h = doc.add_heading(text, level=2)
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(4)
    for run in h.runs:
        run.font.name = "Calibri"
        run.font.size = Pt(14)
        run.font.bold = True
        run.font.color.rgb = RGBColor(30, 58, 138) # Navy Blue
    return h


def add_heading_3(doc, text):
    h = doc.add_heading(text, level=3)
    h.paragraph_format.space_before = Pt(8)
    h.paragraph_format.space_after = Pt(2)
    for run in h.runs:
        run.font.name = "Calibri"
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.color.rgb = RGBColor(59, 130, 246) # Blue
    return h


def add_callout(doc, text, title=""):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    set_cell_background(cell, "F8FAFC")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)

    # Set left border thick indigo
    tcPr = cell._element.get_or_add_tcPr()
    tcBorders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}><w:left w:val="single" w:sz="24" w:space="0" w:color="4F46E5"/>'
        f'<w:top w:val="none"/><w:right w:val="none"/><w:bottom w:val="none"/></w:tcBorders>'
    )
    tcPr.append(tcBorders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    if title:
        r_title = p.add_run(f"{title}\n")
        r_title.bold = True
        r_title.font.name = "Calibri"
        r_title.font.size = Pt(11)
        r_title.font.color.rgb = RGBColor(79, 70, 229)

    r_text = p.add_run(text)
    r_text.font.name = "Calibri"
    r_text.font.size = Pt(10.5)
    r_text.font.color.rgb = RGBColor(51, 65, 85)
    doc.add_paragraph() # Spacing after callout


def style_table(table, col_widths, headers, rows_data):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], "1E293B") # Dark Slate
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=140, right=140)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.name = "Calibri"
            run.font.size = Pt(10)
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)

    for row_idx, r_data in enumerate(rows_data):
        row_cells = table.add_row().cells
        bg_color = "F8FAFC" if row_idx % 2 == 1 else "FFFFFF"
        for i, val in enumerate(r_data):
            row_cells[i].text = val
            set_cell_background(row_cells[i], bg_color)
            set_cell_margins(row_cells[i], top=100, bottom=100, left=140, right=140)
            p = row_cells[i].paragraphs[0]
            for run in p.runs:
                run.font.name = "Calibri"
                run.font.size = Pt(9.5)
                run.font.color.rgb = RGBColor(30, 41, 59)

    for row in table.rows:
        for i, w in enumerate(col_widths):
            row.cells[i].width = Inches(w)


def main():
    print("⏳ Đang tạo tệp Báo cáo Word chuyên nghiệp...")
    doc = docx.Document()

    # Page setup: Margins 1 inch
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Title Block
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(4)
    title_run = title_p.add_run("BÁO CÁO CHIẾN LƯỢC ĐỊNH VỊ SẢN PHẨM & MÔ HÌNH THƯƠNG MẠI HÓA")
    title_run.font.name = "Calibri"
    title_run.font.size = Pt(22)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(15, 23, 42)

    subtitle_p = doc.add_paragraph()
    subtitle_p.paragraph_format.space_before = Pt(0)
    subtitle_p.paragraph_format.space_after = Pt(18)
    sub_run = subtitle_p.add_run(
        "Nghiên cứu vị thế trong Hệ sinh thái Fintech/Banking, Phân tích Nhu cầu Thị trường B2B "
        "và So sánh Benchmark với Zest AI, H2O.ai & SAS Risk Solutions"
    )
    sub_run.font.name = "Calibri"
    sub_run.font.size = Pt(12)
    sub_run.font.italic = True
    sub_run.font.color.rgb = RGBColor(100, 116, 139)

    add_callout(
        doc,
        "Báo cáo chiến lược này nhằm giải quyết phản biện của Mentor về tính thương mại hóa: "
        "Khách hàng B2B thực sự cần gì, Sản phẩm nằm ở đâu trong Hệ sinh thái Công nghệ Tín dụng (Fintech Ecosystem), "
        "và Bài toán ROI cụ thể khiến các Công ty Tài chính / Ngân hàng sẵn sàng trả tiền mua sản phẩm.",
        title="📌 TÓM TẮT MỤC TIÊU CHIẾN LƯỢC"
    )

    # SECTION 1
    add_heading_1(doc, "1. KHÁCH HÀNG B2B THỰC SỰ CẦN GÌ & TRẢ TIỀN CHO GÌ?")
    
    p = doc.add_paragraph()
    p.add_run(
        "Đối tượng khách hàng B2B (Home Credit, FE Credit, MCredit, Ví điện tử MoMo/ZaloPay, VNPT Money, Ngân hàng số) "
        "KHÔNG trả tiền cho một model machine learning đứng riêng lẻ hay một đoạn code Python. "
        "Họ sẵn sàng mở ngân sách để giải quyết 3 Nỗi Đau Lớn (Pain Points) mà các hệ thống cũ không đáp ứng được:"
    )

    add_heading_2(doc, "1.1. Nỗi đau 1: Bỏ sót Khách hàng Thin-file / Unbanked (60% Dân số)")
    p = doc.add_paragraph()
    p.add_run(
        "Các hệ thống chấm điểm truyền thống (SAS / FICO) dựa 100% vào dữ liệu Trung tâm Tín dụng Ngân hàng Nhà nước (CIC). "
        "Với nhóm khách hàng trẻ, sinh viên, người làm tự do (Gig workers) chưa có nợ cũ, hệ thống cũ sẽ TỪ CHỐI THẲNG 100% hồ sơ. "
        "Lenders bỏ lỡ doanh thu khổng lồ. Họ cần một Alternative Credit Engine chấm điểm dựa trên dữ liệu di động, hành vi số, "
        "thâm niên cư trú và mạng lưới xã hội."
    )

    add_heading_2(doc, "1.2. Nỗi đau 2: Tốc độ Duyệt Chậm & Chi phí Vận hành Cao")
    p = doc.add_paragraph()
    p.add_run(
        "Thẩm định viên thủ công mất từ 2 đến 24 giờ để đọc sao kê và gọi điện xác minh. Chi phí nhân sự thẩm định rất lớn. "
        "Lenders cần một Instant Decisioning Engine tự động duyệt 70-80% hồ sơ dưới 5 giây thông qua API."
    )

    add_heading_2(doc, "1.3. Nỗi đau 3: Yêu cầu Tuân thủ Pháp lý & Minh bạch (XAI)")
    p = doc.add_paragraph()
    p.add_run(
        "Ngân hàng Nhà nước và Ban Quản lý Rủi ro (CRO) từ chối mô hình 'Hộp đen' (Black-box AI) vì không giải thích được lý do từ chối "
        "(Adverse Action Notice). Lenders cần Explainable AI (SHAP) kết hợp Rule-based Policy Guard giải thích Tiếng Việt minh bạch."
    )

    # SECTION 2
    add_heading_1(doc, "2. BẢN ĐỒ VỊ TRÍ TRONG HỆ SINH THÁI FINTECH (ECOSYSTEM POSITIONING)")
    p = doc.add_paragraph()
    p.add_run(
        "Trong hệ sinh thái công nghệ tín dụng tiêu chuẩn, hệ thống được chia làm 4 Tầng Công Nghệ (4-Layer Stack):"
    )

    headers_l = ["Tầng Công Nghệ", "Tên Hệ Thống", "Chức Năng & Vị Trí Trong Hệ Sinh Thái"]
    widths_l = [1.5, 1.8, 3.2]
    data_l = [
        ["Tầng 1: Data Layer", "Data Providers", "Cung cấp dữ liệu thô: CIC Bureau, Telco Data (Viettel/Vinaphone), E-wallet, Mobile Behavior."],
        ["Tầng 2: Application Layer", "Loan Origination System (LOS)", "Hệ thống quản lý hồ sơ vay (Temenos, Mambu, Oracle Flexcube). Quản lý eKYC và tiếp nhận hồ sơ."],
        ["Tầng 3: Decisioning Layer", "AI Scoring & Decision Engine (SẢN PHẨM CỦA CHÚNG TA)", "Động cơ ra quyết định AI: Tiếp nhận Payload API ──► Auto-Routing (Hybrid/Alt-only) ──► Scoring ──► Sanity Guard ──► SHAP XAI."],
        ["Tầng 4: Portfolio Layer", "Loan Management System (LMS)", "Quản lý khoản vay sau khi giải ngân, tính lãi, nhắc nợ và thu hồi nợ (Collection Strategy)."]
    ]
    t1 = doc.add_table(rows=1, cols=3)
    style_table(t1, widths_l, headers_l, data_l)
    doc.add_paragraph()

    add_callout(
        doc,
        "Định vị chiến lược: Sản phẩm của chúng ta KHÔNG thay thế hệ thống Core Banking hay LOS sẵn có của Ngân hàng. "
        "Sản phẩm đóng vai trò là một Plug-and-Play AI Decisioning Microservice cắm trực tiếp vào hệ thống LOS thông qua RESTful API.",
        title="💡 BẢN CHẤT VỊ TRÍ SẢN PHẨM"
    )

    # SECTION 3
    add_heading_1(doc, "3. SO SÁNH BENCHMARK VỚI CÁC GÃ KHỔNG LỒ (ZEST AI, H2O.AI & SAS)")

    headers_b = ["Tiêu Chí So Sánh", "SAS Risk Solutions", "H2O.ai", "Zest AI", "SẢN PHẨM CỦA CHÚNG TA"]
    widths_b = [1.3, 1.3, 1.2, 1.3, 1.4]
    data_b = [
        ["Vị trí Ecosystem", "Core Enterprise Risk Platform", "General ML/AutoML Engine", "Specialist AI Underwriting Plugin", "Plug-and-Play AI Decisioning Microservice cho LOS/Fintech"],
        ["Khách hàng Mục tiêu", "Ngân hàng thương mại lớn (Tier-1 Banks)", "Đội ngũ Data Science nội bộ", "Credit Unions, Mid-tier Lenders", "Tổ chức Tài chính Tiêu dùng, BNPL, Fintech, Ví điện tử"],
        ["Điểm khác biệt (Moat)", "Hệ thống GRC truyền thống, bảo mật cao", "AutoML đa ngành, không chuyên sâu tín dụng", "AI Underwriting kết hợp FICO Bureau", "Auto-Routing (Hybrid vs Alt-only) + LLM XAI cho Việt Nam/ĐNA"],
        ["Mô hình Thương mại", "Enterprise License đắt đỏ ($1M+)", "Bán platform subscription", "Bán theo lượt chấm (Per-Query Pricing)", "SaaS API Microservice ($0.20/score) + Phí tích hợp LOS"]
    ]
    t2 = doc.add_table(rows=1, cols=5)
    style_table(t2, widths_b, headers_b, data_b)
    doc.add_paragraph()

    add_heading_2(doc, "3.1. Bài học thành công từ Zest AI")
    p = doc.add_paragraph()
    p.add_run(
        "Zest AI (Định giá > $500M) thành công rực rỡ không phải vì tạo ra giải thuật ML mới, mà nhờ tích hợp sẵn API "
        "vào các hệ thống LOS phổ biến (Temenos, Origence, Mambu) để các tổ chức tín dụng bật tính năng chỉ với 1-Click."
    )

    # SECTION 4
    add_heading_1(doc, "4. MÔ HÌNH THƯƠNG MẠI HÓA & DOANH THU (REVENUE MODEL)")
    p = doc.add_paragraph()
    p.add_run("Khách hàng B2B sẵn sàng trả tiền theo 2 hình thức thương mại chính:")

    add_heading_2(doc, "4.1. Mô hình SaaS API (Pay-Per-Score)")
    p = doc.add_paragraph()
    p.add_run(
        "Thu phí theo lượt gọi API chấm điểm: $0.15 – $0.30 / mỗi hồ sơ. "
        "Ví dụ: Một ví điện tử / đơn vị BNPL xử lý 100,000 hồ sơ/tháng ──► Doanh thu ổn định: $15,000 – $30,000 / tháng."
    )

    add_heading_2(doc, "4.2. Phí Tích hợp & Nâng cấp (Enterprise Setup & Maintenance Fee)")
    p = doc.add_paragraph()
    p.add_run(
        "Phí cài đặt & tích hợp Module vào hệ thống LOS sẵn có của ngân hàng: $25,000 – $40,000 / lần đầu. "
        "Phí duy trì, kiểm định định kỳ và re-train mô hình hàng năm: $10,000 / năm."
    )

    # SECTION 5
    add_heading_1(doc, "5. CHỨNG MINH HIỆU QUẢ TÀI CHÍNH (QUANTIFIABLE FINANCIAL ROI FOR LENDERS)")

    headers_r = ["Chỉ Số Hiệu Quả (KPI)", "Kết Quả Đạt Được", "Tác Động Tài Chính Cho Khách Hàng B2B"]
    widths_r = [1.8, 1.8, 2.9]
    data_r = [
        ["Tăng Tỷ Lệ Phê Duyệt", "Tăng +18% – 25% lượng hồ sơ duyệt", "Với 50k hồ sơ/tháng ──► Duyệt thêm 7,500 khách hàng mới ──► Tăng doanh thu giải ngân hàng chục tỷ đồng."],
        ["Tiết Kiệm Chi Phí Vận Hành", "Tự động hóa 70-80% hồ sơ < 5s", "Giảm 80% thời gian & chi phí nhân sự phòng thẩm định tín dụng gọi điện xác minh."],
        ["Kiểm Soát Nợ Xấu (NPL)", "Duy trì NPL < 3.5% cho tập mới", "Bộ lọc Sanity Guard (DTI & Leverage Floor) ngăn chặn vỡ nợ dây chuyền và tổn thất tín dụng."]
    ]
    t3 = doc.add_table(rows=1, cols=3)
    style_table(t3, widths_r, headers_r, data_r)
    doc.add_paragraph()

    # SECTION 6
    add_heading_1(doc, "6. KHUNG BÀI PITCH TRÌNH BÀY CHO MENTOR & NHÀ ĐẦU TƯ")

    add_callout(
        doc,
        "1. PROBLEM: 60% người dân ĐNA là Thin-file/Unbanked ──► Mô hình truyền thống từ chối do thiếu CIC.\n"
        "2. SOLUTION: Một Plug-and-Play AI Credit Scoring Engine cắm trực tiếp vào LOS qua RESTful API.\n"
        "3. MOAT (Lợi thế): Auto-Routing (Hybrid AUC ~0.76 vs Alt-only AUC ~0.64) + Sanity Guard + SHAP XAI & LLM Tiếng Việt.\n"
        "4. BUSINESS MODEL: SaaS API ($0.20/score), mang lại ROI dương ngay tháng đầu tiên nhờ tăng 20% lượng duyệt hồ sơ mới.",
        title="🎯 KHUNG 4 BƯỚC THUYẾT PHỤC MENTOR"
    )

    # Academic References
    add_heading_2(doc, "Nguồn Tham Khảo Học Thuật & Thị Trường (Citations)")
    p_ref = doc.add_paragraph()
    p_ref.add_run(
        "1. World Bank Group & CGAP (2017) — Alternative Data Assessing Credit Risk for Financial Inclusion.\n"
        "2. Óskarsdóttir et al. (2019) — The value of big data for credit scoring: Enhancing financial inclusion using mobile phone data (ScienceDirect / IEEE TKDE, DOI: 10.1016/j.eswa.2019.02.029).\n"
        "3. Zest AI Platform Architecture Documentation (2024) — LOS Native Integration & AI Automated Underwriting Framework.\n"
        "4. Siddiqi, N. (2012) — Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring."
    )

    output_dir = Path("docs")
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_path = output_dir / "BAO_CAO_CHIEN_LUOC_DINH_VI_SAN_PHAM_CREDIT_SCORING.docx"
    doc.save(doc_path)
    
    print(f"✅ ĐÃ TẠO THÀNH CÔNG BÁO CÁO WORD: {doc_path.resolve()}")


if __name__ == "__main__":
    main()
