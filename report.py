import os
from pathlib import Path
from xml.sax.saxutils import escape

from markdown_it import MarkdownIt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def write_report(markdown: str, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.md").write_text(markdown, encoding="utf-8")
    candidates = [os.getenv("PDF_FONT", ""), "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                  "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]
    font = next((p for p in candidates if p and Path(p).is_file()), None)
    if not font:
        raise FileNotFoundError("PDF_FONT에 한국어 TTF 파일을 지정하세요. Markdown은 저장되었습니다.")
    pdfmetrics.registerFont(TTFont("ReportKorean", font))
    normal = ParagraphStyle("Body", fontName="ReportKorean", fontSize=9.5, leading=15,
                            wordWrap="CJK", spaceAfter=7, alignment=TA_LEFT)
    headings = {level: ParagraphStyle(f"H{level}", parent=normal, fontSize=17 if level == 1 else 12,
                                     leading=23 if level == 1 else 18, spaceBefore=14, spaceAfter=8,
                                     textColor=colors.HexColor("#193A59"), keepWithNext=True)
                for level in range(1, 7)}
    story = []
    style = normal
    body_style = normal
    reference = ParagraphStyle("Reference", parent=normal, fontSize=8, leading=11, spaceAfter=5)
    bullet = False
    for token in MarkdownIt().parse(markdown):
        if token.type == "heading_open":
            style = headings[int(token.tag[1])]
        elif token.type == "list_item_open":
            bullet = True
        elif token.type == "inline":
            # 텍스트만 렌더링하여 모델이 생성한 HTML과 외부 이미지를 실행하지 않는다.
            text = "".join(child.content if child.type in {"text", "code_inline"} else
                           "\n" if child.type in {"softbreak", "hardbreak"} else ""
                           for child in token.children or [])
            story.append(Paragraph(escape(("• " if bullet else "") + text).replace("\n", "<br/>"), style))
            if text.strip() == "REFERENCE":
                body_style = reference
            style = body_style
            bullet = False
        elif token.type == "fence":
            story.append(Paragraph(escape(token.content).replace("\n", "<br/>"), normal))
        elif token.type == "hr":
            story.append(Spacer(1, 10))

    def footer(canvas, doc):
        canvas.setFont("ReportKorean", 8)
        canvas.setFillColor(colors.HexColor("#667085"))
        canvas.drawString(42, 25, "KV cache | 공개정보 기반 다관점 평가")
        canvas.drawRightString(A4[0] - 42, 25, str(doc.page))

    temporary = output / "report.tmp.pdf"
    SimpleDocTemplate(str(temporary), pagesize=A4, rightMargin=42, leftMargin=42,
                      topMargin=38, bottomMargin=42, title="KV cache 다관점 평가").build(
                          story, onFirstPage=footer, onLaterPages=footer)
    temporary.replace(output / "report.pdf")
