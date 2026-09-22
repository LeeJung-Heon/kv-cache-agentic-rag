"""Markdown 보고서를 파일과 한국어 PDF로 저장한다."""

import os
from pathlib import Path
from xml.sax.saxutils import escape

from markdown_it import MarkdownIt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

RESULT_DIR = Path(__file__).resolve().parents[3] / "result"


def write_report(markdown: str, output: Path) -> None:
    """Markdown 원본과 한국어 PDF를 저장한다."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.md").write_text(markdown, encoding="utf-8")
    candidates = [
        os.getenv("PDF_FONT", ""),
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    # 실행 환경마다 폰트 위치가 달라 환경변수를 우선하고 알려진 기본 경로를 차례로 확인한다.
    font = next((path for path in candidates if path and Path(path).is_file()), None)
    if not font:
        raise FileNotFoundError("PDF_FONT에 한국어 TTF 파일을 지정하세요. Markdown은 저장되었습니다.")

    pdfmetrics.registerFont(TTFont("ReportKorean", font))
    normal = ParagraphStyle(
        "Body", fontName="ReportKorean", fontSize=9.5, leading=15,
        wordWrap="CJK", spaceAfter=7, alignment=TA_LEFT,
    )
    headings = {
        level: ParagraphStyle(
            f"H{level}", parent=normal, fontSize=17 if level == 1 else 12,
            leading=23 if level == 1 else 18, spaceBefore=14, spaceAfter=8,
            textColor=colors.HexColor("#193A59"), keepWithNext=True,
        )
        for level in range(1, 7)
    }
    reference = ParagraphStyle("Reference", parent=normal, fontSize=8, leading=11, spaceAfter=5)
    story, style, body_style, bullet = [], normal, normal, False
    # Markdown 토큰을 ReportLab 문단으로 바꿔 제목, 목록, 본문 스타일을 유지한다.
    for token in MarkdownIt().parse(markdown):
        if token.type == "heading_open":
            style = headings[int(token.tag[1])]
        elif token.type == "list_item_open":
            bullet = True
        elif token.type == "inline":
            text = "".join(
                child.content if child.type in {"text", "code_inline"}
                else "\n" if child.type in {"softbreak", "hardbreak"} else ""
                for child in token.children or []
            )
            story.append(Paragraph(escape(("• " if bullet else "") + text).replace("\n", "<br/>"), style))
            if text.strip() == "REFERENCE":
                body_style = reference
            style, bullet = body_style, False
        elif token.type == "fence":
            story.append(Paragraph(escape(token.content).replace("\n", "<br/>"), normal))
        elif token.type == "hr":
            story.append(Spacer(1, 10))

    def footer(canvas, document):
        canvas.setFont("ReportKorean", 8)
        canvas.setFillColor(colors.HexColor("#667085"))
        canvas.drawString(42, 25, "KV cache | 공개정보 기반 다관점 평가")
        canvas.drawRightString(A4[0] - 42, 25, str(document.page))

    # 임시 파일을 완성한 뒤 교체해 실패 시 기존 PDF가 손상되지 않게 한다.
    temporary = output / "report.tmp.pdf"
    SimpleDocTemplate(
        str(temporary), pagesize=A4, rightMargin=42, leftMargin=42,
        topMargin=38, bottomMargin=42, title="KV cache 다관점 평가",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    temporary.replace(output / "report.pdf")


def save_report_pdf(
    report_markdown: str,
    report_evidence_ids: list[str],
) -> Path:
    """인용 ID가 본문에 있는지 확인한 뒤 보고서 파일을 저장한다."""
    missing_ids = [
        evidence_id
        for evidence_id in report_evidence_ids
        if f"[{evidence_id}]" not in report_markdown
    ]
    if missing_ids:
        raise ValueError("보고서에 누락된 근거 ID: " + ", ".join(missing_ids))

    write_report(report_markdown, RESULT_DIR)
    return RESULT_DIR / "report.pdf"
