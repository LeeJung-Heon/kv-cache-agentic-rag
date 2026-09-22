from pathlib import Path

from report import write_report

RESULT_DIR = Path(__file__).resolve().parents[3] / "result"


def save_report_pdf(
    report_markdown: str,
    report_evidence_ids: list[str],
) -> Path:
    missing_ids = [
        evidence_id
        for evidence_id in report_evidence_ids
        if f"[{evidence_id}]" not in report_markdown
    ]
    if missing_ids:
        raise ValueError("보고서에 누락된 근거 ID: " + ", ".join(missing_ids))

    write_report(report_markdown, RESULT_DIR)
    return RESULT_DIR / "report.pdf"
