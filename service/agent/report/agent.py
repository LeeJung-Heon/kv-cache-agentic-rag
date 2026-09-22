"""검증된 에이전트 결과로 Markdown과 PDF 보고서를 생성한다."""

import json
import re

from config.model import get_chat_model
from service.agent.report.pdf import save_report_pdf
from service.agent.report.prompt import REPORT_SYSTEM_PROMPT
from service.schema.state import GraphState

RESULT_FIELDS = (
    "technical_result",
    "market_result",
    "stakeholder_result",
    "domain_result",
    "synthesis_result",
)

def report_agent(state: GraphState) -> dict:
    """최종 보고서를 작성하고 실제 사용된 근거 ID만 반환한다."""
    model = get_chat_model()
    results = {field: state.get(field) for field in RESULT_FIELDS}
    evidence_ids = {
        evidence["id"]
        for result in results.values()
        if result
        for evidence in result["evidence"]
    }
    # 허용 근거 ID를 명시해 모델이 존재하지 않는 출처를 만들지 않도록 제한한다.
    context = {
        "request": state["request"],
        "target_domain": state["target_domain"],
        "technologies": state["technologies"],
        "results": results,
        "allowed_evidence_ids": sorted(evidence_ids),
    }

    response = model.invoke(
        [
            (
                "system",
                f"{REPORT_SYSTEM_PROMPT}\n\n보고서 작성 자료:\n"
                f"{json.dumps(context, ensure_ascii=False)}",
            )
        ]
    )
    report_markdown = str(response.content).strip()
    # 전체 후보가 아니라 보고서 본문에서 실제 인용한 ID만 최종 State에 저장한다.
    report_evidence_ids = sorted(
        evidence_id
        for evidence_id in evidence_ids
        if re.search(rf"\[{re.escape(evidence_id)}\]", report_markdown)
    )
    # Markdown 저장과 PDF 변환은 출력 모듈에 위임한다.
    save_report_pdf(report_markdown, report_evidence_ids)

    return {
        "report_markdown": report_markdown,
        "report_evidence_ids": report_evidence_ids,
    }
