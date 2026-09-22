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
    model = get_chat_model()
    results = {field: state.get(field) for field in RESULT_FIELDS}
    evidence_ids = {
        evidence["id"]
        for result in results.values()
        if result
        for evidence in result["evidence"]
    }
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
    report_evidence_ids = sorted(
        evidence_id
        for evidence_id in evidence_ids
        if re.search(rf"\[{re.escape(evidence_id)}\]", report_markdown)
    )
    save_report_pdf(report_markdown, report_evidence_ids)

    return {
        "report_markdown": report_markdown,
        "report_evidence_ids": report_evidence_ids,
    }
