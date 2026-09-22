import json
from typing import Literal

from pydantic import BaseModel

from config.model import get_chat_model
from service.agent.synthesis.prompt import SYNTHESIS_SYSTEM_PROMPT
from service.schema.state import AgentResult, GraphState


class SynthesisFinding(BaseModel):
    technology_ids: list[str]
    claim: str
    evidence_ids: list[str]
    is_inference: bool


class SynthesisOutput(BaseModel):
    status: Literal["complete", "partial", "error"]
    summary: str
    findings: list[SynthesisFinding]
    limitations: list[str]
    quality_feedback: list[str]

##평가 agent
def synthesis_agent(state: GraphState) -> dict:
    model = get_chat_model().with_structured_output(SynthesisOutput)
    ## 평가 항목
    evaluations = {
        "technical_result": state.get("technical_result"),
        "market_result": state.get("market_result"),
        "stakeholder_result": state.get("stakeholder_result"),
        "domain_result": state.get("domain_result"),
    }

    context = json.dumps(evaluations, ensure_ascii=False)
    result = model.invoke(
        [("system", f"{SYNTHESIS_SYSTEM_PROMPT}\n\n평가 결과:\n{context}")]
    )

    synthesis_result: AgentResult = {
        "status": result.status,
        "summary": result.summary,
        "findings": [finding.model_dump() for finding in result.findings],
        "evidence": [
            evidence
            for evaluation in evaluations.values()
            if evaluation
            for evidence in evaluation["evidence"]
        ],
        "limitations": result.limitations,
    }

    quality_feedback = result.quality_feedback
    if result.status == "complete":
        quality_feedback = []
    elif not quality_feedback:
        quality_feedback = [f"종합 결과 상태가 {result.status}입니다. 재작업이 필요합니다."]

    return {
        "synthesis_result": synthesis_result,
        "quality_feedback": quality_feedback,
        "revision_count": state.get("revision_count", 0) + 1,
    }
