"""관점별 평가를 종합하고 전체 재작업 여부를 판단하는 에이전트."""

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

def synthesis_agent(state: GraphState) -> dict:
    """네 관점의 결과를 종합하고 품질 피드백을 State에 기록한다."""
    model = get_chat_model().with_structured_output(SynthesisOutput)
    # 병렬 평가 결과만 모아 전달하며 새로운 외부 근거는 검색하지 않는다.
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

    # 앞선 에이전트의 근거를 보존해 종합 주장과 최종 보고서가 같은 출처를 참조하게 한다.
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
    # complete이면 재작업 신호를 제거하고, 그 외 상태에는 반드시 분기 근거를 남긴다.
    if result.status == "complete":
        quality_feedback = []
    elif not quality_feedback:
        quality_feedback = [f"종합 결과 상태가 {result.status}입니다. 재작업이 필요합니다."]

    return {
        "synthesis_result": synthesis_result,
        "quality_feedback": quality_feedback,
        # 종합 평가 완료 횟수로 전체 그래프의 재작업 한도를 제어한다.
        "revision_count": state.get("revision_count", 0) + 1,
    }
