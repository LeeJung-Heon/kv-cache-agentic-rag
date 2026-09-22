"""기술 조사 서브그래프 노드. 검색은 코드가 결정하고 분석만 LLM이 수행한다."""
from typing import Literal

from langgraph.graph import END

from service.agent.node.technical.core import (EvidenceError, error_result, normalize_queries, normalize_technical_result,
                                               retry_queries)
from service.agent.node.technical.model import analyze_technical
from service.agent.node.technical.prompts import build_system_prompt
from service.agent.node.technical.retrieval import make_paper_search_tool, retrieve_evidence
from service.agent.node.technical.schema import TechnicalDraft, TechnicalResearchState


def make_technical_nodes(index, model, max_retries: int, *, rules: str = "") -> dict:
    paper_search = make_paper_search_tool(index)
    analyst = model.with_structured_output(TechnicalDraft, method="json_schema", strict=True)
    system = build_system_prompt(rules)

    def retrieve(state: TechnicalResearchState):
        evidence = dict(state.get("technical_evidence", {}))
        retry_count = state.get("technical_retry_count", 0)
        try:
            evidence = retrieve_evidence(paper_search, state, evidence)
            return {"technical_evidence": evidence, "technical_retrieval_failed": False, "technical_retry_count": retry_count}
        except Exception as exc:
            result = error_result(exc)
            previous = state.get("technical_result")
            if previous:
                result["summary"] = previous["summary"]
                result["findings"] = list(previous["findings"])
                result["limitations"] = [*previous["limitations"], *result["limitations"]]
            # 실패 전까지 확보한 근거와 직전 검증 결과를 부모가 기록할 수 있게 보존한다.
            result["evidence"] = [evidence[key] for key in sorted(evidence)]
            return {"technical_evidence": evidence, "technical_retrieval_failed": True, "technical_retry_count": retry_count,
                    "technical_result": result, "technical_queries": [],
                    "technical_missing_items": state.get("technical_missing_items", [])}

    def route_after_retrieval(state: TechnicalResearchState) -> Literal["analyze", "__end__"]:
        return END if state["technical_retrieval_failed"] else "analyze"

    def analyze(state: TechnicalResearchState):
        criteria = state["evaluation_criteria"]["technical"]
        try:
            sources = state.get("technical_evidence", {})
            draft = analyze_technical(analyst, state, sources, system=system)
            try:
                result, missing = normalize_technical_result(draft, sources, state["technologies"], criteria)
            except EvidenceError as exc:
                # 설계서 2.2.4: 인용 검증 실패는 같은 근거로 최대 1회만 수정한다. 두 번째도 실패하면 error로 끝낸다.
                unknown = sorted({key for f in draft.findings for key in f.evidence_ids if key not in sources})
                feedback = [f"이전 응답이 검증에 실패했다: {exc}", f"사용 가능한 evidence ID: {sorted(sources)}"]
                if unknown:
                    feedback.append(f"존재하지 않는 evidence ID를 참조했다: {unknown}")
                draft = analyze_technical(analyst, state, sources, system=system, validation_feedback=feedback)
                result, missing = normalize_technical_result(draft, sources, state["technologies"], criteria)
            partial = result["status"] == "partial"
            if partial and state.get("technical_retry_count", 0) >= max_retries:
                result["limitations"].append(f"기술 재검색 한도({max_retries}회)에 도달하여 남은 항목을 확인하지 못함")
            return {"technical_result": result,
                    "technical_missing_items": missing or (result["limitations"] if partial else []),
                    "technical_queries": normalize_queries(draft.next_queries) if partial else []}
        except Exception as exc:
            return {"technical_result": error_result(exc), "technical_missing_items": [], "technical_queries": []}

    def rewrite_queries(state: TechnicalResearchState):
        queries = normalize_queries(state.get("technical_queries") or []) or retry_queries(state.get("technical_missing_items", []))
        return {"technical_retry_count": state.get("technical_retry_count", 0) + 1, "technical_queries": queries}

    def route_after_analysis(state: TechnicalResearchState) -> Literal["rewrite_queries", "__end__"]:
        if state["technical_result"]["status"] == "partial" and state.get("technical_retry_count", 0) < max_retries:
            return "rewrite_queries"
        return END

    return {"retrieve": retrieve, "analyze": analyze, "rewrite_queries": rewrite_queries,
            "route_after_retrieval": route_after_retrieval, "route_after_analysis": route_after_analysis}
