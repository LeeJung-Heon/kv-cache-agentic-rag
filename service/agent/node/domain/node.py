"""논문 검색과 LLM 평가를 연결하는 도메인 평가 노드."""

from service.schema.state import GraphState

from .model import evaluate_domain
from .retrieval import retrieve_domain


def domain_node(state: GraphState):
    """도메인 근거를 수집해 평가하고 domain_result만 갱신한다."""
    try:
        sources, search_results = retrieve_domain(state)
        result = evaluate_domain(state, sources, search_results)
        return {"domain_result": result}
    except Exception as exc:
        # 한 노드의 예외가 그래프 전체에서 유실되지 않도록 표준 AgentResult로 변환한다.
        return {"domain_result": {
            "status": "error", "summary": "domain_result 실패", "findings": [],
            "evidence": [], "limitations": [f"domain_result: {type(exc).__name__}"],
        }}
