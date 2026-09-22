from service.schema.state import GraphState

from .model import evaluate_domain
from .retrieval import retrieve_domain


def domain_node(state: GraphState):
    try:
        sources, search_results = retrieve_domain(state)
        result = evaluate_domain(state, sources, search_results)
        return {"domain_result": result}
    except Exception as exc:
        return {"domain_result": {
            "status": "error", "summary": "domain_result 실패", "findings": [],
            "evidence": [], "limitations": [f"domain_result: {type(exc).__name__}"],
        }}
