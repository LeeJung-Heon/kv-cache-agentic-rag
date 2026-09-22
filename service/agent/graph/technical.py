"""기술 조사 서브그래프. 부모 그래프에는 컴파일된 그래프를 노드 하나로 연결한다."""
from langgraph.graph import END, START, StateGraph

from service.agent.node.technical import make_technical_nodes
from service.agent.node.technical import model as technical_model
from service.agent.node.technical.schema import TechnicalResearchOutput, TechnicalResearchState


def build_technical_research_graph(index, model=None, max_retries: int = 2, *, rules: str = ""):
    """검색, 분석, 질의 재작성으로 구성된 기술 조사 서브그래프를 만든다."""
    if not 0 <= max_retries <= 5:
        raise ValueError("최대 기술 재검색 횟수는 0~5여야 합니다.")
    if model is None:
        model = technical_model.create_technical_model()
    nodes = make_technical_nodes(index, model, max_retries, rules=rules)
    graph = StateGraph(TechnicalResearchState, output_schema=TechnicalResearchOutput)
    graph.add_node("retrieve", nodes["retrieve"])
    graph.add_node("analyze", nodes["analyze"])
    graph.add_node("rewrite_queries", nodes["rewrite_queries"])
    graph.add_edge(START, "retrieve")
    # 검색 실패는 종료하고, 정상 검색 결과만 근거 검증과 분석으로 넘긴다.
    graph.add_conditional_edges("retrieve", nodes["route_after_retrieval"], ["analyze", END])
    # 근거가 부족하고 재시도 한도가 남은 경우에만 질의를 수정해 다시 검색한다.
    graph.add_conditional_edges("analyze", nodes["route_after_analysis"], ["rewrite_queries", END])
    graph.add_edge("rewrite_queries", "retrieve")
    return graph.compile()
