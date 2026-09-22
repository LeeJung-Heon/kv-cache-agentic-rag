"""KV cache 평가 에이전트의 최종 LangGraph 정의."""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from service.agent.graph.technical import build_technical_research_graph
from service.agent.node.domain import domain_node
from service.agent.node.market import market_node
from service.agent.node.stakeholder import stakeholder_node
from service.agent.report import report_agent
from service.agent.synthesis import synthesis_agent
from service.schema.state import GraphState


def route_after_synthesis(
    state: GraphState,
) -> Literal["technical_agent", "report_agent", "__end__"]:
    """품질 피드백이 있으면 전체 평가를 재실행하되 두 번째 종합에서 종료한다."""
    if not state.get("quality_feedback"):
        return "report_agent"
    # 종합 평가가 두 번 끝난 뒤에도 피드백이 남으면 무한 반복을 막고 종료한다.
    if state.get("revision_count", 0) >= 2:
        return END
    return "technical_agent"


def build_agent_graph(index, technical_model=None, max_technical_retries: int = 2, *, rules: str = ""):
    """전체 평가 그래프를 만들고 컴파일한다.

    기술 조사가 끝나면 시장성·이해관계자·도메인 평가를 병렬 실행한다.
    세 평가가 모두 끝난 뒤 결과를 종합하고 최종 보고서를 생성한다.
    """
    technical_graph = build_technical_research_graph(
        index,
        model=technical_model,
        max_retries=max_technical_retries,
        rules=rules,
    )

    workflow = StateGraph(GraphState)
    workflow.add_node("technical_agent", technical_graph)
    workflow.add_node("market_node", market_node)
    workflow.add_node("stakeholder_node", stakeholder_node)
    workflow.add_node("domain_agent", domain_node)
    workflow.add_node("synthesis_agent", synthesis_agent)
    workflow.add_node("report_agent", report_agent)

    # 모든 후속 평가는 기술 조사에서 확정한 기술 정보와 논문 근거를 입력으로 사용한다.
    workflow.add_edge(START, "technical_agent")

    # Fan-out: 기술 조사 결과를 세 관점이 동시에 사용한다.
    workflow.add_edge("technical_agent", "market_node")
    workflow.add_edge("technical_agent", "stakeholder_node")
    workflow.add_edge("technical_agent", "domain_agent")

    # Fan-in: 세 관점 평가가 모두 끝나야 종합 평가를 시작한다.
    workflow.add_edge(
        ["market_node", "stakeholder_node", "domain_agent"],
        "synthesis_agent",
    )
    workflow.add_conditional_edges(
        "synthesis_agent",
        route_after_synthesis,
        ["technical_agent", "report_agent", END],
    )
    workflow.add_edge("report_agent", END)

    return workflow.compile()
