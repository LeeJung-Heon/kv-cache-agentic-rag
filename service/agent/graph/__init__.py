from .technical import build_technical_research_graph


def build_agent_graph(*args, **kwargs):
    # report 모듈처럼 선택적 실행 의존성이 technical 서브그래프 import까지
    # 막지 않도록 최상위 그래프는 실제 생성 시점에 불러온다.
    from .agent import build_agent_graph as _build_agent_graph

    return _build_agent_graph(*args, **kwargs)


__all__ = ["build_agent_graph", "build_technical_research_graph"]
