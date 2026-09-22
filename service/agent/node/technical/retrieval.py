"""논문 검색 도구와 라운드별 근거 수집. 근거 메타데이터는 검색 결과에서 코드가 만든다."""
from typing import Literal

from langchain_core.tools import tool

from service.agent.node.technical.core import default_queries, normalize_queries
from service.agent.node.technical.schema import TechnicalResearchState
from service.schema.state import Evidence


def make_paper_search_tool(index):
    """`index.search(query, side)`가 id·title·url·page·text를 가진 행을 돌려주면 어떤 인덱스든 쓸 수 있다."""

    @tool("paper_search")
    def paper_search(query: str, side: Literal["sw", "hw"]) -> list[Evidence]:
        """Search one of the two KV-cache papers (sw: DeepSeek-V2 MLA, hw: CXL-PNM) and return the top passages
        as Evidence records with page numbers. Use a specific query naming the mechanism, metric, or condition."""
        return [{"id": row["id"], "source_type": "paper", "title": row["title"], "url": row["url"],
                 "page": row["page"], "published_at": None, "excerpt": row["text"]} for row in index.search(query, side)]

    return paper_search


def retrieve_evidence(paper_search, state: TechnicalResearchState, evidence: dict[str, Evidence]) -> dict[str, Evidence]:
    """첫 라운드는 기술마다 기본 질의로, 재검색은 미충족 항목이 있는 기술의 논문만 수정 질의로 검색해 누적한다."""
    criteria = state["evaluation_criteria"]["technical"]
    queries = normalize_queries(state.get("technical_queries") or [])
    all_ids = {t["id"] for t in state["technologies"]}
    missing_ids = {item.split(":")[0] for item in state.get("technical_missing_items", [])} & all_ids
    defaults = default_queries(state["technologies"], state["target_domain"], criteria)
    for technology in state["technologies"]:
        if queries and missing_ids and technology["id"] not in missing_ids:
            continue
        for query in queries or [defaults[technology["id"]]]:
            for row in paper_search.invoke({"query": query, "side": technology["approach"].lower()}):
                evidence[row["id"]] = row
    return evidence
