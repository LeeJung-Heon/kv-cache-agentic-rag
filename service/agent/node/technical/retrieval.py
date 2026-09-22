"""논문 검색 도구와 라운드별 근거 수집. 근거 메타데이터는 검색 결과에서 코드가 만든다."""
from typing import Literal

from langchain_core.tools import tool

from service.agent.node.technical.core import default_queries, normalize_queries
from service.agent.node.technical.schema import TechnicalResearchState
from service.schema.state import Evidence


def make_paper_search_tool(index):
    """공용 벡터 검색 결과를 기술 에이전트의 Evidence 형식으로 변환한다."""

    @tool("paper_search")
    def paper_search(query: str, side: Literal["sw", "hw"]) -> list[Evidence]:
        """Search indexed KV-cache papers for one technology side and return the top passages as Evidence records.
        Use a specific query naming the mechanism, metric, or condition."""
        # 실제 임베딩과 FAISS 검색은 주입받은 PaperIndex가 담당한다.
        return [{"id": row["id"], "source_type": "paper", "title": row["title"], "url": row["url"],
                 # 문서 manifest의 공개일을 Evidence까지 전달해 최종 보고서 출처에 보존한다.
                 "page": row["page"], "published_at": row.get("published_at"),
                 "excerpt": row["text"]} for row in index.search(query, side)]

    return paper_search


def retrieve_evidence(paper_search, state: TechnicalResearchState, evidence: dict[str, Evidence]) -> dict[str, Evidence]:
    """첫 라운드는 기술마다 기본 질의로, 재검색은 미충족 항목이 있는 기술의 논문만 수정 질의로 검색해 누적한다."""
    criteria = state["evaluation_criteria"]["technical"]
    queries = normalize_queries(state.get("technical_queries") or [])
    all_ids = {t["id"] for t in state["technologies"]}
    missing_ids = {item.split(":")[0] for item in state.get("technical_missing_items", [])} & all_ids
    defaults = default_queries(state["technologies"], state["target_domain"], criteria)
    for technology in state["technologies"]:
        # 재검색에서는 부족하다고 판정된 기술만 대상으로 삼아 불필요한 검색을 줄인다.
        if queries and missing_ids and technology["id"] not in missing_ids:
            continue
        for query in queries or [defaults[technology["id"]]]:
            for row in paper_search.invoke({"query": query, "side": technology["approach"].lower()}):
                # 같은 청크가 여러 질의에서 검색되어도 Evidence ID 기준으로 한 번만 보관한다.
                evidence[row["id"]] = row
    return evidence
