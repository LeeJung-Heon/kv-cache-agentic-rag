"""도메인 평가 기준별로 공용 FAISS 인덱스를 검색한다."""

from service.retrieval.paper_index import get_paper_index
from service.schema.state import GraphState


def retrieve_domain(state: GraphState):
    """기술 조사 근거에 도메인별 검색 결과를 추가한다."""
    index = get_paper_index()
    technologies = state["technologies"]
    criteria = state["evaluation_criteria"]["domain"]
    previous = state.get("technical_result")
    # 기술 조사에서 이미 확보한 근거도 도메인 평가가 재사용할 수 있게 합친다.
    sources = {e["id"]: e for e in (previous or {}).get("evidence", [])}
    search_results = []
    for technology in technologies:
        side = technology["approach"].lower()
        for criterion in criteria:
            # 기술과 평가 기준을 분리해 검색해 어떤 기준의 근거인지 추적 가능하게 남긴다.
            query = (
                f"{technology['name']} | {criterion} | {state['target_domain']} | "
                f"{technology['selection_reason'][:300]} | "
                "measured results baseline experimental conditions limitations tradeoffs"
            )
            evidence_ids = []
            for row in index.search(query, side):
                sources[row["id"]] = {
                    "id": row["id"], "source_type": "paper", "title": row["title"],
                    "url": row["url"], "page": row["page"],
                    # 인덱스 생성 시 보존한 공개일을 버리지 않고 Evidence까지 전달한다.
                    "published_at": row.get("published_at"),
                    "excerpt": row["text"],
                }
                evidence_ids.append(row["id"])
            # LLM에는 근거 본문과 함께 질의별 Evidence ID 연결 정보를 전달한다.
            search_results.append({"technology_id": technology["id"], "criterion": criterion,
                                   "query": query, "evidence_ids": evidence_ids})
    return sources, search_results
