from service.retrieval.paper_index import get_paper_index
from state import GraphState


def retrieve_domain(state: GraphState):
    index = get_paper_index()
    technologies = state["technologies"]
    criteria = state["evaluation_criteria"]["domain"]
    previous = state.get("technical_result")
    sources = {e["id"]: e for e in (previous or {}).get("evidence", [])}
    search_results = []
    for technology in technologies:
        side = technology["approach"].lower()
        for criterion in criteria:
            query = (
                f"{technology['name']} | {criterion} | {state['target_domain']} | "
                f"{technology['selection_reason'][:300]} | "
                "measured results baseline experimental conditions limitations tradeoffs"
            )
            evidence_ids = []
            for row in index.search(query, side):
                sources[row["id"]] = {
                    "id": row["id"], "source_type": "paper", "title": row["title"],
                    "url": row["url"], "page": row["page"], "published_at": None,
                    "excerpt": row["text"],
                }
                evidence_ids.append(row["id"])
            search_results.append({"technology_id": technology["id"], "criterion": criterion,
                                   "query": query, "evidence_ids": evidence_ids})
    return sources, search_results
