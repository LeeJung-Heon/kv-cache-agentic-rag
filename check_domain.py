import json
from copy import deepcopy
from unittest.mock import Mock, patch

from pipeline import initial_state, select_technologies
from service.agent.node.domain import domain_node
from state import AnalysisDraft


EVIDENCE_IDS = {"sw": "chunk-alpha", "hw": "chunk-beta"}


def check_case(state, mode="normal"):
    original = deepcopy(state)
    index = Mock()
    analyst = Mock()

    def search(query, side):
        return [{"id": EVIDENCE_IDS[side], "title": f"{side} paper", "url": f"https://example.org/{side}",
                 "page": 1, "text": f"{side} original excerpt"}]

    def analyze(messages):
        context = json.loads(messages[-1][1])
        technologies = state["technologies"]
        criteria = state["evaluation_criteria"]["domain"]
        assert len(context["search_results"]) == len(technologies) * len(criteria)
        assert len(context["evidence"]) == len(technologies), "반복 검색 청크가 중복됨"
        assert {e["id"] for e in context["evidence"]} == set(EVIDENCE_IDS.values())
        findings = [
            {"technology_ids": [technology["id"]], "criterion": criterion, "claim": "근거 기반 평가",
             "evidence_ids": [EVIDENCE_IDS[technology["approach"].lower()]], "is_inference": True}
            for technology in technologies for criterion in criteria
        ]
        if mode == "one_sided":
            findings = [{"technology_ids": [t["id"] for t in technologies], "criterion": criteria[0],
                         "claim": "양쪽 기술에 대한 주장", "evidence_ids": [EVIDENCE_IDS["sw"]], "is_inference": True}]
        elif mode == "null_summary":
            return AnalysisDraft(status="complete", summary=None, findings=findings, limitations=[], next_queries=[])
        return AnalysisDraft(status="complete", summary="검사 요약", findings=findings, limitations=[], next_queries=[])

    index.search.side_effect = search
    analyst.invoke.side_effect = RuntimeError("SECRET_SENTINEL") if mode == "failure" else analyze
    with patch("service.agent.node.domain.retrieval.get_paper_index", return_value=index), \
         patch("service.agent.node.domain.model.get_analyst", return_value=analyst):
        update = domain_node(state)
    assert state == original, "노드가 입력 State를 직접 변경함"
    assert set(update) == {"domain_result"}, "다른 노드의 결과 필드를 변경함"
    result = update["domain_result"]

    expected = [(technology, criterion) for technology in state["technologies"]
                for criterion in state["evaluation_criteria"]["domain"]]
    assert index.search.call_count == len(expected)
    for call, (technology, criterion) in zip(index.search.call_args_list, expected):
        query, side = call.args
        assert side == technology["approach"].lower()
        assert all(value in query for value in (technology["name"], criterion, state["target_domain"],
                                               technology["selection_reason"][:300]))
    if mode == "normal":
        assert result["status"] == "complete", result
        assert len(result["findings"]) == len(expected)
        assert len(result["evidence"]) == len(state["technologies"])
    elif mode == "one_sided":
        assert result["status"] == "complete" and len(result["findings"]) == 1
    else:
        assert result["status"] == "error", result
        assert "SECRET_SENTINEL" not in json.dumps(result)


def main():
    state = initial_state()
    state.update(select_technologies(state))
    check_case(state)
    state["technologies"] = [
        dict(id="compression-A", name="대체 압축 기술", approach="SW", selection_reason="정밀도를 변경한다"),
        dict(id="memory-B", name="대체 메모리 기술", approach="HW", selection_reason="메모리 계층을 확장한다"),
    ]
    state["target_domain"] = "다른 적용 환경"
    state["evaluation_criteria"]["domain"] = ["운영 복잡도", "이식성"]
    for mode in ("normal", "one_sided", "null_summary", "failure"):
        check_case(state, mode)
    print("PASS: 기술·기준 변경, 검색 분리, 근거 중복 제거, Pydantic null 거부·error, State 보존 (외부 API 없음)")


if __name__ == "__main__":
    main()
