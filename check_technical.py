import json
from copy import deepcopy
from unittest.mock import Mock

from pipeline import initial_state, select_technologies
from service.agent.graph.technical import build_technical_research_graph
from service.agent.node.technical.prompts import ASSESSMENT_DATE
from service.agent.node.technical.schema import TechnicalDraft, TechnicalFindingDraft, TrlDraft

EVIDENCE_IDS = {"sw": "sw_p1_c1", "hw": "hw_p1_c1"}


def check_case(state, mode="normal", retries=2):
    original = deepcopy(state)
    index = Mock()
    model = Mock()
    calls = {"analyze": 0}

    def search(query, side):
        if mode == "search_failure":
            raise RuntimeError("SECRET_SENTINEL")
        return [{"id": EVIDENCE_IDS[side], "title": f"{side} paper", "url": f"https://example.org/{side}",
                 "page": 1, "text": f"{side} original excerpt"}]

    def analyze(messages):
        calls["analyze"] += 1
        if mode == "model_failure":
            raise RuntimeError("SECRET_SENTINEL")
        context = json.loads(messages[-1][1])
        assert context["assessment_date"] == ASSESSMENT_DATE and "quality_feedback" in context
        partial = mode == "always_partial" or (mode == "retry" and calls["analyze"] == 1)
        findings = []
        for technology in context["technologies"]:
            for criterion in context["evaluation_criteria"]:
                if partial and technology["approach"] == "HW" and criterion == "TRL":
                    continue
                trl = TrlDraft(level_or_range="TRL 3~4", as_of=ASSESSMENT_DATE, confidence="중간",
                               unverified_conditions=["운용 근거 미확인"]) if criterion == "TRL" else None
                findings.append(TechnicalFindingDraft(technology_ids=[technology["id"]], criterion=criterion,
                                                      claim=f"{criterion} 근거 확인", evidence_ids=[EVIDENCE_IDS[technology["approach"].lower()]],
                                                      is_inference=criterion == "TRL", trl_assessment=trl))
        return TechnicalDraft(status="partial" if partial else "complete", summary="기술 조사 요약", findings=findings,
                              limitations=["HW TRL 근거 부족"] if partial else [],
                              next_queries=["CXL-PNM maturity evidence"] if partial else [])

    index.search.side_effect = search
    model.with_structured_output.return_value = model
    model.invoke.side_effect = analyze
    output = build_technical_research_graph(index, model, retries, rules="").invoke(state)
    assert state == original, "서브그래프가 입력 State를 직접 변경함"
    assert set(output) == {"technical_result", "technical_retry_count", "technical_queries", "technical_missing_items"}
    result = output["technical_result"]
    if mode == "normal":
        assert result["status"] == "complete" and calls["analyze"] == 1 and output["technical_retry_count"] == 0
        assert len(result["findings"]) == len(state["technologies"]) * len(state["evaluation_criteria"]["technical"])
        assert all(f["trl_assessment"]["basis"] == "공개 정보 기반 추정" for f in result["findings"] if "trl_assessment" in f)
    elif mode == "retry":
        assert result["status"] == "complete" and calls["analyze"] == 2 and output["technical_retry_count"] == 1
        sides = [call.args[1] for call in index.search.call_args_list]
        assert sides == ["sw", "hw", "hw"], "재검색은 미충족 기술의 논문만 검색해야 함"
    elif mode == "always_partial":
        assert result["status"] == "partial" and output["technical_retry_count"] == retries
        assert any("근거 부족" in item for item in result["limitations"])
        assert retries == 0 or any(f"기술 재검색 한도({retries}회)" in item for item in result["limitations"])
    else:
        assert result["status"] == "error" and "SECRET_SENTINEL" not in json.dumps(output)
        assert calls["analyze"] == 0 if mode == "search_failure" else calls["analyze"] == 1


def main():
    state = initial_state()
    state.update(select_technologies(state))
    for mode in ("normal", "retry", "always_partial", "search_failure", "model_failure"):
        check_case(state, mode)
    check_case(state, "always_partial", retries=0)
    state["technologies"] = [
        dict(id="sw_02", name="대체 압축 기술", approach="SW", selection_reason="정밀도를 변경한다"),
        dict(id="hw_02", name="대체 메모리 기술", approach="HW", selection_reason="메모리 계층을 확장한다"),
    ]
    state["target_domain"] = "다른 적용 환경"
    check_case(state)
    print("PASS: 기술 조사 서브그래프 완료·재검색·한도·검색 실패·모델 실패, 출력 격리, State 보존 (외부 API 없음)")


if __name__ == "__main__":
    main()
