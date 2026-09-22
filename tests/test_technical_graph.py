import json

import pytest

from service.agent.graph.technical import build_technical_research_graph
from tests.fixtures import EmptyFirstDraftModelFixture, FailingPaperFixture, ModelFixture, PaperFixture, sample_state


def run(model, retries=2, index=None):
    index = index or PaperFixture()
    graph = build_technical_research_graph(index, model, retries, rules="공통 규칙. ")
    return graph.invoke(sample_state()), index


def test_partial_then_complete_after_one_retry():
    model = ModelFixture()
    result, index = run(model)
    assert result["technical_result"]["status"] == "complete"
    assert result["technical_retry_count"] == 1 and model.calls == 2
    assert result["technical_missing_items"] == [] and result["technical_queries"] == []
    # 첫 라운드는 두 논문 모두, 재검색은 미충족 기술(hw)만 검색한다.
    assert [side for _, side in index.calls] == ["sw", "hw", "hw"]
    assert index.calls[2][0] == "CXL-PNM prototype maturity TRL validation"


def test_retry_limit_keeps_partial_with_limitation():
    model = ModelFixture(always_partial=True)
    result, _ = run(model, retries=2)
    assert result["technical_result"]["status"] == "partial"
    assert result["technical_retry_count"] == 2 and model.calls == 3
    assert "hw_01: TRL" in result["technical_missing_items"]
    assert any("기술 재검색 한도(2회)" in item for item in result["technical_result"]["limitations"])


def test_zero_retries_never_rewrites():
    model = ModelFixture(always_partial=True)
    result, _ = run(model, retries=0)
    assert model.calls == 1 and result["technical_retry_count"] == 0
    assert result["technical_result"]["status"] == "partial"


def test_model_failure_becomes_error_without_secret():
    result, _ = run(ModelFixture(fail=True))
    assert result["technical_result"]["status"] == "error"
    assert "SECRET_SENTINEL" not in json.dumps(result)


def test_output_hides_internal_evidence_dict():
    result, _ = run(ModelFixture())
    assert set(result) == {"technical_result", "technical_retry_count", "technical_queries", "technical_missing_items"}


def test_prompt_receives_common_rules_and_assessment_date():
    model = ModelFixture()
    graph = build_technical_research_graph(PaperFixture(), model, 1, rules="공통 규칙. ")
    graph.invoke(sample_state())
    assert model.contexts[0]["assessment_date"] == "2026-09-21"
    assert model.contexts[1]["missing_items"] == ["hw_01: TRL"]


def test_invalid_retry_range():
    with pytest.raises(ValueError):
        build_technical_research_graph(PaperFixture(), ModelFixture(), 6)


@pytest.mark.parametrize("fail_at, evidence_ids, model_calls", [
    (1, [], 0),
    (2, ["sw_p1_c1"], 0),
    (3, ["hw_p1_c1", "sw_p1_c1"], 1),
])
def test_search_failure_returns_safe_error_and_preserves_progress(fail_at, evidence_ids, model_calls):
    model = ModelFixture()
    output, index = run(model, index=FailingPaperFixture(fail_at))
    result = output["technical_result"]
    assert result["status"] == "error"
    assert "RuntimeError" in result["limitations"][-1]
    assert "SECRET_SENTINEL" not in json.dumps(output)
    assert [row["id"] for row in result["evidence"]] == evidence_ids
    assert model.calls == model_calls and len(index.calls) == fail_at
    assert output["technical_queries"] == []
    assert output["technical_retry_count"] == model_calls
    assert set(output) == {"technical_result", "technical_retry_count", "technical_queries", "technical_missing_items"}
    if fail_at == 3:
        assert result["summary"] == "기술 조사 요약"
        assert len(result["findings"]) == 7
        assert "HW TRL 근거 부족" in result["limitations"]
        assert output["technical_missing_items"] == ["hw_01: TRL"]
    else:
        assert result["findings"] == []
        assert output["technical_missing_items"] == []


def test_parent_graph_stops_after_subgraph_search_failure():
    from pipeline import build_graph

    def unexpected_evaluation(state):
        pytest.fail("검색 오류 후 후속 평가를 실행함")

    nodes = {name: unexpected_evaluation for name in (
        "market_evaluation", "stakeholder_evaluation", "domain_evaluation", "synthesis", "report")}
    nodes["technology_selection"] = lambda state: {"technologies": sample_state()["technologies"]}
    nodes["technical_research"] = build_technical_research_graph(FailingPaperFixture(3), ModelFixture())
    output = build_graph(nodes).invoke(sample_state())
    assert output["technical_result"]["status"] == "error"
    assert len(output["technical_result"]["findings"]) == 7
    assert "technical_evidence" not in output
    assert "SECRET_SENTINEL" not in json.dumps(output)


@pytest.mark.parametrize("next_queries", [[], ["   "], [" ", *["  " + "가" * 600 + "  "] * 8]])
def test_retry_searches_bound_fallback_and_model_queries(next_queries):
    state = sample_state()
    state["evaluation_criteria"]["technical"] = ["원리" + "가" * 600, "성능", "한계", "TRL"]
    model = EmptyFirstDraftModelFixture(next_queries)
    index = PaperFixture()
    output = build_technical_research_graph(index, model, 1).invoke(state)
    retry_calls = index.calls[2:]
    # 기술별로 동일한 최대 4개 질의를 사용하고 한 라운드만 재검색한다.
    assert len(retry_calls) == 8
    assert [side for _, side in retry_calls] == ["sw"] * 4 + ["hw"] * 4
    assert all(query == query.strip() and 0 < len(query) <= 500 for query, _ in retry_calls)
    assert output["technical_retry_count"] == 1
    assert output["technical_result"]["status"] == "complete"


def test_supplied_queries_are_bounded_before_search():
    state = sample_state()
    state["technical_queries"] = ["   ", *["  " + "가" * 600 + "  "] * 8]
    index = PaperFixture()
    output = build_technical_research_graph(index, ModelFixture(), 0).invoke(state)
    assert len(index.calls) == 8
    assert all(query == "가" * 500 for query, _ in index.calls)
    assert output["technical_result"]["status"] == "partial"


def test_invalid_evidence_id_is_corrected_once_with_feedback():
    from tests.fixtures import BadEvidenceFirstModelFixture

    model = BadEvidenceFirstModelFixture()
    result, _ = run(model, retries=0)
    assert result["technical_result"]["status"] == "complete"
    assert model.calls == 2
    assert model.feedback_seen[0] == [] and any("sw_p9_c9" in item for item in model.feedback_seen[1])


def test_invalid_evidence_id_twice_becomes_error():
    from tests.fixtures import BadEvidenceFirstModelFixture

    model = BadEvidenceFirstModelFixture(always_bad=True)
    result, _ = run(model, retries=2)
    assert result["technical_result"]["status"] == "error" and model.calls == 2
    assert "Evidence.id" in result["technical_result"]["limitations"][0]
