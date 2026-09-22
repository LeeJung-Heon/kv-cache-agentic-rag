import json
from collections import Counter
from unittest.mock import patch

from pipeline import (CRITERIA, EvidenceError, build_graph, citation_ids, collect_sources, initial_state, make_nodes,
                      normalize_result, select_technologies)
from service.agent.node.technical import model as technical_model
from service.agent.node.technical.prompts import ASSESSMENT_DATE
from service.agent.node.technical.schema import TechnicalDraft, TechnicalFindingDraft, TrlDraft
from state import AnalysisDraft


class PaperFixture:
    def search(self, query, side):
        return [{"id": f"{side}_p1_c1", "title": f"{side} paper", "url": f"https://example.org/{side}",
                 "page": 1, "text": f"{side} paper evidence"}]


class ModelFixture:
    """기술 조사는 첫 호출만 hw_01 TRL을 빠뜨린 partial, 이후는 complete를 돌려준다."""

    def __init__(self, always_partial=False, fail=False):
        self.technical_calls = 0
        self.always_partial = always_partial
        self.fail = fail

    def with_structured_output(self, *args, **kwargs):
        return self

    def invoke(self, messages):
        context = json.loads(messages[-1][1])
        if "allowed_citation_ids" in context:
            body = "# SUMMARY\n검사 보고서 [sw_p1_c1]\n"
            body += "\n## 4.2 시장성 평가\n초안\n## 4.3 이해관계자 평가\n초안\n## 4.4 도메인 적용 평가\n초안\n# 6. 분석의 한계\n"
            return type("Response", (), {"content": body})()
        criteria = context["evaluation_criteria"]
        technical = "assessment_date" in context
        if technical:
            self.technical_calls += 1
            if self.fail:
                raise RuntimeError("SECRET_SENTINEL")
        partial = technical and (self.always_partial or self.technical_calls == 1)
        findings = []
        for technology in context["technologies"]:
            for criterion in criteria:
                if partial and technology["id"] == "hw_01" and criterion == "TRL":
                    continue
                record = {"technology_ids": [technology["id"]], "criterion": criterion, "claim": f"{criterion} 근거 확인",
                          "evidence_ids": [technology["id"][:2] + "_p1_c1"], "is_inference": criterion == "TRL"}
                if technical:
                    trl = TrlDraft(level_or_range="TRL 3~4", as_of=ASSESSMENT_DATE, confidence="중간",
                                   unverified_conditions=["운용 근거 미확인"]) if criterion == "TRL" else None
                    findings.append(TechnicalFindingDraft(**record, trl_assessment=trl))
                else:
                    findings.append(record)
        if technical:
            return TechnicalDraft(status="partial" if partial else "complete", summary="기술 조사 요약", findings=findings,
                                  limitations=["HW TRL 근거 부족"] if partial else [],
                                  next_queries=["CXL-PNM prototype maturity TRL validation"] if partial else [])
        return AnalysisDraft(status="complete", summary="평가 요약", findings=findings, limitations=[], next_queries=[])


def run_case(model, retries=2, web_error=False):
    seen = []
    # 기술 조사는 전용 모델을 스스로 만들므로 생성 경계만 대역으로 바꾼다.
    with patch.object(technical_model, "create_technical_model", return_value=model):
        nodes = make_nodes(PaperFixture(), model, retries)
    for name, function in list(nodes.items()):
        def traced(state, name=name, function=function):
            if name == "synthesis":
                assert all(field in state for field in ("market_result", "stakeholder_result", "domain_result"))
            seen.append(name)
            return function.invoke(state) if hasattr(function, "invoke") else function(state)
        nodes[name] = traced
    with patch("pipeline.search_web", side_effect=RuntimeError("SECRET_SENTINEL") if web_error else None,
               return_value=[]):
        result = build_graph(nodes, retries).invoke(initial_state())
    return result, Counter(seen)


def main():
    model = ModelFixture()
    result, calls = run_case(model)
    assert calls["technical_research"] == 1 and model.technical_calls == 2 and result["technical_retry_count"] == 1
    assert result["technical_result"]["status"] == "complete" and not result["technical_missing_items"]
    assert any("trl_assessment" in finding for finding in result["technical_result"]["findings"])
    assert all(calls[name] == 1 for name in ("technology_selection", "market_evaluation", "stakeholder_evaluation",
                                             "domain_evaluation", "synthesis", "report"))
    assert result["report_evidence_ids"] == sorted(citation_ids(result["report_markdown"].split("\n# REFERENCE")[0]))
    assert set(result["report_evidence_ids"]) <= collect_sources(result).keys()
    assert "technical_evidence" not in result
    from pipeline import result_markdown
    assert "  - TRL 3~4 (기준일 " in result_markdown(result["technical_result"])

    model = ModelFixture(always_partial=True)
    limited, calls = run_case(model, retries=2)
    assert model.technical_calls == 3 and limited["technical_retry_count"] == 2
    assert limited["technical_result"]["status"] == limited["synthesis_result"]["status"] == "partial"
    assert "hw_01: TRL" in limited["technical_missing_items"]
    assert "기술 재검색 한도(2회)" in limited["report_markdown"]
    model = ModelFixture(always_partial=True)
    disabled, calls = run_case(model, retries=0)
    assert model.technical_calls == 1 and disabled["technical_retry_count"] == 0

    failed, calls = run_case(ModelFixture(fail=True))
    assert failed["technical_result"]["status"] == "error" and calls["report"] == 0 and calls["synthesis"] == 0
    assert "SECRET_SENTINEL" not in json.dumps(failed)
    failed, calls = run_case(ModelFixture(), web_error=True)
    assert failed["market_result"]["status"] == failed["synthesis_result"]["status"] == "error"
    assert calls["report"] == 0 and "SECRET_SENTINEL" not in json.dumps(failed)

    state = initial_state()
    state.update(select_technologies(state))
    malformed = AnalysisDraft(status="complete", summary="invalid", findings=[{
        "technology_ids": ["sw_01"], "criterion": "시장 수요", "claim": "invalid evidence",
        "evidence_ids": ["missing"], "is_inference": False}], limitations=[], next_queries=[])
    try:
        normalize_result(malformed, {}, state, CRITERIA["market"])
    except EvidenceError:
        pass
    else:
        raise AssertionError("존재하지 않는 근거 ID를 수락함")
    print("PASS: 기술 조사 서브그래프 재검색·한도·비활성화, 병렬 합류, 오류 중단, 근거 연결 (외부 API 없음)")


if __name__ == "__main__":
    main()
