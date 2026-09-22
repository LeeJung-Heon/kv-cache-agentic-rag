import json
import unittest

import httpx

from pipeline import RULES, error_result, initial_state, normalize_result, select_technologies
from service.agent.node.market import make_market_node
from service.agent.tavily.query_templates import CRITERIA
from state import AnalysisDraft
from tests.test_tavily_client import FakeSearch, http_error

REUSED = {"id": "sw_p3_c1", "source_type": "paper", "title": "DeepSeek-V2", "url": "https://arxiv.org/abs/2405.04434",
          "page": 3, "published_at": None, "excerpt": "Training cost is reduced and deployment on cloud clusters is described."}
NOT_REUSED = {"id": "hw_p5_c2", "source_type": "paper", "title": "CXL-PNM", "url": "https://example.org/pnm",
              "page": 5, "published_at": None, "excerpt": "Accuracy on long-context benchmarks is preserved."}


def base_state(**extra):
    state = initial_state()
    state.update(select_technologies(state))
    state["technical_result"] = {"status": "complete", "summary": "", "findings": [],
                                 "evidence": [REUSED, NOT_REUSED], "limitations": []}
    state.update(extra)
    return state


def finding(technology_id, criterion, evidence_ids, **fields):
    values = {"technology_ids": [technology_id], "criterion": criterion,
              "claim": f"{criterion} 확인 [{evidence_ids[0]}]", "evidence_ids": evidence_ids, "is_inference": False,
              "claim_type": "fact", "scope": "direct", "stage": "pilot", "stance": "positive"}
    values.update(fields)
    return values


class FakeAnalyst:
    """search_results를 보고 기술 × 기준마다 긍정·부정 finding을 하나씩 만든다. make로 결과를 바꿀 수 있다."""

    def __init__(self, make=None, status="complete", limitations=None):
        self.make, self.status, self.limitations, self.contexts = make, status, limitations or [], []

    def invoke(self, messages):
        context = json.loads(messages[-1][1])
        self.contexts.append(context)
        findings = self.make(context) if self.make else [
            finding(row["technology_id"], row["criterion"], row["evidence_ids"][:1], stance=stance)
            for row in context["search_results"] if row["evidence_ids"] for stance in ("positive", "negative")]
        return AnalysisDraft(status=self.status, summary="시장성 요약", findings=findings,
                             limitations=self.limitations, next_queries=[])


def run(analyst, search=None, state=None):
    search = search or FakeSearch({}, default="commercial_positive")
    node = make_market_node(analyst, rules=RULES, normalize_result=normalize_result, error_result=error_result,
                            search=search)
    return node(state or base_state())["market_result"], search


class MarketEvaluationTest(unittest.TestCase):
    def test_complete_when_all_cells_covered(self):
        result, search = run(FakeAnalyst())
        self.assertEqual(result["status"], "complete", result["limitations"])
        self.assertEqual(len(search.calls), 2 * 3 * 2)  # 기술 2 × 기준 3 × 긍정/부정
        self.assertEqual({f["claim_type"] for f in result["findings"]}, {"fact"})
        self.assertFalse(any("일방적 근거" in item for item in result["limitations"]))

    def test_uses_design_criteria_not_mock(self):
        analyst = FakeAnalyst()
        run(analyst)
        self.assertEqual(analyst.contexts[0]["evaluation_criteria"], CRITERIA["market"])

    def test_reused_evidence_is_filtered_and_separated(self):
        analyst = FakeAnalyst()
        run(analyst)
        context = analyst.contexts[0]
        self.assertEqual([e["id"] for e in context["reused_evidence"]], ["sw_p3_c1"])
        self.assertTrue(all(e["id"].startswith("web_") for e in context["web_evidence"]))

    def test_reused_evidence_can_support_matching_technology_only(self):
        def make(context):
            return [finding("sw_01", "생태계", ["sw_p3_c1"], stance="mixed"),
                    finding("hw_01", "생태계", ["sw_p3_c1"], stance="mixed")]
        result, _ = run(FakeAnalyst(make))
        self.assertEqual([f["technology_ids"] for f in result["findings"]], [["sw_01"]])
        self.assertEqual([e["id"] for e in result["evidence"]], ["sw_p3_c1"])
        self.assertTrue(any("hw_01의 근거가 연결되지 않음" in item for item in result["limitations"]))

    def test_one_sided_stance_is_recorded(self):
        def make(context):
            return [finding(r["technology_id"], r["criterion"], r["evidence_ids"][:1])
                    for r in context["search_results"]]
        result, _ = run(FakeAnalyst(make))
        self.assertEqual(result["status"], "complete")
        self.assertIn("일방적 근거: hw_01 / 생태계에서 긍정 방향 근거만 확인됨, 반대 방향 근거 미확보",
                      result["limitations"])

    def test_invalid_findings_are_dropped_not_error(self):
        def make(context):
            web = context["search_results"][0]["evidence_ids"][:1]
            return [finding("sw_01", "시장 규모·성장성", ["web_0000000000000000"]),  # 수집하지 않은 근거
                    finding("sw_01", "시장 규모·성장성", web, claim_type=None),
                    finding("sw_01", "시장 규모·성장성", web, scope=None),
                    finding("sw_01", "상용화·채택", web, stage=None),
                    finding("sw_01", "시장 수요", web),  # 레포 목업 기준명
                    finding("sw_01", "생태계", web, stance="mixed")]
        result, _ = run(FakeAnalyst(make))
        self.assertEqual(result["status"], "partial")
        self.assertEqual(len(result["findings"]), 1)
        dropped = [item for item in result["limitations"] if item.startswith("검증 실패로 제외")]
        self.assertEqual(len(dropped), 5)
        for reason in ("claim_type 누락", "scope 누락", "stage 누락"):
            self.assertTrue(any(reason in item for item in dropped), reason)

    def test_unknown_citation_in_limitations_is_stripped(self):
        result, _ = run(FakeAnalyst(limitations=["추가 확인 필요 [web_ffffffffffffffff]"]))
        self.assertNotEqual(result["status"], "error")
        self.assertIn("추가 확인 필요", result["limitations"])

    def test_no_results_is_partial_with_withheld_judgement(self):
        result, _ = run(FakeAnalyst(lambda context: []), search=FakeSearch({}, default="empty"))
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["findings"], [])
        withheld = [item for item in result["limitations"] if item.endswith("판단 유보")]
        self.assertEqual(len(withheld), 6)

    def test_one_side_failure_stays_partial_or_complete(self):
        search = FakeSearch({"CXL-PNM adoption barriers delay not deployed": http_error(429)},
                            default="commercial_positive")
        result, _ = run(FakeAnalyst(), search=search)
        self.assertNotEqual(result["status"], "error")
        self.assertTrue(any("부정 근거 수집 실패 (HTTP 429)" in item for item in result["limitations"]))

    def test_all_queries_failed_is_error(self):
        analyst = FakeAnalyst()
        result, _ = run(analyst, search=FakeSearch({}, default=httpx.ConnectError("SECRET_SENTINEL")))
        self.assertEqual(result["status"], "error")
        self.assertEqual(analyst.contexts, [])  # LLM을 호출하지 않는다
        self.assertNotIn("SECRET_SENTINEL", json.dumps(result, ensure_ascii=False))

    def test_llm_failure_is_error(self):
        class Broken:
            def invoke(self, messages):
                raise RuntimeError("SECRET_SENTINEL")
        result, _ = run(Broken())
        self.assertEqual(result["status"], "error")
        self.assertNotIn("SECRET_SENTINEL", json.dumps(result, ensure_ascii=False))

    def test_revision_feedback_is_filtered(self):
        analyst = FakeAnalyst()
        run(analyst, state=base_state(quality_feedback=["시장성: 상용화 근거 보강", "도메인: 전력 수치 조건 누락"]))
        self.assertEqual(analyst.contexts[0]["revision_feedback"], ["시장성: 상용화 근거 보강"])

    def test_result_evidence_only_contains_cited(self):
        result, _ = run(FakeAnalyst())
        cited = {key for f in result["findings"] for key in f["evidence_ids"]}
        self.assertEqual({e["id"] for e in result["evidence"]}, cited)


if __name__ == "__main__":
    unittest.main()
