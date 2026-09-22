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
        self.assertEqual([e["url"] for e in context["reused_evidence"]], [REUSED["url"]])
        self.assertTrue(context["web_evidence"])

    def test_llm_sees_short_refs_and_result_has_real_ids(self):
        analyst = FakeAnalyst()
        result, _ = run(analyst)
        context = analyst.contexts[0]
        refs = [e["id"] for e in context["reused_evidence"] + context["web_evidence"]]
        self.assertEqual(refs, [f"E{i}" for i in range(1, len(refs) + 1)])
        self.assertTrue(all(key.startswith("web_") for f in result["findings"] for key in f["evidence_ids"]))
        self.assertTrue(all("[E" not in f["claim"] and "[web_" in f["claim"] for f in result["findings"]))

    def test_unknown_ref_is_dropped(self):
        def make(context):
            return [finding("hw_01", "생태계", ["E999"], stance="mixed")]
        result, _ = run(FakeAnalyst(make))
        self.assertEqual(result["findings"], [])
        self.assertTrue(any("수집하지 않은 근거 ID E999" in item for item in result["limitations"]))

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
        search = FakeSearch({"CXL processing-near-memory adoption barriers delay not deployed": http_error(429)},
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


def web_ids(context, index=0):
    return context["search_results"][index]["evidence_ids"]


def evidence_by_url(context, fragment):
    return next(e["id"] for e in context["web_evidence"] if fragment in e["url"])


class LabelAndCitationRuleTest(unittest.TestCase):
    """2026-09-22 실제 실행에서 발견한 문제(근거 통째 첨부, 연관 시장을 direct로, 전망을 fact로)에 대한 규칙."""

    def test_claim_without_inline_citation_keeps_evidence_ids(self):
        # 실측에서 LLM은 본문 인용을 넣지 않았다. evidence_ids가 상한 이내면 유지한다.
        def make(context):
            return [finding("hw_01", "생태계", web_ids(context)[:2], claim="인용 없는 주장", stance="mixed")]
        result, _ = run(FakeAnalyst(make))
        self.assertEqual(len(result["findings"][0]["evidence_ids"]), 2)

    def test_more_than_five_citations_is_dropped(self):
        def make(context):
            ids = web_ids(context)
            return [finding("hw_01", "생태계", ids, claim="근거를 통째로 붙인 주장", stance="mixed")]
        result, _ = run(FakeAnalyst(make), search=FakeSearch({}, default="many"))
        self.assertEqual(result["findings"], [])
        self.assertTrue(any("상한 5개 초과" in item for item in result["limitations"]))

    def test_evidence_ids_are_narrowed_to_inline_citations(self):
        def make(context):
            ids = web_ids(context)
            return [finding("hw_01", "생태계", ids, claim=f"주장 [{ids[0]}]", stance="mixed")]
        result, _ = run(FakeAnalyst(make))
        self.assertEqual(len(result["findings"][0]["evidence_ids"]), 1)
        self.assertEqual(len(result["evidence"]), 1)

    def test_unknown_evidence_id_is_named(self):
        def make(context):
            return [finding("hw_01", "생태계", ["web_00000000deadbeef"], stance="mixed")]
        result, _ = run(FakeAnalyst(make))
        self.assertTrue(any("수집하지 않은 근거 ID web_00000000deadbeef" in item for item in result["limitations"]))

    def test_direct_scope_without_technology_term_becomes_adjacent(self):
        def make(context):
            pilot = evidence_by_url(context, "cxl-pilot")  # "CXL memory"만 있고 PNM 고유어 없음
            sample = evidence_by_url(context, "cxl-pnm-sample")  # processing-near-memory 포함
            return [finding("hw_01", "시장 규모·성장성", [pilot], claim=f"CXL 메모리 시장 [{pilot}]", stance="mixed"),
                    finding("hw_01", "생태계", [sample], claim=f"PNM 모듈 [{sample}]", stance="mixed")]
        result, _ = run(FakeAnalyst(make))
        self.assertEqual([f["scope"] for f in result["findings"]], ["adjacent", "direct"])
        self.assertTrue(any("scope direct→adjacent" in item and "hw_01 / 시장 규모·성장성" in item
                            for item in result["limitations"]))

    def test_future_year_or_forecast_term_becomes_forecast(self):
        def make(context):
            ids = web_ids(context)
            return [finding("hw_01", "시장 규모·성장성", ids[:1], claim=f"2030년 123억 달러 [{ids[0]}]", stance="mixed"),
                    finding("hw_01", "생태계", ids[:1], claim=f"연평균 30% 성장 전망 [{ids[0]}]", stance="mixed"),
                    finding("hw_01", "상용화·채택", ids[:1], claim=f"2025년 샘플 출하 [{ids[0]}]", stance="mixed")]
        result, _ = run(FakeAnalyst(make))
        self.assertEqual([f["claim_type"] for f in result["findings"]], ["forecast", "forecast", "fact"])
        self.assertEqual(sum("claim_type fact→forecast" in item for item in result["limitations"]), 2)

    def test_year_like_digits_in_citation_id_are_ignored(self):
        from service.agent.tavily.evaluation import correct_forecast
        f = AnalysisDraft(status="complete", summary="", limitations=[], next_queries=[], findings=[
            finding("hw_01", "생태계", ["web_20270000aaaa1111"], claim="샘플 출하 [web_20270000aaaa1111]")]).findings[0]
        self.assertEqual(correct_forecast(f, "2026-09-21"), [])
        self.assertEqual(f.claim_type, "fact")

    def test_common_prompt_is_sent(self):
        captured = {}

        class Capture(FakeAnalyst):
            def invoke(self, messages):
                captured["system"] = messages[0][1]
                return super().invoke(messages)
        run(Capture())
        self.assertIn("관련 근거를 모두 붙이지 않는다", captured["system"])
