import json
import unittest

import httpx

from pipeline import RULES, error_result, initial_state, normalize_result, select_technologies
from service.agent.node.market import make_market_node
from service.agent.tavily.evaluation import COMMON_PROMPT
from state import AnalysisDraft
from tests.test_tavily_client import FakeSearch, http_error

REUSED = {"id": "sw_p3_c1", "source_type": "paper", "title": "DeepSeek-V2", "url": "https://arxiv.org/abs/2405.04434",
          "page": 3, "published_at": None, "excerpt": "Training cost is reduced and deployment on cloud clusters is described."}
NOT_REUSED = {"id": "hw_p5_c2", "source_type": "paper", "title": "CXL-PNM", "url": "https://example.org/pnm",
              "page": 5, "published_at": None, "excerpt": "Accuracy on long-context benchmarks is preserved."}
CELLS = {(t, c) for t in ("sw_01", "hw_01") for c in ("시장 규모·성장성", "상용화·채택", "생태계")}


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


def cell(context):
    return context["target_technology"]["id"], context["target_criterion"]


def web_refs(context):
    return [e["id"] for e in context["web_evidence"]]


def ref_by_url(context, fragment):
    return next(e["id"] for e in context["web_evidence"] if fragment in e["url"])


def only(technology_id, criterion, make):
    """지정한 칸의 호출에서만 make 결과를 내고 나머지 칸은 빈 findings를 낸다."""
    return lambda context: make(context) if cell(context) == (technology_id, criterion) else []


class FakeAnalyst:
    """칸마다 호출된다. 기본은 첫 웹 근거로 긍정·부정 finding을 하나씩 만든다. make로 결과를 바꿀 수 있다."""

    def __init__(self, make=None, status="complete", limitations=None):
        self.make, self.status, self.limitations, self.contexts = make, status, limitations or [], []

    def invoke(self, messages):
        context = json.loads(messages[-1][1])
        self.contexts.append(context)
        technology_id, criterion = cell(context)
        refs = web_refs(context)
        if self.make:
            findings = self.make(context)
        else:
            findings = [finding(technology_id, criterion, refs[:1], stance=s) for s in ("positive", "negative")] if refs else []
        return AnalysisDraft(status=self.status, summary=f"{technology_id} {criterion} 요약", findings=findings,
                             limitations=self.limitations, next_queries=[])


def run(analyst, search=None, state=None):
    search = search or FakeSearch({}, default="commercial_positive")
    node = make_market_node(analyst, rules=RULES, normalize_result=normalize_result, error_result=error_result,
                            search=search)
    return node(state or base_state())["market_result"], search


class MarketEvaluationTest(unittest.TestCase):
    def test_complete_when_all_cells_covered(self):
        analyst = FakeAnalyst()
        result, search = run(analyst)
        self.assertEqual(result["status"], "complete", result["limitations"])
        self.assertEqual(len(search.calls), 2 * 3 * 2)  # 기술 2 × 기준 3 × 긍정/부정
        self.assertEqual(len(analyst.contexts), 6)  # 칸마다 LLM 1회
        self.assertFalse(any("일방적 근거" in item for item in result["limitations"]))

    def test_each_call_gets_one_cell_and_its_evidence(self):
        analyst = FakeAnalyst()
        run(analyst)
        self.assertEqual({cell(c) for c in analyst.contexts}, CELLS)
        for context in analyst.contexts:
            reused = [e["url"] for e in context["reused_evidence"]]
            # 시장·운영 키워드 필터를 통과한 SW 논문 근거는 SW 칸에만 전달된다.
            self.assertEqual(reused, [REUSED["url"]] if cell(context)[0] == "sw_01" else [])

    def test_llm_sees_short_refs_and_result_has_real_ids(self):
        analyst = FakeAnalyst()
        result, _ = run(analyst)
        for context in analyst.contexts:
            refs = [e["id"] for e in context["reused_evidence"] + context["web_evidence"]]
            self.assertEqual(refs, [f"E{i}" for i in range(1, len(refs) + 1)])
        self.assertTrue(all(key.startswith("web_") for f in result["findings"] for key in f["evidence_ids"]))
        self.assertTrue(all("[E" not in f["claim"] and "[web_" in f["claim"] for f in result["findings"]))

    def test_unknown_ref_is_dropped(self):
        result, _ = run(FakeAnalyst(only("hw_01", "생태계", lambda c: [finding("hw_01", "생태계", ["E999"])])))
        self.assertEqual(result["findings"], [])
        self.assertTrue(any("수집하지 않은 근거 ID E999" in item for item in result["limitations"]))

    def test_finding_for_other_cell_is_dropped(self):
        result, _ = run(FakeAnalyst(only("hw_01", "생태계", lambda c: [finding("sw_01", "생태계", web_refs(c)[:1])])))
        self.assertEqual(result["findings"], [])
        self.assertTrue(any("요청한 칸(hw_01 / 생태계)과 다른 기술·기준" in item for item in result["limitations"]))

    def test_reused_evidence_supports_matching_technology_only(self):
        def make(context):
            if cell(context) == ("sw_01", "생태계"):
                return [finding("sw_01", "생태계", ["E1"], stance="mixed")]  # E1 = 재인용 SW 논문
            if cell(context) == ("hw_01", "생태계"):
                return [finding("hw_01", "생태계", ["sw_p3_c1"], stance="mixed")]  # 실제 ID로 SW 논문 인용
            return []
        result, _ = run(FakeAnalyst(make))
        self.assertEqual([f["technology_ids"] for f in result["findings"]], [["sw_01"]])
        self.assertEqual([e["id"] for e in result["evidence"]], ["sw_p3_c1"])
        self.assertTrue(any("hw_01의 근거가 연결되지 않음" in item for item in result["limitations"]))

    def test_one_sided_stance_is_recorded(self):
        result, _ = run(FakeAnalyst(lambda c: [finding(*cell(c), web_refs(c)[:1])]))
        self.assertEqual(result["status"], "complete")
        self.assertIn("일방적 근거: hw_01 / 생태계에서 긍정 방향 근거만 확인됨, 반대 방향 근거 미확보",
                      result["limitations"])

    def test_invalid_findings_are_dropped_not_error(self):
        def make(context):
            refs, (tid, criterion) = web_refs(context)[:1], cell(context)
            if (tid, criterion) == ("sw_01", "시장 규모·성장성"):
                return [finding(tid, criterion, ["web_0000000000000000"]),  # 수집하지 않은 근거
                        finding(tid, criterion, refs, claim_type=None),
                        finding(tid, criterion, refs, scope=None),
                        finding(tid, "시장 수요", refs)]  # 레포 목업 기준명
            if (tid, criterion) == ("sw_01", "상용화·채택"):
                return [finding(tid, criterion, refs, stage=None)]
            if (tid, criterion) == ("sw_01", "생태계"):
                return [finding(tid, criterion, refs, stance="mixed")]
            return []
        result, _ = run(FakeAnalyst(make))
        self.assertEqual(result["status"], "partial")
        self.assertEqual(len(result["findings"]), 1)
        dropped = [item for item in result["limitations"] if item.startswith("검증 실패로 제외")]
        self.assertEqual(len(dropped), 5)
        for reason in ("수집하지 않은 근거 ID", "claim_type 누락", "scope 누락", "stage 누락", "다른 기술·기준"):
            self.assertTrue(any(reason in item for item in dropped), reason)

    def test_unknown_citation_in_limitations_is_stripped(self):
        result, _ = run(FakeAnalyst(limitations=["추가 확인 필요 [web_ffffffffffffffff]"]))
        self.assertNotEqual(result["status"], "error")
        self.assertIn("추가 확인 필요", result["limitations"])

    def test_no_results_skips_llm_and_withholds_judgement(self):
        analyst = FakeAnalyst()
        result, _ = run(analyst, search=FakeSearch({}, default="empty"),
                        state=base_state(technical_result={"evidence": []}))
        self.assertEqual(analyst.contexts, [])  # 근거 없는 칸은 LLM을 호출하지 않는다
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["findings"], [])
        self.assertEqual(len([item for item in result["limitations"] if item.endswith("판단 유보")]), 6)

    def test_one_side_failure_is_absorbed(self):
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

    def test_all_llm_calls_failed_is_error(self):
        class Broken:
            def invoke(self, messages):
                raise RuntimeError("SECRET_SENTINEL")
        result, _ = run(Broken())
        self.assertEqual(result["status"], "error")
        self.assertNotIn("SECRET_SENTINEL", json.dumps(result, ensure_ascii=False))

    def test_one_cell_llm_failure_is_withheld(self):
        class FlakyAnalyst(FakeAnalyst):
            def invoke(self, messages):
                if cell(json.loads(messages[-1][1])) == ("hw_01", "생태계"):
                    raise RuntimeError("SECRET_SENTINEL")
                return super().invoke(messages)
        result, _ = run(FlakyAnalyst())
        self.assertEqual(result["status"], "partial")
        self.assertIn("hw_01 / 생태계: LLM 분석 실패 (RuntimeError), 판단 유보", result["limitations"])
        self.assertNotIn("SECRET_SENTINEL", json.dumps(result, ensure_ascii=False))

    def test_revision_feedback_is_filtered(self):
        analyst = FakeAnalyst()
        run(analyst, state=base_state(quality_feedback=["시장성: 상용화 근거 보강", "도메인: 전력 수치 조건 누락"]))
        self.assertEqual(analyst.contexts[0]["revision_feedback"], ["시장성: 상용화 근거 보강"])

    def test_result_evidence_only_contains_cited(self):
        result, _ = run(FakeAnalyst())
        cited = {key for f in result["findings"] for key in f["evidence_ids"]}
        self.assertEqual({e["id"] for e in result["evidence"]}, cited)

    def test_summary_joins_cell_summaries(self):
        result, _ = run(FakeAnalyst())
        self.assertIn("sw_01 시장 규모·성장성 요약", result["summary"])
        self.assertIn("hw_01 생태계 요약", result["summary"])


class LabelAndCitationRuleTest(unittest.TestCase):
    """2026-09-22 실제 실행에서 발견한 문제(근거 통째 첨부, 연관 시장을 direct로, 전망을 fact로)에 대한 규칙."""

    def test_claim_without_inline_citation_keeps_evidence_ids(self):
        # 실측에서 LLM은 본문 인용을 넣지 않았다. evidence_ids가 상한 이내면 유지한다.
        result, _ = run(FakeAnalyst(only("hw_01", "생태계", lambda c: [
            finding("hw_01", "생태계", web_refs(c)[:2], claim="인용 없는 주장", stance="mixed")])))
        self.assertEqual(len(result["findings"][0]["evidence_ids"]), 2)

    def test_more_than_five_evidence_is_dropped(self):
        result, _ = run(FakeAnalyst(only("hw_01", "생태계", lambda c: [
            finding("hw_01", "생태계", web_refs(c), claim="근거를 통째로 붙인 주장", stance="mixed")])),
            search=FakeSearch({}, default="many"))
        self.assertEqual(result["findings"], [])
        self.assertTrue(any("근거 6개로 상한 5개 초과" in item for item in result["limitations"]))

    def test_evidence_ids_are_narrowed_to_inline_citations(self):
        def make(context):
            refs = web_refs(context)
            return [finding("hw_01", "생태계", refs, claim=f"주장 [{refs[0]}]", stance="mixed")]
        result, _ = run(FakeAnalyst(only("hw_01", "생태계", make)))
        self.assertEqual(len(result["findings"][0]["evidence_ids"]), 1)
        self.assertEqual(len(result["evidence"]), 1)

    def test_unknown_evidence_id_is_named(self):
        result, _ = run(FakeAnalyst(only("hw_01", "생태계", lambda c: [
            finding("hw_01", "생태계", ["web_00000000deadbeef"], stance="mixed")])))
        self.assertTrue(any("수집하지 않은 근거 ID web_00000000deadbeef" in item for item in result["limitations"]))

    def test_direct_scope_without_technology_term_becomes_adjacent(self):
        def make(context):
            if cell(context) == ("hw_01", "시장 규모·성장성"):
                pilot = ref_by_url(context, "cxl-pilot")  # "CXL memory"만 있고 PNM 고유어 없음
                return [finding("hw_01", "시장 규모·성장성", [pilot], claim=f"CXL 메모리 시장 [{pilot}]", stance="mixed")]
            if cell(context) == ("hw_01", "생태계"):
                sample = ref_by_url(context, "cxl-pnm-sample")  # processing-near-memory 포함
                return [finding("hw_01", "생태계", [sample], claim=f"PNM 모듈 [{sample}]", stance="mixed")]
            return []
        result, _ = run(FakeAnalyst(make))
        self.assertEqual([f["scope"] for f in result["findings"]], ["adjacent", "direct"])
        self.assertTrue(any("scope direct→adjacent" in item and "hw_01 / 시장 규모·성장성" in item
                            for item in result["limitations"]))

    def test_future_year_or_forecast_term_becomes_forecast(self):
        claims = {"시장 규모·성장성": "2030년 123억 달러", "생태계": "연평균 30% 성장 전망", "상용화·채택": "2025년 샘플 출하"}

        def make(context):
            tid, criterion = cell(context)
            if tid != "hw_01":
                return []
            ref = web_refs(context)[0]
            return [finding(tid, criterion, [ref], claim=f"{claims[criterion]} [{ref}]", stance="mixed")]
        result, _ = run(FakeAnalyst(make))
        types = {f["claim"].split(" [")[0]: f["claim_type"] for f in result["findings"]}
        self.assertEqual(types, {"2030년 123억 달러": "forecast", "연평균 30% 성장 전망": "forecast", "2025년 샘플 출하": "fact"})
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
        self.assertIn("target_criterion 하나만 평가", captured["system"])
        self.assertIn("관련 근거를 모두 붙이지 않는다", captured["system"])


if __name__ == "__main__":
    unittest.main()


class GroundingRuleTest(unittest.TestCase):
    """2026-09-22 실측: LLM이 request의 선정 문서 수치를 무관한 USENIX 논문 근거에 붙여 stage=production으로 표시했다."""

    def test_cell_context_excludes_request(self):
        analyst = FakeAnalyst()
        run(analyst)
        self.assertTrue(all("request" not in context for context in analyst.contexts))
        self.assertIn("selection_reason은\n검색 방향을 정하는 입력이며 근거가 아니다", COMMON_PROMPT)

    def run_academic(self, criterion, url_fragment, **fields):
        def make(context):
            ref = ref_by_url(context, url_fragment)
            return [finding("hw_01", criterion, [ref], claim=f"주장 [{ref}]", **fields)]
        return run(FakeAnalyst(only("hw_01", criterion, make)), search=FakeSearch({}, default="academic"))[0]

    def test_commercialization_from_academic_only_is_dropped(self):
        for fragment in ("usenix.org", "arxiv.org"):
            with self.subTest(fragment=fragment):
                result = self.run_academic("상용화·채택", fragment, stage="production", stance="positive")
                self.assertEqual(result["findings"], [])
                self.assertTrue(any("학술 자료만으로 상용화·채택 판단 불가" in item for item in result["limitations"]))

    def test_commercialization_from_vendor_announcement_is_kept(self):
        result = self.run_academic("상용화·채택", "example-semi.com", stage="announced", stance="positive")
        self.assertEqual([f["stage"] for f in result["findings"]], ["announced"])

    def test_stage_from_academic_source_is_cleared(self):
        result = self.run_academic("생태계", "arxiv.org", stage="pilot", stance="mixed")
        self.assertIsNone(result["findings"][0]["stage"])
        self.assertTrue(any("stage pilot→null (학술 자료만으로 상용화 단계 판단 불가)" in item
                            for item in result["limitations"]))

    def test_reused_paper_evidence_is_academic(self):
        from service.agent.tavily.evaluation import is_academic
        self.assertTrue(is_academic(REUSED))
        self.assertTrue(is_academic({**REUSED, "source_type": "web", "url": "https://cs.stanford.edu/paper.pdf"}))
        self.assertFalse(is_academic({**REUSED, "source_type": "web", "url": "https://news.example.com/cxl"}))
