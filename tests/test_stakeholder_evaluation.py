import unittest

from service.agent.node.stakeholder import make_stakeholder_node
from tests.test_market_evaluation import REUSED, FakeAnalyst, base_state, cell, finding, only, web_refs
from tests.test_tavily_client import FakeSearch

FRAMEWORK = {"id": "sw_p9_c2", "source_type": "paper", "title": "MLA conversion", "url": "https://example.org/mla",
             "page": 9, "published_at": None, "excerpt": "The converted model runs on the open-source vLLM framework."}
STAKEHOLDER_CELLS = {(t, c) for t in ("sw_01", "hw_01") for c in ("경쟁 기술 진영", "도입사·개발자", "투자 업계")}


def run(analyst, search=None, state=None):
    search = search or FakeSearch({}, default="commercial_positive")
    node = make_stakeholder_node(analyst, search=search)
    return node(state or base_state())["stakeholder_result"], search


class StakeholderEvaluationTest(unittest.TestCase):
    def test_complete_with_stakeholder_criteria(self):
        analyst = FakeAnalyst()
        result, search = run(analyst)
        self.assertEqual(result["status"], "complete", result["limitations"])
        self.assertEqual({cell(c) for c in analyst.contexts}, STAKEHOLDER_CELLS)
        self.assertEqual(len(search.calls), 12)

    def test_reuse_uses_stakeholder_keywords(self):
        state = base_state()
        state["technical_result"]["evidence"] = [REUSED, FRAMEWORK]
        analyst = FakeAnalyst()
        run(analyst, state=state)
        # 비용·클라우드 청크는 시장성 후보이고, 이해관계자에서는 프레임워크·개발자 청크만 재인용한다.
        sw_context = next(c for c in analyst.contexts if cell(c)[0] == "sw_01")
        self.assertEqual([e["url"] for e in sw_context["reused_evidence"]], [FRAMEWORK["url"]])

    def test_stance_required_and_scope_assigned_by_evidence(self):
        def make(context):
            refs, (tid, criterion) = web_refs(context)[:1], cell(context)
            if (tid, criterion) == ("sw_01", "도입사·개발자"):
                return [finding(tid, criterion, refs, stance=None)]
            if (tid, criterion) == ("sw_01", "투자 업계"):
                return [finding(tid, criterion, refs, stance="mixed", scope="direct", stage=None)]
            return []
        result, _ = run(FakeAnalyst(make))
        self.assertEqual(len(result["findings"]), 1)
        # sw_01 칸에 MLA 고유어가 없는 근거를 인용했으므로 LLM의 direct와 무관하게 adjacent다.
        self.assertEqual(result["findings"][0]["scope"], "adjacent")
        self.assertTrue(any("stance 누락" in item for item in result["limitations"]))
        self.assertEqual(result["status"], "partial")

    def test_market_rules_do_not_apply(self):
        # 시장성 전용 규칙(scope·stage 필수, direct 교정)은 이해관계자에 적용하지 않는다.
        result, _ = run(FakeAnalyst(only("sw_01", "경쟁 기술 진영", lambda c: [
            finding("sw_01", "경쟁 기술 진영", web_refs(c)[:1], scope=None, stage=None, stance="negative")])))
        self.assertEqual(len(result["findings"]), 1)
        self.assertFalse(any("scope direct→adjacent" in item for item in result["limitations"]))

    def test_low_trust_counts_but_fact_becomes_opinion(self):
        def make(context):
            ref = next(e["id"] for e in context["web_evidence"] if "linkedin.com" in e["url"])
            return [finding(*cell(context), [ref], claim="개발자 경험담", stance="positive")]
        result, _ = run(FakeAnalyst(make), search=FakeSearch({}, default="lowtrust"))
        self.assertIn("6칸 중 6칸", result["summary"])  # 이해관계자에서는 개인 의견 근거로 인정
        self.assertEqual({f["claim_type"] for f in result["findings"]}, {"opinion"})
        self.assertTrue(any("claim_type fact→opinion (소셜미디어·개인 블로그 출처만 인용)" in item
                            for item in result["limitations"]))
        self.assertTrue(result["summary"].startswith("이해관계자 평가:"))

    def test_scope_direct_when_technology_term_present(self):
        def make(context):
            ref = next(e["id"] for e in context["web_evidence"] if "cxl-pnm-sample" in e["url"])
            return [finding("hw_01", "투자 업계", [ref], claim="벤더가 CXL-PNM 모듈 투자", scope=None, stance="positive"),
                    finding("hw_01", "투자 업계", [ref], claim="벤더의 기업 전체 투자", scope=None, stance="positive")]
        result, _ = run(FakeAnalyst(only("hw_01", "투자 업계", make)))
        # 근거에 고유어가 있어도 claim이 대상 기술을 말하지 않으면 연관 반응이다.
        self.assertEqual([f["scope"] for f in result["findings"]], ["direct", "adjacent"])

    def test_academic_only_excluded_for_adopters_and_investors(self):
        def make(context):
            ref = next(e["id"] for e in context["web_evidence"] if "arxiv.org" in e["url"])
            return [finding(*cell(context), [ref], stance="positive")]
        result, _ = run(FakeAnalyst(make), search=FakeSearch({}, default="academic"))
        kept = {(f["technology_ids"][0], f["claim"]) for f in result["findings"]}
        self.assertEqual(len(kept), 2)  # 경쟁 기술 진영(sw_01, hw_01)만 남는다
        for criterion in ("도입사·개발자", "투자 업계"):
            self.assertTrue(any(f"학술 자료만으로 {criterion} 반응 판단 불가" in item for item in result["limitations"]))

    def test_revision_feedback_is_filtered(self):
        analyst = FakeAnalyst()
        run(analyst, state=base_state(quality_feedback=["이해관계자: 투자 업계 근거 부족", "시장성: 규모 정의 누락"]))
        self.assertEqual(analyst.contexts[0]["revision_feedback"], ["이해관계자: 투자 업계 근거 부족"])


if __name__ == "__main__":
    unittest.main()
