import unittest

from pipeline import RULES, error_result, normalize_result
from service.agent.node.stakeholder import make_stakeholder_node
from tests.test_market_evaluation import REUSED, FakeAnalyst, base_state, cell, finding, only, web_refs
from tests.test_tavily_client import FakeSearch

FRAMEWORK = {"id": "sw_p9_c2", "source_type": "paper", "title": "MLA conversion", "url": "https://example.org/mla",
             "page": 9, "published_at": None, "excerpt": "The converted model runs on the open-source vLLM framework."}
STAKEHOLDER_CELLS = {(t, c) for t in ("sw_01", "hw_01") for c in ("경쟁 기술 진영", "도입사·개발자", "투자 업계")}


def run(analyst, search=None, state=None):
    search = search or FakeSearch({}, default="commercial_positive")
    node = make_stakeholder_node(analyst, rules=RULES, normalize_result=normalize_result, error_result=error_result,
                                 search=search)
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

    def test_stance_required_and_scope_cleared(self):
        def make(context):
            refs, (tid, criterion) = web_refs(context)[:1], cell(context)
            if (tid, criterion) == ("sw_01", "도입사·개발자"):
                return [finding(tid, criterion, refs, stance=None)]
            if (tid, criterion) == ("sw_01", "투자 업계"):
                return [finding(tid, criterion, refs, stance="mixed", scope="direct", stage=None)]
            return []
        result, _ = run(FakeAnalyst(make))
        self.assertEqual(len(result["findings"]), 1)
        self.assertIsNone(result["findings"][0]["scope"])
        self.assertTrue(any("stance 누락" in item for item in result["limitations"]))
        self.assertEqual(result["status"], "partial")

    def test_market_rules_do_not_apply(self):
        # 시장성 전용 규칙(scope·stage 필수, direct 교정)은 이해관계자에 적용하지 않는다.
        result, _ = run(FakeAnalyst(only("sw_01", "경쟁 기술 진영", lambda c: [
            finding("sw_01", "경쟁 기술 진영", web_refs(c)[:1], scope=None, stage=None, stance="negative")])))
        self.assertEqual(len(result["findings"]), 1)
        self.assertFalse(any("scope direct→adjacent" in item for item in result["limitations"]))

    def test_revision_feedback_is_filtered(self):
        analyst = FakeAnalyst()
        run(analyst, state=base_state(quality_feedback=["이해관계자: 투자 업계 근거 부족", "시장성: 규모 정의 누락"]))
        self.assertEqual(analyst.contexts[0]["revision_feedback"], ["이해관계자: 투자 업계 근거 부족"])


if __name__ == "__main__":
    unittest.main()
