import unittest

from pipeline import RULES, error_result, normalize_result
from service.agent.node.stakeholder import make_stakeholder_node
from service.agent.tavily.query_templates import CRITERIA
from tests.test_market_evaluation import REUSED, FakeAnalyst, base_state, finding
from tests.test_tavily_client import FakeSearch

FRAMEWORK = {"id": "sw_p9_c2", "source_type": "paper", "title": "MLA conversion", "url": "https://example.org/mla",
             "page": 9, "published_at": None, "excerpt": "The converted model runs on the open-source vLLM framework."}


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
        self.assertEqual(analyst.contexts[0]["evaluation_criteria"], CRITERIA["stakeholder"])
        self.assertEqual(len(search.calls), 12)

    def test_reuse_uses_stakeholder_keywords(self):
        state = base_state()
        state["technical_result"]["evidence"] = [REUSED, FRAMEWORK]
        analyst = FakeAnalyst()
        run(analyst, state=state)
        # 비용·클라우드 청크는 시장성 후보이고, 이해관계자에서는 프레임워크·개발자 청크만 재인용한다.
        self.assertEqual([e["url"] for e in analyst.contexts[0]["reused_evidence"]], [FRAMEWORK["url"]])

    def test_stance_required_and_scope_cleared(self):
        def make(context):
            web = context["search_results"][0]["evidence_ids"][:1]
            return [finding("sw_01", "도입사·개발자", web, stance=None),
                    finding("sw_01", "투자 업계", web, stance="mixed", scope="direct", stage=None)]
        result, _ = run(FakeAnalyst(make))
        self.assertEqual(len(result["findings"]), 1)
        self.assertIsNone(result["findings"][0]["scope"])
        self.assertTrue(any("stance 누락" in item for item in result["limitations"]))
        self.assertEqual(result["status"], "partial")

    def test_market_rules_do_not_apply(self):
        # 시장성 전용 규칙(scope·stage 필수)은 이해관계자에 적용하지 않는다.
        def make(context):
            web = context["search_results"][0]["evidence_ids"][:1]
            return [finding("sw_01", "경쟁 기술 진영", web, scope=None, stage=None, stance="negative")]
        result, _ = run(FakeAnalyst(make))
        self.assertEqual(len(result["findings"]), 1)

    def test_revision_feedback_is_filtered(self):
        analyst = FakeAnalyst()
        run(analyst, state=base_state(quality_feedback=["이해관계자: 투자 업계 근거 부족", "시장성: 규모 정의 누락"]))
        self.assertEqual(analyst.contexts[0]["revision_feedback"], ["이해관계자: 투자 업계 근거 부족"])


if __name__ == "__main__":
    unittest.main()
