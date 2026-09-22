import unittest

from service.agent.tavily.query_templates import (CRITERIA, END_DATE, QUERY_TEMPLATES, TECH_ALIASES,
                                                  build_query_pair)


class QueryTemplateTest(unittest.TestCase):
    def test_design_criteria(self):
        # 설계서 3.4절 기준. 레포 목업 CRITERIA와 혼동하지 않는다.
        self.assertEqual(CRITERIA["market"], ["시장 규모·성장성", "상용화·채택", "생태계"])
        self.assertEqual(CRITERIA["stakeholder"], ["경쟁 기술 진영", "도입사·개발자", "투자 업계"])

    def test_every_criterion_has_positive_negative_pair(self):
        for perspective, templates in QUERY_TEMPLATES.items():
            for criterion, template in templates.items():
                with self.subTest(perspective=perspective, criterion=criterion):
                    self.assertIn("{tech}", template["positive"])
                    self.assertIn("{tech}", template["negative"])
                    self.assertNotEqual(template["positive"], template["negative"])
                    self.assertIn(template["topic"], {"general", "news"})

    def test_build_query_pair_substitutes_primary_name(self):
        pair = build_query_pair("market", "상용화·채택", "hw_01")
        self.assertEqual(pair.positive, "CXL-PNM commercial deployment production adoption")
        self.assertEqual(pair.negative, "CXL-PNM adoption barriers delay not deployed")
        self.assertEqual(pair.topic, "news")

    def test_alias_query(self):
        pair = build_query_pair("stakeholder", "투자 업계", "sw_01", alias_index=1)
        self.assertTrue(pair.positive.startswith("Multi-head Latent Attention "))

    def test_invalid_inputs_raise(self):
        with self.assertRaises(KeyError):
            build_query_pair("market", "시장 수요", "hw_01")  # 레포 목업 기준명
        with self.assertRaises(KeyError):
            build_query_pair("domain", "생태계", "hw_01")
        with self.assertRaises(KeyError):
            build_query_pair("market", "생태계", "hw_99")
        with self.assertRaises(IndexError):
            build_query_pair("market", "생태계", "hw_01", alias_index=len(TECH_ALIASES["hw_01"]))

    def test_search_budget(self):
        # 기준 3 × 기술 2 × 관점 2 × 긍정/부정 2 = 1차 검색 24회
        calls = sum(len(t) for t in QUERY_TEMPLATES.values()) * len(TECH_ALIASES) * 2
        self.assertEqual(calls, 24)
        self.assertEqual(END_DATE, "2026-09-21")


if __name__ == "__main__":
    unittest.main()
