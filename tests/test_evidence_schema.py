import re
import unittest

from service.agent.tavily.evidence_schema import (is_after_cutoff, normalize_published_date, web_citation,
                                                  web_evidence_id)


def evidence(url, published_at):
    return {"id": "web_x", "source_type": "web", "title": "t", "url": url, "page": None,
            "published_at": published_at, "excerpt": "e"}


class EvidenceSchemaTest(unittest.TestCase):
    def test_web_id_matches_pipeline_citation_pattern(self):
        key = web_evidence_id("https://example.com/a", "excerpt")
        self.assertRegex(key, r"^web_[a-f0-9]{16}$")
        # pipeline.citation_ids의 웹 ID 패턴과 같아야 보고서 인용이 인식된다.
        self.assertTrue(re.fullmatch(r"web_[a-f0-9]+", key))
        self.assertNotEqual(key, web_evidence_id("https://example.com/a", "other excerpt"))

    def test_normalize_published_date(self):
        self.assertEqual(normalize_published_date("2026-05-01"), "2026-05-01")
        self.assertEqual(normalize_published_date("2026-05-01T09:30:00Z"), "2026-05-01")
        self.assertEqual(normalize_published_date("Fri, 01 May 2026 09:30:00 GMT"), "2026-05-01")
        self.assertIsNone(normalize_published_date(None))
        self.assertIsNone(normalize_published_date("   "))
        self.assertIsNone(normalize_published_date("last week"))

    def test_cutoff(self):
        self.assertFalse(is_after_cutoff("2026-09-21", "2026-09-21"))
        self.assertTrue(is_after_cutoff("2026-09-22", "2026-09-21"))
        self.assertFalse(is_after_cutoff(None, "2026-09-21"))

    def test_web_citation(self):
        self.assertEqual(web_citation(evidence("https://www.example.com/x", "2026-05-01"), "market"),
                         "[시장 자료, example.com 발행 2026-05-01]")
        self.assertEqual(web_citation(evidence("https://news.example.org/y", None), "stakeholder"),
                         "[이해관계자 자료, news.example.org 발행일 미확인]")


if __name__ == "__main__":
    unittest.main()
