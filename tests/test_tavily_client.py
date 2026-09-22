import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from service.agent.tavily.client import (TavilyResponseError, is_weak, parse_results, search_criterion,
                                         search_pair, tavily_search)
from service.agent.tavily.query_templates import build_query_pair

FIXTURES = Path(__file__).parent / "fixtures" / "tavily_responses"
END = "2026-09-21"


def fixture(name):
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def http_error(status):
    request = httpx.Request("POST", "https://api.tavily.com/search")
    return httpx.HTTPStatusError("error", request=request, response=httpx.Response(status, request=request))


class FakeSearch:
    """질의 문자열별로 fixture 이름 또는 예외를 돌려주는 가짜 Tavily."""

    def __init__(self, responses, default="empty"):
        self.responses, self.default, self.calls = responses, default, []

    def __call__(self, query, **kwargs):
        self.calls.append((query, kwargs))
        response = self.responses.get(query, self.default)
        if isinstance(response, Exception):
            raise response
        return fixture(response)


def run_pair(responses):
    search = FakeSearch(responses)
    return search_pair("pos", "neg", topic="news", end_date=END, technology_id="hw_01",
                       criterion="상용화·채택", search=search), search


class ParseResultsTest(unittest.TestCase):
    def test_filters_invalid_rows_and_after_cutoff(self):
        parsed = parse_results(fixture("commercial_positive"), end_date=END)
        urls = [evidence["url"] for evidence, _ in parsed.rows]
        self.assertEqual(urls, ["https://example-semi.com/news/cxl-pnm-sample", "https://www.example-dc.org/blog/cxl-pilot/"])
        self.assertEqual(parsed.after_cutoff, 1)

    def test_missing_published_date_is_none(self):
        rows = dict((e["url"], e) for e, _ in parse_results(fixture("commercial_positive"), end_date=END).rows)
        self.assertIsNone(rows["https://www.example-dc.org/blog/cxl-pilot/"]["published_at"])
        self.assertEqual(rows["https://example-semi.com/news/cxl-pnm-sample"]["published_at"], "2026-03-10")

    def test_evidence_shape_matches_state(self):
        evidence, score = parse_results(fixture("commercial_positive"), end_date=END).rows[0]
        self.assertEqual(set(evidence), {"id", "source_type", "title", "url", "page", "published_at", "excerpt"})
        self.assertEqual(evidence["source_type"], "web")
        self.assertIsNone(evidence["page"])
        self.assertTrue(evidence["id"].startswith("web_"))
        self.assertEqual(score, 0.82)

    def test_malformed_payload_raises(self):
        with self.assertRaises(TavilyResponseError):
            parse_results(fixture("malformed"), end_date=END)


class SearchPairTest(unittest.TestCase):
    def test_both_succeed_and_dedup_keeps_first_direction(self):
        result, search = run_pair({"pos": "commercial_positive", "neg": "commercial_negative"})
        self.assertEqual(len(search.calls), 2)
        self.assertEqual(result.failed, [])
        self.assertEqual(result.limitations, [])
        # www.·끝 슬래시만 다른 중복 URL은 먼저 잡힌 긍정 방향으로 한 번만 남는다.
        pilot = [r for r in result.records if "cxl-pilot" in next(
            e["url"] for e in result.evidence if e["id"] == r["evidence_id"])]
        self.assertEqual(len(pilot), 1)
        self.assertEqual(pilot[0]["direction"], "positive")
        self.assertEqual([r["direction"] for r in result.records], ["positive", "positive", "negative"])
        self.assertEqual(result.after_cutoff, 1)

    def test_passes_topic_and_end_date(self):
        _, search = run_pair({"pos": "empty", "neg": "empty"})
        self.assertEqual(search.calls[0][1], {"topic": "news", "end_date": END, "depth": "basic"})

    def test_one_side_failure_is_limitation_not_error(self):
        result, _ = run_pair({"pos": "commercial_positive", "neg": http_error(429)})
        self.assertEqual(result.failed, ["negative"])
        self.assertTrue(result.evidence)
        self.assertIn("hw_01 / 상용화·채택: 부정 근거 수집 실패 (HTTP 429)", result.limitations)
        self.assertTrue(any("한쪽 수집 실패" in item for item in result.limitations))

    def test_timeout_is_limitation(self):
        result, _ = run_pair({"pos": httpx.ReadTimeout("slow"), "neg": "commercial_negative"})
        self.assertEqual(result.failed, ["positive"])
        self.assertTrue(any("ReadTimeout" in item for item in result.limitations))

    def test_both_fail(self):
        result, _ = run_pair({"pos": http_error(500), "neg": "malformed"})
        self.assertEqual(result.failed, ["positive", "negative"])
        self.assertEqual(result.evidence, [])
        self.assertEqual(len(result.limitations), 1)
        self.assertIn("모두 실패", result.limitations[0])

    def test_error_detail_does_not_leak_message(self):
        result, _ = run_pair({"pos": httpx.ConnectError("SECRET_SENTINEL"), "neg": "empty"})
        self.assertNotIn("SECRET_SENTINEL", json.dumps(result.limitations, ensure_ascii=False))


class WeakAndFallbackTest(unittest.TestCase):
    def test_is_weak(self):
        self.assertTrue(is_weak([]))
        self.assertTrue(is_weak([0.9, 0.2, 0.1]))  # 평균 0.4
        # 입력 순서와 무관하게 score 상위 3개만 본다.
        self.assertFalse(is_weak([0.1, 0.82, 0.05, 0.71, 0.64]))

    def queries(self, alias_index=0):
        pair = build_query_pair("market", "상용화·채택", "hw_01", alias_index)
        return pair.positive, pair.negative

    def test_no_fallback_when_primary_is_strong(self):
        pos, neg = self.queries()
        search = FakeSearch({pos: "commercial_positive", neg: "commercial_negative"})
        result = search_criterion("market", "상용화·채택", "hw_01", search=search)
        self.assertEqual(len(search.calls), 2)
        self.assertFalse(any(r["via_alias"] for r in result.records))

    def test_fallback_with_alias_when_primary_is_weak(self):
        pos, neg = self.queries()
        alias_pos, alias_neg = self.queries(1)
        search = FakeSearch({pos: "weak", neg: "empty", alias_pos: "commercial_positive", alias_neg: "commercial_negative"})
        result = search_criterion("market", "상용화·채택", "hw_01", search=search)
        self.assertEqual(len(search.calls), 4)
        self.assertEqual([q for q, _ in search.calls][2:], [alias_pos, alias_neg])
        self.assertTrue(any(r["via_alias"] for r in result.records))
        self.assertFalse(any("판단 유보" in item for item in result.limitations))

    def test_fallback_is_capped(self):
        search = FakeSearch({}, default="empty")
        result = search_criterion("market", "상용화·채택", "hw_01", search=search, max_fallbacks=1)
        self.assertEqual(len(search.calls), 4)
        self.assertEqual(result.limitations, ["hw_01 / 상용화·채택: 웹 근거 없음, 판단 유보"])

    def test_alias_recovers_after_primary_failure(self):
        pos, neg = self.queries()
        alias_pos, alias_neg = self.queries(1)
        search = FakeSearch({pos: http_error(429), neg: http_error(429), alias_pos: "commercial_positive",
                             alias_neg: "commercial_negative"})
        result = search_criterion("market", "상용화·채택", "hw_01", search=search)
        self.assertTrue(result.evidence)
        self.assertTrue(any("모두 실패" in item for item in result.limitations))
        self.assertFalse(any("판단 유보" in item for item in result.limitations))


class TavilySearchConfigTest(unittest.TestCase):
    def test_missing_api_key_raises_instead_of_being_absorbed(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                tavily_search("q", topic="news", end_date=END)
        search = FakeSearch({"pos": RuntimeError("TAVILY_API_KEY"), "neg": "empty"})
        with self.assertRaises(RuntimeError):
            search_pair("pos", "neg", topic="news", technology_id="hw_01", criterion="상용화·채택", search=search)

    def test_request_payload(self):
        captured = {}

        def fake_post(url, **kwargs):
            captured.update(kwargs, url=url)
            return httpx.Response(200, json={"results": []}, request=httpx.Request("POST", url))

        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}), patch("httpx.post", fake_post):
            self.assertEqual(tavily_search("q", topic="news", end_date=END), {"results": []})
        self.assertEqual(captured["json"]["end_date"], END)
        self.assertEqual(captured["json"]["topic"], "news")
        self.assertEqual(captured["json"]["max_results"], 5)
        self.assertIs(captured["json"]["include_published_date"], True)

    def test_non_json_response(self):
        def fake_post(url, **kwargs):
            return httpx.Response(200, text="<html>", request=httpx.Request("POST", url))

        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}), patch("httpx.post", fake_post):
            with self.assertRaises(TavilyResponseError):
                tavily_search("q", topic="news", end_date=END)


if __name__ == "__main__":
    unittest.main()
