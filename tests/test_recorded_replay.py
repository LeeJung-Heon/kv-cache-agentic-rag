"""2026-09-22 실제 Tavily 응답 축약본(recorded/)을 재생해 검색 흐름을 API 없이 재현한다."""
import json
import unittest
from pathlib import Path

from service.agent.tavily.client import is_weak, parse_results, search_criterion
from service.agent.tavily.query_templates import END_DATE

RECORDED = Path(__file__).parent / "tavily_fixtures" / "recorded"


def load_recorded():
    return {data["request"]["query"]: data for data in
            (json.loads(p.read_text(encoding="utf-8")) for p in sorted(RECORDED.glob("*.json")))}


class ReplaySearch:
    def __init__(self):
        self.recorded, self.calls = load_recorded(), []

    def __call__(self, query, **kwargs):
        self.calls.append(query)
        if query not in self.recorded:
            raise AssertionError(f"녹화되지 않은 질의: {query}")
        return self.recorded[query]["response"]


@unittest.skipUnless(RECORDED.exists() and any(RECORDED.glob("*.json")), "녹화 응답 없음")
class RecordedReplayTest(unittest.TestCase):
    def test_recorded_responses_parse(self):
        for query, data in load_recorded().items():
            with self.subTest(query=query):
                parsed = parse_results(data["response"], end_date=END_DATE)
                self.assertEqual(parsed.after_cutoff, 0)
                self.assertTrue(all(e["published_at"] for e, _ in parsed.rows))  # include_published_date 효과
                self.assertTrue(all(len(e["excerpt"]) <= 200 for e, _ in parsed.rows))  # 축약본만 커밋

    def test_primary_name_swap_reduces_fallback(self):
        # 실측: "CXL-PNM" 1차 검색은 3개 기준 모두 빈약해 12회 호출.
        # "CXL processing-near-memory"를 1차로 바꾸면 상용화·채택만 양방향 보강 검색하고,
        # 시장 규모·성장성은 부정 방향만 빈약해 부정 질의 1회만 보강해 9회가 된다.
        search = ReplaySearch()
        results = {c: search_criterion("market", c, "hw_01", search=search)
                   for c in ("시장 규모·성장성", "상용화·채택", "생태계")}
        self.assertEqual(len(search.calls), 9)
        size = results["시장 규모·성장성"]
        self.assertFalse(any(r["via_alias"] for r in size.records if r["direction"] == "positive"))
        self.assertIn("CXL-PNM market uncertainty slowdown limited demand", search.calls)
        self.assertTrue(any(r["via_alias"] for r in results["상용화·채택"].records))
        self.assertTrue(all(result.evidence for result in results.values()))

    def test_news_scores_are_lower_than_general(self):
        # 실측에서 news 토픽 score는 general보다 전반적으로 낮다. 임계값 0.5에서는 news 기준이 자주 보강 검색된다.
        from statistics import median

        def scores(topic):
            return [r.get("score", 0) for d in load_recorded().values() if d["request"]["topic"] == topic
                    for r in d["response"]["results"]]
        self.assertLess(median(scores("news")), median(scores("general")))
        self.assertTrue(is_weak(scores("news")) or median(scores("news")) < 0.5)

if __name__ == "__main__":
    unittest.main()
