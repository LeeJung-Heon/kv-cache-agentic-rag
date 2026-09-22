"""시장성·이해관계자 에이전트가 공유하는 Tavily 검색 래퍼.

설계서 기준 오류 정책: 질의 실패는 error로 중단하지 않고 limitations에 기록한다.
보강 검색까지 마친 뒤 근거가 없으면 해당 기술·기준을 판단 유보로 기록한다.
API 키 누락 같은 설정 오류는 판단 유보로 숨기지 않고 그대로 올린다.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urlparse, urlunparse

import httpx

from service.agent.tavily.evidence_schema import (QueryDirection, RetrievalRecord, is_after_cutoff,
                                                  normalize_published_date, web_evidence_id)
from service.agent.tavily.query_templates import END_DATE, Perspective, Topic, build_query_pair, search_names
from state import Evidence

TAVILY_URL = "https://api.tavily.com/search"
MAX_RESULTS = 5
MAX_EXCERPT = 6000
# 빈약함 기준: 결과 0건 또는 score 상위 3개 평균이 임계값 미만. 실제 응답을 보고 조정한다.
WEAK_TOP_K = 3
WEAK_SCORE_THRESHOLD = 0.5
DIRECTION_LABELS = {"positive": "긍정", "negative": "부정"}

SearchFn = Callable[..., dict]


class TavilyResponseError(ValueError):
    pass


def tavily_search(query: str, *, topic: Topic, end_date: str, depth: str = "basic",
                  max_results: int = MAX_RESULTS) -> dict:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY가 설정되지 않았습니다.")
    response = httpx.post(TAVILY_URL, timeout=45, headers={"Authorization": f"Bearer {api_key}"},
                          json={"query": query, "topic": topic, "search_depth": depth, "max_results": max_results,
                                "end_date": end_date, "include_raw_content": False,
                                # topic=general에서는 이 값이 없으면 published_date가 오지 않는다.
                                "include_published_date": True})
    response.raise_for_status()
    try:
        return response.json()
    except ValueError as exc:
        raise TavilyResponseError("Tavily 응답이 JSON이 아닙니다.") from exc


@dataclass
class ParsedResults:
    rows: list[tuple[Evidence, float | None]] = field(default_factory=list)
    after_cutoff: int = 0


def parse_results(payload: dict, *, end_date: str) -> ParsedResults:
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise TavilyResponseError("Tavily 응답에 results 목록이 없습니다.")
    parsed = ParsedResults()
    for row in payload["results"]:
        if not isinstance(row, dict):
            continue
        url, excerpt = row.get("url"), row.get("content")
        if not isinstance(url, str) or urlparse(url).scheme not in {"http", "https"}:
            continue
        if not isinstance(excerpt, str) or not excerpt.strip():
            continue
        published_at = normalize_published_date(row.get("published_date"))
        # end_date 파라미터로 1차 차단하지만, 새는 경우를 대비해 후처리로도 거른다.
        if is_after_cutoff(published_at, end_date):
            parsed.after_cutoff += 1
            continue
        excerpt = excerpt[:MAX_EXCERPT]
        score = row.get("score")
        parsed.rows.append(({"id": web_evidence_id(url, excerpt), "source_type": "web",
                             "title": row.get("title") or url, "url": url, "page": None,
                             "published_at": published_at, "excerpt": excerpt},
                            float(score) if isinstance(score, (int, float)) else None))
    return parsed


def canonical_url(url: str) -> str:
    parts = urlparse(url)
    return urlunparse((parts.scheme.lower(), (parts.hostname or "").removeprefix("www."),
                       parts.path.rstrip("/"), "", parts.query, ""))


def failure_detail(exc: Exception) -> str:
    # 요청 본문·인증 헤더가 섞일 수 있는 예외 메시지는 기록하지 않는다.
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return type(exc).__name__


@dataclass
class PairResult:
    evidence: list[Evidence] = field(default_factory=list)
    records: list[RetrievalRecord] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    failed: list[QueryDirection] = field(default_factory=list)
    after_cutoff: int = 0

    @property
    def scores(self) -> list[float]:
        return [r["score"] for r in self.records if r["score"] is not None]

    def merge(self, other: "PairResult") -> None:
        seen = {canonical_url(e["url"]) for e in self.evidence}
        for evidence, record in zip(other.evidence, other.records):
            if canonical_url(evidence["url"]) not in seen:
                seen.add(canonical_url(evidence["url"]))
                self.evidence.append(evidence)
                self.records.append(record)
        self.limitations.extend(other.limitations)
        self.after_cutoff += other.after_cutoff


def search_pair(query_positive: str, query_negative: str, *, topic: Topic, end_date: str = END_DATE,
                depth: str = "basic", technology_id: str, criterion: str, via_alias: bool = False,
                search: SearchFn = tavily_search) -> PairResult:
    result = PairResult()
    seen: set[str] = set()
    target = f"{technology_id} / {criterion}"
    # 긍정 질의를 먼저 실행하므로, 같은 URL이 양쪽에서 잡히면 긍정 방향 기록이 유지된다.
    for direction, query in (("positive", query_positive), ("negative", query_negative)):
        try:
            parsed = parse_results(search(query, topic=topic, end_date=end_date, depth=depth), end_date=end_date)
        except (httpx.HTTPError, TavilyResponseError) as exc:
            result.failed.append(direction)
            result.limitations.append(f"{target}: {DIRECTION_LABELS[direction]} 근거 수집 실패 ({failure_detail(exc)})")
            continue
        result.after_cutoff += parsed.after_cutoff
        for evidence, score in parsed.rows:
            key = canonical_url(evidence["url"])
            if key in seen:
                continue
            seen.add(key)
            result.evidence.append(evidence)
            result.records.append({"evidence_id": evidence["id"], "origin": "web", "technology_id": technology_id,
                                   "criterion": criterion, "direction": direction, "query": query,
                                   "score": score, "via_alias": via_alias})
    if len(result.failed) == 2:
        result.limitations = [f"{target}: 긍정·부정 질의 모두 실패 ({', '.join(result.limitations)})"]
    elif result.failed:
        result.limitations.append(f"{target}: 긍정/부정 근거 중 한쪽 수집 실패로 한쪽 방향 근거만 사용")
    return result


def is_weak(scores: list[float], *, top_k: int = WEAK_TOP_K, threshold: float = WEAK_SCORE_THRESHOLD) -> bool:
    if not scores:
        return True
    top = sorted(scores, reverse=True)[:top_k]
    return sum(top) / len(top) < threshold


def search_criterion(perspective: Perspective, criterion: str, technology_id: str, *, end_date: str = END_DATE,
                     depth: str = "basic", max_fallbacks: int = 1, search: SearchFn = tavily_search) -> PairResult:
    """1차 검색이 빈약할 때만 별칭으로 보강 검색한다. max_fallbacks는 추가 검색 쌍의 최대 횟수다."""
    pair = build_query_pair(perspective, criterion, technology_id)
    result = search_pair(pair.positive, pair.negative, topic=pair.topic, end_date=end_date, depth=depth,
                         technology_id=technology_id, criterion=criterion, search=search)
    aliases = range(1, min(len(search_names(technology_id)), max_fallbacks + 1))
    for alias_index in aliases:
        if not is_weak(result.scores):
            break
        pair = build_query_pair(perspective, criterion, technology_id, alias_index)
        result.merge(search_pair(pair.positive, pair.negative, topic=pair.topic, end_date=end_date, depth=depth,
                                 technology_id=technology_id, criterion=criterion, via_alias=True, search=search))
    # 판단 유보는 질의 실패 여부가 아니라 보강 검색까지 마친 뒤 근거가 없을 때 기록한다.
    if not result.evidence:
        result.limitations.append(f"{technology_id} / {criterion}: 웹 근거 없음, 판단 유보")
    return result
