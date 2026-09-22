"""시장성·이해관계자 에이전트가 공유하는 Tavily 검색 래퍼.

설계서 기준 오류 정책: 질의 실패는 error로 중단하지 않고 limitations에 기록한다.
보강 검색까지 마친 뒤 근거가 없으면 해당 기술·기준을 판단 유보로 기록한다.
API 키 누락 같은 설정 오류는 판단 유보로 숨기지 않고 그대로 올린다.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urlparse, urlunparse

import httpx

from config.config import Settings
from service.agent.tavily.evidence_schema import (QueryDirection, RetrievalRecord, is_after_cutoff,
                                                  normalize_published_date, web_evidence_id)
from service.agent.tavily.query_templates import END_DATE, Perspective, Topic, build_query_pair, search_names
from service.schema.state import Evidence

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


def api_key() -> str:
    # Settings는 .env와 환경변수를 함께 읽는다. 테스트에서는 이 함수를 바꿔 실제 키 사용을 막는다.
    return Settings().tavily_api_key


def tavily_search(query: str, *, topic: Topic, end_date: str, depth: str = "basic",
                  max_results: int = MAX_RESULTS) -> dict:
    key = api_key()
    if not key:
        raise RuntimeError("TAVILY_API_KEY가 설정되지 않았습니다.")
    response = httpx.post(TAVILY_URL, timeout=45, headers={"Authorization": f"Bearer {key}"},
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
    # 보강 검색을 포함한 전체 질의 수. 모든 질의가 실패했는지(도구 오류) 판정에 쓴다.
    attempted: int = 0
    failed_count: int = 0
    # 방향별 질의 결과 score(중복 제거 전). 방향별 보강 검색 여부는 질의 자체의 결과 품질로 판단한다.
    hits: dict[str, list[float]] = field(default_factory=lambda: {"positive": [], "negative": []})

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
        self.attempted += other.attempted
        self.failed_count += other.failed_count
        for direction, scores in other.hits.items():
            self.hits[direction].extend(scores)


def run_queries(queries: list[tuple[QueryDirection, str]], *, topic: Topic, end_date: str = END_DATE,
                depth: str = "basic", technology_id: str, criterion: str, via_alias: bool = False,
                search: SearchFn = tavily_search) -> PairResult:
    result = PairResult()
    seen: set[str] = set()
    target = f"{technology_id} / {criterion}"
    # 긍정 질의를 먼저 실행하므로, 같은 URL이 양쪽에서 잡히면 긍정 방향 기록이 유지된다.
    for direction, query in queries:
        try:
            parsed = parse_results(search(query, topic=topic, end_date=end_date, depth=depth), end_date=end_date)
        except (httpx.HTTPError, TavilyResponseError) as exc:
            result.failed.append(direction)
            result.limitations.append(f"{target}: {DIRECTION_LABELS[direction]} 근거 수집 실패 ({failure_detail(exc)})")
            continue
        result.after_cutoff += parsed.after_cutoff
        result.hits[direction].extend(score for _, score in parsed.rows if score is not None)
        for evidence, score in parsed.rows:
            key = canonical_url(evidence["url"])
            if key in seen:
                continue
            seen.add(key)
            result.evidence.append(evidence)
            result.records.append({"evidence_id": evidence["id"], "origin": "web", "technology_id": technology_id,
                                   "criterion": criterion, "direction": direction, "query": query,
                                   "score": score, "via_alias": via_alias})
    result.attempted, result.failed_count = len(queries), len(result.failed)
    if len(queries) == 2 and len(result.failed) == 2:
        result.limitations = [f"{target}: 긍정·부정 질의 모두 실패 ({', '.join(result.limitations)})"]
    elif len(queries) == 2 and result.failed:
        result.limitations.append(f"{target}: 긍정/부정 근거 중 한쪽 수집 실패로 한쪽 방향 근거만 사용")
    return result


def search_pair(query_positive: str, query_negative: str, **kwargs) -> PairResult:
    return run_queries([("positive", query_positive), ("negative", query_negative)], **kwargs)


def is_weak(scores: list[float], *, top_k: int = WEAK_TOP_K, threshold: float = WEAK_SCORE_THRESHOLD) -> bool:
    if not scores:
        return True
    top = sorted(scores, reverse=True)[:top_k]
    return sum(top) / len(top) < threshold


def search_criterion(perspective: Perspective, criterion: str, technology_id: str, *, end_date: str = END_DATE,
                     depth: str = "basic", max_fallbacks: int = 1, search: SearchFn = tavily_search) -> PairResult:
    """1차 검색이 빈약할 때만 별칭으로 보강 검색한다. max_fallbacks는 추가 검색 쌍의 최대 횟수다.

    양방향 보강 뒤에도 한쪽 방향만 빈약하면 그 방향만 다음 별칭으로 한 번 더 검색한다(칸당 최대 +1회).
    2026-09-22 실측에서 시장 규모·성장성의 부정 질의 score가 낮아 반대 방향 근거가 없었기 때문이다.
    """
    names = search_names(technology_id)
    common = dict(end_date=end_date, depth=depth, technology_id=technology_id, criterion=criterion, search=search)
    pair = build_query_pair(perspective, criterion, technology_id)
    result = search_pair(pair.positive, pair.negative, topic=pair.topic, **common)
    next_alias = 1
    while next_alias < min(len(names), max_fallbacks + 1) and is_weak(result.scores):
        pair = build_query_pair(perspective, criterion, technology_id, next_alias)
        result.merge(search_pair(pair.positive, pair.negative, topic=pair.topic, via_alias=True, **common))
        next_alias += 1
    if next_alias < len(names):
        for direction, other in (("positive", "negative"), ("negative", "positive")):
            if is_weak(result.hits[direction]) and not is_weak(result.hits[other]):
                pair = build_query_pair(perspective, criterion, technology_id, next_alias)
                query = pair.positive if direction == "positive" else pair.negative
                result.merge(run_queries([(direction, query)], topic=pair.topic, via_alias=True, **common))
                break
    # 판단 유보는 질의 실패 여부가 아니라 보강 검색까지 마친 뒤 근거가 없을 때 기록한다.
    if not result.evidence:
        result.limitations.append(f"{technology_id} / {criterion}: 웹 근거 없음, 판단 유보")
    return result
