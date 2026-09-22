"""웹 근거의 내부 수집 기록과 인용 형식.

공용 Evidence 스키마는 바꾸지 않는다. 질의 방향·점수처럼 이 에이전트만 쓰는 정보는
RetrievalRecord로 따로 보관한다. 같은 Evidence ID에 에이전트마다 다른 값을 넣으면
pipeline.collect_sources가 충돌로 처리하기 때문이다.
"""

import hashlib
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from typing import Literal, TypedDict
from urllib.parse import urlparse

from service.schema.state import Evidence

QueryDirection = Literal["positive", "negative"]

CITATION_LABELS = {"market": "시장 자료", "stakeholder": "이해관계자 자료"}


class RetrievalRecord(TypedDict):
    evidence_id: str
    origin: Literal["web", "reused"]  # reused: technical_result.evidence 재인용
    technology_id: str
    criterion: str
    direction: QueryDirection | None  # 재인용 근거는 질의 방향이 없다
    query: str | None
    score: float | None
    via_alias: bool


def web_evidence_id(url: str, excerpt: str) -> str:
    # pipeline.search_web과 같은 규칙이다. 같은 URL의 서로 다른 발췌도 고유 ID로 유지한다.
    return "web_" + hashlib.sha256((url + "\n" + excerpt).encode()).hexdigest()[:16]


def normalize_published_date(raw: str | None) -> str | None:
    """Tavily의 ISO 또는 RFC 2822 날짜를 YYYY-MM-DD로 바꾼다. 해석할 수 없으면 None."""
    if not raw or not raw.strip():
        return None
    raw = raw.strip()
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(raw).date().isoformat()
    except (TypeError, ValueError):
        return None


def is_after_cutoff(published_at: str | None, end_date: str) -> bool:
    # 발행일 미확인 자료는 제외하지 않고 인용에 미확인으로 표시한다.
    if published_at is None:
        return False
    return date.fromisoformat(published_at) > date.fromisoformat(end_date)


def source_domain(url: str) -> str:
    host = urlparse(url).hostname or url
    return host.removeprefix("www.")


def web_citation(evidence: Evidence, perspective: str) -> str:
    """PDF 인용 [SW 문서, PDF p.7]과 대칭인 웹 인용 표기."""
    label = CITATION_LABELS[perspective]
    published = f"발행 {evidence['published_at']}" if evidence["published_at"] else "발행일 미확인"
    return f"[{label}, {source_domain(evidence['url'])} {published}]"
