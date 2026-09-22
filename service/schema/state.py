from typing import Literal, NotRequired, TypedDict


class Technology(TypedDict):
    id: str
    name: str
    approach: Literal["SW", "HW"]
    selection_reason: str


class Evidence(TypedDict):
    id: str
    source_type: Literal["paper", "web"]
    title: str
    url: str
    page: int | None
    published_at: str | None
    excerpt: str


# claim_type은 원문 진술의 성격, is_inference는 에이전트 해석 여부로 서로 독립이다.
# scope·stage·stance가 None이면 해당 관점에 적용되지 않는 항목이다.
ClaimType = Literal["fact", "opinion", "forecast"]
MarketScope = Literal["direct", "adjacent"]
AdoptionStage = Literal["announced", "pilot", "production"]
Stance = Literal["positive", "negative", "mixed", "unknown"]


class Finding(TypedDict):
    technology_ids: list[str]
    claim: str
    evidence_ids: list[str]
    is_inference: bool
    claim_type: ClaimType
    scope: MarketScope | None
    stage: AdoptionStage | None
    stance: Stance | None


class AgentResult(TypedDict):
    status: Literal["complete", "partial", "error"]
    summary: str
    findings: list[Finding]
    evidence: list[Evidence]
    limitations: list[str]


class GraphState(TypedDict):
    request: str
    target_domain: str
    evaluation_criteria: dict[str, list[str]]
    technologies: list[Technology]
    technical_result: NotRequired[AgentResult]
    market_result: NotRequired[AgentResult]
    stakeholder_result: NotRequired[AgentResult]
    domain_result: NotRequired[AgentResult]
    synthesis_result: NotRequired[AgentResult]
    quality_feedback: list[str]
    revision_count: int
    report_markdown: NotRequired[str]
    report_evidence_ids: NotRequired[list[str]]
