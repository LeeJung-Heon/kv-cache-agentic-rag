from typing import Literal, NotRequired, TypedDict

from service.schema.state import AnalysisDraft, DraftFinding


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


class Finding(TypedDict):
    technology_ids: list[str]
    claim: str
    evidence_ids: list[str]
    is_inference: bool


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
    technologies: NotRequired[list[Technology]]
    technical_result: NotRequired[AgentResult]
    technical_retry_count: int
    technical_queries: NotRequired[list[str]]
    technical_missing_items: NotRequired[list[str]]
    market_result: NotRequired[AgentResult]
    stakeholder_result: NotRequired[AgentResult]
    domain_result: NotRequired[AgentResult]
    synthesis_result: NotRequired[AgentResult]
    report_markdown: NotRequired[str]
    report_evidence_ids: NotRequired[list[str]]
