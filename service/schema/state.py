from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict, Field


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


class TrlAssessment(TypedDict):
    level_or_range: str | None
    as_of: str
    confidence: Literal["높음", "중간", "낮음"]
    unverified_conditions: list[str]
    basis: str


class Finding(TypedDict):
    technology_ids: list[str]
    claim: str
    evidence_ids: list[str]
    is_inference: bool
    # 기술 조사 노드의 TRL 항목만 채우는 선택 필드 (설계서 4.1.3)
    trl_assessment: NotRequired[TrlAssessment]


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


# criterion과 next_queries는 LLM 응답 경계에서만 사용하고 공개 AgentResult에는 넣지 않는다.
class DraftFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    technology_ids: list[str]
    criterion: str = Field(description="입력 evaluation_criteria에 있는 항목 하나")
    claim: str
    evidence_ids: list[str]
    is_inference: bool


class AnalysisDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["complete", "partial", "error"]
    summary: str
    findings: list[DraftFinding]
    limitations: list[str]
    next_queries: list[str] = Field(description="기술 조사에서 부족한 근거를 찾을 수정 검색어, 최대 4개. 다른 역할은 빈 목록")
