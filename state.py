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


# criterion과 next_queries는 LLM 응답 경계에서만 사용하고 공개 AgentResult에는 넣지 않는다.
class DraftFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    technology_ids: list[str]
    criterion: str = Field(description="입력 evaluation_criteria에 있는 항목 하나")
    claim: str
    evidence_ids: list[str]
    is_inference: bool
    # strict 구조화 출력은 모든 필드를 required로 요구하므로 기본값 없이 nullable로 둔다.
    claim_type: ClaimType = Field(description="원문 진술의 성격: fact 사실, opinion 의견, forecast 전망")
    scope: MarketScope | None = Field(description="시장성: direct 직접 시장, adjacent 연관 시장. 다른 역할은 null")
    stage: AdoptionStage | None = Field(description="상용화·채택 단계: announced 계획 발표, pilot 실증, production 실제 운영. 해당 없으면 null")
    stance: Stance | None = Field(description="대상 기술에 대한 입장: positive, negative, mixed, unknown. 입장 판단이 아니면 null")


class AnalysisDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["complete", "partial", "error"]
    summary: str
    findings: list[DraftFinding]
    limitations: list[str]
    next_queries: list[str] = Field(description="기술 조사에서 부족한 근거를 찾을 수정 검색어, 최대 4개. 다른 역할은 빈 목록")
