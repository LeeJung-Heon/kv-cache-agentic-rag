"""에이전트가 공유하는 State와 구조화된 LLM 출력 스키마."""

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


# 시장성·이해관계자 판단 구분 (설계서 3.2·3.4). claim_type은 원문 진술의 성격, is_inference는 에이전트 해석 여부로 서로 독립이다.
# None이면 해당 관점에 적용되지 않는 항목이다.
ClaimType = Literal["fact", "opinion", "forecast"]
MarketScope = Literal["direct", "adjacent"]
AdoptionStage = Literal["announced", "pilot", "production"]
Stance = Literal["positive", "negative", "mixed", "unknown"]


class Finding(TypedDict):
    technology_ids: list[str]
    claim: str
    evidence_ids: list[str]
    is_inference: bool
    # 기술 조사 노드의 TRL 항목만 채우는 선택 필드 (설계서 4.1.3)
    trl_assessment: NotRequired[TrlAssessment]
    # AnalysisDraft를 쓰는 노드(시장성·이해관계자·도메인)가 채우는 선택 필드. 기술 조사 노드는 채우지 않는다.
    claim_type: NotRequired[ClaimType | None]
    scope: NotRequired[MarketScope | None]
    stage: NotRequired[AdoptionStage | None]
    stance: NotRequired[Stance | None]


class AgentResult(TypedDict):
    status: Literal["complete", "partial", "error"]
    summary: str
    findings: list[Finding]
    evidence: list[Evidence]
    limitations: list[str]


class GraphState(TypedDict):
    # 실행 중 바뀌지 않는 공통 입력
    request: str
    target_domain: str
    evaluation_criteria: dict[str, list[str]]
    technologies: list[Technology]
    # 각 에이전트는 자신이 담당하는 결과 필드만 갱신한다.
    technical_result: NotRequired[AgentResult]
    market_result: NotRequired[AgentResult]
    stakeholder_result: NotRequired[AgentResult]
    domain_result: NotRequired[AgentResult]
    synthesis_result: NotRequired[AgentResult]
    # 종합 평가가 남긴 피드백과 전체 재작업 횟수
    quality_feedback: list[str]
    revision_count: int
    # 보고서 생성 노드의 최종 출력
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
    # OpenAI strict 변환은 default=None을 제거하고 모든 필드를 required로 보내므로 LLM은 네 필드를 항상 출력한다.
    # 기본값은 기존 코드·fixture가 새 필드 없이도 DraftFinding을 만들 수 있게 하기 위한 것이다.
    claim_type: ClaimType | None = Field(default=None, description="원문 진술의 성격: fact 사실, opinion 의견, forecast 전망")
    scope: MarketScope | None = Field(default=None, description="direct 대상 기술 자체, adjacent 상위 기술·연관 시장. 해당 없으면 null")
    stage: AdoptionStage | None = Field(default=None, description="상용화·채택 단계: announced 계획 발표, pilot 실증, production 실제 운영. 해당 없으면 null")
    stance: Stance | None = Field(default=None, description="대상 기술에 대한 입장: positive, negative, mixed, unknown. 입장 판단이 아니면 null")


class AnalysisDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["complete", "partial", "error"]
    summary: str
    findings: list[DraftFinding]
    limitations: list[str]
    next_queries: list[str] = Field(description="기술 조사에서 부족한 근거를 찾을 수정 검색어, 최대 4개. 다른 역할은 빈 목록")
