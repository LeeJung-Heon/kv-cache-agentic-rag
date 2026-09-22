"""기술 조사 서브그래프 State와 LLM 응답 경계용 Pydantic 모델. 공통 형식은 팀 스키마를 쓴다."""
from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from service.schema.state import AgentResult, Evidence, Technology


class TechnicalResearchState(TypedDict):
    # 부모 GraphState와 공유하는 입력
    request: str
    target_domain: str
    evaluation_criteria: dict[str, list[str]]
    technologies: list[Technology]
    quality_feedback: NotRequired[list[str]]
    revision_count: NotRequired[int]
    # 부모로 쓰는 결과
    technical_result: NotRequired[AgentResult]
    # 아래는 서브그래프 내부 실행 필드다. 부모 GraphState에 없으므로 부모로 흘러가지 않고 재작업마다 새로 시작한다.
    technical_retry_count: NotRequired[int]
    technical_queries: NotRequired[list[str]]
    technical_missing_items: NotRequired[list[str]]
    technical_evidence: NotRequired[dict[str, Evidence]]
    technical_retrieval_failed: NotRequired[bool]


class TechnicalResearchOutput(TypedDict):
    technical_result: AgentResult
    technical_retry_count: int
    technical_queries: list[str]
    technical_missing_items: list[str]


# OpenAI strict JSON schema는 기본값 있는 선택 필드를 거부하므로 None을 명시적으로 받는다.
class TrlDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level_or_range: str | None = Field(description="'TRL 3' 또는 'TRL 4~6' 형식, 판단 불가면 null")
    as_of: str = Field(description="평가 기준일 YYYY-MM-DD, 입력의 assessment_date와 동일")
    confidence: Literal["높음", "중간", "낮음"]
    unverified_conditions: list[str] = Field(description="상위 단계 확인에 필요한 자료")


class TechnicalFindingDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    technology_ids: list[str]
    criterion: str = Field(description="입력 evaluation_criteria에 있는 항목 하나")
    claim: str
    evidence_ids: list[str]
    is_inference: bool
    trl_assessment: TrlDraft | None = Field(description="criterion이 TRL일 때만 채우고 아니면 null")


class TechnicalDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["complete", "partial", "error"]
    summary: str
    findings: list[TechnicalFindingDraft]
    limitations: list[str]
    next_queries: list[str] = Field(description="부족한 근거를 찾을 수정 검색어, 최대 4개")
