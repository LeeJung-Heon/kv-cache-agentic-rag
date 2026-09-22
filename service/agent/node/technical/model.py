"""기술 조사 전용 모델 생성·설정과 LLM 호출. 결과 검증은 core에 맡긴다."""
import json

from langchain_openai import ChatOpenAI

from config import settings
from service.agent.node.technical.prompts import ASSESSMENT_DATE
from service.agent.node.technical.schema import TechnicalDraft, TechnicalResearchState
from service.schema.state import Evidence

# 기술 조사 전용 모델 설정. 모델명은 환경변수가 아니라 여기서 선언하며 인증·엔드포인트만 팀 settings를 따른다.
TECHNICAL_MODEL = "gpt-4.1-mini"
TECHNICAL_MODEL_SETTINGS = {"temperature": 0, "timeout": 120, "max_retries": 0, "max_tokens": 12000}


def create_technical_model() -> ChatOpenAI:
    """settings의 키·엔드포인트로 전용 모델을 만든다. settings에 값이 없으면 SDK가 프로세스 환경변수에서 찾는다."""
    kwargs = dict(TECHNICAL_MODEL_SETTINGS)
    if settings.openai_api_key:
        kwargs["api_key"] = settings.openai_api_key
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return ChatOpenAI(model=TECHNICAL_MODEL, **kwargs)


def analyze_technical(analyst, state: TechnicalResearchState, sources: dict[str, Evidence], *, system: str,
                      validation_feedback: list[str] = ()) -> TechnicalDraft:
    context = {"request": state["request"], "target_domain": state["target_domain"],
               "technologies": state["technologies"], "evaluation_criteria": state["evaluation_criteria"]["technical"],
               "assessment_date": ASSESSMENT_DATE, "evidence": list(sources.values()),
               "technical_result": state.get("technical_result"),
               "missing_items": state.get("technical_missing_items", []),
               "quality_feedback": state.get("quality_feedback", []),
               "revision_count": state.get("revision_count", 0),
               "validation_feedback": list(validation_feedback)}
    return analyst.invoke([("system", system), ("human", json.dumps(context, ensure_ascii=False))])
