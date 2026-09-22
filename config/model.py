"""여러 에이전트가 공유하는 채팅 모델 생성 함수."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from config import settings


def get_chat_model() -> BaseChatModel:
    """환경 설정을 사용해 기본 LLM 클라이언트를 생성한다."""
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        max_tokens=4096,
    )
