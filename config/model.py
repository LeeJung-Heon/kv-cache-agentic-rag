from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from config import settings


def get_chat_model() -> BaseChatModel:
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        max_tokens=4096,
    )
