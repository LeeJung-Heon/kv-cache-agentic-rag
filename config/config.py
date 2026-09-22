from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    tavily_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    openai_base_url: str | None = None
    embedding_device: str | None = None
    pdf_font: str | None = None

    # 서버 연결 정보 대신 로컬 FAISS 인덱스와 메타데이터를 저장할 경로를 관리한다.
    faiss_index_dir: Path = ROOT / ".cache" / "faiss"
    embedding_dimension: int = 1024


settings = Settings()
