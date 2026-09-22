from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
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
    faiss_index_dir: Path = ROOT / "artifacts" / "faiss"
    # 전처리 담당자가 확정한 JSONL 청크를 그대로 색인한다. 여기서 재청킹하지 않는다.
    preprocessed_chunks: Path = ROOT / "database" / "chunks.jsonl"
    document_manifest: Path = ROOT / "database" / "manifest.json"
    include_review_chunks: bool = False
    embedding_model: Literal["BAAI/bge-m3"] = "BAAI/bge-m3"
    embedding_revision: str | None = None
    embedding_dimension: int | None = Field(default=None, gt=0)  # 실제 출력에서 결정, 지정하면 일치 검사
    embedding_batch_size: int = Field(default=8, gt=0)
    retrieval_top_k: int = Field(default=5, gt=0)

    @field_validator("faiss_index_dir", "preprocessed_chunks", "document_manifest")
    @classmethod
    def project_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else ROOT / value


settings = Settings()
