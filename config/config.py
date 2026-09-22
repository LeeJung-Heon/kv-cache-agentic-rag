from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
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
    processed_manifest: Path = ROOT / "data" / "processed" / "documents.json"
    embedding_model: Literal["BAAI/bge-m3"] = "BAAI/bge-m3"
    embedding_revision: str | None = None
    embedding_dimension: int | None = Field(default=None, gt=0)  # 실제 출력에서 결정, 지정하면 일치 검사
    embedding_batch_size: int = Field(default=8, gt=0)
    chunk_size: int = Field(default=400, gt=0)
    chunk_overlap: int = Field(default=0, ge=0)
    chunk_include_section: bool = True
    # 설계서의 비교 모델 모두에서 제목 포함 토큰 한도를 검사한다.
    chunk_tokenizer_models: list[str] = Field(default_factory=lambda: [
        "intfloat/multilingual-e5-large-instruct", "Qwen/Qwen3-Embedding-0.6B",
    ])
    retrieval_top_k: int = Field(default=5, gt=0)

    @field_validator("faiss_index_dir", "processed_manifest")
    @classmethod
    def project_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else ROOT / value

    @model_validator(mode="after")
    def check_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP은 CHUNK_SIZE보다 작아야 합니다.")
        return self


settings = Settings()
