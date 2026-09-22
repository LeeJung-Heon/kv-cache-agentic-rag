from urllib.parse import quote_plus

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "kv_cache"
    db_user: str = "postgres"
    db_password: str = "postgres"
    pgvector_dimension: int = 1024

    @computed_field
    @property
    def database_url(self) -> str:
        user = quote_plus(self.db_user)
        password = quote_plus(self.db_password)
        return (
            f"postgresql+psycopg://{user}:{password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


settings = Settings()
