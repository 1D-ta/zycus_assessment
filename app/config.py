from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    """Environment-backed settings. Paths default to the repo layout."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"
    llm_provider: str = "mock"
    llm_timeout_seconds: float = 45.0
    llm_max_retries: int = 5
    llm_backoff_base_seconds: float = 1.0

    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    knowledge_base_dir: Path = Field(default=PROJECT_ROOT / "knowledge-base")

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_backend: str = "onnx"
    rrf_k: int = 60
    retrieval_top_k: int = 5

    tam_window_days: int = 90
    tam_reference_now: str = ""

    @property
    def tickets_path(self) -> Path:
        return self.data_dir / "tickets.json"

    @property
    def accounts_path(self) -> Path:
        return self.data_dir / "accounts.json"

    @property
    def kb_dir(self) -> Path:
        nested = self.data_dir / "knowledge-base"
        if nested.is_dir():
            return nested
        return self.knowledge_base_dir


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
