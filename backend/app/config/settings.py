from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_chat_model: str = Field(default="gpt-oss:120b")
    ollama_embedding_model: str = Field(default="nomic-embed-text")
    ollama_temperature: float = Field(default=0.0)
    ollama_timeout_seconds: float | None = Field(default=None)
    qdrant_path: str = Field(default="./.data/qdrant")
    qdrant_collection_name: str = Field(default="rag_documents")
    rag_chunk_size: int = Field(default=800, ge=100, le=4000)
    rag_chunk_overlap: int = Field(default=120, ge=0, le=1000)
    rag_top_k_default: int = Field(default=5, ge=1, le=20)
    rag_distance_metric: str = Field(default="cosine")
    log_level: str = Field(default="INFO")
    log_format: Literal["json", "plain"] = Field(default="json")
    log_payload_mode: Literal["metadata", "sanitized", "full"] = Field(default="sanitized")
    log_payload_max_chars: int = Field(default=500, ge=50, le=5000)
    log_include_uvicorn_access: bool = Field(default=True)
    cors_allow_origins: str = Field(default="http://localhost:3000")
    memory_sqlite_path: str = Field(default="./.data/agent_memory.db")
    memory_qdrant_collection: str = Field(default="agent_memory")
    memory_context_limit_tokens: int = Field(default=6000, ge=1000, le=128000)
    memory_keep_recent_turns: int = Field(default=6, ge=2, le=30)
    memory_recent_turn_window: int = Field(default=12, ge=4, le=100)
    memory_top_k: int = Field(default=5, ge=1, le=20)

    @field_validator(
        "ollama_base_url",
        "ollama_chat_model",
        "ollama_embedding_model",
        "qdrant_path",
        "qdrant_collection_name",
        "memory_sqlite_path",
        "memory_qdrant_collection",
    )
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned

    @field_validator("ollama_temperature")
    @classmethod
    def _temperature_range(cls, value: float) -> float:
        if value < 0.0 or value > 2.0:
            raise ValueError("must be between 0.0 and 2.0")
        return value

    @field_validator("ollama_timeout_seconds")
    @classmethod
    def _timeout_positive(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("must be > 0")
        return value

    @field_validator("rag_chunk_overlap")
    @classmethod
    def _chunk_overlap_reasonable(cls, value: int, info) -> int:
        chunk_size = info.data.get("rag_chunk_size")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError("must be smaller than rag_chunk_size")
        return value

    @field_validator("rag_distance_metric")
    @classmethod
    def _normalize_distance_metric(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned

    def cors_allow_origins_list(self) -> list[str]:
        raw = self.cors_allow_origins.strip()
        if not raw:
            return []
        return [item.strip() for item in raw.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
