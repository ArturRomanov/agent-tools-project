from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RagSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_embedding_model: str = Field(default="nomic-embed-text")
    qdrant_path: str = Field(default="./.data/qdrant")
    qdrant_collection_name: str = Field(default="rag_documents")
    rag_chunk_size: int = Field(default=800, ge=100, le=4000)
    rag_chunk_overlap: int = Field(default=120, ge=0, le=1000)
    rag_distance_metric: str = Field(default="cosine")
    log_level: str = Field(default="INFO")
    log_format: Literal["json", "plain"] = Field(default="plain")

    @field_validator("ollama_base_url", "ollama_embedding_model", "qdrant_path", "qdrant_collection_name")
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned

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
