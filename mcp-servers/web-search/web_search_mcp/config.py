from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WebSearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    log_level: str = Field(default="INFO")
    log_format: str = Field(default="plain")
