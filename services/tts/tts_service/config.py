from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    device: str = "auto"
    dtype: str = "auto"
    language: str = "English"
    speaker: str = "Ryan"
    attn_implementation: str = "auto"
    max_text_chars: int = 5000
    cors_origins: str = "http://localhost:3000"

    model_config = {"env_prefix": "TTS_"}


settings = Settings()
