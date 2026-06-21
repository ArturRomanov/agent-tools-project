from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_size: str = "base"
    device: str = "auto"
    compute_type: str = "auto"
    models_dir: str = "./.data/models/whisper"
    max_audio_mb: int = 25
    cors_origins: str = "http://localhost:3000"

    model_config = {"env_prefix": "STT_"}


settings = Settings()
