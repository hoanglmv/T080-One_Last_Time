from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_name: str = "Alternative Credit Scoring POC"
    app_env: Literal["development", "production", "test"] = "development"
    app_port: int = Field(default=8000, ge=1, le=65535)
    app_host: str = "0.0.0.0"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    cors_origins: str = "http://localhost:3000"

    # LLM
    openai_api_key: str = ""
    model_name: str = "gpt-4o-mini"
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    enable_llm_explanations: bool = True
    llm_explanation_timeout_seconds: float = Field(default=8.0, ge=1.0, le=30.0)

    # Credit scoring POC
    credit_model_path: str = "artifacts/models/credit_model.joblib"

    # Database
    database_url: str = "sqlite:///./data/app.db"

    # Vector Store
    chroma_persist_dir: str = "./data/chroma"


def get_settings() -> Settings:
    return Settings()
