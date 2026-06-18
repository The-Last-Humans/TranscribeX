from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRANSCRIBEX_", env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "cpu"
    asr_model: str = "paraformer-zh"
    vad_model: str | None = "fsmn-vad"
    punc_model: str | None = "ct-punc"
    spk_model: str | None = "cam++"
    hub: str | None = None
    preload_model: bool = False
    api_key: str | None = None
    max_upload_mb: int = Field(default=2048, ge=1)
    keep_uploads: bool = False
    work_dir: Path = Path("/tmp/transcribex")
    model_cache_dir: Path | None = Path("/models")
    config_path: Path | None = Path(".transcribex/config.json")
    require_setup: bool = True
    admin_enabled: bool = True
    allowed_origins: list[str] = Field(default_factory=list)
    batch_size_s: int = 300

    @field_validator("vad_model", "punc_model", "spk_model", "hub", "api_key", "config_path", mode="before")
    @classmethod
    def empty_string_to_none(cls, value: object) -> object:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    def apply_cache_environment(self) -> None:
        if not self.model_cache_dir:
            return
        root = str(self.model_cache_dir)
        os.environ.setdefault("MODELSCOPE_CACHE", str(Path(root) / "modelscope"))
        os.environ.setdefault("HF_HOME", str(Path(root) / "huggingface"))


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.apply_cache_environment()
    return settings
