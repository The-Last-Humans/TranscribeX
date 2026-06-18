from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock

from pydantic import BaseModel, Field, field_validator

from transcribex.config import Settings
from transcribex.profiles import ModelProfile, profile_by_id


class RuntimeConfig(BaseModel):
    version: int = 2
    setup_complete: bool = False
    profile_id: str | None = None
    asr_model: str
    vad_model: str | None = "fsmn-vad"
    punc_model: str | None = "ct-punc"
    spk_model: str | None = "cam++"
    hub: str | None = None
    device: str = "cpu"
    batch_size_s: int = Field(default=300, ge=1)
    preload_model: bool = False
    api_key: str | None = None
    max_upload_mb: int = Field(default=2048, ge=1)
    keep_uploads: bool = False
    updated_at: datetime | None = None

    @field_validator("vad_model", "punc_model", "spk_model", "hub", "api_key", mode="before")
    @classmethod
    def empty_string_to_none(cls, value: object) -> object:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value


class RuntimeConfigPatch(BaseModel):
    profile_id: str | None = None
    asr_model: str | None = None
    vad_model: str | None = None
    punc_model: str | None = None
    spk_model: str | None = None
    hub: str | None = None
    device: str | None = None
    batch_size_s: int | None = Field(default=None, ge=1)
    preload_model: bool | None = None
    api_key: str | None = None
    max_upload_mb: int | None = Field(default=None, ge=1)
    keep_uploads: bool | None = None
    setup_complete: bool = True

    @field_validator("vad_model", "punc_model", "spk_model", "hub", "api_key", mode="before")
    @classmethod
    def empty_string_to_none(cls, value: object) -> object:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value


class RuntimeConfigStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.path = settings.config_path
        self._lock = RLock()

    def current(self) -> RuntimeConfig:
        with self._lock:
            if self.path and self.path.exists():
                return RuntimeConfig.model_validate_json(self.path.read_text(encoding="utf-8"))
            return self.default_config(setup_complete=not self.settings.require_setup)

    def configured(self) -> bool:
        return self.current().setup_complete

    def apply(self, patch: RuntimeConfigPatch) -> RuntimeConfig:
        profile = self._profile_from_patch(patch)
        with self._lock:
            data = self.current().model_dump()
            if profile:
                data.update(profile_to_config_values(profile))
                data["profile_id"] = profile.id

            explicit = patch.model_dump(exclude_unset=True)
            explicit.pop("setup_complete", None)
            explicit.pop("profile_id", None)
            for key, value in explicit.items():
                data[key] = value

            data["setup_complete"] = patch.setup_complete
            data["updated_at"] = datetime.now(UTC)
            config = RuntimeConfig.model_validate(data)
            self._write(config)
            return config

    def default_config(self, *, setup_complete: bool) -> RuntimeConfig:
        return RuntimeConfig(
            setup_complete=setup_complete,
            asr_model=self.settings.asr_model,
            vad_model=self.settings.vad_model,
            punc_model=self.settings.punc_model,
            spk_model=self.settings.spk_model,
            hub=self.settings.hub,
            device=self.settings.device,
            batch_size_s=self.settings.batch_size_s,
            preload_model=self.settings.preload_model,
            api_key=self.settings.api_key,
            max_upload_mb=self.settings.max_upload_mb,
            keep_uploads=self.settings.keep_uploads,
        )

    def _profile_from_patch(self, patch: RuntimeConfigPatch) -> ModelProfile | None:
        if not patch.profile_id:
            return None
        profile = profile_by_id(patch.profile_id)
        if not profile:
            raise ValueError(f"Unknown setup profile: {patch.profile_id}")
        return profile

    def _write(self, config: RuntimeConfig) -> None:
        if not self.path:
            raise RuntimeError("TRANSCRIBEX_CONFIG_PATH is disabled; runtime setup cannot be persisted")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(config.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)


def profile_to_config_values(profile: ModelProfile) -> dict[str, object]:
    return {
        "asr_model": profile.asr_model,
        "vad_model": profile.vad_model,
        "punc_model": profile.punc_model,
        "spk_model": profile.spk_model,
        "device": profile.device,
        "batch_size_s": profile.batch_size_s,
    }
