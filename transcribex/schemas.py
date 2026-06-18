from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    default_model: str
    vad_model: str | None
    punc_model: str | None
    spk_model: str | None
    device: str
    model_loaded: bool
    setup_required: bool = False
    configured: bool = True
    config_path: str | None = None


class GPUInfoResponse(BaseModel):
    name: str
    memory_mb: int | None = None


class DeviceFactsResponse(BaseModel):
    os: str
    machine: str
    cpu_count: int | None = None
    memory_gb: float | None = None
    nvidia_gpus: list[GPUInfoResponse] = Field(default_factory=list)


class ModelProfileResponse(BaseModel):
    id: str
    label: str
    asr_model: str
    device: str
    vad_model: str | None = None
    punc_model: str | None = None
    spk_model: str | None = None
    batch_size_s: int
    why: str
    caution: str | None = None


class RuntimeConfigResponse(BaseModel):
    version: int = 2
    setup_complete: bool
    profile_id: str | None = None
    asr_model: str
    vad_model: str | None = None
    punc_model: str | None = None
    spk_model: str | None = None
    hub: str | None = None
    device: str
    batch_size_s: int
    preload_model: bool
    api_key_configured: bool
    max_upload_mb: int
    keep_uploads: bool
    updated_at: datetime | None = None


class SetupStatusResponse(BaseModel):
    setup_required: bool
    configured: bool
    config_path: str | None = None
    current: RuntimeConfigResponse
    facts: DeviceFactsResponse
    recommended_profiles: list[ModelProfileResponse]
    all_profiles: list[ModelProfileResponse]


class SetupApplyRequest(BaseModel):
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


class SetupApplyResponse(BaseModel):
    configured: bool
    setup_required: bool
    current: RuntimeConfigResponse


class Word(BaseModel):
    text: str
    start: float | None = None
    end: float | None = None
    speaker: str | None = None


class Segment(BaseModel):
    id: int
    start: float | None = None
    end: float | None = None
    speaker: str | None = None
    text: str
    words: list[Word] = Field(default_factory=list)


class TranscriptionResponse(BaseModel):
    text: str
    language: str | None = None
    duration: float | None = None
    model: str
    created_at: datetime | None = None
    segments: list[Segment] = Field(default_factory=list)
