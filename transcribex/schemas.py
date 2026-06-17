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
