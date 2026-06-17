from __future__ import annotations

import json
import logging
from functools import partial
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import anyio
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from transcribex import __version__
from transcribex.audio import create_job_dir, prepare_audio, probe_duration, remove_job_dir, save_upload
from transcribex.config import Settings, get_settings
from transcribex.funasr_engine import EngineManager
from transcribex.normalization import format_srt, normalize_funasr_result
from transcribex.schemas import HealthResponse, TranscriptionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()
engine_manager = EngineManager(settings)

app = FastAPI(
    title="TranscribeX",
    version=__version__,
    description="Self-hosted FunASR transcription API with timestamps and speaker diarization.",
)

if settings.allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.on_event("startup")
async def startup() -> None:
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    if settings.preload_model:
        await anyio.to_thread.run_sync(engine_manager.preload_default)


def require_auth(authorization: Annotated[str | None, Header()] = None) -> None:
    if not settings.api_key:
        return
    expected = f"Bearer {settings.api_key}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        default_model=settings.asr_model,
        vad_model=settings.vad_model,
        punc_model=settings.punc_model,
        spk_model=settings.spk_model,
        device=settings.device,
        model_loaded=engine_manager.is_default_loaded(),
    )


@app.post("/v1/audio/transcriptions", response_model=None)
async def create_transcription(
    _: Annotated[None, Depends(require_auth)],
    file: Annotated[UploadFile, File()],
    model: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    response_format: Annotated[str, Form()] = "verbose_json",
    diarize: Annotated[bool, Form()] = True,
    timestamps: Annotated[bool, Form()] = True,
    hotword: Annotated[str | None, Form()] = None,
    speaker_map: Annotated[str | None, Form()] = None,
) -> Response:
    response_format = response_format.strip().lower()
    if response_format not in {"verbose_json", "json", "text", "srt"}:
        raise HTTPException(
            status_code=400,
            detail="response_format must be one of: verbose_json, json, text, srt",
        )

    parsed_speaker_map = parse_speaker_map(speaker_map)
    job_dir = create_job_dir(settings.work_dir)
    try:
        uploaded = await save_upload(file, job_dir, settings.max_upload_mb)
        audio_path = await anyio.to_thread.run_sync(prepare_audio, uploaded, job_dir)
        duration = await anyio.to_thread.run_sync(probe_duration, audio_path)
        spec = engine_manager.default_spec(model=model, diarize=diarize)
        engine = engine_manager.get_engine(spec)
        raw_result = await anyio.to_thread.run_sync(
            partial(engine.transcribe, audio_path, language=language, hotword=hotword)
        )
        normalized = normalize_funasr_result(
            raw_result,
            model=spec.asr_model,
            duration=duration,
            language=language,
            include_speakers=diarize,
            include_timestamps=timestamps,
            speaker_map=parsed_speaker_map,
        )
        normalized.created_at = datetime.now(UTC)
    except RuntimeError as exc:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if not settings.keep_uploads:
            remove_job_dir(job_dir)

    return render_response(normalized, response_format)


def parse_speaker_map(value: str | None) -> dict[str, str] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="speaker_map must be valid JSON") from exc
    if not isinstance(parsed, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()):
        raise HTTPException(status_code=400, detail="speaker_map must be a JSON object of string keys and values")
    return parsed


def render_response(result: TranscriptionResponse, response_format: str) -> Response:
    if response_format == "text":
        return PlainTextResponse(result.text)
    if response_format == "srt":
        return PlainTextResponse(format_srt(result.segments), media_type="application/x-subrip")
    if response_format == "json":
        return JSONResponse({"text": result.text})
    return JSONResponse(result.model_dump(mode="json"))


def app_factory() -> FastAPI:
    return app
