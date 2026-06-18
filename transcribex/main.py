from __future__ import annotations

import json
import logging
from functools import partial
from datetime import UTC, datetime
from importlib.resources import files
from typing import Annotated

import anyio
from fastapi import Body, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response

from transcribex import __version__
from transcribex.audio import create_job_dir, prepare_audio, probe_duration, remove_job_dir, save_upload
from transcribex.config import get_settings
from transcribex.funasr_engine import EngineManager
from transcribex.normalization import format_srt, normalize_funasr_result
from transcribex.profiles import all_profiles, detect_device_facts, facts_to_dict, profile_to_dict, recommend_profiles
from transcribex.runtime_config import RuntimeConfig, RuntimeConfigPatch, RuntimeConfigStore
from transcribex.schemas import (
    HealthResponse,
    RuntimeConfigResponse,
    SetupApplyRequest,
    SetupApplyResponse,
    SetupStatusResponse,
    TranscriptionResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()
config_store = RuntimeConfigStore(settings)
engine_manager = EngineManager()

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
    current = config_store.current()
    if current.preload_model:
        await anyio.to_thread.run_sync(partial(engine_manager.preload_default, current))


def require_auth(authorization: Annotated[str | None, Header()] = None) -> None:
    api_key = config_store.current().api_key
    if not api_key:
        return
    expected = f"Bearer {api_key}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


def require_management_auth(authorization: Annotated[str | None, Header()] = None) -> None:
    api_key = config_store.current().api_key
    if not api_key:
        return
    expected = f"Bearer {api_key}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/admin")


@app.get("/admin", include_in_schema=False)
async def admin() -> HTMLResponse:
    if not settings.admin_enabled:
        raise HTTPException(status_code=404, detail="Admin UI is disabled")
    return HTMLResponse(read_web_asset("admin.html"), media_type="text/html")


@app.get("/assets/admin.css", include_in_schema=False)
async def admin_css() -> Response:
    if not settings.admin_enabled:
        raise HTTPException(status_code=404, detail="Admin UI is disabled")
    return Response(read_web_asset("admin.css"), media_type="text/css")


@app.get("/assets/admin.js", include_in_schema=False)
async def admin_js() -> Response:
    if not settings.admin_enabled:
        raise HTTPException(status_code=404, detail="Admin UI is disabled")
    return Response(read_web_asset("admin.js"), media_type="application/javascript")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    current = config_store.current()
    return HealthResponse(
        status="ok",
        version=__version__,
        default_model=current.asr_model,
        vad_model=current.vad_model,
        punc_model=current.punc_model,
        spk_model=current.spk_model,
        device=current.device,
        model_loaded=engine_manager.is_default_loaded(current),
        setup_required=not current.setup_complete,
        configured=current.setup_complete,
        config_path=str(config_store.path) if config_store.path else None,
    )


@app.get("/v1/setup/status", response_model=SetupStatusResponse)
async def setup_status() -> SetupStatusResponse:
    return build_setup_status()


@app.post("/v1/setup/apply", response_model=SetupApplyResponse)
async def apply_setup(
    _: Annotated[None, Depends(require_management_auth)],
    request: Annotated[SetupApplyRequest, Body()],
) -> SetupApplyResponse:
    try:
        current = config_store.apply(RuntimeConfigPatch.model_validate(request.model_dump(exclude_unset=True)))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return SetupApplyResponse(
        configured=current.setup_complete,
        setup_required=not current.setup_complete,
        current=runtime_config_response(current),
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
    current = config_store.current()
    job_dir = create_job_dir(settings.work_dir)
    try:
        uploaded = await save_upload(file, job_dir, current.max_upload_mb)
        audio_path = await anyio.to_thread.run_sync(prepare_audio, uploaded, job_dir)
        duration = await anyio.to_thread.run_sync(probe_duration, audio_path)
        spec = engine_manager.default_spec(current, model=model, diarize=diarize)
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
        if not current.keep_uploads:
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


def build_setup_status() -> SetupStatusResponse:
    current = config_store.current()
    facts = detect_device_facts()
    return SetupStatusResponse(
        setup_required=not current.setup_complete,
        configured=current.setup_complete,
        config_path=str(config_store.path) if config_store.path else None,
        current=runtime_config_response(current),
        facts=facts_to_dict(facts),
        recommended_profiles=[profile_to_dict(profile) for profile in recommend_profiles(facts)],
        all_profiles=[profile_to_dict(profile) for profile in all_profiles()],
    )


def runtime_config_response(config: RuntimeConfig) -> RuntimeConfigResponse:
    return RuntimeConfigResponse(
        version=config.version,
        setup_complete=config.setup_complete,
        profile_id=config.profile_id,
        asr_model=config.asr_model,
        vad_model=config.vad_model,
        punc_model=config.punc_model,
        spk_model=config.spk_model,
        hub=config.hub,
        device=config.device,
        batch_size_s=config.batch_size_s,
        preload_model=config.preload_model,
        api_key_configured=bool(config.api_key),
        max_upload_mb=config.max_upload_mb,
        keep_uploads=config.keep_uploads,
        updated_at=config.updated_at,
    )


def read_web_asset(name: str) -> str:
    return files("transcribex.web").joinpath(name).read_text(encoding="utf-8")


def app_factory() -> FastAPI:
    return app
