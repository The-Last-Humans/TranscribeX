from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from fastapi import HTTPException, UploadFile


def create_job_dir(work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    job_dir = work_dir / uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=False)
    return job_dir


async def save_upload(upload: UploadFile, job_dir: Path, max_upload_mb: int) -> Path:
    suffix = Path(upload.filename or "audio").suffix or ".audio"
    target = job_dir / f"input{suffix}"
    size = 0
    limit = max_upload_mb * 1024 * 1024

    with target.open("wb") as out:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise HTTPException(
                    status_code=413,
                    detail=f"Uploaded file exceeds TRANSCRIBEX_MAX_UPLOAD_MB={max_upload_mb}",
                )
            out.write(chunk)

    if size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    return target


def prepare_audio(input_path: Path, job_dir: Path) -> Path:
    output_path = job_dir / "audio_16k_mono.wav"
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        if input_path.suffix.lower() == ".wav":
            return input_path
        raise HTTPException(
            status_code=500,
            detail="ffmpeg is required for non-WAV uploads but was not found",
        )

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    run_command(command, "ffmpeg failed to convert uploaded audio")
    return output_path


def probe_duration(audio_path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, OSError):
        return None
    try:
        return float(completed.stdout.strip())
    except ValueError:
        return None


def remove_job_dir(job_dir: Path) -> None:
    shutil.rmtree(job_dir, ignore_errors=True)


def run_command(command: list[str], failure_message: str) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"{failure_message}: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or exc.stdout or "").strip()
        detail = failure_message if not stderr else f"{failure_message}: {stderr}"
        raise HTTPException(status_code=400, detail=detail) from exc


def copy_stream(source: BinaryIO, target: Path) -> None:
    with target.open("wb") as out:
        shutil.copyfileobj(source, out)
