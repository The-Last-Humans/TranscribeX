from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GPUInfo:
    name: str
    memory_mb: int | None


@dataclass(frozen=True)
class DeviceFacts:
    os: str
    machine: str
    cpu_count: int | None
    memory_gb: float | None
    nvidia_gpus: list[GPUInfo]


@dataclass(frozen=True)
class ModelProfile:
    id: str
    label: str
    asr_model: str
    device: str
    vad_model: str | None
    punc_model: str | None
    spk_model: str | None
    batch_size_s: int
    why: str
    caution: str | None = None


def detect_device_facts() -> DeviceFacts:
    return DeviceFacts(
        os=platform.system(),
        machine=platform.machine(),
        cpu_count=os.cpu_count(),
        memory_gb=memory_gb(),
        nvidia_gpus=detect_nvidia_gpus(),
    )


def all_profiles() -> list[ModelProfile]:
    return [
        ModelProfile(
            id="cpu-balanced",
            label="CPU balanced Chinese meetings",
            asr_model="paraformer-zh",
            device="cpu",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            spk_model="cam++",
            batch_size_s=300,
            why="Stable default for Chinese meeting transcription without NVIDIA GPU.",
        ),
        ModelProfile(
            id="cpu-multilingual",
            label="CPU multilingual and Cantonese",
            asr_model="iic/SenseVoiceSmall",
            device="cpu",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            spk_model="cam++",
            batch_size_s=300,
            why="Better first choice for mixed Chinese, English, Cantonese, or emotion/event tags.",
        ),
        ModelProfile(
            id="gpu-balanced",
            label="GPU balanced default",
            asr_model="iic/SenseVoiceSmall",
            device="cuda",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            spk_model="cam++",
            batch_size_s=300,
            why="Strong general-purpose profile when an NVIDIA GPU is available.",
        ),
        ModelProfile(
            id="gpu-chinese",
            label="GPU fast Mandarin production",
            asr_model="paraformer-zh",
            device="cuda",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            spk_model="cam++",
            batch_size_s=300,
            why="Fast mature route for Mandarin-focused meeting traffic.",
        ),
        ModelProfile(
            id="gpu-accuracy",
            label="GPU higher accuracy LLM-ASR",
            asr_model="FunAudioLLM/Fun-ASR-Nano-2512",
            device="cuda",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            spk_model="cam++",
            batch_size_s=120,
            why="Higher-accuracy LLM-based ASR candidate for users with enough VRAM.",
            caution="Use GPU with at least 16GB VRAM; expect heavier dependencies and slower cold start.",
        ),
        ModelProfile(
            id="gpu-experimental",
            label="GPU experimental Qwen3-ASR",
            asr_model="Qwen3-ASR-0.6B",
            device="cuda",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            spk_model="cam++",
            batch_size_s=120,
            why="Evaluation path for new Qwen3-ASR models.",
            caution="Treat as experimental until validated with the local service image and real audio.",
        ),
    ]


def profile_by_id(profile_id: str) -> ModelProfile | None:
    return next((profile for profile in all_profiles() if profile.id == profile_id), None)


def recommend_profiles(facts: DeviceFacts | None = None) -> list[ModelProfile]:
    facts = facts or detect_device_facts()
    profiles = {profile.id: profile for profile in all_profiles()}
    if not facts.nvidia_gpus:
        choices = [profiles["cpu-balanced"], profiles["cpu-multilingual"]]
        if facts.os == "Darwin" and facts.machine in {"arm64", "aarch64"}:
            choices[0] = replace(
                choices[0],
                caution="Docker on Apple Silicon normally runs this service on CPU, not Apple GPU/MPS.",
            )
        if facts.memory_gb is not None and facts.memory_gb < 16:
            choices[0] = replace(
                choices[0],
                caution="Memory is below 16GB; keep one transcription at a time and avoid LLM-ASR.",
            )
        return choices

    max_vram = max((gpu.memory_mb or 0) for gpu in facts.nvidia_gpus)
    choices = [profiles["gpu-balanced"], profiles["gpu-chinese"]]
    if max_vram >= 16_000:
        choices.append(profiles["gpu-accuracy"])
    else:
        choices.append(
            replace(
                profiles["gpu-accuracy"],
                caution="Detected VRAM below 16GB; only try this profile for experiments.",
            )
        )
    return choices


def facts_to_dict(facts: DeviceFacts) -> dict[str, Any]:
    data = asdict(facts)
    data["nvidia_gpus"] = [asdict(gpu) for gpu in facts.nvidia_gpus]
    return data


def profile_to_dict(profile: ModelProfile) -> dict[str, Any]:
    return asdict(profile)


def detect_nvidia_gpus() -> list[GPUInfo]:
    if not shutil.which("nvidia-smi"):
        return []
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=8)
    except (OSError, subprocess.SubprocessError):
        return []

    gpus: list[GPUInfo] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if not parts or not parts[0]:
            continue
        memory = None
        if len(parts) > 1:
            try:
                memory = int(parts[1])
            except ValueError:
                memory = None
        gpus.append(GPUInfo(name=parts[0], memory_mb=memory))
    return gpus


def memory_gb() -> float | None:
    system = platform.system()
    if system == "Darwin":
        sysctl = shutil.which("sysctl") or "/usr/sbin/sysctl"
        value = command_result([sysctl, "-n", "hw.memsize"])
        if value and value.isdigit():
            return round(int(value) / (1024**3), 1)
    if system == "Linux":
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024**2), 1)
    return None


def command_result(command: list[str]) -> str | None:
    if not shutil.which(command[0]):
        return None
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=8)
    except (OSError, subprocess.SubprocessError):
        return None
    return (completed.stdout or completed.stderr).strip() or None
