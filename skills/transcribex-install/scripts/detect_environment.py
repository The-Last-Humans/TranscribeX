#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class GPU:
    name: str
    memory_mb: int | None


@dataclass
class Profile:
    id: str
    label: str
    asr_model: str
    device: str
    why: str
    caution: str | None = None


def main() -> int:
    args = parse_args()
    report = detect()
    if args.pretty:
        print_pretty(report)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect local TranscribeX install environment.")
    parser.add_argument("--pretty", action="store_true", help="Print a human-readable recommendation summary.")
    return parser.parse_args()


def detect() -> dict[str, Any]:
    system = platform.system()
    machine = platform.machine()
    docker = command_result(["docker", "--version"])
    compose = command_result(["docker", "compose", "version"])
    docker_running = command_ok(["docker", "info"])
    gpus = detect_nvidia_gpus()
    mem_gb = memory_gb()
    env = read_env(Path(".env"))

    facts = {
        "os": system,
        "machine": machine,
        "cpu_count": os.cpu_count(),
        "memory_gb": mem_gb,
        "docker": docker,
        "docker_compose": compose,
        "docker_running": docker_running,
        "nvidia_gpus": [asdict(gpu) for gpu in gpus],
        "existing_env": {key: env[key] for key in sorted(env) if key.startswith("TRANSCRIBEX_")},
    }
    return {
        "facts": facts,
        "recommended_profiles": [asdict(profile) for profile in recommend(system, machine, mem_gb, gpus)],
        "all_profiles": [asdict(profile) for profile in all_profiles()],
    }


def command_result(command: list[str]) -> str | None:
    if not shutil.which(command[0]):
        return None
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=8)
    except (OSError, subprocess.SubprocessError):
        return None
    return (completed.stdout or completed.stderr).strip() or None


def command_ok(command: list[str]) -> bool:
    if not shutil.which(command[0]):
        return False
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=8)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def detect_nvidia_gpus() -> list[GPU]:
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
    gpus: list[GPU] = []
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
        gpus.append(GPU(name=parts[0], memory_mb=memory))
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


def read_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def all_profiles() -> list[Profile]:
    return [
        Profile(
            id="cpu-balanced",
            label="CPU balanced Chinese meetings",
            asr_model="paraformer-zh",
            device="cpu",
            why="Best stable default for Chinese meeting transcription without NVIDIA GPU.",
        ),
        Profile(
            id="cpu-multilingual",
            label="CPU multilingual and Cantonese",
            asr_model="iic/SenseVoiceSmall",
            device="cpu",
            why="Better first choice for mixed Chinese/English/Cantonese or emotion/event tags.",
        ),
        Profile(
            id="gpu-balanced",
            label="GPU balanced default",
            asr_model="iic/SenseVoiceSmall",
            device="cuda",
            why="Strong general-purpose model when NVIDIA GPU is available.",
        ),
        Profile(
            id="gpu-chinese",
            label="GPU fast Mandarin production",
            asr_model="paraformer-zh",
            device="cuda",
            why="Fast mature route for Mandarin-focused meeting traffic.",
        ),
        Profile(
            id="gpu-accuracy",
            label="GPU higher accuracy LLM-ASR",
            asr_model="FunAudioLLM/Fun-ASR-Nano-2512",
            device="cuda",
            why="Higher-accuracy LLM-based ASR candidate for users with enough VRAM.",
            caution="Use GPU with at least 16GB VRAM; expect heavier dependencies and slower cold start.",
        ),
        Profile(
            id="gpu-experimental",
            label="GPU experimental Qwen3-ASR",
            asr_model="Qwen3-ASR-0.6B",
            device="cuda",
            why="Evaluation path for new Qwen3-ASR models.",
            caution="Treat as experimental until validated with the local service image and real audio.",
        ),
    ]


def recommend(system: str, machine: str, mem_gb: float | None, gpus: list[GPU]) -> list[Profile]:
    profiles = {profile.id: profile for profile in all_profiles()}
    if not gpus:
        choices = [profiles["cpu-balanced"], profiles["cpu-multilingual"]]
        if system == "Darwin" and machine in {"arm64", "aarch64"}:
            choices[0].caution = "Docker on Apple Silicon normally runs this service on CPU, not Apple GPU/MPS."
        if mem_gb is not None and mem_gb < 16:
            choices[0].caution = "Memory is below 16GB; keep one transcription at a time and avoid LLM-ASR."
        return choices

    max_vram = max((gpu.memory_mb or 0) for gpu in gpus)
    choices = [profiles["gpu-balanced"], profiles["gpu-chinese"]]
    if max_vram >= 16_000:
        choices.append(profiles["gpu-accuracy"])
    else:
        profiles["gpu-accuracy"].caution = "Detected VRAM below 16GB; only try this for experiments."
        choices.append(profiles["gpu-accuracy"])
    return choices


def print_pretty(report: dict[str, Any]) -> None:
    facts = report["facts"]
    print("Environment")
    print(f"  OS: {facts['os']} {facts['machine']}")
    print(f"  CPU cores: {facts['cpu_count']}")
    memory = f"{facts['memory_gb']} GB" if facts["memory_gb"] is not None else "unknown"
    print(f"  Memory: {memory}")
    print(f"  Docker: {facts['docker'] or 'not found'}")
    print(f"  Docker Compose: {facts['docker_compose'] or 'not found'}")
    print(f"  Docker running: {facts['docker_running']}")
    if facts["nvidia_gpus"]:
        print("  NVIDIA GPUs:")
        for gpu in facts["nvidia_gpus"]:
            memory = f"{gpu['memory_mb']} MB" if gpu["memory_mb"] else "unknown memory"
            print(f"    - {gpu['name']} ({memory})")
    else:
        print("  NVIDIA GPUs: none detected")

    print("\nRecommended profiles")
    for index, profile in enumerate(report["recommended_profiles"], start=1):
        print(f"  {index}. {profile['id']} - {profile['label']}")
        print(f"     model={profile['asr_model']} device={profile['device']}")
        print(f"     {profile['why']}")
        if profile.get("caution"):
            print(f"     Caution: {profile['caution']}")


if __name__ == "__main__":
    raise SystemExit(main())
