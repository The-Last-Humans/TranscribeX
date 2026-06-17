#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


PROFILES = {
    "cpu-balanced": {"TRANSCRIBEX_ASR_MODEL": "paraformer-zh", "TRANSCRIBEX_DEVICE": "cpu"},
    "cpu-multilingual": {"TRANSCRIBEX_ASR_MODEL": "iic/SenseVoiceSmall", "TRANSCRIBEX_DEVICE": "cpu"},
    "gpu-balanced": {"TRANSCRIBEX_ASR_MODEL": "iic/SenseVoiceSmall", "TRANSCRIBEX_DEVICE": "cuda"},
    "gpu-chinese": {"TRANSCRIBEX_ASR_MODEL": "paraformer-zh", "TRANSCRIBEX_DEVICE": "cuda"},
    "gpu-accuracy": {"TRANSCRIBEX_ASR_MODEL": "FunAudioLLM/Fun-ASR-Nano-2512", "TRANSCRIBEX_DEVICE": "cuda"},
    "gpu-experimental": {"TRANSCRIBEX_ASR_MODEL": "Qwen3-ASR-0.6B", "TRANSCRIBEX_DEVICE": "cuda"},
}

DEFAULTS = {
    "TRANSCRIBEX_VAD_MODEL": "fsmn-vad",
    "TRANSCRIBEX_PUNC_MODEL": "ct-punc",
    "TRANSCRIBEX_SPK_MODEL": "cam++",
    "TRANSCRIBEX_PRELOAD_MODEL": "false",
    "TRANSCRIBEX_API_KEY": "",
    "TRANSCRIBEX_MAX_UPLOAD_MB": "2048",
    "TRANSCRIBEX_KEEP_UPLOADS": "false",
    "TRANSCRIBEX_WORK_DIR": "/tmp/transcribex",
    "TRANSCRIBEX_MODEL_CACHE_DIR": "/models",
}


def main() -> int:
    args = parse_args()
    updates = DEFAULTS.copy()
    if args.profile:
        updates.update(PROFILES[args.profile])
    if args.asr_model:
        updates["TRANSCRIBEX_ASR_MODEL"] = args.asr_model
    if args.device:
        updates["TRANSCRIBEX_DEVICE"] = args.device
    if args.disable_speaker:
        updates["TRANSCRIBEX_SPK_MODEL"] = ""
    if args.preload_model is not None:
        updates["TRANSCRIBEX_PRELOAD_MODEL"] = "true" if args.preload_model else "false"
    if args.api_key is not None:
        updates["TRANSCRIBEX_API_KEY"] = args.api_key

    path = Path(args.env_file)
    existing = read_env_lines(path)
    merged = merge_env(existing, updates)
    if args.dry_run:
        print("".join(merged), end="")
        return 0
    path.write_text("".join(merged), encoding="utf-8")
    print(f"Wrote {path}")
    print(f"TRANSCRIBEX_ASR_MODEL={updates['TRANSCRIBEX_ASR_MODEL']}")
    print(f"TRANSCRIBEX_DEVICE={updates['TRANSCRIBEX_DEVICE']}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write TranscribeX .env settings.")
    parser.add_argument("--profile", choices=sorted(PROFILES), help="Recommended install profile")
    parser.add_argument("--asr-model", help="Override ASR model")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Override device")
    parser.add_argument("--disable-speaker", action="store_true", help="Disable cam++ speaker diarization")
    parser.add_argument("--preload-model", dest="preload_model", action="store_true", default=None)
    parser.add_argument("--no-preload-model", dest="preload_model", action="store_false")
    parser.add_argument("--api-key", help="Optional API bearer token")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_env_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def merge_env(lines: list[str], updates: dict[str, str]) -> list[str]:
    remaining = updates.copy()
    output: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            output.append(line)
            continue
        key, _ = stripped.split("=", 1)
        if key in remaining:
            output.append(f"{key}={remaining.pop(key)}\n")
        else:
            output.append(line if line.endswith("\n") else line + "\n")
    if output and output[-1].strip():
        output.append("\n")
    for key, value in remaining.items():
        output.append(f"{key}={value}\n")
    return output


if __name__ == "__main__":
    raise SystemExit(main())
