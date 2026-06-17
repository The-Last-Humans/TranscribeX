---
name: transcribex-install
description: Install and configure the TranscribeX Docker transcription service. Use when the user wants to install TranscribeX, choose a FunASR model based on local hardware, configure CPU/GPU deployment, write the .env file, build/start Docker Compose, or verify the service before using the transcription skill.
---

# TranscribeX Install

Guide the user from environment detection to a working TranscribeX Docker service.

## Workflow

1. Run the environment detector from the TranscribeX project root:

```bash
python3 skills/transcribex-install/scripts/detect_environment.py --pretty
```

2. Explain the top 2-3 recommended profiles and their tradeoffs. Ask the user to choose one profile before writing configuration.
3. Apply the selected profile:

```bash
python3 skills/transcribex-install/scripts/configure_env.py --profile <profile>
```

4. Build and start the service:

```bash
docker compose up --build -d
```

5. Verify health:

```bash
curl -fsS http://127.0.0.1:8000/health
```

6. Tell the user how to invoke `$transcribex-transcribe` for actual transcription.

## Profiles

Use the detector output as the source of truth. These are the supported profile IDs:

| Profile | Primary model | Use when |
|---|---|---|
| `cpu-balanced` | `paraformer-zh` | Chinese meetings on CPU, stable default. |
| `cpu-multilingual` | `iic/SenseVoiceSmall` | CPU users with Chinese plus English/Cantonese/other supported languages. |
| `gpu-balanced` | `iic/SenseVoiceSmall` | NVIDIA GPU users who want a strong general first model. |
| `gpu-chinese` | `paraformer-zh` | NVIDIA GPU users focused on Mandarin meetings and speed. |
| `gpu-accuracy` | `FunAudioLLM/Fun-ASR-Nano-2512` | NVIDIA GPU users willing to spend more VRAM/latency for stronger ASR. |
| `gpu-experimental` | `Qwen3-ASR-0.6B` | Evaluation only; confirm compatibility before production. |

## Recommendation Rules

- Prefer `cpu-balanced` on Mac Docker, Apple Silicon, or machines without NVIDIA GPU.
- Prefer `cpu-multilingual` when the user mentions Cantonese, English-heavy meetings, or mixed languages and has no NVIDIA GPU.
- Prefer `gpu-balanced` as the default NVIDIA GPU recommendation.
- Offer `gpu-accuracy` when VRAM is at least 16GB and the user prioritizes recognition quality over throughput.
- Treat `Qwen3-ASR-0.6B`, `Qwen3-ASR-1.7B`, and `GLM-ASR-Nano` as evaluation candidates unless the user explicitly wants to experiment.
- Keep `cam++` enabled for speaker diarization unless the user says they do not need speakers.

## Configuration Script

`scripts/configure_env.py` writes `.env` in the current project root.

Examples:

```bash
python3 skills/transcribex-install/scripts/configure_env.py --profile cpu-balanced
python3 skills/transcribex-install/scripts/configure_env.py --profile gpu-accuracy --device cuda
python3 skills/transcribex-install/scripts/configure_env.py --asr-model iic/SenseVoiceSmall --device cpu
```

The script preserves unknown existing `.env` keys and updates TranscribeX keys.

## Safety

Do not overwrite `.env` until the user has chosen a profile or explicit model.

Do not run `docker compose up --build -d` until after showing the chosen profile and the main `.env` values that will be used.

If Docker is missing or not running, stop after explaining the prerequisite. Do not try to install Docker automatically unless the user explicitly asks.
