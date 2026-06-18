# TranscribeX

TranscribeX is a self-hosted meeting transcription service built around FunASR. It exposes an OpenAI-style transcription endpoint, a v2 management UI with first-run setup, and companion Codex skills so agents can configure the service and submit audio files to the local API.

The default pipeline targets Chinese meetings:

```text
audio -> ffmpeg normalization -> FunASR VAD -> ASR -> punctuation -> speaker diarization -> JSON/SRT/text
```

Default models:

- ASR: `paraformer-zh`
- VAD: `fsmn-vad`
- punctuation: `ct-punc`
- speaker diarization: `cam++`

## Quick Start

### Packaged Docker Image

The intended v2 deployment path is a ready-to-run Docker image. Users do not need a source checkout:

```bash
docker run -d --name transcribex \
  -p 8000:8000 \
  -v transcribex-models:/models \
  -v transcribex-config:/config \
  ghcr.io/the-last-humans/transcribex:latest
```

Open the management UI:

```text
http://127.0.0.1:8000/admin
```

On first start, the UI shows a setup wizard. It detects the container device facts, recommends model profiles, and writes the selected runtime configuration to `/config/config.json`.

The repository includes a GitHub Actions workflow that publishes this image to GHCR on pushes to `main`, version tags, or manual workflow dispatch.

### Build From Source

```bash
cp .env.example .env
docker compose up --build
```

The first request downloads FunASR models into the `models` volume, so it can take several minutes.

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Setup status:

```bash
curl http://127.0.0.1:8000/v1/setup/status
```

Transcribe a file:

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@meeting.wav \
  -F response_format=verbose_json
```

If `TRANSCRIBEX_API_KEY` is set, include:

```bash
-H "Authorization: Bearer $TRANSCRIBEX_API_KEY"
```

## Agent Install Prompt

Copy this prompt into Codex or another local coding agent to clone TranscribeX from GitHub and install it on the current machine:

```text
Install TranscribeX from GitHub and verify it is ready for local meeting transcription.

Repository: https://github.com/The-Last-Humans/TranscribeX

Steps:
1. Clone the repository if it is not already present:
   git clone https://github.com/The-Last-Humans/TranscribeX.git
2. Enter the project directory and inspect README.md plus skills/transcribex-install/SKILL.md.
3. Run:
   python3 skills/transcribex-install/scripts/detect_environment.py --pretty
4. Choose the best supported profile for this machine. Prefer cpu-balanced unless an NVIDIA GPU is available; use cpu-multilingual when Chinese/English/Cantonese mixed meetings are important.
5. Write .env with the selected profile:
   python3 skills/transcribex-install/scripts/configure_env.py --profile <profile>
6. Build and start the Docker service:
   docker compose up --build -d
7. Verify:
   curl -fsS http://127.0.0.1:8000/health
8. Complete v2 setup. Prefer API setup so the same flow works against a packaged Docker image:
   python3 skills/transcribex-install/scripts/setup_service.py --auto
   If the user wants a browser UI, open http://127.0.0.1:8000/admin instead.
9. Install the bundled skills by copying or symlinking skills/transcribex-install and skills/transcribex-transcribe into ${CODEX_HOME:-$HOME/.codex}/skills if the user wants Codex to invoke them automatically.
10. Show the user how to transcribe an audio file:
   python3 skills/transcribex-transcribe/scripts/transcribe_audio.py /path/to/meeting.wav --format markdown

Do not install Docker automatically unless the user explicitly asks. If Docker is missing or not running, explain the prerequisite and stop.
```

## API

### `GET /health`

Returns service status, current model defaults, and whether the model has been loaded.

The response also includes `setup_required`, `configured`, and `config_path`.

### `GET /v1/setup/status`

Returns:

- current persisted runtime configuration
- first-run setup status
- detected device facts
- recommended profiles
- all supported profiles

### `POST /v1/setup/apply`

Applies setup from a profile and optional overrides. If the current config has an API key, this endpoint requires `Authorization: Bearer <key>`.

```bash
curl http://127.0.0.1:8000/v1/setup/apply \
  -H "Content-Type: application/json" \
  -d '{"profile_id":"cpu-balanced","setup_complete":true}'
```

To configure a running service from an agent:

```bash
python3 skills/transcribex-install/scripts/setup_service.py --auto
python3 skills/transcribex-install/scripts/setup_service.py --profile cpu-multilingual
```

### `POST /v1/audio/transcriptions`

Multipart form fields:

| Field | Required | Default | Description |
|---|---:|---|---|
| `file` | yes | | Audio/video file to transcribe. |
| `model` | no | `TRANSCRIBEX_ASR_MODEL` | FunASR ASR model name or path. |
| `language` | no | | Optional language hint passed to FunASR. |
| `response_format` | no | `verbose_json` | `verbose_json`, `json`, `text`, or `srt`. |
| `diarize` | no | `true` | Include speaker labels when configured. |
| `timestamps` | no | `true` | Include segment timestamps. |
| `hotword` | no | | FunASR hotword string, for example `项目名 20 产品名 20`. |
| `speaker_map` | no | | JSON map for display names, for example `{"SPEAKER_00":"张三","SPEAKER_01":"李四"}`. |

Verbose response shape:

```json
{
  "text": "完整转录文本",
  "language": "zh",
  "duration": 128.42,
  "model": "paraformer-zh",
  "segments": [
    {
      "id": 0,
      "start": 0.4,
      "end": 3.8,
      "speaker": "SPEAKER_00",
      "text": "我们先看一下今天的议程。",
      "words": []
    }
  ]
}
```

## Configuration

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `TRANSCRIBEX_DEVICE` | `cpu` | `cpu`, `cuda`, or another torch device string. |
| `TRANSCRIBEX_ASR_MODEL` | `paraformer-zh` | Default ASR model. Try `iic/SenseVoiceSmall` for mixed Chinese, English, Cantonese, emotion/event tags. |
| `TRANSCRIBEX_VAD_MODEL` | `fsmn-vad` | VAD model. Empty string disables VAD. |
| `TRANSCRIBEX_PUNC_MODEL` | `ct-punc` | Punctuation model. Empty string disables punctuation. |
| `TRANSCRIBEX_SPK_MODEL` | `cam++` | Speaker diarization model. Empty string disables diarization. |
| `TRANSCRIBEX_PRELOAD_MODEL` | `false` | Load default model during startup instead of first request. |
| `TRANSCRIBEX_API_KEY` | empty | Optional bearer token required by API. |
| `TRANSCRIBEX_MAX_UPLOAD_MB` | `2048` | Upload size limit checked after saving. |
| `TRANSCRIBEX_KEEP_UPLOADS` | `false` | Keep temporary uploaded and converted files for debugging. |
| `TRANSCRIBEX_WORK_DIR` | `/tmp/transcribex` | Temporary working directory. |
| `TRANSCRIBEX_MODEL_CACHE_DIR` | `/models` | Model cache root used by Docker. |
| `TRANSCRIBEX_CONFIG_PATH` | `.transcribex/config.json` locally, `/config/config.json` in Docker | Runtime setup JSON written by the UI/API. Empty disables persisted setup. |
| `TRANSCRIBEX_REQUIRE_SETUP` | `true` | When no runtime config exists, mark the service as requiring first-run setup. |
| `TRANSCRIBEX_ADMIN_ENABLED` | `true` | Serve the packaged management UI at `/admin`. |

Runtime setup values override these defaults after `/v1/setup/apply` writes the config file.

## GPU

Use a CUDA PyTorch wheel at build time and run with NVIDIA Container Toolkit:

```bash
docker build \
  --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
  -t transcribex:cuda .

docker run --gpus all -p 8000:8000 \
  -e TRANSCRIBEX_DEVICE=cuda \
  -v transcribex-models:/models \
  -v transcribex-config:/config \
  transcribex:cuda
```

## Companion Skill

The Codex transcription skill lives at `skills/transcribex-transcribe`. Install it by copying or symlinking that folder into your Codex skills directory, then invoke `$transcribex-transcribe`.

The skill uses `TRANSCRIBEX_URL` and `TRANSCRIBEX_API_KEY` when present. Default URL is `http://127.0.0.1:8000`.

The install/setup skill lives at `skills/transcribex-install`. It can either write `.env` for source builds or configure an already running packaged image through the setup API:

```bash
python3 skills/transcribex-install/scripts/setup_service.py --show
python3 skills/transcribex-install/scripts/setup_service.py --auto
```

## Local Development

Install only lightweight test dependencies if you only want to run unit tests:

```bash
python3 -m unittest discover -s tests
```

Install the full service locally:

```bash
python3 -m pip install torch torchaudio
python3 -m pip install -e .
uvicorn transcribex.main:app --reload
```

## Notes

Single-channel speaker diarization is inherently probabilistic. For best results, use close microphones, reduce background noise, and prefer separate participant tracks when your meeting platform can export them.
