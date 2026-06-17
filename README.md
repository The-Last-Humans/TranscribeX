# TranscribeX

TranscribeX is a self-hosted meeting transcription service built around FunASR. It exposes an OpenAI-style transcription endpoint and includes a companion Codex skill so agents can submit audio files to the local service.

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

```bash
cp .env.example .env
docker compose up --build
```

The first request downloads FunASR models into the `models` volume, so it can take several minutes.

Health check:

```bash
curl http://127.0.0.1:8000/health
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

## API

### `GET /health`

Returns service status, current model defaults, and whether the model has been loaded.

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

## GPU

Use a CUDA PyTorch wheel at build time and run with NVIDIA Container Toolkit:

```bash
docker build \
  --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
  -t transcribex:cuda .

docker run --gpus all -p 8000:8000 \
  -e TRANSCRIBEX_DEVICE=cuda \
  -v transcribex-models:/models \
  transcribex:cuda
```

## Companion Skill

The Codex skill lives at `skills/transcribex-transcribe`. Install it by copying or symlinking that folder into your Codex skills directory, then invoke `$transcribex-transcribe`.

The skill uses `TRANSCRIBEX_URL` and `TRANSCRIBEX_API_KEY` when present. Default URL is `http://127.0.0.1:8000`.

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
