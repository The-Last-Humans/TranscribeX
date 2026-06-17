---
name: transcribex-transcribe
description: Transcribe local meeting audio or video files through a running TranscribeX FunASR service. Use when the user asks to transcribe, diarize, generate timestamps, convert meeting recordings to text, or prepare a transcript for later AI summary/analysis without sending audio to public cloud transcription services.
---

# TranscribeX Transcribe

Use the local TranscribeX API service to turn meeting recordings into text with speaker labels and segment timestamps.

## Workflow

1. Get the local audio/video file path from the user if it was not provided.
2. Use `TRANSCRIBEX_URL` when set; otherwise use `http://127.0.0.1:8000`.
3. Check service health before long uploads:

```bash
BASE_URL="${TRANSCRIBEX_URL:-http://127.0.0.1:8000}"
curl -fsS "$BASE_URL/health"
```

4. Run the bundled client script from this skill directory:

```bash
python3 scripts/transcribe_audio.py "/path/to/meeting.m4a" --format markdown
```

5. Return the transcript to the user, or save it when they ask for a file.
6. If the user asks for a summary or analysis, use the returned transcript as source material and preserve timestamp references when useful.

## Client Script

`scripts/transcribe_audio.py` uploads a file to `/v1/audio/transcriptions`.

Common options:

```bash
python3 scripts/transcribe_audio.py meeting.wav \
  --base-url http://127.0.0.1:8000 \
  --format markdown \
  --output meeting.transcript.md
```

Use `--format json`, `--format text`, or `--format srt` for other outputs.

Use `--api-key` or `TRANSCRIBEX_API_KEY` when the service requires bearer auth.

Use `--model iic/SenseVoiceSmall` to override the service default for a request.

Use `--hotword "项目名 20 产品名 20"` when domain-specific terms need help.

Use `--speaker-map '{"SPEAKER_00":"Alice","SPEAKER_01":"Bob"}'` after the user identifies speakers.

## Output Expectations

The service returns speaker labels like `SPEAKER_00` unless a speaker map is supplied. Treat these labels as machine guesses, especially for noisy rooms, overlapping speech, or similar voices.

Segment timestamps are the primary timing surface. Do not promise word-level precision unless the API response includes word timestamps.

## Troubleshooting

- If health check fails, tell the user to start the service with `docker compose up`.
- If the first transcription is slow, explain that FunASR models are being downloaded and cached.
- If diarization quality is poor, suggest closer microphones, cleaner audio, or a speaker map after manual identification.
