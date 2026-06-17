#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4


def main() -> int:
    args = parse_args()
    audio_path = Path(args.audio)
    if not audio_path.exists() or not audio_path.is_file():
        print(f"Audio file not found: {audio_path}", file=sys.stderr)
        return 2

    response_format = "verbose_json" if args.format in {"markdown", "md"} else args.format
    fields = {
        "response_format": response_format,
        "diarize": str(args.diarize).lower(),
        "timestamps": str(args.timestamps).lower(),
    }
    optional_fields = {
        "model": args.model,
        "language": args.language,
        "hotword": args.hotword,
        "speaker_map": args.speaker_map,
    }
    fields.update({key: value for key, value in optional_fields.items() if value})

    try:
        body = MultipartBody(fields, "file", audio_path)
        request = urllib.request.Request(
            url=f"{args.base_url.rstrip('/')}/v1/audio/transcriptions",
            data=body,
            method="POST",
            headers={
                "Content-Type": body.content_type,
                "Content-Length": str(body.content_length),
            },
        )
        api_key = args.api_key or os.environ.get("TRANSCRIBEX_API_KEY")
        if api_key:
            request.add_header("Authorization", f"Bearer {api_key}")

        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            payload = response.read()
            charset = response.headers.get_content_charset() or "utf-8"
            rendered = render_payload(payload.decode(charset), args.format)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"TranscribeX API error {exc.code}: {detail}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Failed to connect to TranscribeX at {args.base_url}: {exc.reason}", file=sys.stderr)
        return 1

    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload an audio file to TranscribeX.")
    parser.add_argument("audio", help="Path to an audio or video file")
    parser.add_argument("--base-url", default=os.environ.get("TRANSCRIBEX_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--format", choices=["markdown", "md", "json", "verbose_json", "text", "srt"], default="markdown")
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--hotword", default=None)
    parser.add_argument("--speaker-map", default=None)
    parser.add_argument("--no-diarize", dest="diarize", action="store_false")
    parser.add_argument("--no-timestamps", dest="timestamps", action="store_false")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.set_defaults(diarize=True, timestamps=True)
    return parser.parse_args()


class MultipartBody:
    def __init__(self, fields: dict[str, str], file_field: str, file_path: Path) -> None:
        self.fields = fields
        self.file_field = file_field
        self.file_path = file_path
        self.boundary = f"----TranscribeX{uuid4().hex}"
        self.content_type = f"multipart/form-data; boundary={self.boundary}"
        self._field_chunks = self._build_field_chunks()
        self._file_header = self._build_file_header()
        self._file_footer = b"\r\n"
        self._closing = f"--{self.boundary}--\r\n".encode()
        self.content_length = (
            sum(len(chunk) for chunk in self._field_chunks)
            + len(self._file_header)
            + self.file_path.stat().st_size
            + len(self._file_footer)
            + len(self._closing)
        )

    def __iter__(self) -> Iterator[bytes]:
        yield from self._field_chunks
        yield self._file_header
        with self.file_path.open("rb") as file:
            while True:
                chunk = file.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk
        yield self._file_footer
        yield self._closing

    def _build_field_chunks(self) -> list[bytes]:
        chunks: list[bytes] = []
        for name, value in self.fields.items():
            chunks.append(f"--{self.boundary}\r\n".encode())
            chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            chunks.append(str(value).encode("utf-8"))
            chunks.append(b"\r\n")
        return chunks

    def _build_file_header(self) -> bytes:
        content_type = mimetypes.guess_type(str(self.file_path))[0] or "application/octet-stream"
        return (
            f"--{self.boundary}\r\n"
            f'Content-Disposition: form-data; name="{self.file_field}"; '
            f'filename="{self.file_path.name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode()


def render_payload(text: str, output_format: str) -> str:
    if output_format not in {"markdown", "md"}:
        return text
    data = json.loads(text)
    return to_markdown(data)


def to_markdown(data: dict) -> str:
    lines = ["# Transcript", ""]
    meta = []
    if data.get("model"):
        meta.append(f"model: `{data['model']}`")
    if data.get("duration") is not None:
        meta.append(f"duration: `{data['duration']:.2f}s`")
    if data.get("language"):
        meta.append(f"language: `{data['language']}`")
    if meta:
        lines.extend(["; ".join(meta), ""])

    segments = data.get("segments") or []
    if segments:
        for segment in segments:
            prefix = ""
            if segment.get("start") is not None and segment.get("end") is not None:
                prefix += f"[{format_clock(segment['start'])} - {format_clock(segment['end'])}] "
            if segment.get("speaker"):
                prefix += f"{segment['speaker']}: "
            lines.append(f"{prefix}{segment.get('text', '')}".rstrip())
    else:
        lines.append(data.get("text", ""))
    lines.append("")
    return "\n".join(lines)


def format_clock(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


if __name__ == "__main__":
    raise SystemExit(main())
