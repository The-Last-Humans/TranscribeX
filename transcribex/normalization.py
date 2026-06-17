from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from transcribex.schemas import Segment, TranscriptionResponse, Word


def normalize_funasr_result(
    raw_result: Any,
    *,
    model: str,
    duration: float | None,
    language: str | None,
    include_speakers: bool,
    include_timestamps: bool,
    speaker_map: dict[str, str] | None = None,
) -> TranscriptionResponse:
    items = _as_result_items(raw_result)
    text = _join_text([str(item.get("text", "")).strip() for item in items if isinstance(item, dict)])
    segments: list[Segment] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        sentence_info = item.get("sentence_info")
        if isinstance(sentence_info, list) and sentence_info:
            segments.extend(
                _segments_from_sentence_info(
                    sentence_info,
                    include_speakers=include_speakers,
                    include_timestamps=include_timestamps,
                    speaker_map=speaker_map,
                    start_index=len(segments),
                )
            )

    if not segments and text:
        segments.append(
            Segment(
                id=0,
                start=0.0 if include_timestamps else None,
                end=duration if include_timestamps else None,
                speaker=None,
                text=text,
                words=[],
            )
        )

    if segments and not text:
        text = _join_text([segment.text for segment in segments])

    return TranscriptionResponse(
        text=text,
        language=language or _first_language(items),
        duration=duration,
        model=model,
        segments=segments,
    )


def _as_result_items(raw_result: Any) -> list[Any]:
    if isinstance(raw_result, list):
        return raw_result
    if isinstance(raw_result, dict):
        return [raw_result]
    return []


def _segments_from_sentence_info(
    sentence_info: list[Any],
    *,
    include_speakers: bool,
    include_timestamps: bool,
    speaker_map: dict[str, str] | None,
    start_index: int,
) -> list[Segment]:
    segments: list[Segment] = []
    for offset, sentence in enumerate(sentence_info):
        if not isinstance(sentence, dict):
            continue
        speaker = None
        if include_speakers:
            speaker = _speaker_label(sentence)
            if speaker and speaker_map:
                speaker = speaker_map.get(speaker, speaker_map.get(str(sentence.get("spk")), speaker))

        start = _funasr_ms_to_seconds(sentence.get("start")) if include_timestamps else None
        end = _funasr_ms_to_seconds(sentence.get("end")) if include_timestamps else None
        words = _words_from_timestamp(sentence.get("timestamp"), sentence.get("text", ""), speaker) if include_timestamps else []
        segments.append(
            Segment(
                id=start_index + offset,
                start=start,
                end=end,
                speaker=speaker,
                text=str(sentence.get("text", "")).strip(),
                words=words,
            )
        )
    return _merge_adjacent_segments(segments)


def _words_from_timestamp(raw_timestamp: Any, text: Any, speaker: str | None) -> list[Word]:
    if not isinstance(raw_timestamp, list):
        return []
    tokens = _split_text_like_funasr(text)
    words: list[Word] = []
    for index, pair in enumerate(raw_timestamp):
        if not _is_pair(pair):
            continue
        token = tokens[index] if index < len(tokens) else ""
        words.append(
            Word(
                text=token,
                start=_funasr_ms_to_seconds(pair[0]),
                end=_funasr_ms_to_seconds(pair[1]),
                speaker=speaker,
            )
        )
    return words


def _split_text_like_funasr(text: Any) -> list[str]:
    value = str(text or "").strip()
    if not value:
        return []
    if " " in value:
        return [part for part in value.split() if part]
    return list(value)


def _is_pair(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) >= 2 and _is_number(value[0]) and _is_number(value[1])


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _funasr_ms_to_seconds(value: Any) -> float | None:
    if not _is_number(value):
        return None
    return round(float(value) / 1000.0, 3)


def _speaker_label(sentence: dict[str, Any]) -> str | None:
    for key in ("speaker", "spk", "speaker_id"):
        value = sentence.get(key)
        if value is None:
            continue
        if isinstance(value, int):
            return f"SPEAKER_{value:02d}"
        text = str(value)
        if text.isdigit():
            return f"SPEAKER_{int(text):02d}"
        if text.startswith("SPEAKER_"):
            return text
        return text
    return None


def _merge_adjacent_segments(segments: list[Segment]) -> list[Segment]:
    if not segments:
        return []
    merged: list[Segment] = []
    for segment in segments:
        if not segment.text:
            continue
        previous = merged[-1] if merged else None
        if (
            previous
            and previous.speaker == segment.speaker
            and previous.end is not None
            and segment.start is not None
            and segment.start - previous.end <= 0.3
        ):
            previous.text = _join_text([previous.text, segment.text])
            previous.end = segment.end
            previous.words.extend(segment.words)
        else:
            segment.id = len(merged)
            merged.append(segment)
    return merged


def _join_text(parts: Iterable[str]) -> str:
    cleaned = [part.strip() for part in parts if part and part.strip()]
    if not cleaned:
        return ""
    text = ""
    for part in cleaned:
        if not text:
            text = part
        elif _looks_cjk(text[-1]) or _looks_cjk(part[0]):
            text += part
        else:
            text += " " + part
    return text


def _looks_cjk(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def _first_language(items: list[Any]) -> str | None:
    for item in items:
        if isinstance(item, dict):
            for key in ("language", "lang"):
                value = item.get(key)
                if value:
                    return str(value)
    return None


def format_srt(segments: list[Segment]) -> str:
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        start = _srt_time(segment.start or 0.0)
        end = _srt_time(segment.end or segment.start or 0.0)
        label = f"{segment.speaker}: " if segment.speaker else ""
        blocks.append(f"{index}\n{start} --> {end}\n{label}{segment.text}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _srt_time(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
