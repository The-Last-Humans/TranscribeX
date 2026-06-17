from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from transcribex.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    asr_model: str
    vad_model: str | None
    punc_model: str | None
    spk_model: str | None
    device: str
    hub: str | None


class FunASREngine:
    def __init__(self, spec: ModelSpec, batch_size_s: int) -> None:
        self.spec = spec
        self.batch_size_s = batch_size_s
        self._model: Any | None = None
        self._load_lock = Lock()
        self._infer_lock = Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                from funasr import AutoModel
            except Exception as exc:  # pragma: no cover - depends on deployment image
                raise RuntimeError(
                    "FunASR is not installed or failed to import. Install service dependencies first."
                ) from exc

            kwargs: dict[str, Any] = {
                "model": self.spec.asr_model,
                "device": self.spec.device,
            }
            if self.spec.vad_model:
                kwargs["vad_model"] = self.spec.vad_model
            if self.spec.punc_model:
                kwargs["punc_model"] = self.spec.punc_model
            if self.spec.spk_model:
                kwargs["spk_model"] = self.spec.spk_model
            if self.spec.hub:
                kwargs["hub"] = self.spec.hub

            logger.info("Loading FunASR model: %s", kwargs)
            self._model = AutoModel(**kwargs)
            logger.info("FunASR model loaded")

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
        hotword: str | None = None,
    ) -> Any:
        self.load()
        generate_kwargs: dict[str, Any] = {
            "input": str(audio_path),
            "batch_size_s": self.batch_size_s,
        }
        if language:
            generate_kwargs["language"] = language
        if hotword:
            generate_kwargs["hotword"] = hotword

        assert self._model is not None
        with self._infer_lock:
            return self._model.generate(**generate_kwargs)


class EngineManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._engines: dict[ModelSpec, FunASREngine] = {}
        self._lock = Lock()

    def default_spec(self, *, model: str | None = None, diarize: bool = True) -> ModelSpec:
        return ModelSpec(
            asr_model=model or self.settings.asr_model,
            vad_model=self.settings.vad_model,
            punc_model=self.settings.punc_model,
            spk_model=self.settings.spk_model if diarize else None,
            device=self.settings.device,
            hub=self.settings.hub,
        )

    def get_engine(self, spec: ModelSpec) -> FunASREngine:
        with self._lock:
            engine = self._engines.get(spec)
            if engine is None:
                engine = FunASREngine(spec, batch_size_s=self.settings.batch_size_s)
                self._engines[spec] = engine
            return engine

    def preload_default(self) -> None:
        self.get_engine(self.default_spec()).load()

    def is_default_loaded(self) -> bool:
        engine = self._engines.get(self.default_spec())
        return bool(engine and engine.loaded)
