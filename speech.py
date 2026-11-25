"""Local Kokoro speech synthesis for completed translations."""

from __future__ import annotations

from io import BytesIO
import logging
import os
import wave
from threading import Lock

import numpy as np

logger = logging.getLogger(__name__)

MODEL_ID = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24_000

VOICE_BY_LANGUAGE = {
    "en": ("a", "af_heart"),
    "ja": ("j", "jf_alpha"),
    "zhCN": ("z", "zf_xiaobei"),
    "zhTW": ("z", "zf_xiaobei"),
}

_model = None
_pipelines = {}
_model_lock = Lock()


def speech_model_is_loaded() -> bool:
    return _model is not None


class SpeechModelError(RuntimeError):
    """A safe, user-facing local speech synthesis failure."""


def _get_pipeline(target_language: str):
    global _model

    lang_code, _voice = VOICE_BY_LANGUAGE[target_language]
    if lang_code in _pipelines:
        return _pipelines[lang_code]

    with _model_lock:
        if lang_code in _pipelines:
            return _pipelines[lang_code]

        try:
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            import torch
            from kokoro import KModel, KPipeline
        except ImportError as error:
            raise SpeechModelError(
                "Local voice support is missing. Run `uv sync` and restart FastAPI."
            ) from error

        try:
            if _model is None:
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"

                _model = KModel(repo_id=MODEL_ID).to(device).eval()

            pipeline = KPipeline(lang_code=lang_code, repo_id=MODEL_ID, model=_model)
            _pipelines[lang_code] = pipeline
            return pipeline
        except ImportError as error:
            _model = None
            raise SpeechModelError(
                "The local voice language data is missing. Run `uv sync` and restart FastAPI."
            ) from error
        except Exception as error:
            _model = None
            logger.exception("Could not load the local Kokoro voice model")
            raise SpeechModelError(
                "The local voice model could not load. Connect to the internet for its first download, then retry."
            ) from error


def synthesize_wav(text: str, target_language: str) -> bytes:
    """Generate in-memory WAV audio without saving the translation or audio."""
    if target_language not in VOICE_BY_LANGUAGE:
        raise SpeechModelError("That target language does not have a local voice.")

    _lang_code, voice = VOICE_BY_LANGUAGE[target_language]
    pipeline = _get_pipeline(target_language)

    try:
        chunks = [
            audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            for _graphemes, _phonemes, audio in pipeline(text, voice=voice)
            if audio is not None
        ]
    except Exception as error:
        logger.exception("Kokoro failed to synthesize a local translation")
        raise SpeechModelError(
            "The local voice could not speak this translation. For first use, connect to the internet to download the model, then retry."
        ) from error

    if not chunks:
        raise SpeechModelError("The local voice did not produce any audio. Try again.")

    samples = np.clip(np.concatenate(chunks).reshape(-1), -1.0, 1.0)
    pcm_audio = (samples * 32767).astype("<i2", copy=False).tobytes()
    output = BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm_audio)

    return output.getvalue()
