"""Local, in-memory Whisper transcription for spoken translation snippets."""

from __future__ import annotations

from io import BytesIO
import logging
from threading import Lock

logger = logging.getLogger(__name__)

MODEL_NAME = "base"

_model = None
_model_state = "lazy"
_model_lock = Lock()


class SpeechToTextError(RuntimeError):
    """A safe, user-facing local transcription error."""

    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message)
        self.status_code = status_code


def transcription_model_status() -> dict[str, str | bool]:
    return {
        "transcription_model_loaded": _model is not None,
        "transcription_model_state": _model_state,
    }


def _get_model():
    global _model, _model_state

    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        _model_state = "loading"
        try:
            import ctranslate2
            from faster_whisper import WhisperModel

            supported = ctranslate2.get_supported_compute_types("cpu")
            compute_type = next(
                (
                    option
                    for option in ("int8", "int8_float32", "float32")
                    if option in supported
                ),
                "float32",
            )
            _model = WhisperModel(
                MODEL_NAME,
                device="cpu",
                compute_type=compute_type,
                cpu_threads=4,
                num_workers=1,
            )
            _model_state = "loaded"
            return _model
        except ImportError as error:
            _model_state = "error"
            raise SpeechToTextError(
                "Local speech recognition is unavailable. Run `uv sync` and restart FastAPI."
            ) from error
        except Exception as error:
            _model_state = "error"
            logger.exception("Could not load the local speech recognition model")
            raise SpeechToTextError(
                "The local speech model could not load. Connect to the internet for its first download, then retry."
            ) from error


def transcribe_audio(
    audio: bytes,
    source_language: str = "auto",
) -> dict[str, str | float]:
    if not audio:
        raise SpeechToTextError("No audio was recorded. Try speaking again.", 422)

    model = _get_model()
    language = None if source_language == "auto" else source_language

    try:
        segments, info = model.transcribe(
            BytesIO(audio),
            language=language,
            task="transcribe",
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt=(
                "Genshin Impact, Тейват, Мондштадт, Ли Юэ, Инадзума, "
                "Сумеру, Фонтейн, Натлан."
            ),
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as error:
        logger.exception("Could not transcribe the recorded speech")
        raise SpeechToTextError(
            "The audio could not be transcribed. Try a shorter, clearer recording.",
            422,
        ) from error

    if not text:
        raise SpeechToTextError("I couldn't hear clear speech. Try speaking again.", 422)

    return {
        "text": text,
        "detected_language": info.language,
        "language_probability": float(info.language_probability),
    }
