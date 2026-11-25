from contextlib import asynccontextmanager
from threading import BoundedSemaphore
from typing import Literal
import json
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from openai import APIError
from pydantic import BaseModel, Field, field_validator
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from main import (
    LANGUAGES,
    LLAMA_URL,
    LLAMA_MODEL,
    TranslationResources,
    load_translation_resources,
    build_prompt,
    generate_translation_stream,
    retrieve_terms,
    semantic_model_is_loaded,
    translation_model_status,
    translate,
)
from model import RankedTerm
from speech import SpeechModelError, speech_model_is_loaded, synthesize_wav
from stt import SpeechToTextError, transcription_model_status, transcribe_audio
from pronunciation import PronunciationError, transliterate


TargetLanguage = Literal["en", "ja", "zhCN", "zhTW"]
SourceLanguage = Literal["auto", "ru"]
logger = logging.getLogger(__name__)
MAX_AUDIO_BYTES = 12 * 1024 * 1024
SUPPORTED_AUDIO_TYPES = {
    "audio/webm",
    "audio/mp4",
    "audio/ogg",
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
}


class TranslationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    target_language: TargetLanguage
    source_language: SourceLanguage = "auto"

    model_config = {"extra": "forbid"}

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must contain at least one non-space character")
        return value


class TranslationResponse(BaseModel):
    original: str
    target_language: TargetLanguage
    translation: str
    retrieved_terms: list[RankedTerm]


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    target_language: TargetLanguage

    model_config = {"extra": "forbid"}

    @field_validator("text")
    @classmethod
    def speech_text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must contain at least one non-space character")
        return value


class PronunciationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    target_language: Literal["ja", "zhCN", "zhTW"]

    model_config = {"extra": "forbid"}

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must contain at least one non-space character")
        return value


class TranscriptionResponse(BaseModel):
    text: str
    detected_language: str
    language_probability: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.resources = load_translation_resources()
    # Bound simultaneous model calls so a local client cannot queue unlimited work.
    app.state.translation_slots = BoundedSemaphore(4)
    app.state.speech_slots = BoundedSemaphore(1)
    app.state.transcription_slots = BoundedSemaphore(1)
    yield
    app.state.resources = None


app = FastAPI(
    title="Genshin Translation API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:1420",
        "http://127.0.0.1:1420",
        "tauri://localhost",
        "http://tauri.localhost",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/api/v1/health")
def health(request: Request) -> dict[str, str | int | bool]:
    resources: TranslationResources = request.app.state.resources
    dictionary_ready = bool(resources.term_store)
    semantic_loaded = semantic_model_is_loaded()
    speech_loaded = speech_model_is_loaded()
    model_status = translation_model_status()
    ready = dictionary_ready and bool(model_status["translation_model_loaded"])
    return {
        "status": "ready" if ready else "degraded",
        "api_online": True,
        "translation_model_url": LLAMA_URL,
        "translation_model_name": LLAMA_MODEL,
        "dictionary_ready": dictionary_ready,
        "dictionary_entries": len(resources.term_store),
        "semantic_model_loaded": semantic_loaded,
        "semantic_model_state": "loaded" if semantic_loaded else "lazy",
        "speech_model_loaded": speech_loaded,
        "speech_model_state": "loaded" if speech_loaded else "lazy",
        **transcription_model_status(),
        **model_status,
    }


@app.get("/api/v1/languages")
def languages() -> dict[str, list[dict[str, str]]]:
    return {
        "languages": [
            {"code": code, "name": name}
            for code, name in LANGUAGES.items()
        ]
    }


@app.post("/api/v1/speech")
def create_speech(body: SpeechRequest, request: Request) -> Response:
    slots: BoundedSemaphore = request.app.state.speech_slots
    if not slots.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="The local voice is busy. Try again shortly.",
            headers={"Retry-After": "1"},
        )

    try:
        wav_audio = synthesize_wav(body.text, body.target_language)
    except SpeechModelError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    finally:
        slots.release()

    return Response(
        content=wav_audio,
        media_type="audio/wav",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/v1/pronunciations")
def create_pronunciation(body: PronunciationRequest) -> dict[str, str]:
    try:
        pronunciation = transliterate(body.text, body.target_language)
    except PronunciationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return {
        "pronunciation": pronunciation,
        "target_language": body.target_language,
        "label": "romaji" if body.target_language == "ja" else "pinyin",
    }


@app.post("/api/v1/transcriptions", response_model=TranscriptionResponse)
async def create_transcription(
    request: Request,
    source_language: SourceLanguage = "auto",
) -> TranscriptionResponse:
    content_type = request.headers.get("content-type", "").split(";", maxsplit=1)[0].strip().lower()
    if content_type not in SUPPORTED_AUDIO_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Use a browser-recorded audio clip and try again.",
        )

    try:
        content_length = int(request.headers.get("content-length", "0"))
    except ValueError:
        content_length = 0
    if content_length > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Keep each recording under 30 seconds.")

    slots: BoundedSemaphore = request.app.state.transcription_slots
    if not slots.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="The local speech recognizer is busy. Try again shortly.",
            headers={"Retry-After": "1"},
        )

    try:
        audio = bytearray()
        async for chunk in request.stream():
            audio.extend(chunk)
            if len(audio) > MAX_AUDIO_BYTES:
                raise HTTPException(status_code=413, detail="Keep each recording under 30 seconds.")

        result = await run_in_threadpool(transcribe_audio, bytes(audio), source_language)
    except SpeechToTextError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error
    finally:
        slots.release()

    return TranscriptionResponse.model_validate(result)


@app.post("/api/v1/translations", response_model=TranslationResponse)
def create_translation(
    body: TranslationRequest,
    request: Request,
) -> TranslationResponse:
    resources: TranslationResources = request.app.state.resources
    slots: BoundedSemaphore = request.app.state.translation_slots
    if not slots.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="The local translation service is busy. Try again shortly.",
            headers={"Retry-After": "1"},
        )
    try:
        result = translate(
            query=body.text,
            target_language=body.target_language,
            resources=resources,
            source_language=body.source_language,
        )
        if not result["translation"].strip():
            raise HTTPException(
                status_code=502,
                detail="The local model returned an empty translation. Check the model server and try again.",
            )
    except APIError as error:
        raise HTTPException(
            status_code=502,
            detail="The local model server could not complete the translation.",
        ) from error
    finally:
        slots.release()

    return TranslationResponse.model_validate(result)


@app.post("/api/v1/translations/stream")
def create_translation_stream(
    body: TranslationRequest,
    request: Request,
) -> StreamingResponse:
    resources: TranslationResources = request.app.state.resources
    slots: BoundedSemaphore = request.app.state.translation_slots
    if not slots.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="The local translation service is busy. Try again shortly.",
            headers={"Retry-After": "1"},
        )

    released = False

    def release_slot() -> None:
        nonlocal released
        if not released:
            released = True
            slots.release()

    def events():
        try:
            retrieved_terms = retrieve_terms(body.text, resources)
            yield json.dumps({
                "type": "terms",
                "retrieved_terms": [term.model_dump(mode="json") for term in retrieved_terms],
            }, ensure_ascii=False) + "\n"

            prompt = build_prompt(
                body.text,
                [match.term for match in retrieved_terms],
                body.target_language,
                body.source_language,
            )
            parts: list[str] = []
            for delta in generate_translation_stream(prompt):
                parts.append(delta)
                yield json.dumps({"type": "token", "text": delta}, ensure_ascii=False) + "\n"

            translation = "".join(parts)
            if not translation.strip():
                yield json.dumps({
                    "type": "error",
                    "detail": "The local model returned an empty translation. Check the model server and try again.",
                }, ensure_ascii=False) + "\n"
                return
            yield json.dumps({"type": "done", "translation": translation}, ensure_ascii=False) + "\n"
        except APIError as error:
            logger.error("Translation model request failed (%s)", type(error).__name__)
            yield json.dumps({
                "type": "error",
                "detail": "The local model server could not complete the translation.",
            }, ensure_ascii=False) + "\n"
        except Exception as error:
            logger.error("Unexpected translation stream failure (%s)", type(error).__name__)
            yield json.dumps({
                "type": "error",
                "detail": "The local translation service could not complete the translation.",
            }, ensure_ascii=False) + "\n"
        finally:
            release_slot()

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        background=BackgroundTask(release_slot),
    )
