"""
OpenAI transcription service for audio chunks.
"""

import random
import time
from pathlib import Path

import requests

from app.core.config import settings
from app.core.logging import logger


def _sleep_for_retry(attempt: int) -> None:
    """Exponential backoff with jitter."""
    if attempt <= 1:
        return
    delay = float(2 ** (attempt - 2)) + random.uniform(0.0, 0.4)
    time.sleep(delay)


def _error_with_response_context(prefix: str, response: requests.Response | None = None, exc: Exception | None = None) -> RuntimeError:
    status = response.status_code if response is not None else "n/a"
    error_type = exc.__class__.__name__ if exc is not None else "HTTPError"
    return RuntimeError(f"{prefix}; status={status}; error_type={error_type}")


def transcribe_chunk(chunk_path: Path, model: str, job_id: str) -> str:
    """
    Send a single audio chunk to OpenAI for transcription.

    Includes bounded retry for transient errors.
    Returns the transcript text.
    """
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    timeout = settings.OPENAI_TIMEOUT
    last_error: Exception | None = None

    for attempt in range(1, settings.MAX_RETRY_ATTEMPTS + 1):
        _sleep_for_retry(attempt)
        try:
            with chunk_path.open("rb") as audio_file:
                files = {"file": (chunk_path.name, audio_file, "audio/wav")}
                data = {"model": model}
                headers = {"Authorization": f"Bearer {api_key}"}
                response = requests.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=timeout,
                )

            if response.status_code in settings.RETRYABLE_STATUS_CODES and attempt < settings.MAX_RETRY_ATTEMPTS:
                logger.warning(
                    "openai_retry job_id=%s attempt=%s status=%s chunk=%s",
                    job_id, attempt, response.status_code, chunk_path.name,
                )
                continue

            response.raise_for_status()
            payload = response.json()
            text = payload.get("text", "").strip()
            if text:
                return text
            continue

        except requests.exceptions.RequestException as exc:
            last_error = exc
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError))
            if status_code in settings.RETRYABLE_STATUS_CODES:
                retryable = True

            if retryable and attempt < settings.MAX_RETRY_ATTEMPTS:
                logger.warning(
                    "openai_retry job_id=%s attempt=%s status=%s error=%s chunk=%s",
                    job_id, attempt, status_code, exc.__class__.__name__, chunk_path.name,
                )
                continue

            if getattr(exc, "response", None) is not None:
                raise _error_with_response_context("OpenAI transcription failed", exc.response, exc) from exc
            raise RuntimeError(f"OpenAI transcription failed; error_type={exc.__class__.__name__}") from exc

    last_error_type = last_error.__class__.__name__ if last_error is not None else "Unknown"
    raise RuntimeError(f"OpenAI transcription failed after retries; error_type={last_error_type}")
