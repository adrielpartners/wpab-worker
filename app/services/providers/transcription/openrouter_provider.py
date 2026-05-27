"""
OpenRouter transcription provider.

OpenRouter's speech-to-text endpoint accepts JSON with base64-encoded audio,
not the multipart file upload used by OpenAI and Groq.
"""

import base64
import random
import time
from pathlib import Path

import requests

from app.core.config import settings
from app.core.logging import logger


class OpenRouterProvider:
    """Transcription provider for OpenRouter STT models."""

    def __init__(self, api_key: str, endpoint: str = 'https://openrouter.ai/api'):
        self.api_key = api_key
        self.endpoint = endpoint.rstrip('/').removesuffix('/v1')

    def _sleep_for_retry(self, attempt: int) -> None:
        if attempt <= 1:
            return
        time.sleep(float(2 ** (attempt - 2)) + random.uniform(0.0, 0.4))

    def _truncate(self, value: str, limit: int = 500) -> str:
        return value if len(value) <= limit else f"{value[:limit]}..."

    def _error_with_context(self, prefix: str, response=None, exc=None) -> RuntimeError:
        status = response.status_code if response is not None else 'n/a'
        snippet = ''
        if response is not None:
            snippet = self._truncate((response.text or '').strip().replace('\n', ' '))
        if not snippet and exc is not None:
            snippet = self._truncate(str(exc))
        return RuntimeError(f"{prefix}; status={status}; body={snippet}")

    def transcribe_chunk(self, chunk_path: Path, model: str, job_id: str) -> str:
        url = f"{self.endpoint}/v1/audio/transcriptions"
        timeout = (settings.DOWNLOAD_CONNECT_TIMEOUT, settings.DOWNLOAD_READ_TIMEOUT)
        last_error: Exception | None = None

        for attempt in range(1, settings.MAX_RETRY_ATTEMPTS + 1):
            self._sleep_for_retry(attempt)
            try:
                audio_data = base64.b64encode(chunk_path.read_bytes()).decode('ascii')
                response = requests.post(
                    url,
                    headers={
                        'Authorization': f'Bearer {self.api_key}',
                        'Content-Type': 'application/json',
                        'HTTP-Referer': settings.PUBLIC_BASE_URL,
                        'X-Title': 'WP Audio Buddy Worker',
                    },
                    json={
                        'model': model,
                        'input_audio': {
                            'data': audio_data,
                            'format': _audio_format(chunk_path),
                        },
                    },
                    timeout=timeout,
                )

                if response.status_code in settings.RETRYABLE_STATUS_CODES and attempt < settings.MAX_RETRY_ATTEMPTS:
                    logger.warning(
                        "openrouter_retry job_id=%s attempt=%s status=%s chunk=%s",
                        job_id, attempt, response.status_code, chunk_path.name,
                    )
                    continue

                response.raise_for_status()
                text = response.json().get('text', '').strip()
                if text:
                    return text
                last_error = RuntimeError("OpenRouter returned empty transcription text")

            except requests.exceptions.RequestException as exc:
                last_error = exc
                status_code = getattr(getattr(exc, 'response', None), 'status_code', None)
                retryable = isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError))
                if status_code in settings.RETRYABLE_STATUS_CODES:
                    retryable = True

                if retryable and attempt < settings.MAX_RETRY_ATTEMPTS:
                    logger.warning(
                        "openrouter_retry job_id=%s attempt=%s status=%s error=%s chunk=%s",
                        job_id, attempt, status_code, exc.__class__.__name__, chunk_path.name,
                    )
                    continue

                if getattr(exc, 'response', None) is not None:
                    raise self._error_with_context('OpenRouter transcription failed', exc.response, exc) from exc
                raise RuntimeError(f"OpenRouter transcription failed; error={self._truncate(str(exc))}") from exc

        raise RuntimeError(f"OpenRouter transcription failed after retries; error={self._truncate(str(last_error))}")


def _audio_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip('.')
    return suffix if suffix in {'wav', 'mp3', 'flac', 'm4a', 'ogg', 'webm', 'aac'} else 'mp3'
