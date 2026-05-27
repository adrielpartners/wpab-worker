"""
OpenAI-compatible transcription provider.

This provider works with any OpenAI-compatible API endpoint,
including OpenAI, Groq, DeepSeek, OpenRouter (text only), and
any other service that implements the same `/v1/audio/transcriptions` format.
"""

import random
import time
from pathlib import Path

import requests

from app.core.config import settings
from app.core.logging import logger


class OpenAICompatProvider:
    """Transcription provider for OpenAI-compatible APIs."""

    def __init__(self, api_key: str, endpoint: str):
        self.api_key = api_key
        self.endpoint = endpoint.rstrip('/')

    def _sleep_for_retry(self, attempt: int) -> None:
        if attempt <= 1:
            return
        delay = float(2 ** (attempt - 2)) + random.uniform(0.0, 0.4)
        time.sleep(delay)

    def _truncate(self, value: str, limit: int = 500) -> str:
        if len(value) <= limit:
            return value
        return f"{value[:limit]}..."

    def _error_with_context(self, prefix: str, response=None, exc=None) -> RuntimeError:
        status = response.status_code if response is not None else 'n/a'
        snippet = ''
        if response is not None:
            snippet = self._truncate((response.text or '').strip().replace('\n', ' '))
        if not snippet and exc is not None:
            snippet = self._truncate(str(exc))
        return RuntimeError(f"{prefix}; status={status}; body={snippet}")

    def transcribe_chunk(self, chunk_path: Path, model: str, job_id: str) -> str:
        """
        Transcribe a single audio chunk via the OpenAI-compatible API.

        Includes bounded retry for transient errors (429, 5xx, network timeouts).
        Returns the transcript text.
        """
        url = f"{self.endpoint}/v1/audio/transcriptions"
        timeout = (settings.DOWNLOAD_CONNECT_TIMEOUT, settings.DOWNLOAD_READ_TIMEOUT)
        last_error: Exception | None = None

        for attempt in range(1, settings.MAX_RETRY_ATTEMPTS + 1):
            self._sleep_for_retry(attempt)
            try:
                with chunk_path.open('rb') as audio_file:
                    files = {'file': (chunk_path.name, audio_file, 'audio/wav')}
                    data = {'model': model}
                    headers = {'Authorization': f'Bearer {self.api_key}'}
                    response = requests.post(
                        url,
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
                text = payload.get('text', '').strip()
                if text:
                    return text
                continue

            except requests.exceptions.RequestException as exc:
                last_error = exc
                status_code = getattr(getattr(exc, 'response', None), 'status_code', None)
                retryable = isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError))
                if status_code in settings.RETRYABLE_STATUS_CODES:
                    retryable = True

                if retryable and attempt < settings.MAX_RETRY_ATTEMPTS:
                    logger.warning(
                        "openai_retry job_id=%s attempt=%s status=%s error=%s chunk=%s",
                        job_id, attempt, status_code, exc.__class__.__name__, chunk_path.name,
                    )
                    continue

                if getattr(exc, 'response', None) is not None:
                    raise self._error_with_context('OpenAI transcription failed', exc.response, exc) from exc
                raise RuntimeError(f"OpenAI transcription failed; error={self._truncate(str(exc))}") from exc

        raise RuntimeError(f"OpenAI transcription failed after retries; error={self._truncate(str(last_error))}")