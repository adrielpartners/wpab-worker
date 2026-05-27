"""
Signed callback client for posting results back to WordPress.
"""

import json
import random
import time

import requests

from app.core.config import settings
from app.core.logging import logger
from app.core.security import make_callback_signature


def _sleep_for_retry(attempt: int) -> None:
    if attempt <= 1:
        return
    delay = float(2 ** (attempt - 2)) + random.uniform(0.0, 0.4)
    time.sleep(delay)


def _truncate(value: str, limit: int = 500) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def _error_with_response_context(prefix: str, response: requests.Response | None = None, exc: Exception | None = None) -> RuntimeError:
    status = response.status_code if response is not None else "n/a"
    snippet = ""
    if response is not None:
        snippet = _truncate((response.text or "").strip().replace("\n", " "))
    if not snippet and exc is not None:
        snippet = _truncate(str(exc))
    return RuntimeError(f"{prefix}; status={status}; body={snippet}")


def send_callback(
    callback_url: str,
    payload: dict,
    site_id: str,
    job_id_for_log: str,
) -> None:
    """
    Send a signed callback to WordPress.

    Signs the payload with the site's shared secret.
    Retries on transient failures up to CALLBACK_RETRY_ATTEMPTS times.
    """
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    signature = make_callback_signature(raw, site_id)

    if signature is None:
        raise RuntimeError(f"Cannot send callback: no secret configured for site_id={site_id}")

    headers = {
        "Content-Type": "application/json",
        "X-WPAB-Signature": signature,
    }
    timeout = settings.CALLBACK_TIMEOUT

    last_error: Exception | None = None
    for attempt in range(1, settings.CALLBACK_RETRY_ATTEMPTS + 1):
        _sleep_for_retry(attempt)
        try:
            response = requests.post(callback_url, data=raw, headers=headers, timeout=timeout)

            if response.status_code in settings.RETRYABLE_STATUS_CODES and attempt < settings.CALLBACK_RETRY_ATTEMPTS:
                logger.warning(
                    "callback_retry job_id=%s attempt=%s status=%s",
                    job_id_for_log, attempt, response.status_code,
                )
                continue

            response.raise_for_status()
            logger.info("callback_success job_id=%s attempt=%s", job_id_for_log, attempt)
            return

        except requests.exceptions.RequestException as exc:
            last_error = exc
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError))
            if status_code in settings.RETRYABLE_STATUS_CODES:
                retryable = True

            if retryable and attempt < settings.CALLBACK_RETRY_ATTEMPTS:
                logger.warning(
                    "callback_retry job_id=%s attempt=%s status=%s error=%s",
                    job_id_for_log, attempt, status_code, exc.__class__.__name__,
                )
                continue

            if getattr(exc, "response", None) is not None:
                raise _error_with_response_context("Callback failed", exc.response, exc) from exc
            raise RuntimeError(f"Callback failed; error={_truncate(str(exc))}") from exc

    raise RuntimeError(f"Callback failed after retries; error={_truncate(str(last_error))}")