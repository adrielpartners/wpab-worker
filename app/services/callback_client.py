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


def _error_with_response_context(prefix: str, response: requests.Response | None = None, exc: Exception | None = None) -> RuntimeError:
    status = response.status_code if response is not None else "n/a"
    error_type = exc.__class__.__name__ if exc is not None else "HTTPError"
    return RuntimeError(f"{prefix}; status={status}; error_type={error_type}")


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
    timestamp = int(time.time())
    callback_payload = dict(payload)
    callback_payload["timestamp"] = timestamp
    if site_id:
        callback_payload["site_id"] = site_id
    raw = json.dumps(callback_payload, separators=(",", ":")).encode("utf-8")
    signature = make_callback_signature(raw, site_id, timestamp)

    if signature is None:
        raise RuntimeError(f"Cannot send callback: no secret configured for site_id={site_id}")

    headers = {
        "Content-Type": "application/json",
        "X-WPAB-Site-ID": site_id,
        "X-WPAB-Timestamp": str(timestamp),
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
            raise RuntimeError(f"Callback failed; error_type={exc.__class__.__name__}") from exc

    last_error_type = last_error.__class__.__name__ if last_error is not None else "Unknown"
    raise RuntimeError(f"Callback failed after retries; error_type={last_error_type}")
