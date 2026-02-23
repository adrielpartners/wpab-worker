import json
import logging
import os
import random
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from rq import get_current_job

from hmac_util import make_signature

logger = logging.getLogger(__name__)

WORK_ROOT = Path(os.getenv("WPAB_WORK_ROOT", "/work"))
DEFAULT_MODEL = os.getenv("WPAB_DEFAULT_MODEL", "gpt-4o-mini-transcribe")
DEFAULT_CHUNK_SECONDS = int(os.getenv("WPAB_DEFAULT_CHUNK_SECONDS", "720"))
MAX_CHUNK_SECONDS = 900
MAX_DOWNLOAD_MB = int(os.getenv("WPAB_MAX_DOWNLOAD_MB", "200"))
MAX_DOWNLOAD_BYTES = MAX_DOWNLOAD_MB * 1024 * 1024
DOWNLOAD_CONNECT_TIMEOUT_SECONDS = float(os.getenv("WPAB_DOWNLOAD_CONNECT_TIMEOUT_SECONDS", "10"))
DOWNLOAD_READ_TIMEOUT_SECONDS = float(os.getenv("WPAB_DOWNLOAD_READ_TIMEOUT_SECONDS", "60"))
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRY_ATTEMPTS = 4


def _safe_chunk_seconds(value: int | None) -> int:
    if value is None:
        return DEFAULT_CHUNK_SECONDS
    return max(1, min(int(value), MAX_CHUNK_SECONDS))


def _source_extension(audio_url: str) -> str:
    path = urlparse(audio_url).path
    ext = Path(path).suffix
    return ext if ext else ".bin"


def _source_url_domain(audio_url: str) -> str:
    return (urlparse(audio_url).netloc or "unknown").strip()[:255]


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


def _sleep_for_retry(attempt: int) -> None:
    if attempt <= 1:
        return
    delay = float(2 ** (attempt - 2)) + random.uniform(0.0, 0.4)
    time.sleep(delay)


def _download_audio_with_guardrails(audio_url: str, source_file: Path, job_id: str) -> tuple[int, float]:
    parsed = urlparse(audio_url)
    if parsed.scheme not in {"http", "https"}:
        raise RuntimeError(f"Unsupported URL scheme '{parsed.scheme}'. Only http/https are allowed")

    source_domain = _source_url_domain(audio_url)
    timeout = (DOWNLOAD_CONNECT_TIMEOUT_SECONDS, DOWNLOAD_READ_TIMEOUT_SECONDS)
    start = time.time()
    bytes_downloaded = 0

    with requests.get(audio_url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        content_length_raw = response.headers.get("Content-Length")
        content_type = (response.headers.get("Content-Type") or "").strip().lower()

        if content_type and not (
            content_type.startswith("audio/")
            or content_type.startswith("video/")
            or content_type.startswith("application/octet-stream")
        ):
            logger.warning(
                "download_content_type_unexpected job_id=%s source_url_domain=%s content_type=%s",
                job_id,
                source_domain,
                content_type,
            )

        if content_length_raw:
            try:
                content_length = int(content_length_raw)
            except ValueError:
                content_length = None
            if content_length is not None and content_length > MAX_DOWNLOAD_BYTES:
                raise RuntimeError(
                    f"file too large: {round(content_length / (1024 * 1024), 2)}MB exceeds max {MAX_DOWNLOAD_MB}MB"
                )
        else:
            content_length = None

        logger.info(
            "download_start job_id=%s source_url_domain=%s content_length=%s",
            job_id,
            source_domain,
            content_length,
        )

        with source_file.open("wb") as out:
            for piece in response.iter_content(chunk_size=1024 * 1024):
                if not piece:
                    continue
                bytes_downloaded += len(piece)
                if bytes_downloaded > MAX_DOWNLOAD_BYTES:
                    raise RuntimeError(f"file too large: exceeds max {MAX_DOWNLOAD_MB}MB")
                out.write(piece)

    elapsed = time.time() - start
    logger.info(
        "download_done job_id=%s source_url_domain=%s bytes_downloaded=%s elapsed_seconds=%.3f",
        job_id,
        source_domain,
        bytes_downloaded,
        elapsed,
    )
    return bytes_downloaded, elapsed


def _run_ffmpeg_chunking(source_path: Path, chunk_dir: Path, chunk_seconds: int) -> list[Path]:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = chunk_dir / "chunk_%04d.wav"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        str(output_pattern),
    ]
    logger.info("Running chunk command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    chunks = sorted(chunk_dir.glob("chunk_*.wav"))
    if not chunks:
        raise RuntimeError("No chunks created by ffmpeg")
    return chunks


def _ffprobe_duration_seconds(path: Path) -> float | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return float(result.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return None


def _transcribe_chunk(chunk_path: Path, model: str, job_id: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    timeout = (DOWNLOAD_CONNECT_TIMEOUT_SECONDS, DOWNLOAD_READ_TIMEOUT_SECONDS)
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
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

            if response.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRY_ATTEMPTS:
                logger.warning(
                    "openai_retry job_id=%s attempt=%s status=%s chunk=%s",
                    job_id,
                    attempt,
                    response.status_code,
                    chunk_path.name,
                )
                continue

            response.raise_for_status()
            payload = response.json()
            return payload.get("text", "").strip()
        except requests.exceptions.RequestException as exc:
            last_error = exc
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError))
            if status_code in RETRYABLE_STATUS_CODES:
                retryable = True

            if retryable and attempt < MAX_RETRY_ATTEMPTS:
                logger.warning(
                    "openai_retry job_id=%s attempt=%s status=%s error=%s chunk=%s",
                    job_id,
                    attempt,
                    status_code,
                    exc.__class__.__name__,
                    chunk_path.name,
                )
                continue

            if getattr(exc, "response", None) is not None:
                raise _error_with_response_context("OpenAI transcription failed", exc.response, exc) from exc
            raise RuntimeError(f"OpenAI transcription failed; error={_truncate(str(exc))}") from exc

    raise RuntimeError(f"OpenAI transcription failed after retries; error={_truncate(str(last_error))}")


def _post_callback(callback_url: str, payload: dict) -> None:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    signature = make_signature(raw)
    headers = {
        "Content-Type": "application/json",
        "X-WPAB-Signature": signature,
    }
    timeout = (DOWNLOAD_CONNECT_TIMEOUT_SECONDS, DOWNLOAD_READ_TIMEOUT_SECONDS)
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        _sleep_for_retry(attempt)
        try:
            response = requests.post(callback_url, data=raw, headers=headers, timeout=timeout)
            if response.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRY_ATTEMPTS:
                logger.warning(
                    "callback_retry job_id=%s attempt=%s status=%s",
                    payload.get("job_id"),
                    attempt,
                    response.status_code,
                )
                continue
            response.raise_for_status()
            return
        except requests.exceptions.RequestException as exc:
            last_error = exc
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError))
            if status_code in RETRYABLE_STATUS_CODES:
                retryable = True
            if retryable and attempt < MAX_RETRY_ATTEMPTS:
                logger.warning(
                    "callback_retry job_id=%s attempt=%s status=%s error=%s",
                    payload.get("job_id"),
                    attempt,
                    status_code,
                    exc.__class__.__name__,
                )
                continue
            if getattr(exc, "response", None) is not None:
                raise _error_with_response_context("Callback failed", exc.response, exc) from exc
            raise RuntimeError(f"Callback failed; error={_truncate(str(exc))}") from exc
    raise RuntimeError(f"Callback failed after retries; error={_truncate(str(last_error))}")


def process_transcription_job(
    attachment_id: int,
    audio_url: str,
    site_url: str,
    callback_url: str,
    transcription_model: str | None = None,
    chunk_seconds: int | None = None,
    job_id: str | None = None,
):
    del site_url  # accepted for parity with API contract

    started = time.time()
    model = transcription_model or DEFAULT_MODEL
    chunk_seconds = _safe_chunk_seconds(chunk_seconds)

    if not job_id:
        current_job = get_current_job()
        if current_job:
            job_id = current_job.id

    if not job_id:
        raise RuntimeError("job_id is required")

    job_dir = WORK_ROOT / "jobs" / job_id
    chunk_dir = job_dir / "chunks"
    source_file = job_dir / f"source{_source_extension(audio_url)}"
    job_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting job_id=%s attachment_id=%s", job_id, attachment_id)
    source_domain = _source_url_domain(audio_url)

    current_job = get_current_job()
    if current_job:
        current_job.meta["attachment_id"] = attachment_id
        current_job.meta["source_url_domain"] = source_domain
        current_job.meta["audio_url"] = _truncate(audio_url, limit=200)
        current_job.save_meta()

    logger.info(
        "job_start job_id=%s attachment_id=%s source_url_domain=%s max_download_mb=%s keep_files=%s",
        job_id,
        attachment_id,
        source_domain,
        MAX_DOWNLOAD_MB,
        os.getenv("WPAB_KEEP_JOB_FILES", "0") == "1",
    )

    try:
        _download_audio_with_guardrails(audio_url, source_file, job_id)

        chunks = _run_ffmpeg_chunking(source_file, chunk_dir, chunk_seconds)
        chunk_sizes = [chunk.stat().st_size for chunk in chunks]
        logger.info(
            "chunking_done job_id=%s num_chunks=%s chunk_size_min=%s chunk_size_max=%s",
            job_id,
            len(chunks),
            min(chunk_sizes),
            max(chunk_sizes),
        )
        transcript_parts = []

        for index, chunk in enumerate(chunks, start=1):
            logger.info("chunk_transcribe_start job_id=%s chunk_index=%s total_chunks=%s", job_id, index, len(chunks))
            text = _transcribe_chunk(chunk, model, job_id)
            transcript_parts.append(text)

        total_duration = _ffprobe_duration_seconds(source_file)
        if total_duration is None:
            total_duration = 0.0
            for chunk in chunks:
                chunk_dur = _ffprobe_duration_seconds(chunk)
                if chunk_dur is not None:
                    total_duration += chunk_dur

        done_payload = {
            "attachment_id": attachment_id,
            "status": "done",
            "transcript": "\n\n".join(transcript_parts),
            "seconds": int(round(total_duration)),
            "model": model,
            "job_id": job_id,
            "error": None,
        }
        _post_callback(callback_url, done_payload)
        logger.info(
            "job_end job_id=%s status=success runtime_seconds=%.3f transcript_chars=%s",
            job_id,
            time.time() - started,
            len(done_payload["transcript"]),
        )
        return done_payload

    except Exception as exc:
        logger.exception("job_end job_id=%s status=failed runtime_seconds=%.3f", job_id, time.time() - started)
        error_payload = {
            "attachment_id": attachment_id,
            "status": "error",
            "transcript": "",
            "seconds": 0,
            "model": model,
            "job_id": job_id,
            "error": str(exc),
        }
        try:
            _post_callback(callback_url, error_payload)
        except Exception:
            logger.exception("Failed to send error callback for job_id=%s", job_id)
        raise
    finally:
        if os.getenv("WPAB_KEEP_JOB_FILES", "0") != "1":
            shutil.rmtree(job_dir, ignore_errors=True)
