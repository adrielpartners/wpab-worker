"""
Job orchestration service — coordinates the full processing pipeline.
"""

import shutil
import time
from pathlib import Path

from rq import get_current_job

from app.core.config import settings
from app.core.logging import logger
from app.models.payloads import CallbackSuccess, CallbackFailure
from app.services.audio_download_service import download_audio
from app.services.chunking_service import chunk_audio
from app.services.transcription_service import transcribe_chunk
from app.services.result_assembly_service import assemble_transcript
from app.services.callback_client import send_callback


def _safe_error(exc: Exception) -> str:
    """Map internal exceptions to callback-safe error messages."""
    message = str(exc).lower()
    if "file too large" in message:
        return "The audio file is larger than the worker limit."
    if "unsupported url scheme" in message or "invalid url" in message:
        return "The audio download URL is invalid."
    if "download" in message or "status=403" in message or "status=404" in message or exc.__class__.__module__.startswith("requests"):
        return "The audio file could not be downloaded."
    if "ffmpeg" in message or "chunk" in message or "no chunks" in message:
        return "The audio file could not be prepared for transcription."
    if "openai" in message or "transcription" in message:
        return "The audio could not be transcribed."
    if "callback" in message:
        return "The worker could not send the result callback."
    return "The transcription job failed."


def run_transcription_job(
    attachment_id: int,
    audio_url: str,
    callback_url: str,
    site_id: str = "",
    model: str | None = None,
    chunk_seconds: int | None = None,
    job_uuid: str = "",
    job_id: str | None = None,
) -> dict:
    """
    Execute a full transcription job pipeline.

    This function is designed to be called by an RQ worker.
    """
    started = time.time()
    resolved_model = model or settings.DEFAULT_MODEL
    resolved_chunk_seconds = max(1, min(int(chunk_seconds or settings.DEFAULT_CHUNK_SECONDS), settings.MAX_CHUNK_SECONDS))

    # Resolve job_id.
    if not job_id:
        current_job = get_current_job()
        if current_job:
            job_id = current_job.id
    if not job_id:
        raise RuntimeError("job_id is required")

    # Prepare directories.
    job_dir = Path(settings.WORK_ROOT) / "jobs" / job_id
    chunk_dir = job_dir / "chunks"
    job_dir.mkdir(parents=True, exist_ok=True)

    # Store metadata in RQ job.
    current_job = get_current_job()
    if current_job:
        from urllib.parse import urlparse
        source_domain = (urlparse(audio_url).netloc or "unknown").strip()[:255]
        current_job.meta["attachment_id"] = attachment_id
        current_job.meta["source_url_domain"] = source_domain
        current_job.save_meta()

    logger.info(
        "job_start job_id=%s attachment_id=%s model=%s max_download_mb=%s",
        job_id, attachment_id, resolved_model, settings.MAX_DOWNLOAD_MB,
    )

    try:
        # Phase 1: Download
        download_result = download_audio(audio_url, job_dir, job_id)

        # Phase 2: Chunk
        chunks = chunk_audio(download_result.source_path, chunk_dir, resolved_chunk_seconds)

        # Phase 3: Transcribe each chunk
        chunk_transcripts = []
        for index, chunk in enumerate(chunks, start=1):
            logger.info(
                "chunk_transcribe_start job_id=%s chunk_index=%s total_chunks=%s",
                job_id, index, len(chunks),
            )
            text = transcribe_chunk(chunk, resolved_model, job_id)
            chunk_transcripts.append(text)

        # Phase 4: Assemble
        assembly = assemble_transcript(chunks, chunk_transcripts, download_result.source_path, resolved_model, job_id)

        # Phase 5: Send success callback
        success_payload = CallbackSuccess(
            job_id=job_id,
            status="done",
            attachment_id=attachment_id,
            job_uuid=job_uuid,
            transcript=assembly.transcript,
            model=resolved_model,
            seconds=int(round(assembly.total_duration)),
        )
        send_callback(callback_url, success_payload.model_dump(), site_id, job_id)

        elapsed = time.time() - started
        logger.info(
            "job_end job_id=%s status=success runtime_seconds=%.3f transcript_chars=%s",
            job_id, elapsed, len(assembly.transcript),
        )
        return success_payload.model_dump()

    except Exception as exc:
        elapsed = time.time() - started
        logger.exception("job_end job_id=%s status=failed runtime_seconds=%.3f", job_id, elapsed)

        error_payload = CallbackFailure(
            job_id=job_id,
            status="error",
            attachment_id=attachment_id,
            job_uuid=job_uuid,
            error=_safe_error(exc),
        )
        try:
            send_callback(callback_url, error_payload.model_dump(), site_id, job_id)
        except Exception:
            logger.exception("Failed to send error callback for job_id=%s", job_id)
        raise

    finally:
        if not settings.KEEP_JOB_FILES:
            shutil.rmtree(job_dir, ignore_errors=True)
