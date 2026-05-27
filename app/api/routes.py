"""
FastAPI application routes.
"""

import time as time_module

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError
from redis import Redis
from rq import Queue
from rq.job import Job

from app.core.config import settings
from app.core.logging import logger
from app.core.security import verify_request
from app.models.payloads import TranscribeRequest, JobStatusResponse
from app.services.job_service import run_transcription_job

router = APIRouter()


# --- Redis / RQ connection (lazy init) ---

_redis_conn: Redis | None = None
_queue: Queue | None = None


def get_redis() -> Redis:
    global _redis_conn
    if _redis_conn is None:
        _redis_conn = Redis.from_url(settings.REDIS_URL)
    return _redis_conn


def get_queue() -> Queue:
    global _queue
    if _queue is None:
        _queue = Queue(settings.QUEUE_NAME, connection=get_redis())
    return _queue


# --- Helpers ---


def _safe_truncate(value: str | None, limit: int = 200) -> str | None:
    if value is None:
        return None
    return value if len(value) <= limit else f"{value[:limit]}..."


def _fmt_dt(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, time_module.struct_time):
        return time_module.strftime("%Y-%m-%dT%H:%M:%SZ", value)
    return str(value)


def _normalize_status(status: str) -> str:
    mapping = {
        "queued": "queued",
        "started": "running",
        "deferred": "queued",
        "finished": "completed",
        "failed": "failed",
        "stopped": "failed",
    }
    return mapping.get(status, status)


def _raise_error(status_code: int, code: str, message: str) -> None:
    raise HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


# --- Endpoints ---


@router.get("/health")
def health():
    return {"ok": True}


@router.post("/v1/jobs/transcribe")
async def submit_transcription_job(request: Request):
    """
    Receive a signed transcription job, verify it, and enqueue it.
    Returns immediately — processing happens in the background worker.
    """
    raw_body = await request.body()
    x_signature = request.headers.get("x-wpab-signature", "")
    x_site_id = request.headers.get("x-wpab-site-id", "")
    x_timestamp = request.headers.get("x-wpab-timestamp", "")

    if not x_signature or not x_site_id or not x_timestamp:
        _raise_error(401, "INVALID_SIGNATURE", "Missing signature headers.")

    try:
        timestamp = int(x_timestamp)
    except ValueError:
        _raise_error(401, "TIMESTAMP_EXPIRED", "Invalid request timestamp.")

    if not verify_request(
        raw_body,
        x_signature,
        x_site_id,
        timestamp,
    ):
        _raise_error(401, "INVALID_SIGNATURE", "Invalid or missing signature.")

    try:
        payload = TranscribeRequest.model_validate_json(raw_body)
    except ValidationError:
        _raise_error(422, "INVALID_PAYLOAD", "Invalid transcription job payload.")
    except ValueError:
        _raise_error(400, "INVALID_PAYLOAD", "Invalid JSON body.")

    if payload.site_id and payload.site_id != x_site_id:
        _raise_error(401, "INVALID_SIGNATURE", "Payload site does not match signed site.")

    if payload.timestamp and payload.timestamp != timestamp:
        _raise_error(401, "INVALID_SIGNATURE", "Payload timestamp does not match signed timestamp.")

    queue = get_queue()
    rq_job = queue.enqueue(
        run_transcription_job,
        attachment_id=payload.attachment_id,
        audio_url=payload.audio_url,
        callback_url=payload.callback_url,
        site_id=x_site_id,
        model=payload.model,
        chunk_seconds=payload.chunk_seconds,
        job_uuid=payload.job_uuid,
        job_id=str(payload.job_id),
        provider=payload.provider,
        provider_config=payload.provider_config,
        job_timeout=settings.QUEUE_DEFAULT_TIMEOUT,
        result_ttl=settings.JOB_RESULT_TTL,
    )

    logger.info(
        "job_enqueued rq_job_id=%s attachment_id=%s site_id=%s",
        rq_job.id, payload.attachment_id, x_site_id,
    )

    return {"ok": True, "data": {"job_id": rq_job.id, "status": "accepted"}}


@router.get("/v1/jobs/{job_id}")
def get_job_status(job_id: str):
    """Get the current status of a transcription job."""
    try:
        job = Job.fetch(job_id, connection=get_redis())
    except Exception:
        _raise_error(404, "JOB_NOT_FOUND", "Job not found.")

    status = job.get_status(refresh=True)
    meta = job.meta or {}
    exc_info = job.exc_info or ""
    error_summary = _safe_truncate(exc_info.splitlines()[-1] if exc_info else None, limit=500)

    return JobStatusResponse(
        job_id=job.id,
        status=_normalize_status(status),
        enqueued_at=_fmt_dt(job.enqueued_at),
        started_at=_fmt_dt(job.started_at),
        ended_at=_fmt_dt(job.ended_at),
        error=error_summary,
        attachment_id=meta.get("attachment_id"),
    ).model_dump(exclude_none=True)
