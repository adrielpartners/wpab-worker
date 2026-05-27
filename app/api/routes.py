"""
FastAPI application routes.
"""

import time as time_module

from fastapi import APIRouter, Header, HTTPException, Request
from redis import Redis
from rq import Queue
from rq.job import Job
from rq.registry import FailedJobRegistry, StartedJobRegistry

from app.core.config import settings
from app.core.logging import logger
from app.core.security import verify_request
from app.models.payloads import TranscribeRequest, TranscribeResponse, ErrorResponse, JobStatusResponse
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
    x_timestamp = request.headers.get("x-wpab-timestamp", "0")

    # Parse timestamp header
    try:
        timestamp = int(x_timestamp)
    except ValueError:
        timestamp = 0

    # Verify HMAC
    if not verify_request(raw_body, x_signature, x_site_id, timestamp):
        raise HTTPException(status_code=401, detail="Invalid or missing signature")

    # Parse and validate payload
    try:
        payload_data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Extract fields from the payload (the TranscribeRequest model)
    attachment_id = payload_data.get("attachment_id", 0)
    audio_url = payload_data.get("audio_url", "")
    callback_url = payload_data.get("callback_url", "")
    site_id = payload_data.get("site_id", x_site_id)
    job_uuid = payload_data.get("job_uuid", "")
    model = payload_data.get("model", None)
    chunk_seconds = payload_data.get("chunk_seconds", None)
    job_id_from_payload = payload_data.get("job_id", None)

    if not attachment_id or not audio_url or not callback_url:
        raise HTTPException(status_code=400, detail="Missing required fields: attachment_id, audio_url, callback_url")

    # Enqueue the job
    queue = get_queue()
    rq_job = queue.enqueue(
        run_transcription_job,
        attachment_id=attachment_id,
        audio_url=audio_url,
        callback_url=callback_url,
        site_id=site_id,
        model=model,
        chunk_seconds=chunk_seconds,
        job_uuid=job_uuid,
        job_id=str(job_id_from_payload) if job_id_from_payload else None,
        job_timeout=settings.QUEUE_DEFAULT_TIMEOUT,
        result_ttl=settings.JOB_RESULT_TTL,
    )

    logger.info(
        "job_enqueued rq_job_id=%s attachment_id=%s site_id=%s",
        rq_job.id, attachment_id, site_id,
    )

    return {"ok": True, "data": {"job_id": rq_job.id}}


@router.get("/v1/jobs/{job_id}")
def get_job_status(job_id: str):
    """Get the current status of a transcription job."""
    try:
        job = Job.fetch(job_id, connection=get_redis())
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found")

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


@router.get("/v1/admin/queue")
def queue_status():
    """Get aggregate queue statistics."""
    redis_conn = get_redis()
    started_registry = StartedJobRegistry(name=settings.QUEUE_NAME, connection=redis_conn)
    failed_registry = FailedJobRegistry(name=settings.QUEUE_NAME, connection=redis_conn)
    q = get_queue()

    return {
        "queue_name": settings.QUEUE_NAME,
        "queued_count": len(q),
        "started_count": len(started_registry),
        "failed_count": len(failed_registry),
    }


@router.get("/v1/admin/failed")
def failed_jobs(limit: int = 20):
    """List recently failed jobs."""
    safe_limit = max(1, min(limit, 100))
    redis_conn = get_redis()
    failed_registry = FailedJobRegistry(name=settings.QUEUE_NAME, connection=redis_conn)
    job_ids = failed_registry.get_job_ids()[:safe_limit]

    items = []
    for jid in job_ids:
        try:
            job = Job.fetch(jid, connection=redis_conn)
        except Exception:
            continue
        exc_info = job.exc_info or ""
        items.append({
            "job_id": job.id,
            "failed_at": _fmt_dt(job.ended_at),
            "error": _safe_truncate(exc_info.splitlines()[-1] if exc_info else None, limit=300),
        })

    return {"queue_name": settings.QUEUE_NAME, "failed": items}