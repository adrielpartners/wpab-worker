import logging
import os
from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException, Request
from redis import Redis
from rq import Queue
from rq.job import Job
from rq.registry import FailedJobRegistry, StartedJobRegistry

from hmac_util import verify_signature
from jobs import process_transcription_job

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
QUEUE_NAME = os.getenv("RQ_QUEUE", "transcriptions")
redis_conn = Redis.from_url(REDIS_URL)
queue = Queue(name=QUEUE_NAME, connection=redis_conn)


@app.get("/health")
def health():
    return {"ok": True}


def _safe_truncate(value: str | None, limit: int = 200) -> str | None:
    if value is None:
        return None
    return value if len(value) <= limit else f"{value[:limit]}..."


def _fmt_dt(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)


def _normalize_status(status: str) -> str:
    if status in {"queued", "started", "finished", "failed"}:
        return status
    if status == "deferred":
        return "queued"
    if status == "stopped":
        return "failed"
    return status


@app.get("/v1/queue")
def queue_status():
    started_registry = StartedJobRegistry(name=QUEUE_NAME, connection=redis_conn)
    failed_registry = FailedJobRegistry(name=QUEUE_NAME, connection=redis_conn)
    return {
        "queue_name": QUEUE_NAME,
        "queued_count": len(queue),
        "started_count": len(started_registry),
        "failed_count": len(failed_registry),
    }


@app.get("/v1/job/{job_id}")
def job_status(job_id: str):
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found")

    status = job.get_status(refresh=True)
    meta = job.meta or {}
    exc_info = job.exc_info or ""
    error_summary = _safe_truncate(exc_info.splitlines()[-1] if exc_info else None, limit=500)

    return {
        "job_id": job.id,
        "status": _normalize_status(status),
        "enqueued_at": _fmt_dt(job.enqueued_at),
        "started_at": _fmt_dt(job.started_at),
        "ended_at": _fmt_dt(job.ended_at),
        "error": error_summary,
        "attachment_id": meta.get("attachment_id"),
        "source_url_domain": _safe_truncate(meta.get("source_url_domain"), limit=100),
        "audio_url": _safe_truncate(meta.get("audio_url"), limit=200),
    }


@app.get("/v1/failed")
def failed_jobs(limit: int = 20):
    safe_limit = max(1, min(limit, 100))
    failed_registry = FailedJobRegistry(name=QUEUE_NAME, connection=redis_conn)
    job_ids = failed_registry.get_job_ids()[:safe_limit]
    items = []
    for jid in job_ids:
        try:
            job = Job.fetch(jid, connection=redis_conn)
        except Exception:
            continue
        exc_info = job.exc_info or ""
        items.append(
            {
                "job_id": job.id,
                "failed_at": _fmt_dt(job.ended_at),
                "error": _safe_truncate(exc_info.splitlines()[-1] if exc_info else None, limit=300),
            }
        )
    return {"queue_name": QUEUE_NAME, "failed": items}


@app.post("/v1/transcribe")
async def transcribe(request: Request, x_wpab_signature: str = Header(default="")):
    raw_body = await request.body()
    if not verify_signature(raw_body, x_wpab_signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    payload = await request.json()

    job = queue.enqueue(
        process_transcription_job,
        attachment_id=payload["attachment_id"],
        audio_url=payload["audio_url"],
        site_url=payload.get("site_url", ""),
        callback_url=payload["callback_url"],
        transcription_model=payload.get("transcription_model"),
        chunk_seconds=payload.get("chunk_seconds"),
        job_timeout="2h",
        result_ttl=86400,
    )

    logger.info("Enqueued transcription job %s", job.id)
    return {"job_id": job.id}
