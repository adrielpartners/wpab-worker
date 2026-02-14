import logging
import os
from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from redis import Redis
from redis.exceptions import RedisError
from rq import Queue, Worker
from rq.job import Job
from rq.registry import FailedJobRegistry

from hmac_util import verify_signature
from jobs import canary_job, process_transcription_job

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
QUEUE_NAME = os.getenv("RQ_QUEUE", "transcriptions")
STATUS_TOKEN = os.getenv("WPAB_STATUS_TOKEN", "")

redis_conn = Redis.from_url(REDIS_URL)
queue = Queue(name=QUEUE_NAME, connection=redis_conn)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_token_guard(token: str | None) -> None:
    if STATUS_TOKEN and token != STATUS_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _redis_probe_connection() -> Redis:
    return Redis.from_url(REDIS_URL, socket_connect_timeout=1, socket_timeout=1, retry_on_timeout=False)


def _ready_payload() -> dict:
    payload = {
        "ok": False,
        "redis_ok": False,
        "queue_name": QUEUE_NAME,
        "queue_length": 0,
        "workers_count": 0,
        "failed_count": 0,
        "time": _now_iso(),
    }
    try:
        probe = _redis_probe_connection()
        payload["redis_ok"] = bool(probe.ping())
        q = Queue(name=QUEUE_NAME, connection=probe)
        payload["queue_length"] = int(q.count)
        payload["workers_count"] = len(Worker.all(connection=probe, queue=q))
        payload["failed_count"] = int(FailedJobRegistry(name=QUEUE_NAME, connection=probe).count)
        payload["ok"] = payload["redis_ok"]
    except (RedisError, OSError, ValueError) as exc:
        logger.warning("ready check failed: %s", exc)
    return payload


def _sanitize_object(value):
    if isinstance(value, dict):
        redacted = {}
        for key, val in value.items():
            if str(key).lower() == "transcript":
                redacted[key] = "[redacted]"
            else:
                redacted[key] = _sanitize_object(val)
        return redacted
    if isinstance(value, list):
        return [_sanitize_object(item) for item in value]
    return value


def _job_summary(job: Job) -> dict:
    info = {
        "job_id": job.id,
        "status": job.get_status(refresh=False),
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "enqueued_at": job.enqueued_at.isoformat() if job.enqueued_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "ended_at": job.ended_at.isoformat() if job.ended_at else None,
        "meta": _sanitize_object(job.meta or {}),
    }
    if job.result is not None:
        if isinstance(job.result, (dict, list)):
            info["result"] = _sanitize_object(job.result)
        else:
            info["result"] = {"value": str(job.result)[:300]}
    else:
        info["result"] = None
    return info


@app.get("/")
def dashboard() -> HTMLResponse:
    html = """<!doctype html>
<html>
<head><meta charset=\"utf-8\"><title>WPAB Worker Status</title>
<style>
body{font-family:system-ui,-apple-system,sans-serif;padding:20px;max-width:900px;margin:0 auto}
pre{background:#111;color:#d8f3dc;padding:12px;border-radius:8px;overflow:auto}
button{padding:8px 12px;border-radius:6px;border:1px solid #444;cursor:pointer}
.small{color:#666;font-size:12px}
</style></head>
<body>
<h1>WPAB Worker Status</h1>
<p class=\"small\">Polls <code>/readyz</code> every 5 seconds.</p>
<button id=\"refresh\">Refresh now</button>
<pre id=\"out\">Loading...</pre>
<script>
async function update(){
  try{
    const res=await fetch('/readyz',{cache:'no-store'});
    const json=await res.json();
    document.getElementById('out').textContent=JSON.stringify(json,null,2);
  }catch(err){
    document.getElementById('out').textContent='Error: '+String(err);
  }
}
document.getElementById('refresh').addEventListener('click',update);
update();
setInterval(update,5000);
</script>
</body></html>"""
    return HTMLResponse(content=html)


@app.get("/health")
def health():
    return {"ok": True}



@app.get("/readyz")
def readyz():
    return _ready_payload()


@app.get("/api/status")
def api_status(token: str | None = Query(default=None)):
    _status_token_guard(token)
    ready = _ready_payload()
    try:
        q = Queue(name=QUEUE_NAME, connection=redis_conn)
        recent_ids = q.job_ids[-10:]
        jobs = []
        for job_id in recent_ids:
            job = Job.fetch(job_id, connection=redis_conn)
            jobs.append({"job_id": job.id, "status": job.get_status(refresh=False)})
    except Exception as exc:
        logger.warning("status listing failed: %s", exc)
        jobs = []
    ready["recent_jobs"] = jobs
    return ready


@app.post("/api/canary")
def api_canary(token: str | None = Query(default=None)):
    _status_token_guard(token)
    job = queue.enqueue(canary_job, job_timeout=30, result_ttl=3600)
    return {"job_id": job.id}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str, token: str | None = Query(default=None)):
    _status_token_guard(token)
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_summary(job)


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
