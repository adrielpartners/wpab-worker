import logging
import os

from fastapi import FastAPI, Header, HTTPException, Request
from redis import Redis
from rq import Queue

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
