"""
RQ worker entrypoint for processing transcription jobs.

Run with:
    rq worker wpab-transcription --url redis://localhost:6379

The import at the top ensures RQ can resolve run_transcription_job
when it deserializes queued jobs.
"""

from app.core.logging import logger
from app.core.config import settings
from app.services.job_service import run_transcription_job  # noqa: F401 — registered for RQ

logger.info(
    "Worker process starting. queue=%s redis=%s",
    settings.QUEUE_NAME,
    settings.REDIS_URL,
)