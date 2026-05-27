"""
RQ worker entrypoint for processing transcription jobs.

Run with:
    rq worker wpab-transcription --url redis://localhost:6379
"""

from app.core.logging import logger
from app.core.config import settings

logger.info(
    "Worker process starting. queue=%s redis=%s",
    settings.QUEUE_NAME,
    settings.REDIS_URL,
)