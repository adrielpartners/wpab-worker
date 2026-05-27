"""
FastAPI application entry point for wpab-worker.

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8080

Or via Docker:
    docker compose up
"""

from fastapi import FastAPI

from app.core.config import settings
from app.core.logging import logger
from app.api.routes import router

app = FastAPI(
    title="WP AB Worker",
    description="Backend worker service for WP Audio Buddy transcription processing",
    version="0.2.0",
)

app.include_router(router)

logger.info(
    "Starting WP AB Worker. env=%s host=%s port=%s",
    settings.ENV,
    settings.HOST,
    settings.PORT,
)