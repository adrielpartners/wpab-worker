"""
FastAPI application entry point for wpab-worker.

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8080

Or via Docker:
    docker compose up
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import logger
from app.api.routes import router

app = FastAPI(
    title="WP AB Worker",
    description="Backend worker service for WP Audio Buddy transcription processing",
    version="0.2.0",
)

app.include_router(router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {}
    code = detail.get("code", "HTTP_ERROR")
    message = detail.get("message", "Request failed.")
    return JSONResponse(
        status_code=exc.status_code,
        content={"ok": False, "error": {"code": code, "message": message}},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "ok": False,
            "error": {
                "code": "INVALID_PAYLOAD",
                "message": "Invalid request payload.",
            },
        },
    )

logger.info(
    "Starting WP AB Worker. env=%s host=%s port=%s",
    settings.ENV,
    settings.HOST,
    settings.PORT,
)
