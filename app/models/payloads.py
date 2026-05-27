"""
Pydantic models for worker API request and response payloads.
"""

from pydantic import BaseModel, Field
from typing import Optional


class TranscribeRequest(BaseModel):
    """Inbound job submission payload from WP Audio Buddy."""

    job_id: int
    job_uuid: str
    attachment_id: int
    operation: str = "transcribe"
    audio_url: str
    callback_url: str
    model: Optional[str] = None
    chunk_seconds: Optional[int] = None
    timestamp: int = 0
    site_id: Optional[str] = None


class TranscribeResponse(BaseModel):
    """Response returned immediately after enqueueing a job."""

    ok: bool = True
    data: dict = Field(default_factory=lambda: {"job_id": ""})


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    ok: bool = False
    error: dict = Field(default_factory=lambda: {"code": "UNKNOWN", "message": "An error occurred"})


class JobStatusResponse(BaseModel):
    """Job status returned to WordPress."""

    job_id: str
    status: str
    enqueued_at: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    error: Optional[str] = None
    attachment_id: Optional[int] = None


class CallbackSuccess(BaseModel):
    """Payload sent to WordPress on successful transcription."""

    attachment_id: int
    status: str = "done"
    transcript: str
    seconds: int = 0
    model: str
    job_uuid: str = ""
    timestamp: int = 0


class CallbackFailure(BaseModel):
    """Payload sent to WordPress on failed transcription."""

    attachment_id: int
    status: str = "error"
    transcript: str = ""
    seconds: int = 0
    model: str = ""
    job_uuid: str = ""
    error: str
    timestamp: int = 0