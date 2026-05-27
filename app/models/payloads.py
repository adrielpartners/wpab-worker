"""
Pydantic models for worker API request and response payloads.
"""

from typing import Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, PositiveInt, field_validator


class TranscribeRequest(BaseModel):
    """Inbound job submission payload from WP Audio Buddy."""

    job_id: PositiveInt
    job_uuid: str
    attachment_id: PositiveInt
    operation: str = "transcribe"
    audio_url: str
    callback_url: str
    model: Optional[str] = None
    chunk_seconds: Optional[PositiveInt] = None
    timestamp: int = 0
    site_id: Optional[str] = None

    @field_validator("operation")
    @classmethod
    def validate_operation(cls, value: str) -> str:
        if value != "transcribe":
            raise ValueError("operation must be transcribe")
        return value

    @field_validator("job_uuid")
    @classmethod
    def validate_job_uuid(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("job_uuid is required")
        return value

    @field_validator("audio_url", "callback_url")
    @classmethod
    def validate_http_url(cls, value: str) -> str:
        value = value.strip()
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("URL must be http or https")
        return value

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value = value.strip()
        if not value:
            return None
        allowed = {"gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"}
        if value not in allowed:
            raise ValueError("unsupported transcription model")
        return value


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

    job_id: str
    status: str = "done"
    attachment_id: int
    job_uuid: str = ""
    transcript: str
    model: str
    seconds: int = 0
    timestamp: Optional[int] = None
    site_id: Optional[str] = None


class CallbackFailure(BaseModel):
    """Payload sent to WordPress on failed transcription."""

    job_id: str
    status: str = "error"
    attachment_id: int
    job_uuid: str = ""
    error: str
    timestamp: Optional[int] = None
    site_id: Optional[str] = None
