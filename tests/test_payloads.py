import pytest
from pydantic import ValidationError

from app.models.payloads import TranscribeRequest


def valid_payload(**overrides):
    payload = {
        "job_id": 123,
        "job_uuid": "abc-123",
        "attachment_id": 456,
        "operation": "transcribe",
        "audio_url": "https://example.com/audio.mp3",
        "callback_url": "https://example.com/wp-json/wpab/v1/worker-callback",
        "model": "gpt-4o-mini-transcribe",
        "chunk_seconds": 600,
    }
    payload.update(overrides)
    return payload


def test_transcribe_request_accepts_valid_payload():
    payload = TranscribeRequest.model_validate(valid_payload())

    assert payload.job_id == 123
    assert payload.attachment_id == 456
    assert payload.operation == "transcribe"
    assert payload.audio_url == "https://example.com/audio.mp3"


def test_transcribe_request_rejects_wrong_operation():
    with pytest.raises(ValidationError):
        TranscribeRequest.model_validate(valid_payload(operation="summarize"))


def test_transcribe_request_rejects_invalid_model():
    with pytest.raises(ValidationError):
        TranscribeRequest.model_validate(valid_payload(model="not a real model"))


def test_transcribe_request_accepts_openrouter_payload():
    payload = TranscribeRequest.model_validate(
        valid_payload(
            provider="openrouter",
            model="openai/whisper-large-v3",
            chunk_seconds=55,
            provider_config={
                "endpoint": "https://openrouter.ai/api",
                "api_key": "must-not-be-kept",
            },
        )
    )

    assert payload.provider == "openrouter"
    assert payload.model == "openai/whisper-large-v3"
    assert payload.provider_config == {"endpoint": "https://openrouter.ai/api"}


def test_transcribe_request_rejects_unsupported_provider():
    with pytest.raises(ValidationError):
        TranscribeRequest.model_validate(valid_payload(provider="nope"))


def test_transcribe_request_rejects_invalid_url():
    with pytest.raises(ValidationError):
        TranscribeRequest.model_validate(valid_payload(audio_url="file:///etc/passwd"))
