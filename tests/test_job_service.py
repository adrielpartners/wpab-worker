from app.services.job_service import _safe_error
from app.models.payloads import CallbackFailure


def test_safe_error_does_not_expose_provider_body():
    error = _safe_error(RuntimeError("OpenAI transcription failed; status=500; body=provider secret detail"))

    assert error == "The audio could not be transcribed."


def test_safe_error_maps_large_audio():
    error = _safe_error(RuntimeError("file too large: exceeds max 200MB"))

    assert error == "The audio file is larger than the worker limit."


def test_failure_callback_format():
    payload = CallbackFailure(
        job_id="123",
        job_uuid="abc-123",
        attachment_id=456,
        error="The audio file could not be downloaded.",
        timestamp=111,
        site_id="site-a",
    ).model_dump(exclude_none=True)

    assert payload == {
        "job_id": "123",
        "status": "error",
        "attachment_id": 456,
        "job_uuid": "abc-123",
        "error": "The audio file could not be downloaded.",
        "timestamp": 111,
        "site_id": "site-a",
    }
