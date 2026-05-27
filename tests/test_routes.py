import json
import time

from fastapi.testclient import TestClient

from app.api import routes
from app.core import security
from app.main import app


class FakeRqJob:
    id = "rq-job-1"


class FakeQueue:
    def __init__(self):
        self.kwargs = None

    def enqueue(self, func, **kwargs):
        self.kwargs = kwargs
        return FakeRqJob()


def test_admin_routes_are_not_exposed():
    client = TestClient(app)

    assert client.get("/v1/admin/queue").status_code == 404
    assert client.get("/v1/admin/failed").status_code == 404


def test_invalid_signature_uses_error_envelope(monkeypatch):
    monkeypatch.setattr(security.settings, "ALLOWED_SITES_RAW", "site-a=shared-secret")
    client = TestClient(app)

    response = client.post(
        "/v1/jobs/transcribe",
        content=b"{}",
        headers={
            "X-WPAB-Site-ID": "site-a",
            "X-WPAB-Timestamp": str(int(time.time())),
            "X-WPAB-Signature": "bad",
        },
    )

    assert response.status_code == 401
    assert response.json() == {
        "ok": False,
        "error": {
            "code": "INVALID_SIGNATURE",
            "message": "Invalid or missing signature.",
        },
    }


def test_valid_signed_request_enqueues(monkeypatch):
    fake_queue = FakeQueue()
    monkeypatch.setattr(security.settings, "ALLOWED_SITES_RAW", "site-a=shared-secret")
    monkeypatch.setattr(routes, "get_queue", lambda: fake_queue)

    body = {
        "job_id": 123,
        "job_uuid": "abc-123",
        "attachment_id": 456,
        "operation": "transcribe",
        "audio_url": "https://example.com/audio.mp3?wpab_sig=keep%2Fexact",
        "callback_url": "https://example.com/wp-json/wpab/v1/worker-callback",
        "model": "openai/whisper-large-v3",
        "provider": "openrouter",
        "provider_config": {
            "endpoint": "https://openrouter.ai/api",
            "api_key": "must-not-be-kept",
        },
        "chunk_seconds": 55,
        "timestamp": 0,
        "site_id": "site-a",
    }
    timestamp = int(time.time())
    body["timestamp"] = timestamp
    raw = json.dumps(body, separators=(",", ":")).encode("utf-8")
    signature = security.sign_request_payload(
        timestamp,
        "site-a",
        raw,
        "shared-secret",
    )

    client = TestClient(app)
    response = client.post(
        "/v1/jobs/transcribe",
        content=raw,
        headers={
            "Content-Type": "application/json",
            "X-WPAB-Site-ID": "site-a",
            "X-WPAB-Timestamp": str(timestamp),
            "X-WPAB-Signature": signature,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True, "data": {"job_id": "rq-job-1", "status": "accepted"}}
    assert fake_queue.kwargs["attachment_id"] == 456
    assert fake_queue.kwargs["job_id"] == "123"
    assert fake_queue.kwargs["audio_url"] == "https://example.com/audio.mp3?wpab_sig=keep%2Fexact"
    assert fake_queue.kwargs["provider"] == "openrouter"
    assert fake_queue.kwargs["provider_config"] == {"endpoint": "https://openrouter.ai/api"}
    assert fake_queue.kwargs["model"] == "openai/whisper-large-v3"
    assert fake_queue.kwargs["chunk_seconds"] == 55
