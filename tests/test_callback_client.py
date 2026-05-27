import json

from app.services import callback_client


class FakeResponse:
    status_code = 200

    def raise_for_status(self):
        return None


def test_send_callback_includes_full_hmac_headers(monkeypatch):
    captured = {}

    monkeypatch.setattr(callback_client, "make_callback_signature", lambda raw, site_id, timestamp: "abc123")

    def fake_post(url, data, headers, timeout):
        captured["url"] = url
        captured["data"] = data
        captured["headers"] = headers
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(callback_client.requests, "post", fake_post)

    callback_client.send_callback(
        "https://example.com/wp-json/wpab/v1/worker-callback",
        {"job_id": "123", "job_uuid": "abc-123", "attachment_id": 456, "status": "done"},
        "site-a",
        "job-1",
    )

    body = json.loads(captured["data"].decode("utf-8"))
    assert captured["headers"]["X-WPAB-Site-ID"] == "site-a"
    assert captured["headers"]["X-WPAB-Signature"] == "abc123"
    assert captured["headers"]["X-WPAB-Timestamp"].isdigit()
    assert body["timestamp"] == int(captured["headers"]["X-WPAB-Timestamp"])
    assert body["site_id"] == captured["headers"]["X-WPAB-Site-ID"]
    assert body["status"] == "done"
