import hashlib
import hmac
import time

from app.core import security


def test_verify_request_accepts_plugin_signature(monkeypatch):
    body = b'{"job_id":1}'
    site_id = "site-a"
    timestamp = int(time.time())
    secret = "shared-secret"

    monkeypatch.setattr(security.settings, "ALLOWED_SITES_RAW", f"{site_id}={secret}")
    signature = security.sign_request_payload(timestamp, site_id, body, secret)

    assert security.verify_request(
        body,
        signature,
        site_id,
        timestamp,
    )


def test_verify_request_rejects_invalid_signature(monkeypatch):
    body = b'{"job_id":1}'
    site_id = "site-a"
    timestamp = int(time.time())
    secret = "shared-secret"

    monkeypatch.setattr(security.settings, "ALLOWED_SITES_RAW", f"{site_id}={secret}")

    assert not security.verify_request(
        body,
        "bad-signature",
        site_id,
        timestamp,
    )


def test_verify_request_rejects_stale_timestamp(monkeypatch):
    body = b'{"job_id":1}'
    site_id = "site-a"
    timestamp = int(time.time()) - security.settings.REQUEST_TIMESTAMP_TOLERANCE - 1
    secret = "shared-secret"

    monkeypatch.setattr(security.settings, "ALLOWED_SITES_RAW", f"{site_id}={secret}")
    signature = security.sign_request_payload(timestamp, site_id, body, secret)

    assert not security.verify_request(body, signature, site_id, timestamp)


def test_make_callback_signature_uses_plugin_payload(monkeypatch):
    site_id = "site-a"
    timestamp = int(time.time())
    body = f'{{"status":"done","timestamp":{timestamp},"site_id":"site-a"}}'.encode("utf-8")
    secret = "shared-secret"

    monkeypatch.setattr(security.settings, "ALLOWED_SITES_RAW", f"{site_id}={secret}")
    signature = security.make_callback_signature(body, site_id, timestamp)
    expected = hmac.new(
        secret.encode("utf-8"),
        f"{timestamp}\n{site_id}\n{body.decode('utf-8')}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    assert signature == expected
    assert security.verify_request(
        body,
        signature,
        site_id,
        timestamp,
    )
