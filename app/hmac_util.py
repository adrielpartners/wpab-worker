import hashlib
import hmac
import os


def get_shared_secret() -> str:
    secret = os.getenv("WPAB_SHARED_SECRET", "")
    if not secret:
        raise RuntimeError("WPAB_SHARED_SECRET is required")
    return secret


def make_signature(payload: bytes, secret: str | None = None) -> str:
    key = (secret or get_shared_secret()).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def verify_signature(payload: bytes, provided_signature: str) -> bool:
    if not provided_signature:
        return False
    expected = make_signature(payload)
    return hmac.compare_digest(expected, provided_signature.strip())
