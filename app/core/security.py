"""
HMAC request signing and verification with multi-site support.

All WordPress -> worker and worker -> WordPress communication
uses SHA-256 HMAC signatures with site-specific secrets.
"""

import hashlib
import hmac
import time

from app.core.config import settings
from app.core.logging import logger


def get_site_secret(site_id: str) -> str | None:
    """Look up the shared secret for a given site ID."""
    return settings.allowed_sites.get(site_id, None)


def sign_payload(payload: bytes, secret: str) -> str:
    """Generate an HMAC-SHA256 hex signature for the given payload."""
    return hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify that the provided signature matches the expected HMAC."""
    expected = sign_payload(payload, secret)
    return hmac.compare_digest(expected, signature.strip())


def verify_request(
    payload: bytes,
    signature: str,
    site_id: str,
    timestamp: int,
) -> bool:
    """
    Full request verification: site ID lookup, timestamp tolerance, HMAC check.

    Returns True if the request is valid, False otherwise.
    """
    site_secret = get_site_secret(site_id)
    if site_secret is None:
        logger.warning("Unknown site_id=%s", site_id)
        return False

    now = int(time.time())
    if abs(now - timestamp) > settings.REQUEST_TIMESTAMP_TOLERANCE:
        logger.warning(
            "stale_timestamp site_id=%s timestamp=%s now=%s tolerance=%s",
            site_id,
            timestamp,
            now,
            settings.REQUEST_TIMESTAMP_TOLERANCE,
        )
        return False

    if not verify_signature(payload, signature, site_secret):
        logger.warning("invalid_signature site_id=%s", site_id)
        return False

    return True


def make_callback_signature(payload: bytes, site_id: str) -> str | None:
    """
    Generate a signature suitable for callback to the given site.

    This uses the site's shared secret so the WordPress plugin can verify it.
    """
    site_secret = get_site_secret(site_id)
    if site_secret is None:
        return None
    return sign_payload(payload, site_secret)