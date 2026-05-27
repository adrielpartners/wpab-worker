"""
Safe audio download from WordPress signed URLs.
"""

import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from app.core.config import settings
from app.core.logging import logger


class DownloadResult:
    """Result of a successful audio download."""

    def __init__(self, source_path: Path, bytes_downloaded: int, elapsed: float, source_domain: str):
        self.source_path = source_path
        self.bytes_downloaded = bytes_downloaded
        self.elapsed = elapsed
        self.source_domain = source_domain


def download_audio(audio_url: str, job_dir: Path, job_id: str) -> DownloadResult:
    """
    Download audio from a signed WordPress URL with guardrails.

    Raises RuntimeError on validation failure, oversized file, or network error.
    """
    parsed = urlparse(audio_url)
    if parsed.scheme not in {"http", "https"}:
        raise RuntimeError(f"Unsupported URL scheme '{parsed.scheme}'. Only http/https are allowed")

    source_domain = (parsed.netloc or "unknown").strip()[:255]
    ext = Path(parsed.path).suffix or ".bin"
    source_file = job_dir / f"source{ext}"

    timeout = (settings.DOWNLOAD_CONNECT_TIMEOUT, settings.DOWNLOAD_READ_TIMEOUT)
    start = time.time()
    bytes_downloaded = 0

    with requests.get(audio_url, stream=True, timeout=timeout) as response:
        if response.status_code in {403, 404}:
            logger.warning(
                "download_signed_url_rejected job_id=%s source_url_domain=%s status=%s",
                job_id,
                source_domain,
                response.status_code,
            )
            raise RuntimeError(f"audio download failed: status={response.status_code}")
        response.raise_for_status()
        content_length_raw = response.headers.get("Content-Length")
        content_type = (response.headers.get("Content-Type") or "").strip().lower()

        if content_type and not (
            content_type.startswith("audio/")
            or content_type.startswith("video/")
            or content_type.startswith("application/octet-stream")
        ):
            logger.warning(
                "download_content_type_unexpected job_id=%s source_url_domain=%s content_type=%s",
                job_id,
                source_domain,
                content_type,
            )

        if content_length_raw:
            try:
                content_length = int(content_length_raw)
            except ValueError:
                content_length = None
            if content_length is not None and content_length > settings.MAX_DOWNLOAD_BYTES:
                raise RuntimeError(
                    f"file too large: {round(content_length / (1024 * 1024), 2)}MB"
                    f" exceeds max {settings.MAX_DOWNLOAD_MB}MB"
                )

        logger.info(
            "download_start job_id=%s source_url_domain=%s content_length=%s",
            job_id,
            source_domain,
            content_length_raw,
        )

        with source_file.open("wb") as out:
            for piece in response.iter_content(chunk_size=1024 * 1024):
                if not piece:
                    continue
                bytes_downloaded += len(piece)
                if bytes_downloaded > settings.MAX_DOWNLOAD_BYTES:
                    raise RuntimeError(f"file too large: exceeds max {settings.MAX_DOWNLOAD_MB}MB")
                out.write(piece)

    elapsed = time.time() - start
    logger.info(
        "download_done job_id=%s source_url_domain=%s bytes_downloaded=%s elapsed_seconds=%.3f",
        job_id,
        source_domain,
        bytes_downloaded,
        elapsed,
    )

    return DownloadResult(
        source_path=source_file,
        bytes_downloaded=bytes_downloaded,
        elapsed=elapsed,
        source_domain=source_domain,
    )
