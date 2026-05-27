"""
Configuration module for wpab-worker.

All environment variables are loaded and validated centrally.
No module should read os.getenv() directly.
"""

import os
from typing import Dict


class Settings:
    """Application settings loaded from environment variables."""

    # Environment
    ENV: str = os.getenv("WPAB_WORKER_ENV", "development")
    LOG_LEVEL: str = os.getenv("WPAB_LOG_LEVEL", "INFO")
    PUBLIC_BASE_URL: str = os.getenv("WPAB_WORKER_PUBLIC_BASE_URL", "http://localhost:8080")

    # Redis / Queue
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    QUEUE_NAME: str = os.getenv("WPAB_QUEUE_NAME", "wpab-transcription")
    QUEUE_DEFAULT_TIMEOUT: str = os.getenv("WPAB_QUEUE_DEFAULT_TIMEOUT", "2h")
    JOB_RESULT_TTL: int = int(os.getenv("WPAB_JOB_RESULT_TTL", "86400"))

    # Provider API keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

    # OpenAI (legacy, also used by OpenAI provider)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_TIMEOUT: float = float(os.getenv("WPAB_OPENAI_TIMEOUT_SECONDS", "300"))

    # Audio download
    MAX_DOWNLOAD_MB: int = int(os.getenv("WPAB_MAX_DOWNLOAD_MB", "200"))
    MAX_DOWNLOAD_BYTES: int = MAX_DOWNLOAD_MB * 1024 * 1024
    DOWNLOAD_CONNECT_TIMEOUT: float = float(os.getenv("WPAB_DOWNLOAD_CONNECT_TIMEOUT_SECONDS", "10"))
    DOWNLOAD_READ_TIMEOUT: float = float(os.getenv("WPAB_DOWNLOAD_READ_TIMEOUT_SECONDS", "60"))

    # Audio chunking
    DEFAULT_CHUNK_SECONDS: int = int(os.getenv("WPAB_DEFAULT_CHUNK_SECONDS", "720"))
    MAX_CHUNK_SECONDS: int = 900

    # Transcription
    DEFAULT_MODEL: str = os.getenv("WPAB_DEFAULT_MODEL", "gpt-4o-mini-transcribe")
    MAX_RETRY_ATTEMPTS: int = int(os.getenv("WPAB_MAX_RETRY_ATTEMPTS", "4"))
    RETRYABLE_STATUS_CODES: set = {429, 500, 502, 503, 504}

    # Callback
    CALLBACK_TIMEOUT: float = float(os.getenv("WPAB_CALLBACK_TIMEOUT_SECONDS", "30"))
    CALLBACK_RETRY_ATTEMPTS: int = int(os.getenv("WPAB_CALLBACK_RETRY_ATTEMPTS", "4"))

    # HMAC / Security
    REQUEST_TIMESTAMP_TOLERANCE: int = int(os.getenv("WPAB_REQUEST_TIMESTAMP_TOLERANCE_SECONDS", "300"))

    # Sites: format "site_id1=secret1,site_id2=secret2"
    ALLOWED_SITES_RAW: str = os.getenv("WPAB_ALLOWED_SITES", "")

    # Temp storage
    WORK_ROOT: str = os.getenv("WPAB_WORK_ROOT", "/work")
    KEEP_JOB_FILES: bool = os.getenv("WPAB_KEEP_JOB_FILES", "0") == "1"

    # Server
    HOST: str = os.getenv("WPAB_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("WPAB_PORT", "8080"))

    # Cleanup
    RETENTION_HOURS: int = int(os.getenv("WPAB_RETENTION_HOURS", "168"))
    CLEANUP_INTERVAL_SECONDS: int = int(os.getenv("WPAB_CLEANUP_INTERVAL_SECONDS", "3600"))

    @property
    def allowed_sites(self) -> Dict[str, str]:
        """Parse the WPAB_ALLOWED_SITES env var into a dict of site_id -> secret."""
        result: Dict[str, str] = {}
        raw = self.ALLOWED_SITES_RAW.strip()
        if raw:
            for pair in raw.split(","):
                pair = pair.strip()
                if "=" in pair:
                    site_id, secret = pair.split("=", 1)
                    result[site_id.strip()] = secret.strip()
        return result

    def validate_production(self) -> None:
        """Validate that required settings are present in production."""
        if self.ENV == "production":
            if not any([self.OPENAI_API_KEY, self.GROQ_API_KEY, self.DEEPGRAM_API_KEY, self.OPENROUTER_API_KEY]):
                raise RuntimeError("At least one provider API key is required in production")
            if not self.ALLOWED_SITES_RAW:
                raise RuntimeError("WPAB_ALLOWED_SITES is required in production")
            if not self.allowed_sites:
                raise RuntimeError("WPAB_ALLOWED_SITES must contain at least one site_id=secret pair")


settings = Settings()
