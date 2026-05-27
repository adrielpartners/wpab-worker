"""
Cleanup daemon entrypoint — runs periodically to remove stale job files.

Run standalone:
    python3 -m app.workers.cleanup_daemon
"""

import time

from app.core.config import settings
from app.core.logging import logger
from app.services.cleanup_service import cleanup_stale


def main() -> None:
    logger.info(
        "cleanup_daemon_start retention_hours=%s check_interval=%s",
        settings.RETENTION_HOURS,
        settings.CLEANUP_INTERVAL_SECONDS,
    )
    while True:
        try:
            removed = cleanup_stale()
            if removed:
                logger.info("cleanup_cycle_removed=%s", removed)
        except Exception:
            logger.exception("cleanup_cycle_error")
        time.sleep(settings.CLEANUP_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()