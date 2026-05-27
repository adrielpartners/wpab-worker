"""
Temporary file cleanup service.

Removes old job directories and stale files to prevent disk growth.
"""

import shutil
import time
from pathlib import Path

from app.core.config import settings
from app.core.logging import logger


def cleanup_job_directory(job_dir: Path) -> None:
    """Remove a single job directory."""
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
        logger.info("cleanup_removed path=%s", job_dir)


def cleanup_stale() -> int:
    """Remove all job directories older than RETENTION_HOURS. Returns count removed."""
    jobs_root = Path(settings.WORK_ROOT) / "jobs"
    if not jobs_root.exists():
        return 0

    cutoff = time.time() - (settings.RETENTION_HOURS * 3600)
    removed = 0

    for path in jobs_root.iterdir():
        try:
            mtime = path.stat().st_mtime
            if mtime < cutoff:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
                removed += 1
                logger.info("cleanup_stale_removed path=%s", path)
        except FileNotFoundError:
            continue
        except Exception:
            logger.exception("cleanup_error path=%s", path)

    return removed