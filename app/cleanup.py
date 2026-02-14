import logging
import os
import shutil
import time
from pathlib import Path

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WORK_ROOT = Path(os.getenv("WPAB_WORK_ROOT", "/work"))
RETENTION_HOURS = int(os.getenv("WPAB_RETENTION_HOURS", "168"))
CHECK_INTERVAL_SECONDS = 3600


def cleanup_once() -> None:
    jobs_root = WORK_ROOT / "jobs"
    if not jobs_root.exists():
        return

    cutoff = time.time() - (RETENTION_HOURS * 3600)
    for path in jobs_root.iterdir():
        try:
            mtime = path.stat().st_mtime
            if mtime < cutoff:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
                logger.info("Deleted old path: %s", path)
        except FileNotFoundError:
            continue
        except Exception:
            logger.exception("Failed deleting path: %s", path)


def main() -> None:
    logger.info("Starting cleanup daemon. retention_hours=%s", RETENTION_HOURS)
    while True:
        cleanup_once()
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
