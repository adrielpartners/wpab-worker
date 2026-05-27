"""
FFmpeg-based audio chunking service.
"""

import subprocess
import time
from pathlib import Path

from app.core.config import settings
from app.core.logging import logger


def probe_duration(path: Path) -> float | None:
    """Get the duration of an audio file in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return float(result.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return None


def chunk_audio(source_path: Path, chunk_dir: Path, chunk_seconds: int | None = None) -> list[Path]:
    """
    Split an audio file into chunks using ffmpeg segment mode.

    Returns a sorted list of chunk file paths.
    Raises RuntimeError if chunking fails.
    """
    seconds = chunk_seconds or settings.DEFAULT_CHUNK_SECONDS
    seconds = max(1, min(int(seconds), settings.MAX_CHUNK_SECONDS))

    chunk_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = chunk_dir / "chunk_%04d.mp3"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        "48k",
        "-f",
        "segment",
        "-segment_time",
        str(seconds),
        str(output_pattern),
    ]

    logger.info("chunking_start source=%s cmd=%s", source_path.name, " ".join(cmd))
    start = time.time()

    subprocess.run(cmd, check=True)

    chunks = sorted(chunk_dir.glob("chunk_*.mp3"))
    if not chunks:
        raise RuntimeError("No chunks created by ffmpeg")

    elapsed = time.time() - start
    chunk_sizes = [c.stat().st_size for c in chunks]
    logger.info(
        "chunking_done num_chunks=%s chunk_size_min=%s chunk_size_max=%s elapsed_seconds=%.3f",
        len(chunks),
        min(chunk_sizes),
        max(chunk_sizes),
        elapsed,
    )

    return chunks
