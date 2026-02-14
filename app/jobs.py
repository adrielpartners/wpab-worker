import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from rq import get_current_job

from hmac_util import make_signature

logger = logging.getLogger(__name__)

WORK_ROOT = Path(os.getenv("WPAB_WORK_ROOT", "/work"))
DEFAULT_MODEL = os.getenv("WPAB_DEFAULT_MODEL", "gpt-4o-mini-transcribe")
DEFAULT_CHUNK_SECONDS = int(os.getenv("WPAB_DEFAULT_CHUNK_SECONDS", "720"))
MAX_CHUNK_SECONDS = 900


def _safe_chunk_seconds(value: int | None) -> int:
    if value is None:
        return DEFAULT_CHUNK_SECONDS
    return max(1, min(int(value), MAX_CHUNK_SECONDS))


def _source_extension(audio_url: str) -> str:
    path = urlparse(audio_url).path
    ext = Path(path).suffix
    return ext if ext else ".bin"


def _run_ffmpeg_chunking(source_path: Path, chunk_dir: Path, chunk_seconds: int) -> list[Path]:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = chunk_dir / "chunk_%04d.wav"

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
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        str(output_pattern),
    ]
    logger.info("Running chunk command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    chunks = sorted(chunk_dir.glob("chunk_*.wav"))
    if not chunks:
        raise RuntimeError("No chunks created by ffmpeg")
    return chunks


def _ffprobe_duration_seconds(path: Path) -> float | None:
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


def _transcribe_chunk(chunk_path: Path, model: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    with chunk_path.open("rb") as audio_file:
        files = {"file": (chunk_path.name, audio_file, "audio/wav")}
        data = {"model": model}
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers=headers,
            data=data,
            files=files,
            timeout=300,
        )

    response.raise_for_status()
    payload = response.json()
    return payload.get("text", "").strip()


def _post_callback(callback_url: str, payload: dict) -> None:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    signature = make_signature(raw)
    headers = {
        "Content-Type": "application/json",
        "X-WPAB-Signature": signature,
    }
    response = requests.post(callback_url, data=raw, headers=headers, timeout=60)
    response.raise_for_status()


def process_transcription_job(
    attachment_id: int,
    audio_url: str,
    site_url: str,
    callback_url: str,
    transcription_model: str | None = None,
    chunk_seconds: int | None = None,
    job_id: str | None = None,
):
    del site_url  # accepted for parity with API contract

    started = time.time()
    model = transcription_model or DEFAULT_MODEL
    chunk_seconds = _safe_chunk_seconds(chunk_seconds)

    if not job_id:
        current_job = get_current_job()
        if current_job:
            job_id = current_job.id

    if not job_id:
        raise RuntimeError("job_id is required")

    job_dir = WORK_ROOT / "jobs" / job_id
    chunk_dir = job_dir / "chunks"
    source_file = job_dir / f"source{_source_extension(audio_url)}"
    job_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting job_id=%s attachment_id=%s", job_id, attachment_id)

    try:
        with requests.get(audio_url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with source_file.open("wb") as out:
                for piece in response.iter_content(chunk_size=1024 * 1024):
                    if piece:
                        out.write(piece)

        chunks = _run_ffmpeg_chunking(source_file, chunk_dir, chunk_seconds)
        transcript_parts = []

        for index, chunk in enumerate(chunks, start=1):
            logger.info("Transcribing chunk %s/%s for job %s", index, len(chunks), job_id)
            text = _transcribe_chunk(chunk, model)
            transcript_parts.append(text)

        total_duration = _ffprobe_duration_seconds(source_file)
        if total_duration is None:
            total_duration = 0.0
            for chunk in chunks:
                chunk_dur = _ffprobe_duration_seconds(chunk)
                if chunk_dur is not None:
                    total_duration += chunk_dur

        done_payload = {
            "attachment_id": attachment_id,
            "status": "done",
            "transcript": "\n\n".join(transcript_parts),
            "seconds": int(round(total_duration)),
            "model": model,
            "job_id": job_id,
            "error": None,
        }
        _post_callback(callback_url, done_payload)
        logger.info("Job completed in %.2fs job_id=%s", time.time() - started, job_id)
        return done_payload

    except Exception as exc:
        logger.exception("Job failed job_id=%s", job_id)
        error_payload = {
            "attachment_id": attachment_id,
            "status": "error",
            "transcript": "",
            "seconds": 0,
            "model": model,
            "job_id": job_id,
            "error": str(exc),
        }
        try:
            _post_callback(callback_url, error_payload)
        except Exception:
            logger.exception("Failed to send error callback for job_id=%s", job_id)
        raise
    finally:
        if os.getenv("WPAB_KEEP_JOB_FILES", "0") != "1":
            shutil.rmtree(job_dir, ignore_errors=True)
