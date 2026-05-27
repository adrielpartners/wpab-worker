"""
Stitch chunk transcripts into a coherent final result.
"""

from pathlib import Path

from app.core.logging import logger
from app.services.chunking_service import probe_duration


class AssemblyResult:
    """The assembled transcript result ready for callback."""

    def __init__(self, transcript: str, total_duration: float, num_chunks: int, model: str):
        self.transcript = transcript
        self.total_duration = total_duration
        self.num_chunks = num_chunks
        self.model = model


def assemble_transcript(
    chunk_paths: list[Path],
    chunk_transcripts: list[str],
    source_path: Path,
    model: str,
    job_id: str,
) -> AssemblyResult:
    """
    Stitch ordered chunk transcripts into a single result.

    Args:
        chunk_paths: Ordered list of chunk file paths.
        chunk_transcripts: Ordered list of transcript texts.
        source_path: Path to the original source audio (for duration).
        model: The AI model used.
        job_id: For logging context.

    Returns:
        AssemblyResult with the stitched transcript and metadata.
    """
    if len(chunk_paths) != len(chunk_transcripts):
        raise RuntimeError(
            f"Chunk count mismatch: {len(chunk_paths)} files vs {len(chunk_transcripts)} transcripts"
        )

    # Filter empty transcripts.
    parts = [t for t in chunk_transcripts if t.strip()]
    if not parts:
        raise RuntimeError("All chunk transcripts were empty")

    transcript = "\n\n".join(parts)

    # Determine total duration.
    total_duration = probe_duration(source_path)
    if total_duration is None:
        total_duration = 0.0
        for chunk in chunk_paths:
            dur = probe_duration(chunk)
            if dur is not None:
                total_duration += dur

    logger.info(
        "assembly_done job_id=%s num_chunks=%s total_chars=%s duration_seconds=%s",
        job_id,
        len(chunk_paths),
        len(transcript),
        int(round(total_duration)),
    )

    return AssemblyResult(
        transcript=transcript,
        total_duration=total_duration,
        num_chunks=len(chunk_paths),
        model=model,
    )