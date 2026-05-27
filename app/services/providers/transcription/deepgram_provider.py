"""
Deepgram transcription provider.

Uses the native Deepgram API for audio transcription.
"""

from pathlib import Path

import requests

from app.core.config import settings
from app.core.logging import logger


class DeepgramProvider:
    """Transcription provider using the Deepgram API."""

    def __init__(self, api_key: str, endpoint: str = 'https://api.deepgram.com'):
        self.api_key = api_key
        self.endpoint = endpoint.rstrip('/')

    def transcribe_chunk(self, chunk_path: Path, model: str, job_id: str) -> str:
        """
        Transcribe a single audio chunk via Deepgram's listen endpoint.
        """
        url = f"{self.endpoint}/v1/listen?model={model}&punctuate=true"
        headers = {'Authorization': f'Token {self.api_key}'}
        timeout = settings.DOWNLOAD_READ_TIMEOUT

        with open(chunk_path, 'rb') as f:
            response = requests.post(url, headers=headers, data=f, timeout=timeout)

        response.raise_for_status()
        data = response.json()
        transcript = data['results']['channels'][0]['alternatives'][0]['transcript'] or ''
        return transcript.strip()