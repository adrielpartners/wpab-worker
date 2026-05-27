"""
Provider registry for wpab-worker transcription providers.

Maps provider slugs to their implementation classes and configuration.
Each provider reads its API key from its own environment variable.
"""

import importlib
import os
from urllib.parse import urlparse

from app.core.logging import logger


PROVIDERS = {
    'openai': {
        'label': 'OpenAI',
        'module': 'app.services.providers.transcription.openai_compat_provider',
        'class': 'OpenAICompatProvider',
        'env_key': 'OPENAI_API_KEY',
        'default_endpoint': 'https://api.openai.com',
        'default_model': 'gpt-4o-mini-transcribe',
        'default_chunk_seconds': 660,
    },
    'groq': {
        'label': 'Groq',
        'module': 'app.services.providers.transcription.openai_compat_provider',
        'class': 'OpenAICompatProvider',
        'env_key': 'GROQ_API_KEY',
        'default_endpoint': 'https://api.groq.com/openai/v1',
        'default_model': 'whisper-large-v3-turbo',
        'default_chunk_seconds': 660,
    },
    'openrouter': {
        'label': 'OpenRouter',
        'module': 'app.services.providers.transcription.openrouter_provider',
        'class': 'OpenRouterProvider',
        'env_key': 'OPENROUTER_API_KEY',
        'default_endpoint': 'https://openrouter.ai/api',
        'default_model': 'openai/whisper-large-v3',
        'default_chunk_seconds': 55,
    },
    'deepgram': {
        'label': 'Deepgram',
        'module': 'app.services.providers.transcription.deepgram_provider',
        'class': 'DeepgramProvider',
        'env_key': 'DEEPGRAM_API_KEY',
        'default_endpoint': 'https://api.deepgram.com',
        'default_model': 'nova-2',
        'default_chunk_seconds': 660,
    },
}


def get_provider_info(provider_slug: str) -> dict:
    """Return registry metadata for a transcription provider."""
    info = PROVIDERS.get(provider_slug)
    if not info:
        raise ValueError(f"Unsupported transcription provider: {provider_slug}")
    return info


def _endpoint_from_config(default_endpoint: str, provider_config: dict | None) -> str:
    endpoint = str((provider_config or {}).get('endpoint') or default_endpoint).strip().rstrip('/')
    parsed = urlparse(endpoint)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        raise ValueError("Invalid provider endpoint")
    return endpoint


def get_transcription_provider(provider_slug: str, provider_config: dict | None = None):
    """
    Factory: returns a provider instance for the given slug.

    The provider is pre-configured with its API key from the corresponding
    environment variable and its default endpoint URL.
    """
    info = get_provider_info(provider_slug)

    api_key = os.getenv(info['env_key'], '')
    if not api_key:
        raise RuntimeError(f"{info['env_key']} is required for provider '{provider_slug}'")

    endpoint = _endpoint_from_config(info['default_endpoint'], provider_config)

    module = importlib.import_module(info['module'])
    cls = getattr(module, info['class'])

    logger.info(
        "provider_resolved slug=%s class=%s endpoint=%s",
        provider_slug,
        info['class'],
        endpoint,
    )

    return cls(
        api_key=api_key,
        endpoint=endpoint,
    )
