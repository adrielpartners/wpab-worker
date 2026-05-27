"""
Provider registry for wpab-worker transcription providers.

Maps provider slugs to their implementation classes and configuration.
Each provider reads its API key from its own environment variable.
"""

import importlib
import os
from pathlib import Path

from app.core.logging import logger


PROVIDERS = {
    'openai': {
        'label': 'OpenAI',
        'module': 'app.services.providers.transcription.openai_compat_provider',
        'class': 'OpenAICompatProvider',
        'env_key': 'OPENAI_API_KEY',
        'default_endpoint': 'https://api.openai.com',
        'default_model': 'gpt-4o-mini-transcribe',
    },
    'groq': {
        'label': 'Groq',
        'module': 'app.services.providers.transcription.openai_compat_provider',
        'class': 'OpenAICompatProvider',
        'env_key': 'GROQ_API_KEY',
        'default_endpoint': 'https://api.groq.com/openai/v1',
        'default_model': 'whisper-large-v3-turbo',
    },
    'deepgram': {
        'label': 'Deepgram',
        'module': 'app.services.providers.transcription.deepgram_provider',
        'class': 'DeepgramProvider',
        'env_key': 'DEEPGRAM_API_KEY',
        'default_endpoint': 'https://api.deepgram.com',
        'default_model': 'nova-2',
    },
}


def get_transcription_provider(provider_slug: str):
    """
    Factory: returns a provider instance for the given slug.

    The provider is pre-configured with its API key from the corresponding
    environment variable and its default endpoint URL.
    """
    info = PROVIDERS.get(provider_slug)
    if not info:
        raise ValueError(f"Unsupported transcription provider: {provider_slug}")

    api_key = os.getenv(info['env_key'], '')
    if not api_key:
        raise RuntimeError(f"{info['env_key']} is required for provider '{provider_slug}'")

    # Import the provider class dynamically
    module = importlib.import_module(info['module'])
    cls = getattr(module, info['class'])

    logger.info(
        "provider_resolved slug=%s class=%s endpoint=%s",
        provider_slug,
        info['class'],
        info['default_endpoint'],
    )

    return cls(
        api_key=api_key,
        endpoint=info['default_endpoint'],
    )