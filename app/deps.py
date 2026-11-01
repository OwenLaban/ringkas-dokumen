"""Dependency FastAPI yang bisa dioverride saat test."""

from functools import lru_cache

from .config import get_settings
from .ollama import OllamaClient


@lru_cache
def get_client() -> OllamaClient:
    settings = get_settings()
    return OllamaClient(settings.ollama_url, timeout=settings.request_timeout)
