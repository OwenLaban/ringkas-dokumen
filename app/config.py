"""Konfigurasi aplikasi dari environment variable.

Tidak ada secret di sini. Semua nilai punya default yang aman untuk model
lokal 1B–3B dengan context window kecil.
"""

import os
from dataclasses import dataclass
from functools import lru_cache


def _int_env(nama: str, default: int) -> int:
    nilai = os.getenv(nama)
    if nilai is None:
        return default
    try:
        return int(nilai)
    except ValueError as error:
        raise ValueError(f"{nama} harus berupa angka, bukan {nilai!r}") from error


@dataclass(frozen=True)
class Settings:
    ollama_url: str = "http://localhost:11434"
    default_model: str = "qwen2.5:1.5b"
    num_ctx: int = 2048
    chunk_tokens: int = 600
    chunk_overlap: int = 80
    max_output_tokens: int = 600
    request_timeout: int = 300
    max_file_bytes: int = 1_000_000

    @property
    def batas_input(self) -> int:
        """Token yang tersedia untuk dokumen setelah output dan prompt cadangan."""
        return self.num_ctx - self.max_output_tokens - 300


@lru_cache
def get_settings() -> Settings:
    return Settings(
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        default_model=os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
        num_ctx=_int_env("NUM_CTX", 2048),
        chunk_tokens=_int_env("CHUNK_TOKENS", 600),
        chunk_overlap=_int_env("CHUNK_OVERLAP", 80),
        max_output_tokens=_int_env("MAX_OUTPUT_TOKENS", 600),
        request_timeout=_int_env("REQUEST_TIMEOUT", 300),
        max_file_bytes=_int_env("MAX_FILE_BYTES", 1_000_000),
    )
