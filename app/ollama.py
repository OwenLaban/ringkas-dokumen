"""Klien HTTP kecil untuk Ollama.

Dipakai oleh aplikasi dan bisa diganti dengan fake client di test, sehingga
test tidak membutuhkan Ollama atau model yang terpasang.
"""

from dataclasses import dataclass

import requests


class OllamaError(RuntimeError):
    """Ollama tidak bisa dihubungi atau mengembalikan jawaban tidak valid."""


@dataclass
class HasilGenerate:
    teks: str
    eval_count: int = 0
    eval_duration_ns: int = 0
    total_duration_ns: int = 0


class OllamaClient:
    def __init__(self, base_url: str, timeout: int = 300) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def daftar_model(self) -> list[str]:
        """Kembalikan nama model yang terpasang di Ollama."""
        try:
            response = self._session.get(
                f"{self.base_url}/api/tags", timeout=5
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise OllamaError(
                f"Tidak bisa menghubungi Ollama di {self.base_url}. "
                "Pastikan `ollama serve` berjalan."
            ) from error

        body = response.json()
        return [model["name"] for model in body.get("models", [])]

    def generate(
        self,
        prompt: str,
        model: str,
        num_ctx: int = 2048,
        num_predict: int = 300,
        temperature: float = 0.0,
    ) -> HasilGenerate:
        """Jalankan satu permintaan non-streaming ke Ollama."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
        }

        try:
            response = self._session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
        except requests.Timeout as error:
            raise OllamaError(
                f"Ollama melewati batas waktu {self.timeout} detik."
            ) from error
        except requests.RequestException as error:
            raise OllamaError(
                f"Gagal menghubungi Ollama di {self.base_url}."
            ) from error

        if response.status_code >= 400:
            detail = response.text[:300]
            raise OllamaError(
                f"Ollama menolak permintaan (HTTP {response.status_code}): {detail}"
            )

        try:
            body = response.json()
        except ValueError as error:
            raise OllamaError("Respons Ollama bukan JSON.") from error

        teks = (body.get("response") or "").strip()
        if not teks:
            raise OllamaError("Ollama mengembalikan jawaban kosong.")

        return HasilGenerate(
            teks=teks,
            eval_count=int(body.get("eval_count", 0) or 0),
            eval_duration_ns=int(body.get("eval_duration", 0) or 0),
            total_duration_ns=int(body.get("total_duration", 0) or 0),
        )
