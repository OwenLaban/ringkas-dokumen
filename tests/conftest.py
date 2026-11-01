"""Fake Ollama client untuk test tanpa model atau jaringan."""

import pytest

from app.ollama import HasilGenerate, OllamaError


class FakeClient:
    def __init__(self, jawaban=None, error: Exception | None = None) -> None:
        self.prompts: list[str] = []
        self.error = error
        self.jawaban = jawaban or (lambda _prompt, nomor: f"Ringkasan {nomor}.")

    def daftar_model(self) -> list[str]:
        if self.error:
            raise self.error
        return ["fake:1b"]

    def generate(
        self,
        prompt: str,
        model: str,
        num_ctx: int = 2048,
        num_predict: int = 300,
        temperature: float = 0.0,
    ) -> HasilGenerate:
        if self.error:
            raise self.error
        self.prompts.append(prompt)
        teks = self.jawaban(prompt, len(self.prompts))
        return HasilGenerate(
            teks=teks,
            eval_count=12,
            eval_duration_ns=1_000_000_000,
            total_duration_ns=1_500_000_000,
        )


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def fake_client_mati() -> FakeClient:
    return FakeClient(error=OllamaError("Ollama tidak berjalan"))
