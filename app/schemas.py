"""Schema request/response API."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Strategi = Literal["auto", "stuff", "map_reduce", "refine"]


class RingkasRequest(BaseModel):
    teks: str = Field(
        min_length=1,
        max_length=200_000,
        description="Dokumen yang akan diringkas.",
    )
    strategi: Strategi = Field(
        default="auto",
        description="`auto` memilih stuff jika muat, map_reduce jika tidak.",
    )
    model: str | None = Field(
        default=None,
        description="Nama model Ollama. Default dari environment OLLAMA_MODEL.",
    )

    @field_validator("teks")
    @classmethod
    def teks_tidak_hanya_spasi(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("teks tidak boleh kosong")
        return value


class RingkasResponse(BaseModel):
    ringkasan: str
    strategi: str
    model: str
    jumlah_chunk: int
    token_input: int
    token_output: int
    batas_input: int
    token_per_detik: float | None = None
    latency_s: float
    peringatan: list[str] = Field(default_factory=list)


class KesehatanResponse(BaseModel):
    status: str
    ollama: str
    model_default: str
    detail: str | None = None


class ModelResponse(BaseModel):
    model: list[str]
