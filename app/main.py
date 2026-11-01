"""Entry point FastAPI untuk API summarizer dokumen lokal."""

from dataclasses import asdict

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from .config import Settings, get_settings
from .deps import get_client
from .ollama import OllamaClient, OllamaError
from .schemas import (
    KesehatanResponse,
    ModelResponse,
    RingkasRequest,
    RingkasResponse,
    Strategi,
)
from .summarizer import RingkasanTidakValid, ringkas_dokumen

app = FastAPI(
    title="ringkas-dokumen",
    description=(
        "API summarizer dokumen berbahasa Indonesia dengan model lokal Ollama. "
        "Mendukung chunking, map-reduce, refine, dan validasi output."
    ),
    version="0.1.0",
)

EKSTENSI_DIDUKUNG = {".txt", ".md"}


@app.exception_handler(OllamaError)
async def tangani_ollama_error(_request, error: OllamaError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(error)})


@app.exception_handler(RingkasanTidakValid)
async def tangani_ringkasan_error(
    _request, error: RingkasanTidakValid
) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(error)})


@app.exception_handler(ValueError)
async def tangani_value_error(_request, error: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(error)})


@app.get("/", include_in_schema=False)
def akar() -> dict:
    return {"nama": "ringkas-dokumen", "dokumentasi": "/docs"}


@app.get("/health", response_model=KesehatanResponse)
def kesehatan(
    client: OllamaClient = Depends(get_client),
    settings: Settings = Depends(get_settings),
) -> KesehatanResponse:
    try:
        client.daftar_model()
    except OllamaError as error:
        return KesehatanResponse(
            status="degraded",
            ollama="down",
            model_default=settings.default_model,
            detail=str(error),
        )
    return KesehatanResponse(
        status="ok",
        ollama="ok",
        model_default=settings.default_model,
    )


@app.get("/models", response_model=ModelResponse)
def daftar_model(client: OllamaClient = Depends(get_client)) -> ModelResponse:
    return ModelResponse(model=client.daftar_model())


@app.post("/ringkas", response_model=RingkasResponse)
def ringkas(
    request: RingkasRequest,
    client: OllamaClient = Depends(get_client),
    settings: Settings = Depends(get_settings),
) -> RingkasResponse:
    hasil = ringkas_dokumen(
        request.teks,
        client,
        model=request.model,
        strategi=request.strategi,
        settings=settings,
    )
    return RingkasResponse(**asdict(hasil))


@app.post("/ringkas/file", response_model=RingkasResponse)
async def ringkas_file(
    file: UploadFile = File(...),
    strategi: Strategi = Query(default="auto"),
    model: str | None = Query(default=None),
    client: OllamaClient = Depends(get_client),
    settings: Settings = Depends(get_settings),
) -> RingkasResponse:
    nama = (file.filename or "").lower()
    ekstensi = "." + nama.rsplit(".", 1)[-1] if "." in nama else ""
    if ekstensi not in EKSTENSI_DIDUKUNG:
        raise HTTPException(
            status_code=400,
            detail="Format file harus .txt atau .md",
        )

    isi = await file.read(settings.max_file_bytes + 1)
    if len(isi) > settings.max_file_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File melebihi batas {settings.max_file_bytes} byte.",
        )

    try:
        teks = isi.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise HTTPException(
            status_code=400,
            detail="File harus berisi teks UTF-8.",
        ) from error

    hasil = ringkas_dokumen(
        teks,
        client,
        model=model,
        strategi=strategi,
        settings=settings,
    )
    return RingkasResponse(**asdict(hasil))
