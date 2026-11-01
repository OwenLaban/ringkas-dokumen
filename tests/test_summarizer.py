import pytest

from app.chunking import hitung_token, potong_teks
from app.config import Settings
from app.summarizer import (
    RingkasanTidakValid,
    pilih_strategi,
    ringkas_dokumen,
    ringkas_map_reduce,
    ringkas_refine,
    validasi_ringkasan,
)

SETTINGS = Settings(
    num_ctx=800,
    chunk_tokens=150,
    chunk_overlap=20,
    max_output_tokens=100,
)
TEKS_PANJANG = " ".join(f"fakta{i}" for i in range(1200))
TEKS_PENDEK = "Proyek Solar Farm Sumba berkapasitas 25 MW mulai beroperasi 2026."


def test_pilih_strategi_auto():
    assert pilih_strategi(100, 400) == "stuff"
    assert pilih_strategi(500, 400) == "map_reduce"


def test_pilih_strategi_stuff_terlalu_panjang():
    with pytest.raises(ValueError):
        pilih_strategi(500, 400, diminta="stuff")


def test_pilih_strategi_tidak_dikenal():
    with pytest.raises(ValueError):
        pilih_strategi(100, 400, diminta="ajaib")


def test_ringkas_dokumen_stuff(fake_client):
    hasil = ringkas_dokumen(
        TEKS_PENDEK,
        fake_client,
        model="fake:1b",
        settings=SETTINGS,
    )

    assert hasil.strategi == "stuff"
    assert hasil.jumlah_chunk == 1
    assert len(fake_client.prompts) == 1
    assert hasil.ringkasan.startswith("Ringkasan")
    assert hasil.token_output > 0


def test_ringkas_map_reduce_jumlah_panggilan(fake_client):
    jumlah_chunk = len(
        potong_teks(TEKS_PANJANG, SETTINGS.chunk_tokens, SETTINGS.chunk_overlap)
    )
    assert jumlah_chunk > 1

    ringkasan, chunk_dipakai = ringkas_map_reduce(
        TEKS_PANJANG,
        fake_client,
        model="fake:1b",
        settings=SETTINGS,
    )

    assert chunk_dipakai == jumlah_chunk
    assert len(fake_client.prompts) == jumlah_chunk + 1
    assert ringkasan


def test_ringkas_refine_memakai_ringkasan_sebelumnya(fake_client):
    jumlah_chunk = len(
        potong_teks(TEKS_PANJANG, SETTINGS.chunk_tokens, SETTINGS.chunk_overlap)
    )

    ringkasan, chunk_dipakai = ringkas_refine(
        TEKS_PANJANG,
        fake_client,
        model="fake:1b",
        settings=SETTINGS,
    )

    assert chunk_dipakai == jumlah_chunk
    assert len(fake_client.prompts) == jumlah_chunk
    assert "Ringkasan sementara" in fake_client.prompts[1]
    assert ringkasan


def test_ringkas_dokumen_menolak_kosong(fake_client):
    with pytest.raises(ValueError):
        ringkas_dokumen("   ", fake_client, settings=SETTINGS)


def test_validasi_ringkasan_kosong():
    with pytest.raises(RingkasanTidakValid):
        validasi_ringkasan("  \n ", max_token=20)


def test_validasi_ringkasan_terlalu_panjang_dipotong():
    teks = " ".join(f"kata{i}" for i in range(100))

    bersih, dipotong = validasi_ringkasan(teks, max_token=10)

    assert dipotong is True
    assert hitung_token(bersih) <= 10


def test_validasi_ringkasan_membersihkan_code_fence():
    bersih, dipotong = validasi_ringkasan("```\nRingkasan singkat.\n```", max_token=20)

    assert bersih == "Ringkasan singkat."
    assert dipotong is False


def test_validasi_ringkasan_membersihkan_pembuka():
    bersih, _ = validasi_ringkasan(
        "Berikut ringkasan akhir dokumen:\n\nIsi ringkasan.",
        max_token=20,
    )
    assert bersih == "Isi ringkasan."

    bersih, _ = validasi_ringkasan(
        "Berikut adalah ringkasan akhir:\n\nIsi kedua.",
        max_token=20,
    )
    assert bersih == "Isi kedua."
