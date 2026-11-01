from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from app.deps import get_client
from app.main import app


@pytest.fixture
def api(fake_client):
    app.dependency_overrides[get_client] = lambda: fake_client
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_health_ok(api):
    response = api.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["ollama"] == "ok"


def test_health_degraded(fake_client_mati):
    app.dependency_overrides[get_client] = lambda: fake_client_mati
    with TestClient(app) as client:
        response = client.get("/health")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["ollama"] == "down"


def test_ringkas_pendek(api):
    response = api.post(
        "/ringkas",
        json={"teks": "Dokumen pendek tentang proyek surya."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["strategi"] == "stuff"
    assert body["jumlah_chunk"] == 1
    assert body["model"] == "qwen2.5:1.5b"
    assert body["ringkasan"]


def test_ringkas_panjang_memilih_map_reduce(api):
    teks = " ".join(f"kata{i}" for i in range(2500))

    response = api.post("/ringkas", json={"teks": teks})

    assert response.status_code == 200
    body = response.json()
    assert body["strategi"] == "map_reduce"
    assert body["jumlah_chunk"] > 1


def test_ringkas_tolak_teks_kosong(api):
    response = api.post("/ringkas", json={"teks": "   "})

    assert response.status_code == 422


def test_ringkas_file_txt(api):
    response = api.post(
        "/ringkas/file",
        files={"file": ("dokumen.txt", BytesIO(b"Proyek surya 25 MW."), "text/plain")},
    )

    assert response.status_code == 200
    assert response.json()["ringkasan"]


def test_ringkas_file_ekstensi_salah(api):
    response = api.post(
        "/ringkas/file",
        files={"file": ("dokumen.pdf", BytesIO(b"%PDF"), "application/pdf")},
    )

    assert response.status_code == 400


def test_ringkas_saat_ollama_mati(fake_client_mati):
    app.dependency_overrides[get_client] = lambda: fake_client_mati
    with TestClient(app) as client:
        response = client.post("/ringkas", json={"teks": "Dokumen pendek."})
    app.dependency_overrides.clear()

    assert response.status_code == 503
