"""Inti summarizer: pilih strategi, chunking, map-reduce, refine, validasi."""

import time
from dataclasses import dataclass, field
from typing import Protocol

from .chunking import hitung_token, potong_teks
from .config import Settings, get_settings
from .ollama import HasilGenerate


class LLMClient(Protocol):
    def generate(
        self,
        prompt: str,
        model: str,
        num_ctx: int = 2048,
        num_predict: int = 300,
        temperature: float = 0.0,
    ) -> HasilGenerate: ...


class RingkasanTidakValid(ValueError):
    """Model mengembalikan ringkasan kosong atau tidak bisa dipakai."""


PROMPT_STUFF = """Ringkas dokumen berikut dalam bahasa Indonesia.

Aturan:
- Pertahankan angka, nama, tanggal, keputusan, dan syarat penting.
- Jangan menambahkan fakta yang tidak ada di dokumen.
- Tulis ringkasan padat, bukan daftar kalimat pembuka.
- Maksimal 250 kata.
- Balas hanya dengan ringkasan, tanpa markdown.

Dokumen:
{teks}
"""

PROMPT_MAP = """Ringkas bagian {i} dari {n} dokumen berikut.

Aturan:
- Pertahankan angka, nama, tanggal, keputusan, dan syarat penting.
- Jangan menambahkan fakta baru.
- Tulis ringkasan padat dalam bahasa Indonesia.
- Maksimal 150 kata.
- Balas hanya dengan ringkasan bagian ini.

Bagian dokumen:
{teks}
"""

PROMPT_REDUCE = """Gabungkan ringkasan bagian berikut menjadi satu ringkasan akhir.

Aturan:
- Hilangkan pengulangan, tetapi jangan membuang fakta penting.
- Pertahankan angka, nama, tanggal, keputusan, dan syarat.
- Jangan menambahkan fakta baru.
- Tulis dalam bahasa Indonesia, padat dan runtut.
- Maksimal 250 kata.
- Balas hanya dengan ringkasan akhir.

Ringkasan bagian:
{ringkasan}
"""

PROMPT_REFINE = """Perbarui ringkasan sementara dengan informasi dari bagian dokumen baru.

Aturan:
- Pertahankan fakta penting dari ringkasan lama.
- Tambahkan fakta baru yang relevan.
- Jangan mengulang kalimat dan jangan menambahkan fakta di luar dokumen.
- Tulis ringkasan yang sudah diperbarui dalam bahasa Indonesia.
- Maksimal 250 kata.
- Balas hanya dengan ringkasan.

Ringkasan sementara:
{ringkasan}

Bagian dokumen baru:
{teks}
"""


@dataclass
class HasilRingkasan:
    ringkasan: str
    strategi: str
    model: str
    jumlah_chunk: int
    token_input: int
    token_output: int
    batas_input: int
    token_per_detik: float | None
    latency_s: float
    peringatan: list[str] = field(default_factory=list)


class _Pencatat:
    """Wrapper yang menjumlahkan statistik token dari semua panggilan model."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.eval_count = 0
        self.eval_duration_ns = 0

    def generate(self, *args, **kwargs) -> HasilGenerate:
        hasil = self.client.generate(*args, **kwargs)
        self.eval_count += hasil.eval_count
        self.eval_duration_ns += hasil.eval_duration_ns
        return hasil


def pilih_strategi(token_dokumen: int, batas_input: int, diminta: str = "auto") -> str:
    """Pilih strategi ringkasan berdasarkan ukuran dokumen."""
    if batas_input <= 0:
        raise ValueError("batas_input harus > 0")
    if diminta not in {"auto", "stuff", "map_reduce", "refine"}:
        raise ValueError(f"strategi tidak dikenal: {diminta!r}")

    if diminta == "auto":
        return "stuff" if token_dokumen <= batas_input else "map_reduce"
    if diminta == "stuff" and token_dokumen > batas_input:
        raise ValueError(
            "Dokumen terlalu panjang untuk strategi stuff. "
            "Gunakan map_reduce atau refine."
        )
    return diminta


def _generate(
    client: LLMClient,
    prompt: str,
    model: str,
    settings: Settings,
) -> str:
    hasil = client.generate(
        prompt,
        model,
        num_ctx=settings.num_ctx,
        num_predict=settings.max_output_tokens,
    )
    return hasil.teks


def _reduce_hierarkis(
    bagian: list[str],
    client: LLMClient,
    model: str,
    settings: Settings,
) -> str:
    """Gabungkan ringkasan parsial; pecah lagi jika masih terlalu panjang."""
    if not bagian:
        raise RingkasanTidakValid("Tidak ada ringkasan bagian untuk digabung.")

    teks = "\n\n".join(bagian)
    if hitung_token(teks) <= settings.batas_input:
        prompt = PROMPT_REDUCE.format(ringkasan=teks)
        return _generate(client, prompt, model, settings)

    if len(bagian) == 1:
        dipotong = potong_teks(teks, settings.batas_input)[0]
        prompt = PROMPT_REDUCE.format(ringkasan=dipotong)
        return _generate(client, prompt, model, settings)

    tengah = len(bagian) // 2
    kiri = _reduce_hierarkis(bagian[:tengah], client, model, settings)
    kanan = _reduce_hierarkis(bagian[tengah:], client, model, settings)
    return _reduce_hierarkis([kiri, kanan], client, model, settings)


def ringkas_stuff(
    teks: str,
    client: LLMClient,
    model: str,
    settings: Settings,
) -> str:
    """Ringkas seluruh dokumen dalam satu panggilan."""
    prompt = PROMPT_STUFF.format(teks=teks)
    return _generate(client, prompt, model, settings)


def ringkas_map_reduce(
    teks: str,
    client: LLMClient,
    model: str,
    settings: Settings,
) -> tuple[str, int]:
    """Ringkas tiap chunk, lalu gabungkan hasilnya secara hierarkis."""
    potongan = potong_teks(teks, settings.chunk_tokens, settings.chunk_overlap)
    if not potongan:
        raise RingkasanTidakValid("Dokumen tidak menghasilkan chunk.")

    ringkasan_bagian = []
    for nomor, bagian in enumerate(potongan, start=1):
        prompt = PROMPT_MAP.format(i=nomor, n=len(potongan), teks=bagian)
        ringkasan_bagian.append(_generate(client, prompt, model, settings))

    ringkasan = _reduce_hierarkis(ringkasan_bagian, client, model, settings)
    return ringkasan, len(potongan)


def ringkas_refine(
    teks: str,
    client: LLMClient,
    model: str,
    settings: Settings,
) -> tuple[str, int]:
    """Perbarui ringkasan bertahap untuk setiap chunk."""
    potongan = potong_teks(teks, settings.chunk_tokens, settings.chunk_overlap)
    if not potongan:
        raise RingkasanTidakValid("Dokumen tidak menghasilkan chunk.")

    prompt_awal = PROMPT_MAP.format(i=1, n=len(potongan), teks=potongan[0])
    ringkasan = _generate(client, prompt_awal, model, settings)

    for bagian in potongan[1:]:
        prompt = PROMPT_REFINE.format(ringkasan=ringkasan, teks=bagian)
        ringkasan = _generate(client, prompt, model, settings)

    return ringkasan, len(potongan)


def validasi_ringkasan(teks: str, max_token: int) -> tuple[str, bool]:
    """Bersihkan dan validasi output model.

    Kembalikan ``(ringkasan, dipotong)``. Ringkasan kosong dianggap gagal;
    ringkasan yang terlalu panjang dipotong agar tetap muat di batas output.
    """
    bersih = teks.strip()
    if bersih.startswith("```") and bersih.endswith("```"):
        baris = bersih.splitlines()
        if len(baris) >= 2:
            bersih = "\n".join(baris[1:-1]).strip()

    # Buang label/pembuka yang sering ditambahkan model.
    baris = bersih.splitlines()
    if baris and baris[0].strip().lower().startswith(
        ("berikut ringkasan", "berikut adalah ringkasan", "ringkasan akhir:", "ringkasan:")
    ):
        bersih = "\n".join(baris[1:]).strip()

    if not bersih:
        raise RingkasanTidakValid("Model mengembalikan ringkasan kosong.")

    if hitung_token(bersih) > max_token:
        potongan = potong_teks(bersih, max_token)
        return potongan[0].strip(), True
    return bersih, False


def ringkas_dokumen(
    teks: str,
    client: LLMClient,
    model: str | None = None,
    strategi: str = "auto",
    settings: Settings | None = None,
) -> HasilRingkasan:
    """Orkestrasi lengkap: pilih strategi, jalankan, validasi, ukur."""
    if not teks.strip():
        raise ValueError("Dokumen tidak boleh kosong.")

    settings = settings or get_settings()
    model = model or settings.default_model
    batas = settings.batas_input
    token_input = hitung_token(teks)
    dipilih = pilih_strategi(token_input, batas, strategi)

    pencatat = _Pencatat(client)
    mulai = time.perf_counter()

    if dipilih == "stuff":
        ringkasan = ringkas_stuff(teks, pencatat, model, settings)
        jumlah_chunk = 1
    elif dipilih == "map_reduce":
        ringkasan, jumlah_chunk = ringkas_map_reduce(
            teks, pencatat, model, settings
        )
    else:
        ringkasan, jumlah_chunk = ringkas_refine(teks, pencatat, model, settings)

    detik = time.perf_counter() - mulai
    bersih, dipotong = validasi_ringkasan(ringkasan, settings.max_output_tokens)
    peringatan: list[str] = []
    if dipotong:
        peringatan.append(
            "Ringkasan melewati batas output dan dipotong otomatis."
        )

    token_per_detik = None
    if pencatat.eval_count and pencatat.eval_duration_ns:
        token_per_detik = round(
            pencatat.eval_count / (pencatat.eval_duration_ns / 1_000_000_000),
            2,
        )

    return HasilRingkasan(
        ringkasan=bersih,
        strategi=dipilih,
        model=model,
        jumlah_chunk=jumlah_chunk,
        token_input=token_input,
        token_output=hitung_token(bersih),
        batas_input=batas,
        token_per_detik=token_per_detik,
        latency_s=round(detik, 2),
        peringatan=peringatan,
    )
