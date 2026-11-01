# 📄 ringkas-dokumen

API summarizer dokumen berbahasa Indonesia yang berjalan sepenuhnya dengan
**model lokal via Ollama**. Dokumen panjang dipecah menjadi chunk, diringkas
dengan strategi **map-reduce** atau **refine**, lalu outputnya divalidasi sebelum
dikembalikan.

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-000000)](https://ollama.com/)
[![Tests](https://img.shields.io/badge/tests-24%20passed-brightgreen)](tests/)

## Masalah

Model lokal dengan context window kecil tidak bisa menerima dokumen panjang
sekaligus. Memotong dokumen begitu saja juga berbahaya: angka, nama, tanggal,
dan keputusan penting bisa hilang di batas potongan.

Proyek ini menunjukkan cara yang lebih aman:

1. hitung token, bukan karakter;
2. potong dokumen dengan overlap;
3. ringkas per bagian, lalu gabungkan (**map-reduce**), atau perbarui ringkasan
   bertahap (**refine**);
4. validasi output sebelum dikirim ke client.

## Fitur

- **Chunking berbasis token** (`tiktoken`) dengan overlap yang bisa diatur.
- **Tiga strategi**: `stuff`, `map_reduce`, dan `refine`; `auto` memilih
  berdasarkan ukuran dokumen.
- **Reduce hierarkis** jika ringkasan bagian masih terlalu panjang.
- **Validasi output**: membersihkan code fence/preamble, menolak output kosong,
  dan memotong output yang melewati batas.
- **Upload file** `.txt` dan `.md` (UTF-8) dengan batas ukuran.
- **Error handling**: Ollama mati → `503`, input salah → `400/422`.
- **Test tanpa Ollama** memakai fake client, sehingga CI tidak butuh model.
- **Benchmark 3 model lokal** dengan metrik latency, throughput, kompresi, dan
  cakupan fakta kunci.

## Arsitektur

```text
Client
  │  POST /ringkas  (JSON)  atau  /ringkas/file  (txt/md)
  ▼
FastAPI
  │
  ▼
ringkas_dokumen()
  ├─ hitung token + pilih strategi
  ├─ potong_teks(max_token, overlap)
  ├─ stuff / map_reduce / refine
  │      └── Ollama /api/generate
  ├─ validasi_ringkasan()
  └─ respons JSON + statistik
```

## Quickstart

Prasyarat: Python 3.11+, [Ollama](https://ollama.com/), dan model 1B–3B.

```powershell
ollama pull qwen2.5:1.5b

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

uvicorn app.main:app --reload
```

Buka dokumentasi interaktif di <http://127.0.0.1:8000/docs>.

Konfigurasi opsional disalin dari `.env.example` ke `.env` (mis. ganti model
atau `NUM_CTX`). Tidak ada API key yang dibutuhkan.

## API

### `GET /health`

```json
{
  "status": "ok",
  "ollama": "ok",
  "model_default": "qwen2.5:1.5b",
  "detail": null
}
```

### `POST /ringkas`

```powershell
$body = @{
  teks = Get-Content data/contoh_dokumen.txt -Raw
  strategi = "map_reduce"
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/ringkas `
  -ContentType "application/json" `
  -Body $body
```

Contoh respons:

```json
{
  "ringkasan": "Rapat Proyek Solar Farm Sumba: kapasitas 25 MW, ...",
  "strategi": "map_reduce",
  "model": "qwen2.5:1.5b",
  "jumlah_chunk": 2,
  "token_input": 927,
  "token_output": 394,
  "batas_input": 1148,
  "token_per_detik": 16.88,
  "latency_s": 69.83,
  "peringatan": []
}
```

### `POST /ringkas/file`

```powershell
curl.exe -X POST "http://127.0.0.1:8000/ringkas/file?strategi=map_reduce" `
  -F "file=@data/contoh_dokumen.txt"
```

Format yang didukung: `.txt` dan `.md`, encoding UTF-8, maksimal 1 MB
(dapat diubah lewat `MAX_FILE_BYTES`).

## Strategi

| Strategi | Cara kerja | Kapan dipakai | Kelebihan | Kekurangan |
|---|---|---|---|---|
| `stuff` | Seluruh dokumen dalam satu prompt | Dokumen muat di `batas_input` | Paling cepat, konteks utuh | Tidak bisa dokumen panjang |
| `map_reduce` | Ringkas tiap chunk, lalu gabungkan | Dokumen panjang | Paralel secara konsep, stabil | Fakta bisa hilang di tahap reduce |
| `refine` | Ringkasan diperbarui per chunk | Dokumen panjang, butuh alur naratif | Detail awal bisa dipertahankan | Berurutan, lebih lambat |
| `auto` | `stuff` jika muat, selain itu `map_reduce` | Default | Simpel | Tidak selalu optimal |

`batas_input = NUM_CTX - MAX_OUTPUT_TOKENS - 300` (cadangan untuk prompt).

## Hasil Benchmark

Dokumen uji: notulen rapat 927 token. Strategi `map_reduce`, 2 chunk
(600 token, overlap 80), temperature 0, `num_ctx` 2048. Setiap model melakukan
warm-up sebelum pengukuran.

| Model | Latency | Token/detik | Kompresi | Fakta kunci |
|---|---:|---:|---:|---:|
| `llama3.2:1b` | 61,50 s | 11,50 | 6,22× | 5/8 |
| **`qwen2.5:1.5b`** | **69,83 s** | **16,88** | **2,35×** | **7/8** |
| `gemma2:2b` | 102,79 s | 8,81 | 2,19× | 6/8 |

**Pilihan default: `qwen2.5:1.5b`.** Model ini paling seimbang: ringkasan
paling lengkap (7/8 fakta), throughput tertinggi, dan tetap muat di VRAM 2 GB.
`llama3.2:1b` paling cepat dan paling ringkas, tetapi kehilangan banyak fakta.
`gemma2:2b` lambat karena sebagian besar berjalan di CPU.

Laporan lengkap, termasuk fakta yang hilang dan keterbatasan pengukuran:
[`docs/laporan-benchmark.md`](docs/laporan-benchmark.md). Hasil mentah:
[`hasil_benchmark.json`](hasil_benchmark.json).

## Test

```powershell
pip install -r requirements-dev.txt
pytest -q
```

Hasil saat ini: **24 passed**. Test memakai `FakeClient`, jadi tidak memerlukan
Ollama, model, atau jaringan.

## Struktur

```text
ringkas-dokumen/
├── app/
│   ├── main.py          # endpoint FastAPI
│   ├── summarizer.py    # strategi, prompt, validasi
│   ├── chunking.py      # hitung token + potong teks
│   ├── ollama.py        # HTTP client Ollama
│   ├── config.py        # environment variable
│   ├── schemas.py       # Pydantic
│   └── deps.py          # dependency injection
├── tests/               # 24 test dengan fake client
├── scripts/benchmark.py # benchmark 3 model
├── data/                # dokumen contoh
├── docs/                # laporan benchmark
├── requirements.txt
└── .env.example
```

## Keterbatasan

- Baru mendukung teks `.txt`/`.md`; PDF dan DOCX belum.
- Belum ada autentikasi, rate limit, atau queue; cocok untuk pemakaian lokal.
- Evaluasi kualitas memakai 8 fakta kunci dan pencocokan kata kunci, belum
  LLM-as-judge atau evaluasi semantik.
- Benchmark memakai satu dokumen dan satu run per model; latency bisa berubah
  karena cold start, thermal throttling, dan beban CPU/GPU.
- Tokenizer `tiktoken` dipakai sebagai perkiraan; tokenizer asli tiap model
  (Qwen, Llama, Gemma) bisa berbeda, sehingga `num_ctx` tetap perlu dipantau.
- Ringkasan tetap bisa mengandung kesalahan fakta. Validasi saat ini hanya
  memeriksa bentuk output, bukan kebenaran isi.

## Pengembangan Berikutnya

- Ekstraksi PDF/DOCX dan OCR.
- Streaming respons per chunk.
- Cache ringkasan berdasarkan hash dokumen.
- Evaluasi LLM-as-judge + daftar fakta per dokumen.
- Antrean job dan penyimpanan hasil untuk dokumen besar.

## Lisensi

MIT — lihat [`LICENSE`](LICENSE).
