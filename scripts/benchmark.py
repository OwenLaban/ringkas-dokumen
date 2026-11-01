"""Benchmark 3 model lokal Ollama untuk tugas summarizer.

Jalankan dari root repo:

    python -m scripts.benchmark

Model dijalankan pada dokumen yang sama, lalu diukur latency, throughput,
rasio kompresi, dan berapa fakta kunci yang masih muncul di ringkasan.
"""

import argparse
import json
import time
from pathlib import Path

from app.config import Settings
from app.ollama import OllamaClient, OllamaError
from app.summarizer import ringkas_dokumen
from app.chunking import hitung_token

MODEL_DEFAULT = ["llama3.2:1b", "qwen2.5:1.5b", "gemma2:2b"]

# Setiap fakta punya beberapa bentuk penulisan yang diterima.
FAKTA_KUNCI = {
    "kapasitas_25_mw": ["25 mw", "25mw"],
    "investasi_410_miliar": ["410 miliar", "rp410", "rp 410"],
    "cod_september_2026": ["30 september 2026", "september 2026"],
    "mitra_surya_teknik": ["surya teknik"],
    "lokasi_sumba": ["sumba"],
    "ppa_pln_20_tahun": ["pln", "ppa"],
    "baterai_10_mwh": ["10 mwh", "10mwh"],
    "tambah_3_insinyur": ["3 insinyur", "tiga insinyur"],
}


def fakta_tertahan(ringkasan: str) -> tuple[int, list[str]]:
    teks = ringkasan.lower()
    ditemukan = [
        nama
        for nama, varian in FAKTA_KUNCI.items()
        if any(bentuk in teks for bentuk in varian)
    ]
    return len(ditemukan), ditemukan


def jalankan_benchmark(
    models: list[str],
    dokumen: str,
    url: str = "http://localhost:11434",
) -> dict:
    settings = Settings(
        ollama_url=url,
        request_timeout=600,
        max_output_tokens=600,
        chunk_tokens=600,
        chunk_overlap=80,
    )
    client = OllamaClient(url, timeout=settings.request_timeout)
    hasil = []

    for model in models:
        print(f"[{model}] warm-up ...", flush=True)
        try:
            client.generate(
                "Balas hanya dengan kata: siap",
                model,
                num_ctx=settings.num_ctx,
                num_predict=5,
            )
        except OllamaError as error:
            hasil.append({"model": model, "error": str(error)})
            print(f"[{model}] gagal: {error}", flush=True)
            continue

        print(f"[{model}] meringkas ...", flush=True)
        mulai = time.perf_counter()
        try:
            ringkasan = ringkas_dokumen(
                dokumen,
                client,
                model=model,
                strategi="map_reduce",
                settings=settings,
            )
        except (OllamaError, ValueError) as error:
            hasil.append({"model": model, "error": str(error)})
            print(f"[{model}] gagal: {error}", flush=True)
            continue

        jumlah_fakta, fakta = fakta_tertahan(ringkasan.ringkasan)
        hasil.append(
            {
                "model": model,
                "latency_s": round(time.perf_counter() - mulai, 2),
                "strategi": ringkasan.strategi,
                "jumlah_chunk": ringkasan.jumlah_chunk,
                "token_input": ringkasan.token_input,
                "token_output": ringkasan.token_output,
                "token_per_detik": ringkasan.token_per_detik,
                "kompresi": round(
                    ringkasan.token_input / max(ringkasan.token_output, 1), 2
                ),
                "fakta_tertahan": jumlah_fakta,
                "total_fakta": len(FAKTA_KUNCI),
                "fakta_ditemukan": fakta,
                "peringatan": ringkasan.peringatan,
                "ringkasan": ringkasan.ringkasan,
            }
        )

    return {
        "waktu": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dokumen_token": hitung_token(dokumen),
        "model": models,
        "hasil": hasil,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODEL_DEFAULT)
    parser.add_argument(
        "--dokumen",
        type=Path,
        default=Path("data/contoh_dokumen.txt"),
    )
    parser.add_argument("--output", type=Path, default=Path("hasil_benchmark.json"))
    args = parser.parse_args()

    dokumen = args.dokumen.read_text(encoding="utf-8")
    hasil = jalankan_benchmark(args.models, dokumen)
    args.output.write_text(
        json.dumps(hasil, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n| Model | Latency | Token/detik | Kompresi | Fakta |")
    print("|---|---:|---:|---:|---:|")
    for item in hasil["hasil"]:
        if "error" in item:
            print(f"| {item['model']} | gagal | - | - | - |")
            continue
        print(
            f"| {item['model']} | {item['latency_s']} s | "
            f"{item['token_per_detik'] or '-'} | {item['kompresi']}x | "
            f"{item['fakta_tertahan']}/{item['total_fakta']} |"
        )
    print(f"\nHasil mentah: {args.output}")


if __name__ == "__main__":
    main()
