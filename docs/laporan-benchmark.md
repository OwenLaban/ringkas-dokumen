# Laporan Benchmark — Proyek #4

Tanggal: 1 November 2026  
Hardware: Intel Core i7-1195G7, RAM 7,7 GB, NVIDIA MX450 2 GB VRAM  
Ollama: 0.40.13  
Model: `llama3.2:1b`, `qwen2.5:1.5b`, `gemma2:2b`

---

## 1. Tujuan

Membandingkan 2–3 model lokal pada satu tugas nyata: **meringkas notulen rapat
berbahasa Indonesia** dengan strategi map-reduce. Yang diukur: latency,
throughput, rasio kompresi, dan berapa fakta kunci yang masih muncul.

Ini juga menjadi laporan tradeoff untuk kelulusan Fase 2 roadmap.

---

## 2. Metode

- Dokumen: [`data/contoh_dokumen.txt`](../data/contoh_dokumen.txt), 927 token.
- Strategi: `map_reduce`.
- Chunk: 600 token, overlap 80 → 2 chunk.
- `num_ctx`: 2048; `max_output_tokens`: 600; temperature 0.
- Setiap model melakukan satu panggilan warm-up sebelum pengukuran.
- Latency diukur di sekitar pemanggilan `ringkas_dokumen()` (map + reduce).
- Throughput dihitung dari `eval_count / eval_duration` yang dilaporkan Ollama.
- Kompresi = token input / token output.
- Kualitas = jumlah fakta kunci yang ditemukan dari 8 fakta:

| Fakta | Bentuk yang diterima |
|---|---|
| Kapasitas 25 MW | `25 mw`, `25mw` |
| Investasi Rp410 miliar | `410 miliar`, `rp410`, `rp 410` |
| Target COD 30 September 2026 | `30 september 2026`, `september 2026` |
| Mitra PT Surya Teknik | `surya teknik` |
| Lokasi Sumba | `sumba` |
| PPA dengan PLN 20 tahun | `pln`, `ppa` |
| Baterai 10 MWh | `10 mwh`, `10mwh` |
| Penambahan 3 insinyur | `3 insinyur`, `tiga insinyur` |

Pencocokan dilakukan pada teks ringkasan yang sudah di-lowercase. Metode ini
sederhana dan bisa melewatkan fakta yang ditulis dengan istilah berbeda.

---

## 3. Hasil

| Model | Latency | Token/detik | Kompresi | Fakta kunci |
|---|---:|---:|---:|---:|
| `llama3.2:1b` | 61,50 s | 11,50 | 6,22× | 5/8 |
| **`qwen2.5:1.5b`** | **69,83 s** | **16,88** | **2,35×** | **7/8** |
| `gemma2:2b` | 102,79 s | 8,81 | 2,19× | 6/8 |

### Fakta yang ditemukan

| Fakta | llama3.2:1b | qwen2.5:1.5b | gemma2:2b |
|---|:---:|:---:|:---:|
| Kapasitas 25 MW | ✅ | ✅ | ✅ |
| Investasi Rp410 miliar | ✅ | ✅ | ✅ |
| COD 30 September 2026 | ✅ | ✅ | ❌ |
| PT Surya Teknik | ❌ | ✅ | ✅ |
| Lokasi Sumba | ✅ | ✅ | ✅ |
| PPA/PLN | ✅ | ✅ | ✅ |
| Baterai 10 MWh | ❌ | ❌ | ❌ |
| Tambah 3 insinyur | ❌ | ✅ | ✅ |

### Catatan output

- **llama3.2:1b** menghasilkan ringkasan paling pendek (149 token output).
  Ringkasannya rapi, tetapi terlalu agresif memadatkan: Surya Teknik, baterai,
  dan keputusan penambahan insinyur hilang. Preamble “Berikut adalah ringkasan
  akhir…” berhasil dibersihkan oleh validator output.
- **qwen2.5:1.5b** menghasilkan ringkasan paling lengkap (394 token output) dan
  menemukan 7/8 fakta. Satu-satunya fakta yang hilang adalah baterai 10 MWh.
  Ringkasan berbentuk bullet dan memuat keputusan rapat dengan baik. Model juga
  sempat menulis “PPN” pada satu bullet, meskipun di bullet lain menulis “PPA”
  dengan benar — contoh kecil inkonsistensi model.
- **gemma2:2b** menghasilkan ringkasan naratif yang enak dibaca (423 token),
  tetapi melewatkan target COD dan baterai. Latency-nya paling tinggi karena
  sebagian besar inferensi berjalan di CPU saat VRAM 2 GB penuh.

---

## 4. Analisis Tradeoff

### Kecepatan

`llama3.2:1b` paling cepat (61,50 s), diikuti `qwen2.5:1.5b` (69,83 s), lalu
`gemma2:2b` (102,79 s). Namun latency total dipengaruhi jumlah panggilan model:
map-reduce memanggil model 3 kali (2 map + 1 reduce). Throughput per token lebih
adil dibandingkan: Qwen 16,88 token/detik, Llama 11,50, Gemma 8,81.

### Kualitas

Qwen menang di cakupan fakta (7/8) dan tetap paling cepat di antara model yang
ringkasannya lengkap. Llama lebih cepat, tetapi kehilangan 3 fakta penting.
Gemma berada di tengah kualitas, tetapi biaya waktunya sekitar 1,5 kali Qwen.

### Biaya

Semua model berjalan lokal, jadi **biaya marginal per request mendekati nol**
(tanpa biaya API). Biaya sebenarnya ada di listrik, waktu, dan hardware. Untuk
pemakaian pribadi dengan MX450 2 GB, Qwen 1,5B adalah kompromi terbaik.

### Kesesuaian hardware

- `llama3.2:1b` dan `qwen2.5:1.5b` (quantized) muat di VRAM 2 GB.
- `gemma2:2b` cenderung melimpah ke CPU. Pada benchmark Minggu 11, Gemma terukur
  sekitar 94% CPU / 6% GPU; pola yang sama menjelaskan latency tingginya di sini.

---

## 5. Keputusan

**Model default: `qwen2.5:1.5b`.**

Alasannya:

1. kualitas ringkasan paling baik (7/8 fakta);
2. throughput tertinggi (16,88 token/detik);
3. latency total masih wajar untuk dokumen 927 token;
4. muat di VRAM 2 GB.

`llama3.2:1b` tetap berguna untuk ringkasan cepat yang tidak menuntut detail.
`gemma2:2b` tidak dijadikan default karena latency-nya paling tinggi pada
hardware ini.

---

## 6. Keterbatasan

- Hanya satu dokumen dan satu run per model; belum ada pengulangan untuk
  mengukur variansi.
- Pencocokan fakta berbasis kata kunci, bukan evaluasi semantik.
- Latency bisa dipengaruhi cold start meskipun sudah ada warm-up; model tetap
  dimuat ulang jika Ollama membuangnya dari memori.
- Jumlah chunk hanya 2, sehingga jalur reduce hierarkis belum teruji pada
  dokumen yang jauh lebih panjang.
- Tokenizer `tiktoken` (`o200k_base`) berbeda dari tokenizer asli tiap model,
  jadi angka token adalah perkiraan, bukan hitungan persis model.
- Kualitas ringkasan belum diperiksa oleh human rater independen.

---

## 7. Pelajaran

- **Ringkasan panjang belum tentu lebih baik, tetapi ringkasan terlalu pendek
  kehilangan fakta.** Llama memadatkan 6,22× dan kehilangan 3 fakta; Qwen
  memadatkan 2,35× dan hanya kehilangan 1 fakta.
- **Map-reduce punya titik kehilangan di tahap map.** Baterai 10 MWh hilang
  sebelum tahap reduce, jadi memperbesar batas output tidak otomatis
  menyelamatkannya. Prompt map perlu menekankan jenis fakta yang wajib
  dipertahankan.
- **Jumlah parameter bukan penentu tunggal.** Gemma 2B lebih lambat daripada
  Qwen 1,5B karena kuantisasi dan keterbatasan VRAM.
- **Validasi bentuk ≠ validasi isi.** Kode bisa memastikan output tidak kosong
  dan tidak melewati batas token, tetapi tidak bisa menjamin kebenaran fakta.
  Untuk itu perlu ground truth dan evaluasi terpisah.

---

## 8. Reproduksi

```powershell
ollama pull llama3.2:1b
ollama pull qwen2.5:1.5b
ollama pull gemma2:2b

python -m scripts.benchmark
```

Hasil mentah tersimpan di `hasil_benchmark.json`.
