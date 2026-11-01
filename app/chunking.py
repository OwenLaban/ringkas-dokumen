"""Pemotongan teks berbasis token.

Model menghitung token, bukan kata atau karakter. Karena itu chunking di sini
memakai tokenizer yang sama untuk menghitung batas dan memotong teks.
"""

import tiktoken

ENCODING = "o200k_base"


def hitung_token(teks: str, encoding: str = ENCODING) -> int:
    """Kembalikan jumlah token dari sebuah teks."""
    encoder = tiktoken.get_encoding(encoding)
    return len(encoder.encode(teks))


def potong_teks(
    teks: str,
    max_token: int,
    overlap: int = 0,
    encoding: str = ENCODING,
) -> list[str]:
    """Potong teks menjadi chunk yang tidak melebihi ``max_token``.

    ``overlap`` mengulang sebagian token di awal chunk berikutnya agar kalimat
    atau konteks yang terpotong tidak hilang sepenuhnya.
    """
    if max_token <= 0:
        raise ValueError("max_token harus > 0")
    if overlap < 0 or overlap >= max_token:
        raise ValueError("overlap harus >= 0 dan < max_token")
    if not teks:
        return []

    encoder = tiktoken.get_encoding(encoding)
    token_ids = encoder.encode(teks)
    langkah = max_token - overlap
    potongan: list[str] = []

    for mulai in range(0, len(token_ids), langkah):
        akhir = min(mulai + max_token, len(token_ids))
        potongan.append(encoder.decode(token_ids[mulai:akhir]))
        if akhir >= len(token_ids):
            break

    return potongan
