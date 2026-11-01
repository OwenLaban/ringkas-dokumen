from app.chunking import hitung_token, potong_teks

TEKS = " ".join(f"kata{i}" for i in range(60))


def test_hitung_token_positif():
    assert hitung_token("Halo dunia") > 0


def test_teks_kosong_menghasilkan_list_kosong():
    assert potong_teks("", max_token=10) == []


def test_potongan_tidak_melebihi_batas():
    potongan = potong_teks(TEKS, max_token=10)

    assert len(potongan) > 1
    assert all(hitung_token(bagian) <= 10 for bagian in potongan)


def test_overlap_menambah_potongan():
    tanpa_overlap = potong_teks(TEKS, max_token=10, overlap=0)
    dengan_overlap = potong_teks(TEKS, max_token=10, overlap=4)

    assert len(dengan_overlap) > len(tanpa_overlap)


def test_validasi_parameter():
    try:
        potong_teks(TEKS, max_token=0)
    except ValueError:
        pass
    else:
        raise AssertionError("max_token 0 harus ditolak")

    try:
        potong_teks(TEKS, max_token=10, overlap=10)
    except ValueError:
        pass
    else:
        raise AssertionError("overlap >= max_token harus ditolak")
