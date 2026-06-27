#!/usr/bin/env python
# coding: utf-8
"""
01_case_base.py  —  TAHAP 1: Membangun Case Base
=================================================
Tujuan : Mengumpulkan & menyiapkan (cleaning) corpus putusan yang bersih.

Langkah kerja yang diimplementasikan (sesuai SubCPMK-3):
  i.   Seleksi & Unduh         -> lihat src/scraper_real.py (scraping asli) atau
                                   src/generate_sample_casebase.py (data sampel demo)
  ii.  Konversi & Ekstraksi    -> sudah dilakukan, hasil ada di data/_source_pdf_text/*.txt
  iii. Pembersihan             -> DILAKUKAN DI SINI (hapus header/footer/watermark,
                                   normalisasi spasi & karakter, tokenisasi)
  iv.  Validasi                -> cek kelengkapan isi (>=80%), catat ke cleaning.log
  v.   Output                  -> data/raw/*.txt (bersih) + logs/cleaning.log
"""

import os
import re
import glob

_THIS_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
BASE_DIR = os.path.join(_THIS_DIR, "..")
SRC_DIR = os.path.join(BASE_DIR, "data", "_source_pdf_text")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# Pola header/footer/watermark yang umum muncul pada dokumen putusan hasil
# konversi PDF->text (garis pembatas, nomor halaman, disclaimer situs, dll.)
# -----------------------------------------------------------------------
NOISE_PATTERNS = [
    r"^=+$",                                                  # garis pembatas =====
    r"^DIREKTORI PUTUSAN MAHKAMAH AGUNG REPUBLIK INDONESIA$",
    r"^putusan\.mahkamahagung\.go\.id$",
    r"^-{2,}\s*Halaman\s+\d+\s+dari\s+\d+\s*-{2,}$",          # marker halaman
    r"^Disclaimer:.*$",
    r"^akademik mata kuliah Penalaran Komputer.*$",
    r"^resmi Mahkamah Agung RI\. Lihat README\.md\.$",
]
NOISE_RE = re.compile("|".join(NOISE_PATTERNS), re.IGNORECASE)


def hapus_header_footer(text: str) -> str:
    """Hapus baris-baris noise (header/footer/watermark/page marker)."""
    lines = text.split("\n")
    cleaned = [ln for ln in lines if not NOISE_RE.match(ln.strip())]
    return "\n".join(cleaned)


def normalisasi(text: str) -> str:
    """Normalisasi spasi & karakter. Teks asli (case asli) dipertahankan untuk
    NER/metadata-extraction pada Tahap 2; normalisasi spasi dilakukan di sini,
    sedangkan lower-casing dilakukan terpisah saat membentuk fitur teks (Tahap 2/3)
    agar nama pihak & nomor perkara tetap terbaca apa adanya."""
    text = re.sub(r"[ \t]+", " ", text)          # spasi/tab ganda -> satu spasi
    text = re.sub(r"\n{3,}", "\n\n", text)        # baris kosong berlebih
    text = "\n".join(ln.strip() for ln in text.split("\n"))
    return text.strip()


def validasi_kelengkapan(original: str, cleaned: str) -> float:
    """Rasio panjang teks bersih terhadap teks asli (proxy validasi >=80%)."""
    if len(original) == 0:
        return 0.0
    return len(cleaned) / len(original)


def main():
    files = sorted(glob.glob(os.path.join(SRC_DIR, "*.txt")))
    log_lines = []
    n_ok, n_warn = 0, 0

    for fp in files:
        fname = os.path.basename(fp)
        with open(fp, "r", encoding="utf-8") as f:
            original = f.read()

        no_header = hapus_header_footer(original)
        cleaned = normalisasi(no_header)
        rasio = validasi_kelengkapan(original, cleaned)

        status = "OK" if rasio >= 0.80 else "WARNING(<80%)"
        if status == "OK":
            n_ok += 1
        else:
            n_warn += 1

        out_path = os.path.join(RAW_DIR, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        log_lines.append(f"{fname}\tpanjang_asli={len(original)}\tpanjang_bersih={len(cleaned)}\trasio={rasio:.2%}\tstatus={status}")

    log_path = os.path.join(LOG_DIR, "cleaning.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("LOG PEMBERSIHAN DATA - TAHAP 1: MEMBANGUN CASE BASE\n")
        f.write(f"Total dokumen diproses : {len(files)}\n")
        f.write(f"Status OK (>=80%)      : {n_ok}\n")
        f.write(f"Status WARNING (<80%)  : {n_warn}\n")
        f.write("-" * 90 + "\n")
        f.write("\n".join(log_lines))

    print(f"[OK] {len(files)} dokumen dibersihkan -> {RAW_DIR}")
    print(f"[OK] Validasi: {n_ok} OK, {n_warn} WARNING (lihat {log_path})")
    assert len(files) >= 30, "Jumlah dokumen kurang dari syarat minimal (30)."
    print(f"[OK] Syarat minimal 30 dokumen terpenuhi ({len(files)} dokumen).")


if __name__ == "__main__":
    main()
