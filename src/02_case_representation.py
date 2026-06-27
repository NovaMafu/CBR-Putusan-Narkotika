#!/usr/bin/env python
# coding: utf-8
"""
02_case_representation.py  —  TAHAP 2: Case Representation
============================================================
Tujuan: Representasikan setiap putusan (file di data/raw/*.txt) ke dalam
struktur data terorganisir (cases.csv).

Langkah kerja:
  i.   Ekstraksi Metadata     -> regex: no_perkara, tanggal, jenis_perkara, pasal, pihak
  ii.  Ekstraksi Konten Kunci -> ringkasan_fakta (DUDUK PERKARA), argumen hukum (PERTIMBANGAN HUKUM)
  iii. Feature Engineering    -> panjang teks (jumlah kata), bag-of-words top terms
  iv.  Penyimpanan            -> data/processed/cases.csv
"""

import os
import re
import glob
import json
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
BASE_DIR = os.path.join(_THIS_DIR, "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

RE_NO_PERKARA = re.compile(r"Nomor\s+([0-9]+/Pid\.Sus/\d{4}/PN[\w.\s]+?)\n")
RE_PIHAK = re.compile(r"Nama lengkap\s*:\s*(.+)")
RE_PASAL = re.compile(r"(Pasal\s+\d+\s+ayat\s*\(\d+\)\s+UU No\.\s*\d+\s*Tahun\s*\d{4}\s*tentang\s*Narkotika)")
RE_TANGGAL_FILE = re.compile(r"(\d{2}-\d{2}-\d{4})")
RE_PENGADILAN = re.compile(r"Pengadilan Negeri\s+([A-Z\s]+?)\s+yang mengadili")
RE_PENUNTUT = re.compile(r"Penuntut Umum\s+([^\n]+?)\s+dalam surat dakwaannya")
RE_BARANG_BUKTI = re.compile(r"barang bukti berupa\s+(.+?)\s+dengan berat/jumlah kurang lebih\s+([\d.]+)\s+(\w+)")
RE_VONIS = re.compile(r"dengan\s+(pidana penjara selama[^;]+|menjalani rehabilitasi[^;]+);")

SECTION_DUDUK = re.compile(r"DUDUK PERKARA\n\n(.*?)\n\nPERTIMBANGAN HUKUM", re.DOTALL)
SECTION_PERTIMBANGAN = re.compile(r"PERTIMBANGAN HUKUM\n\n(.*?)\n\nMENGADILI", re.DOTALL)


def klasifikasi_outcome(vonis_text: str) -> str:
    """Turunkan label kategori outcome dari kalimat amar putusan (untuk
    dijadikan target klasifikasi/ground-truth pada Tahap 3 & 4)."""
    if vonis_text is None:
        return "Tidak Diketahui"
    t = vonis_text.lower()
    if "rehabilitasi" in t:
        return "Rehabilitasi"
    m = re.search(r"selama\s+(\d+)\s*\(\w+\)\s*tahun", t)
    if m:
        tahun = int(m.group(1))
        if tahun < 1:
            return "Pidana Penjara < 1 Tahun"
        elif tahun <= 4:
            return "Pidana Penjara 1-4 Tahun"
        elif tahun <= 8:
            return "Pidana Penjara 4-8 Tahun"
        else:
            return "Pidana Penjara > 8 Tahun"
    if "bulan" in t:
        return "Pidana Penjara < 1 Tahun"
    return "Tidak Diketahui"


def ekstrak_metadata(case_id: int, fname: str, text: str) -> dict:
    no_perkara = RE_NO_PERKARA.search(text)
    pihak = RE_PIHAK.search(text)
    pasal = RE_PASAL.search(text)
    pengadilan = RE_PENGADILAN.search(text)
    penuntut = RE_PENUNTUT.search(text)
    barang = RE_BARANG_BUKTI.search(text)
    vonis = RE_VONIS.search(text)
    duduk = SECTION_DUDUK.search(text)
    pertimbangan = SECTION_PERTIMBANGAN.search(text)

    ringkasan_fakta = " ".join(duduk.group(1).split()) if duduk else ""
    argumen_hukum = " ".join(pertimbangan.group(1).split()) if pertimbangan else ""

    barang_bukti_str = (
        f"{barang.group(1)} {barang.group(2)} {barang.group(3)}" if barang else ""
    )

    vonis_text = vonis.group(1) if vonis else None

    full_text_for_features = f"{ringkasan_fakta} {argumen_hukum}"
    jumlah_kata = len(full_text_for_features.split())

    return {
        "case_id": case_id,
        "filename": fname,
        "no_perkara": no_perkara.group(1).strip() if no_perkara else None,
        "pengadilan": ("PN " + pengadilan.group(1).strip()) if pengadilan else None,
        "jenis_perkara": "Pidana Khusus Narkotika dan Psikotropika",
        "pasal": pasal.group(1) if pasal else None,
        "pihak_terdakwa": pihak.group(1).strip() if pihak else None,
        "penuntut_umum": penuntut.group(1).strip() if penuntut else None,
        "barang_bukti": barang_bukti_str,
        "ringkasan_fakta": ringkasan_fakta,
        "argumen_hukum": argumen_hukum,
        "amar_putusan": vonis_text,
        "outcome_label": klasifikasi_outcome(vonis_text),
        "jumlah_kata": jumlah_kata,
        "text_full": full_text_for_features,
    }


def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.txt")))
    rows = []
    gagal = []

    for i, fp in enumerate(files, start=1):
        fname = os.path.basename(fp)
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read()
        try:
            row = ekstrak_metadata(i, fname, text)
            if not row["no_perkara"] or not row["pasal"]:
                gagal.append(fname)
            rows.append(row)
        except Exception as e:
            gagal.append(f"{fname} (error: {e})")

    df = pd.DataFrame(rows)

    # Feature engineering tambahan: bag-of-words ringkas (top kata kunci hukum)
    keywords = ["narkotika", "psikotropika", "rehabilitasi", "penjara", "denda",
                "barang bukti", "tanpa hak", "melawan hukum"]
    for kw in keywords:
        col = "bow_" + kw.replace(" ", "_")
        df[col] = df["text_full"].str.lower().str.count(kw)

    csv_path = os.path.join(PROC_DIR, "cases.csv")
    json_path = os.path.join(PROC_DIR, "cases.json")
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    print(f"[OK] {len(df)} kasus diekstrak metadata -> {csv_path}")
    print(f"[OK] Kolom: {list(df.columns)}")
    if gagal:
        print(f"[WARNING] {len(gagal)} dokumen gagal ekstraksi penuh: {gagal}")
    else:
        print("[OK] Seluruh dokumen berhasil diekstrak metadata lengkap.")

    print("\nDistribusi outcome_label:")
    print(df["outcome_label"].value_counts())


if __name__ == "__main__":
    main()
