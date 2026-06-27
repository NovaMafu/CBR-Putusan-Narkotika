#!/usr/bin/env python
# coding: utf-8
"""
scraper_real.py — Scraper SUNGGUHAN untuk Direktori Putusan Mahkamah Agung RI
================================================================================
PENTING: Jalankan skrip ini dari komputer/Google Colab Anda sendiri yang
memiliki akses internet PENUH dan dapat menjalankan browser/requests biasa.
Skrip ini TIDAK akan berjalan di sandbox eksekusi kode pada chat ini, karena
sandbox tersebut memblokir akses ke domain selain beberapa domain paket (pip/
npm/github) — dan situs putusan3.mahkamahagung.go.id juga menerapkan deteksi
bot terhadap traffic non-browser tanpa header yang lengkap/IP residensial.

Cara pakai:
    pip install requests beautifulsoup4 pdfminer.six lxml
    python scraper_real.py --kategori narkotika-dan-psikotropika-1 \
                            --pengadilan mahkamah-agung \
                            --jumlah 30 \
                            --out ../data/_source_pdf_text

Setelah selesai, jalankan notebook 01_case_base.ipynb seperti biasa — kode
pembersihan & seluruh tahap berikutnya TIDAK perlu diubah karena format file
output (.txt per kasus) identik dengan data sampel demo.
"""

import os
import re
import time
import argparse
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://putusan3.mahkamahagung.go.id"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8",
}


def get_listing_url(kategori: str, pengadilan: str, page: int) -> str:
    base = f"{BASE_URL}/direktori/index/pengadilan/{pengadilan}/kategori/{kategori}.html"
    if page > 1:
        base = base.replace(".html", f"/page/{page}.html")
    return base


def ambil_link_putusan(listing_html: str):
    soup = BeautifulSoup(listing_html, "lxml")
    links = []
    for a in soup.select("a[href*='/putusan/']"):
        href = a.get("href")
        if href and href not in links:
            links.append(href)
    return links


def ambil_teks_putusan(detail_url: str, session: requests.Session) -> str:
    """Buka halaman detail putusan, lalu unduh & ekstrak teks dari PDF putusan
    (umumnya tersedia melalui tombol 'Download PDF' pada halaman detail)."""
    resp = session.get(detail_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    pdf_link_tag = soup.select_one("a[href$='.pdf'], a:-soup-contains('Download')")
    if not pdf_link_tag or not pdf_link_tag.get("href"):
        # fallback: ambil konten ringkasan/abstrak yang tampil di halaman detail
        konten = soup.select_one("#content") or soup.select_one(".entry-content")
        return konten.get_text("\n", strip=True) if konten else ""

    pdf_url = pdf_link_tag["href"]
    if pdf_url.startswith("/"):
        pdf_url = BASE_URL + pdf_url

    pdf_resp = session.get(pdf_url, headers=HEADERS, timeout=60)
    pdf_resp.raise_for_status()

    tmp_pdf = "_tmp_putusan.pdf"
    with open(tmp_pdf, "wb") as f:
        f.write(pdf_resp.content)

    try:
        from pdfminer.high_level import extract_text
        teks = extract_text(tmp_pdf)
    finally:
        if os.path.exists(tmp_pdf):
            os.remove(tmp_pdf)

    return teks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kategori", default="narkotika-dan-psikotropika-1")
    parser.add_argument("--pengadilan", default="mahkamah-agung")
    parser.add_argument("--jumlah", type=int, default=30)
    parser.add_argument("--out", default="../data/_source_pdf_text")
    parser.add_argument("--delay", type=float, default=2.0,
                         help="Delay antar-request (detik) — sopan terhadap server, hindari blokir.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    session = requests.Session()

    semua_link = []
    page = 1
    while len(semua_link) < args.jumlah:
        url = get_listing_url(args.kategori, args.pengadilan, page)
        print(f"[INFO] Mengambil daftar putusan: {url}")
        resp = session.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            print(f"[WARNING] Gagal mengambil halaman {page} (status {resp.status_code}), berhenti.")
            break
        links = ambil_link_putusan(resp.text)
        if not links:
            print("[WARNING] Tidak ada link ditemukan di halaman ini, berhenti.")
            break
        semua_link.extend(links)
        page += 1
        time.sleep(args.delay)

    semua_link = semua_link[: args.jumlah]
    print(f"[INFO] Total {len(semua_link)} link putusan akan diunduh.")

    for i, link in enumerate(semua_link, start=1):
        detail_url = link if link.startswith("http") else BASE_URL + link
        try:
            teks = ambil_teks_putusan(detail_url, session)
            if not teks.strip():
                print(f"[WARNING] Teks kosong untuk {detail_url}, dilewati.")
                continue
            out_path = os.path.join(args.out, f"case_{i:03d}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(teks)
            print(f"[OK] ({i}/{len(semua_link)}) Disimpan -> {out_path}")
        except Exception as e:
            print(f"[ERROR] Gagal mengambil {detail_url}: {e}")
        time.sleep(args.delay)

    print("\n[SELESAI] Jalankan notebook 01_case_base.ipynb untuk membersihkan data ini.")


if __name__ == "__main__":
    main()
