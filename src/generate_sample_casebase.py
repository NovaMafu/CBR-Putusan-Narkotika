"""
generate_sample_casebase.py
-----------------------------------------------------------------------------
TUJUAN
Skrip ini MEMBANGUN /data/raw/*.txt yang merepresentasikan dokumen putusan
pengadilan domain "Pidana Khusus Narkotika dan Psikotropika".

CATATAN PENTING (baca README.md bagian "Tentang Dataset"):
Website resmi Direktori Putusan Mahkamah Agung RI (putusan3.mahkamahagung.go.id)
menerapkan bot-detection yang memblokir automated scraping dari lingkungan
sandbox/CI tanpa browser asli. Oleh karena itu, dataset pada repository ini
dibangun dengan generator terstruktur (rule + template + randomisasi) yang
mereplikasi format, struktur, dan distribusi linguistik dokumen putusan asli
(berdasarkan observasi terhadap ratusan entri direktori publik), sehingga
seluruh tahap pipeline CBR (cleaning, metadata extraction, retrieval, reuse,
evaluation) dapat dijalankan end-to-end dan diuji secara nyata.

Script scraper SUNGGUHAN (functional, bukan simulasi) untuk mengambil data asli
dari Direktori Putusan MA RI disediakan terpisah di src/scraper_real.py.
Jalankan scraper itu dari komputer/Colab yang memiliki akses internet penuh
(bukan dari sandbox ini) untuk MENGGANTI isi data/_source_pdf_text/ dengan
dokumen asli (hasil konversi PDF->text), lalu jalankan ulang notebook 01-05
tanpa perlu mengubah kode apa pun karena format file output identik.

Output skrip ini merepresentasikan hasil Langkah Kerja 1.ii (Konversi & Ekstraksi
Teks) pada Tahap 1: teks mentah HASIL KONVERSI, BELUM dibersihkan (masih ada
header/footer/watermark/disclaimer). Proses Pembersihan (1.iii) dilakukan oleh
notebook 01_case_base.ipynb, yang membaca dari data/_source_pdf_text/ dan
menulis hasil bersih ke data/raw/*.txt sesuai struktur folder yang diminta.
"""

import os
import random
import textwrap

random.seed(42)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "_source_pdf_text")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Pools data untuk randomisasi (mengikuti pola yang umum ditemukan pada
# entri publik Direktori Putusan MA RI kategori Narkotika dan Psikotropika)
# ---------------------------------------------------------------------------

PENGADILAN = [
    "PN JAKARTA SELATAN", "PN SURABAYA", "PN MALANG", "PN DENPASAR",
    "PN SURAKARTA", "PN MEDAN", "PN BANDUNG", "PN SEMARANG",
    "PN MAKASSAR", "PN PALEMBANG", "PN YOGYAKARTA", "PN PEKANBARU",
]

NAMA_DEPAN = [
    "Ahmad", "Budi", "Candra", "Dedi", "Eko", "Fajar", "Gilang", "Hendra",
    "Irwan", "Joko", "Komarudin", "Lukman", "Maulana", "Nanda", "Oki",
    "Putra", "Rian", "Sandi", "Taufik", "Umar", "Wawan", "Yusuf", "Zainal",
]
NAMA_BELAKANG = [
    "Saputra", "Pratama", "Wijaya", "Setiawan", "Hidayat", "Kusuma",
    "Gunawan", "Hermawan", "Susanto", "Rahman", "Firmansyah", "Nugroho",
]

PASAL_POOL = [
    ("Pasal 114 ayat (1) UU No. 35 Tahun 2009 tentang Narkotika", "kepemilikan/pengedaran"),
    ("Pasal 112 ayat (1) UU No. 35 Tahun 2009 tentang Narkotika", "kepemilikan/penguasaan"),
    ("Pasal 127 ayat (1) UU No. 35 Tahun 2009 tentang Narkotika", "penyalahgunaan untuk diri sendiri"),
    ("Pasal 132 ayat (1) UU No. 35 Tahun 2009 tentang Narkotika", "percobaan/permufakatan jahat"),
    ("Pasal 111 ayat (1) UU No. 35 Tahun 2009 tentang Narkotika", "kepemilikan narkotika golongan I bentuk tanaman"),
]

BARANG_BUKTI = [
    ("sabu-sabu (metamfetamina)", (0.3, 25.0), "gram"),
    ("ganja kering", (5.0, 500.0), "gram"),
    ("pil ekstasi", (5, 120), "butir"),
    ("tembakau sintetis", (2.0, 40.0), "gram"),
]

PENUNTUT_UMUM = [
    "Dewi Kartika, S.H.", "Rudi Hartono, S.H., M.H.", "Siti Maryam, S.H.",
    "Anton Wibowo, S.H.", "Lina Marlina, S.H., M.H.", "Bambang Sutejo, S.H.",
]

# Distribusi outcome (label) - dibuat agar berkorelasi dengan jenis pasal &
# berat barang bukti, sehingga pola "mirip kasus -> mirip vonis" benar2 ada
# (penting supaya tahap retrieval & solution reuse punya sinyal yang masuk akal).

def pilih_outcome(pasal_idx, berat_norm):
    """Tentukan kategori vonis berdasarkan pasal & 'keparahan' barang bukti."""
    if pasal_idx == 2:  # Pasal 127 - penyalahguna
        return random.choices(
            ["Rehabilitasi", "Pidana Penjara < 1 Tahun"], weights=[0.65, 0.35]
        )[0]
    if berat_norm < 0.33:
        return random.choices(
            ["Pidana Penjara < 1 Tahun", "Pidana Penjara 1-4 Tahun"], weights=[0.4, 0.6]
        )[0]
    elif berat_norm < 0.66:
        return random.choices(
            ["Pidana Penjara 1-4 Tahun", "Pidana Penjara 4-8 Tahun"], weights=[0.5, 0.5]
        )[0]
    else:
        return random.choices(
            ["Pidana Penjara 4-8 Tahun", "Pidana Penjara > 8 Tahun"], weights=[0.45, 0.55]
        )[0]

LAMA_PIDANA = {
    "Rehabilitasi": "menjalani rehabilitasi medis dan sosial",
    "Pidana Penjara < 1 Tahun": "pidana penjara selama 8 (delapan) bulan dan denda Rp1.000.000,00",
    "Pidana Penjara 1-4 Tahun": "pidana penjara selama 2 (dua) tahun 6 (enam) bulan dan denda Rp800.000.000,00",
    "Pidana Penjara 4-8 Tahun": "pidana penjara selama 5 (lima) tahun dan denda Rp1.000.000.000,00",
    "Pidana Penjara > 8 Tahun": "pidana penjara selama 10 (sepuluh) tahun dan denda Rp1.500.000.000,00",
}

N_DOCS = 40


def buat_nomor_perkara(idx, tahun, pengadilan_code):
    return f"{100 + idx}/Pid.Sus/{tahun}/{pengadilan_code}"


def pengadilan_code(nama):
    mapping = {
        "PN JAKARTA SELATAN": "PN Jkt.Sel", "PN SURABAYA": "PN Sby",
        "PN MALANG": "PN Mlg", "PN DENPASAR": "PN Dps", "PN SURAKARTA": "PN Skt",
        "PN MEDAN": "PN Mdn", "PN BANDUNG": "PN Bdg", "PN SEMARANG": "PN Smg",
        "PN MAKASSAR": "PN Mks", "PN PALEMBANG": "PN Plg", "PN YOGYAKARTA": "PN Yyk",
        "PN PEKANBARU": "PN Pbr",
    }
    return mapping[nama]


def buat_dokumen(idx):
    pengadilan = random.choice(PENGADILAN)
    tahun = random.choice([2021, 2022, 2023, 2024])
    nama_terdakwa = f"{random.choice(NAMA_DEPAN)} {random.choice(NAMA_BELAKANG)}"
    alias = random.choice(["", " Alias " + random.choice(NAMA_DEPAN).upper()])
    nama_terdakwa_full = nama_terdakwa + alias

    pasal_idx = random.randint(0, len(PASAL_POOL) - 1)
    pasal_text, deskripsi_pasal = PASAL_POOL[pasal_idx]

    barang, rng, unit = random.choice(BARANG_BUKTI)
    berat = round(random.uniform(*rng), 2)
    berat_norm = (berat - rng[0]) / (rng[1] - rng[0])

    outcome = pilih_outcome(pasal_idx, berat_norm)
    amar = LAMA_PIDANA[outcome]

    no_perkara = buat_nomor_perkara(idx, tahun, pengadilan_code(pengadilan))
    tanggal = f"{random.randint(1,28):02d}-{random.randint(1,12):02d}-{tahun}"
    jaksa = random.choice(PENUNTUT_UMUM)

    ringkasan_fakta = textwrap.dedent(f"""\
        Bahwa pada hari dan tanggal sebagaimana tersebut di atas, Terdakwa
        {nama_terdakwa_full} ditangkap oleh petugas Kepolisian berdasarkan informasi
        masyarakat terkait dugaan tindak pidana narkotika di wilayah hukum
        {pengadilan}. Pada saat dilakukan penangkapan dan penggeledahan badan
        maupun tempat tinggal Terdakwa, ditemukan barang bukti berupa
        {barang} dengan berat/jumlah kurang lebih {berat} {unit} yang disimpan
        Terdakwa tanpa hak atau melawan hukum. Barang bukti tersebut kemudian
        diuji di laboratorium forensik dan hasilnya dinyatakan positif
        mengandung zat narkotika golongan I sebagaimana dimaksud dalam
        Undang-Undang Nomor 35 Tahun 2009 tentang Narkotika.
        """)

    argumen_hukum = textwrap.dedent(f"""\
        Menimbang, bahwa Penuntut Umum {jaksa} dalam surat dakwaannya
        menyatakan Terdakwa telah melakukan tindak pidana sebagaimana diatur dan
        diancam pidana dalam {pasal_text} terkait {deskripsi_pasal}.
        Menimbang, bahwa unsur "tanpa hak atau melawan hukum" telah terpenuhi
        berdasarkan keterangan saksi, barang bukti, dan hasil uji laboratorium.
        Majelis Hakim berpendapat bahwa seluruh unsur dalam dakwaan telah
        terbukti secara sah dan meyakinkan menurut hukum.
        """)

    header = textwrap.dedent(f"""\
        =====================================================================
        DIREKTORI PUTUSAN MAHKAMAH AGUNG REPUBLIK INDONESIA
        putusan.mahkamahagung.go.id
        =====================================================================
        P U T U S A N
        Nomor {no_perkara}

        DEMI KEADILAN BERDASARKAN KETUHANAN YANG MAHA ESA

        Pengadilan Negeri {pengadilan.replace('PN ', '')} yang mengadili perkara pidana
        dengan acara pemeriksaan biasa dalam tingkat pertama menjatuhkan putusan
        sebagai berikut dalam perkara Terdakwa:

        Nama lengkap   : {nama_terdakwa_full}
        Tempat lahir   : {pengadilan.replace('PN ', '').title()}
        Umur/tgl lahir : {random.randint(19,45)} Tahun
        Jenis kelamin  : {random.choice(['Laki-laki','Perempuan'])}
        Kebangsaan     : Indonesia
        Tempat tinggal : {pengadilan.replace('PN ', '').title()}
        Agama          : {random.choice(['Islam','Kristen','Hindu','Katolik','Buddha'])}
        Pekerjaan      : {random.choice(['Buruh','Wiraswasta','Pengangguran','Karyawan Swasta','Sopir'])}
        """)

    duduk_perkara = "DUDUK PERKARA\n\n" + ringkasan_fakta

    pertimbangan = "PERTIMBANGAN HUKUM\n\n" + argumen_hukum

    amar_putusan = textwrap.dedent(f"""\
        MENGADILI

        1. Menyatakan Terdakwa {nama_terdakwa_full} telah terbukti secara sah dan
           meyakinkan bersalah melakukan tindak pidana narkotika sebagaimana
           diatur dalam {pasal_text};
        2. Menjatuhkan pidana kepada Terdakwa tersebut dengan {amar};
        3. Menetapkan barang bukti berupa {barang} dirampas untuk dimusnahkan;
        4. Membebankan biaya perkara kepada Terdakwa.
        """)

    footer = textwrap.dedent(f"""\
        Demikianlah diputuskan dalam rapat permusyawaratan Majelis Hakim
        Pengadilan Negeri {pengadilan.replace('PN ', '')} pada hari kerja, diucapkan dalam
        persidangan yang terbuka untuk umum.

        Panitera Pengganti                         Ketua Majelis Hakim


        -------------------------------- Halaman {idx+1} dari 1 --------------------------------
        Disclaimer: Dokumen ini merupakan data sampel terstruktur untuk keperluan
        akademik mata kuliah Penalaran Komputer (SubCPMK-3), BUKAN dokumen putusan
        resmi Mahkamah Agung RI. Lihat README.md.
        """)

    full_text = "\n".join([
        header, duduk_perkara, pertimbangan, amar_putusan, footer
    ])

    meta = {
        "case_id": idx,
        "no_perkara": no_perkara,
        "tanggal": tanggal,
        "pengadilan": pengadilan,
        "jenis_perkara": "Pidana Khusus Narkotika dan Psikotropika",
        "pasal": pasal_text,
        "pihak_terdakwa": nama_terdakwa_full,
        "penuntut_umum": jaksa,
        "barang_bukti": f"{barang} {berat} {unit}",
        "outcome_label": outcome,
    }
    return full_text, meta


def main():
    metas = []
    for i in range(1, N_DOCS + 1):
        text, meta = buat_dokumen(i)
        fname = f"case_{i:03d}.txt"
        with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as f:
            f.write(text)
        meta["filename"] = fname
        metas.append(meta)

    print(f"[OK] {len(metas)} dokumen raw disimpan di: {OUT_DIR}")

    # simpan ground-truth generator metadata (untuk dibandingkan dgn hasil ekstraksi regex)
    import json
    gt_path = os.path.join(os.path.dirname(__file__), "..", "data", "_generator_ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)
    print(f"[OK] Ground-truth generator metadata: {gt_path}")


if __name__ == "__main__":
    main()
