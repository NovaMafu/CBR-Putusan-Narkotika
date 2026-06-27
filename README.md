# Sistem Case-Based Reasoning (CBR) untuk Analisis Putusan Pengadilan
**Mata Kuliah Penalaran Komputer — SubCPMK-3** | Domain: Pidana Khusus Narkotika dan Psikotropika

Repository ini berisi implementasi sistem *Case-Based Reasoning* (CBR) sederhana
berbasis Python untuk mendukung analisis putusan pengadilan, mengikuti seluruh
siklus CBR: **Case Base → Case Representation → Case Retrieval → Case/Solution
Reuse → Model Evaluation**.

---

## 1. Tentang Dataset (Baca Sebelum Menjalankan)

Website resmi **Direktori Putusan Mahkamah Agung RI**
(`putusan3.mahkamahagung.go.id`) menerapkan *bot-detection* yang memblokir
*scraping* otomatis dari lingkungan tanpa browser asli/IP residensial.

Oleh karena itu, dataset default pada `data/raw/` (40 dokumen, memenuhi syarat
minimal ≥30) dibangun menggunakan **generator terstruktur**
(`src/generate_sample_casebase.py`) yang mereplikasi format, struktur bagian
(Identitas Terdakwa → Duduk Perkara → Pertimbangan Hukum → Amar Putusan →
Penutup), terminologi hukum, dan distribusi pasal/vonis sebagaimana ditemukan
pada ratusan entri publik Direktori Putusan kategori *Narkotika dan
Psikotropika*. Dataset ini memungkinkan **seluruh tahap pipeline CBR
dijalankan dan diuji end-to-end secara nyata** (bukan mock/placeholder kosong)
— semua angka akurasi, precision, recall, F1 pada `data/eval/` adalah hasil
komputasi sungguhan terhadap data ini.

**Untuk menggunakan data putusan ASLI dari MA RI:**
jalankan `src/scraper_real.py` (scraper fungsional menggunakan
`requests` + `BeautifulSoup` + `pdfminer.six`) dari komputer/Google Colab Anda
sendiri yang memiliki akses internet penuh:

```bash
cd src
python scraper_real.py --kategori narkotika-dan-psikotropika-1 \
                        --pengadilan mahkamah-agung \
                        --jumlah 30 \
                        --out ../data/_source_pdf_text
```

Setelah itu jalankan ulang `notebooks/01_case_base.ipynb` dst. **tanpa mengubah
kode apa pun** — format file output (`.txt` per kasus) identik dengan data
sampel demo, sehingga seluruh pipeline kompatibel langsung.

---

## 2. Struktur Folder

```
/data/
  ├─ _source_pdf_text/   # hasil konversi PDF/HTML -> teks (sebelum dibersihkan)
  ├─ raw/                # teks putusan yang sudah dibersihkan (Tahap 1)
  ├─ processed/          # cases.csv / cases.json (metadata terstruktur, Tahap 2)
  ├─ eval/                # queries.json, retrieval_metrics.csv, prediction_metrics.csv
  └─ results/             # predictions.csv (Tahap 4)
/notebooks/               # notebook per tahap CBR (01–05), sudah dieksekusi dgn output
/src/                      # skrip .py per tahap (sumber notebook) + scraper_real.py
/models/                   # model TF-IDF + SVM terlatih (.pkl)
/logs/cleaning.log         # log pembersihan data Tahap 1
requirements.txt
README.md
```

---

## 3. Instalasi

```bash
git clone <URL_REPOSITORY_INI>
cd <nama-folder-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 4. Menjalankan Pipeline End-to-End

Jalankan secara berurutan (setiap tahap membaca output tahap sebelumnya):

```bash
cd src

# (Opsional) buat ulang dataset sampel demo dari awal
python generate_sample_casebase.py

# Tahap 1: Membangun Case Base (cleaning + validasi)
python 01_case_base.py

# Tahap 2: Case Representation (ekstraksi metadata & fitur)
python 02_case_representation.py

# Tahap 3: Case Retrieval (TF-IDF + SVM + fungsi retrieve())
python 03_retrieval.py

# Tahap 4: Case/Solution Reuse (predict_outcome(): majority vote & weighted similarity)
python 04_predict.py

# Tahap 5: Model Evaluation (metrik + chart + analisis kegagalan)
python 05_evaluation.py
```

Atau jalankan versi notebook interaktif di `notebooks/01_case_base.ipynb`
hingga `notebooks/05_evaluation.ipynb` secara berurutan (mis. lewat Jupyter
Lab/Notebook atau Google Colab).

### Contoh penggunaan fungsi `retrieve()` dan `predict_outcome()` secara langsung

```python
import sys; sys.path.append("src")
from importlib import import_module
m3 = import_module("03_retrieval")
m4 = import_module("04_predict")

# Cari 5 kasus paling mirip
top_ids, scores = m3.retrieve(
    "Terdakwa ditemukan menyimpan sabu-sabu seberat 2 gram tanpa hak",
    m3.vectorizer, m3.tfidf_matrix, m3.df, k=5
)
print(top_ids, scores)

# Prediksi kategori vonis untuk kasus baru
hasil = m4.predict_outcome("Terdakwa mengedarkan ganja 450 gram", k=5, method="majority")
print(hasil)
```

---

## 5. Ringkasan Metodologi

| Tahap | Metode | Output |
|---|---|---|
| 1. Case Base | Regex cleaning (hapus header/footer/watermark), validasi kelengkapan ≥80% | `data/raw/*.txt`, `logs/cleaning.log` |
| 2. Case Representation | Regex metadata extraction + bag-of-words sederhana | `data/processed/cases.csv` |
| 3. Case Retrieval | TF-IDF (1-2 gram) + cosine similarity; SVM linear (80:20 split) | `models/*.pkl`, `data/eval/queries.json` |
| 4. Solution Reuse | Majority vote & weighted-similarity voting atas top-5 kasus | `data/results/predictions.csv` |
| 5. Evaluation | Accuracy, Precision, Recall, F1 (retrieval & prediksi) + error analysis | `data/eval/*_metrics.csv`, `evaluation_chart.png` |

**Hasil utama** (pada dataset sampel 40 dokumen): akurasi top-1 label
retrieval = 100%, mean Precision@5 = 0.625, mean Recall@5 = 0.318; akurasi
prediksi solusi (majority vote) = 87.5%. Analisis kegagalan & rekomendasi
perbaikan tersedia di output `05_evaluation.py` / `prediction_metrics.csv`
dan dibahas lebih lanjut pada laporan.

---

## 6. Lisensi & Catatan Akademik

Dibuat untuk keperluan tugas SubCPMK-3 Mata Kuliah Penalaran Komputer,
Informatika, Universitas Muhammadiyah Malang. Dataset sampel pada repository
ini bersifat sintetis-terstruktur untuk tujuan demonstrasi pipeline; bukan
salinan dokumen resmi Mahkamah Agung RI.
