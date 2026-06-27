#!/usr/bin/env python
# coding: utf-8
"""
04_predict.py  —  TAHAP 4: Case/Solution Reuse
=================================================
Tujuan: Gunakan putusan lama (top-k hasil retrieval) sebagai dasar prediksi
"solusi" (kategori vonis / ringkasan amar putusan) untuk kasus baru.

Langkah kerja:
  i.   Ekstrak Solusi      -> {case_id: solusi_text} dari kolom amar_putusan
  ii.  Algoritma Prediksi  -> Majority Vote DAN Weighted Similarity (keduanya
                              diimplementasikan agar bisa dibandingkan pada Tahap 5)
  iii. Implementasi Fungsi -> predict_outcome(query) -> str
  iv.  Demo Manual         -> 5 contoh kasus baru -> predict_outcome() -> bandingkan
  v.   Output              -> data/results/predictions.csv
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse

_THIS_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
BASE_DIR = os.path.join(_THIS_DIR, "..")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
EVAL_DIR = os.path.join(BASE_DIR, "data", "eval")
os.makedirs(RESULTS_DIR, exist_ok=True)

with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)
tfidf_matrix = scipy.sparse.load_npz(os.path.join(MODEL_DIR, "tfidf_matrix.npz"))
df = pd.read_pickle(os.path.join(MODEL_DIR, "cases_df.pkl"))

# i. Ekstrak solusi: {case_id: solusi_text}
case_solutions = dict(zip(df["case_id"], df["amar_putusan"]))
case_outcome_label = dict(zip(df["case_id"], df["outcome_label"]))


def preprocess(text):
    return str(text).lower()


def retrieve(query: str, k: int = 5):
    q_vec = vectorizer.transform([preprocess(query)])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_k_idx = np.argsort(sims)[::-1][:k]
    top_k_case_ids = df.iloc[top_k_idx]["case_id"].tolist()
    top_k_scores = sims[top_k_idx].tolist()
    return top_k_case_ids, top_k_scores


def predict_outcome(query: str, k: int = 5, method: str = "majority") -> dict:
    """
    def predict_outcome(query: str) -> str
        top_k = retrieve(query, k=5)
        solutions = [case_solutions[c] for c in top_k]
        # Terapkan voting / weighting -> pilih satu ringkasan
        return predicted_solution

    method:
      - "majority" : Majority vote atas outcome_label dari top-k kasus.
      - "weighted" : Weighted similarity (bobot = skor cosine similarity).
    """
    top_k, scores = retrieve(query, k=k)
    labels = [case_outcome_label[c] for c in top_k]
    solutions = [case_solutions[c] for c in top_k]

    if method == "majority":
        counts = Counter(labels)
        predicted_label, _ = counts.most_common(1)[0]
    elif method == "weighted":
        weighted = {}
        for lbl, score in zip(labels, scores):
            weighted[lbl] = weighted.get(lbl, 0.0) + score
        predicted_label = max(weighted, key=weighted.get)
    else:
        raise ValueError("method harus 'majority' atau 'weighted'")

    # ambil contoh solusi konkret (amar putusan) dari kasus top-1 yang labelnya
    # cocok dengan prediksi, sebagai representasi "solusi" yang dapat dibaca manusia
    predicted_solution_text = next(
        (sol for lbl, sol in zip(labels, solutions) if lbl == predicted_label),
        solutions[0]
    )

    return {
        "predicted_outcome_label": predicted_label,
        "predicted_solution_text": predicted_solution_text,
        "top_k_case_ids": top_k,
        "similarity_scores": [round(s, 4) for s in scores],
        "method": method,
    }


def main():
    # ---- iv. Demo manual: 5 contoh kasus baru ----
    demo_queries = [
        "Terdakwa ditemukan menyimpan sabu-sabu seberat 0.5 gram untuk dikonsumsi sendiri tanpa hak.",
        "Terdakwa kedapatan mengedarkan ganja kering seberat 450 gram kepada beberapa pembeli.",
        "Terdakwa ditangkap membawa pil ekstasi sebanyak 80 butir di sebuah klub malam.",
        "Terdakwa adalah pengguna narkotika yang belum pernah terlibat kasus pengedaran sebelumnya.",
        "Terdakwa terbukti menyimpan ganja seberat 480 gram dengan maksud untuk diedarkan kembali.",
    ]

    rows = []
    for i, q in enumerate(demo_queries, start=1):
        pred_majority = predict_outcome(q, k=5, method="majority")
        pred_weighted = predict_outcome(q, k=5, method="weighted")
        rows.append({
            "query_id": f"DEMO{i:02d}",
            "query_text": q,
            "predicted_solution_majority": pred_majority["predicted_outcome_label"],
            "predicted_solution_weighted": pred_weighted["predicted_outcome_label"],
            "top_5_case_ids": pred_majority["top_k_case_ids"],
            "amar_putusan_acuan": pred_majority["predicted_solution_text"],
        })
        print(f"[DEMO {i}] Query: {q[:60]}...")
        print(f"          -> Majority: {pred_majority['predicted_outcome_label']} | Weighted: {pred_weighted['predicted_outcome_label']}")

    # ---- Jalankan juga atas seluruh queries.json dari Tahap 3 (evaluasi resmi di Tahap 5) ----
    with open(os.path.join(EVAL_DIR, "queries.json"), "r", encoding="utf-8") as f:
        eval_queries = json.load(f)

    for q in eval_queries:
        pred = predict_outcome(q["query_text"], k=5, method="majority")
        rows.append({
            "query_id": q["query_id"],
            "query_text": q["query_text"],
            "predicted_solution_majority": pred["predicted_outcome_label"],
            "predicted_solution_weighted": predict_outcome(q["query_text"], k=5, method="weighted")["predicted_outcome_label"],
            "top_5_case_ids": pred["top_k_case_ids"],
            "amar_putusan_acuan": pred["predicted_solution_text"],
        })

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "predictions.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\n[OK] Hasil prediksi disimpan -> {out_path} ({len(out_df)} baris)")


if __name__ == "__main__":
    main()
