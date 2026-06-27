#!/usr/bin/env python
# coding: utf-8
"""
03_retrieval.py  —  TAHAP 3: Case Retrieval
=============================================
Tujuan: Temukan kasus lama yang paling mirip dengan query kasus baru.

Pendekatan yang dipilih (sesuai opsi pada SubCPMK-3):
  - Representasi vektor : TF-IDF (sklearn TfidfVectorizer)
  - Model retrieval     : Support Vector Machine (SVM, kernel linear) pada
                           representasi TF-IDF untuk classification/retrieval
                           terhadap label outcome_label (sebagai proxy "jenis solusi").
  - Splitting data      : 80:20 (train:test), stratified.
  - Fungsi retrieve(query, k) -> top-k case_id berdasarkan cosine similarity.

Output:
  - models/tfidf_vectorizer.pkl, models/svm_model.pkl
  - data/eval/queries.json  (5-10 query uji + ground-truth case_id)
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, classification_report

_THIS_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
BASE_DIR = os.path.join(_THIS_DIR, "..")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
EVAL_DIR = os.path.join(BASE_DIR, "data", "eval")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

CASES_CSV = os.path.join(PROC_DIR, "cases.csv")

RANDOM_STATE = 42


def preprocess(text: str) -> str:
    """Pre-process sederhana: lower-case (stopword removal Bahasa Indonesia
    disengaja diminimalkan karena istilah hukum seperti 'tanpa hak' penting
    sebagai sinyal retrieval)."""
    return str(text).lower()


def load_cases():
    df = pd.read_csv(CASES_CSV)
    df["text_clean"] = df["text_full"].apply(preprocess)
    return df


def build_tfidf(df):
    vectorizer = TfidfVectorizer(
        max_features=3000, ngram_range=(1, 2), min_df=1, sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(df["text_clean"])
    return vectorizer, tfidf_matrix


def train_svm(tfidf_matrix, labels):
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        tfidf_matrix, labels, df_index_placeholder(labels),
        test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    model = SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    return model, idx_train, idx_test, acc, report


def df_index_placeholder(labels):
    return list(range(len(labels)))


# ---------------------------------------------------------------------------
# FUNGSI RETRIEVAL UTAMA (sesuai spesifikasi tugas)
# ---------------------------------------------------------------------------
def retrieve(query: str, vectorizer: TfidfVectorizer, tfidf_matrix, df: pd.DataFrame, k: int = 5):
    """
    def retrieve(query: str, k: int = 5) -> List[case_id]
      1) Pre-process query
      2) Hitung vektor query
      3) Hitung cosine-similarity dengan semua case vectors
      4) Kembalikan top-k case_id
    """
    q_clean = preprocess(query)
    q_vec = vectorizer.transform([q_clean])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_k_idx = np.argsort(sims)[::-1][:k]
    top_k_case_ids = df.iloc[top_k_idx]["case_id"].tolist()
    top_k_scores = sims[top_k_idx].tolist()
    return top_k_case_ids, top_k_scores


def buat_query_uji(df: pd.DataFrame, n_query=8, seed=RANDOM_STATE):
    """Siapkan query uji + ground-truth case_id (kasus itu sendiri = paling mirip
    dengan dirinya / kasus dgn outcome_label sama dianggap relevan)."""
    rng = np.random.RandomState(seed)
    sample = df.sample(n=n_query, random_state=seed)
    queries = []
    for _, row in sample.iterrows():
        # query = ringkasan fakta saja (mensimulasikan kasus baru yang fakta-nya mirip)
        query_text = row["ringkasan_fakta"][:400]
        relevan = df[df["outcome_label"] == row["outcome_label"]]["case_id"].tolist()
        queries.append({
            "query_id": f"Q{int(row['case_id']):03d}",
            "query_text": query_text,
            "ground_truth_case_id": int(row["case_id"]),
            "ground_truth_relevant_case_ids": relevan,
            "ground_truth_outcome_label": row["outcome_label"],
        })
    return queries


def main():
    df = load_cases()
    vectorizer, tfidf_matrix = build_tfidf(df)
    labels = df["outcome_label"].values

    model, idx_train, idx_test, acc, report = train_svm(tfidf_matrix, labels)
    print(f"[OK] SVM trained. Test accuracy (80:20 split): {acc:.3f}")
    print(report)

    with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODEL_DIR, "svm_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    df.to_pickle(os.path.join(MODEL_DIR, "cases_df.pkl"))
    import scipy.sparse
    scipy.sparse.save_npz(os.path.join(MODEL_DIR, "tfidf_matrix.npz"), tfidf_matrix)

    # ---- Demo pemanggilan retrieve() ----
    contoh_query = ("Terdakwa ditemukan menyimpan sabu-sabu seberat 2 gram tanpa hak "
                     "saat dilakukan penggeledahan oleh petugas Kepolisian")
    top_ids, scores = retrieve(contoh_query, vectorizer, tfidf_matrix, df, k=5)
    print(f"\n[DEMO] retrieve() untuk query contoh -> top-5 case_id: {top_ids}")
    print(f"[DEMO] skor cosine similarity        : {[round(s,3) for s in scores]}")

    # ---- Siapkan & simpan query uji ----
    queries = buat_query_uji(df, n_query=8)
    with open(os.path.join(EVAL_DIR, "queries.json"), "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] {len(queries)} query uji disimpan -> {os.path.join(EVAL_DIR, 'queries.json')}")


if __name__ == "__main__":
    main()
