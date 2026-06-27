#!/usr/bin/env python
# coding: utf-8
"""
05_evaluation.py  —  TAHAP 5: Model Evaluation
=================================================
Tujuan: Ukur dan analisis performa retrieval & prediksi.

Langkah kerja:
  i.   Evaluasi Retrieval  -> Accuracy, Precision, Recall, F1-score
       (def eval_retrieval(queries, ground_truth, k))
  ii.  Visualisasi & Laporan -> tabel metrik per model, bar chart, error analysis
  iii. Output -> data/eval/retrieval_metrics.csv, data/eval/prediction_metrics.csv
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
BASE_DIR = os.path.join(_THIS_DIR, "..")
MODEL_DIR = os.path.join(BASE_DIR, "models")
EVAL_DIR = os.path.join(BASE_DIR, "data", "eval")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
os.makedirs(EVAL_DIR, exist_ok=True)

with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)
tfidf_matrix = scipy.sparse.load_npz(os.path.join(MODEL_DIR, "tfidf_matrix.npz"))
df = pd.read_pickle(os.path.join(MODEL_DIR, "cases_df.pkl"))
case_outcome_label = dict(zip(df["case_id"], df["outcome_label"]))


def preprocess(text):
    return str(text).lower()


def retrieve(query: str, k: int = 5):
    q_vec = vectorizer.transform([preprocess(query)])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_k_idx = np.argsort(sims)[::-1][:k]
    return df.iloc[top_k_idx]["case_id"].tolist(), sims[top_k_idx].tolist()


def eval_retrieval(queries, k=5):
    """
    def eval_retrieval(queries, ground_truth, k):
        # loop setiap query -> hitung metrics

    Untuk setiap query, hasil retrieve() dibandingkan terhadap himpunan
    ground_truth_relevant_case_ids (kasus lain dengan outcome_label yang sama).
    Sebuah case dianggap "true positive" jika ia masuk top-k DAN berada dalam
    himpunan relevan ground-truth.
    """
    per_query_rows = []
    y_true_all, y_pred_all = [], []  # untuk akurasi label tunggal (top-1 match)

    for q in queries:
        top_k, scores = retrieve(q["query_text"], k=k)
        relevan_set = set(q["ground_truth_relevant_case_ids"])

        tp = len(set(top_k) & relevan_set)
        precision = tp / k
        recall = tp / len(relevan_set) if relevan_set else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        top1_label = case_outcome_label[top_k[0]]
        true_label = q["ground_truth_outcome_label"]
        y_true_all.append(true_label)
        y_pred_all.append(top1_label)

        per_query_rows.append({
            "query_id": q["query_id"],
            "k": k,
            "precision_at_k": round(precision, 3),
            "recall_at_k": round(recall, 3),
            "f1_at_k": round(f1, 3),
            "top1_label_match": int(top1_label == true_label),
            "retrieved_case_ids": top_k,
        })

    overall_accuracy = accuracy_score(y_true_all, y_pred_all)
    overall = {
        "mean_precision_at_k": round(np.mean([r["precision_at_k"] for r in per_query_rows]), 3),
        "mean_recall_at_k": round(np.mean([r["recall_at_k"] for r in per_query_rows]), 3),
        "mean_f1_at_k": round(np.mean([r["f1_at_k"] for r in per_query_rows]), 3),
        "top1_label_accuracy": round(overall_accuracy, 3),
    }
    return pd.DataFrame(per_query_rows), overall


def main():
    with open(os.path.join(EVAL_DIR, "queries.json"), "r", encoding="utf-8") as f:
        queries = json.load(f)

    per_query_df, overall = eval_retrieval(queries, k=5)
    per_query_df.to_csv(os.path.join(EVAL_DIR, "retrieval_metrics.csv"), index=False)

    print("[OK] Metrik retrieval per-query:")
    print(per_query_df[["query_id", "precision_at_k", "recall_at_k", "f1_at_k", "top1_label_match"]])
    print("\n[OK] Ringkasan metrik retrieval keseluruhan:")
    for k_, v_ in overall.items():
        print(f"     {k_}: {v_}")

    # ---- Evaluasi prediksi (Tahap 4 hasil) ----
    pred_path = os.path.join(RESULTS_DIR, "predictions.csv")
    pred_df = pd.read_csv(pred_path)
    # cocokkan hanya baris yang berasal dari queries.json (punya ground truth)
    qid_to_truth = {q["query_id"]: q["ground_truth_outcome_label"] for q in queries}
    matched = pred_df[pred_df["query_id"].isin(qid_to_truth.keys())].copy()
    matched["ground_truth_outcome_label"] = matched["query_id"].map(qid_to_truth)

    y_true = matched["ground_truth_outcome_label"]
    y_pred_majority = matched["predicted_solution_majority"]
    y_pred_weighted = matched["predicted_solution_weighted"]

    metrics_rows = []
    for nama, y_pred in [("Majority Vote", y_pred_majority), ("Weighted Similarity", y_pred_weighted)]:
        metrics_rows.append({
            "metode_prediksi": nama,
            "accuracy": round(accuracy_score(y_true, y_pred), 3),
            "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 3),
            "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 3),
            "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 3),
        })
    pred_metrics_df = pd.DataFrame(metrics_rows)
    pred_metrics_df.to_csv(os.path.join(EVAL_DIR, "prediction_metrics.csv"), index=False)

    print("\n[OK] Metrik prediksi (Case Solution Reuse):")
    print(pred_metrics_df)

    # ---- Analisis kegagalan (error analysis) ----
    gagal = matched[matched["predicted_solution_majority"] != matched["ground_truth_outcome_label"]]
    print(f"\n[ANALISIS KEGAGALAN] {len(gagal)} dari {len(matched)} query (majority vote) salah diprediksi.")
    if len(gagal) > 0:
        print(gagal[["query_id", "predicted_solution_majority", "ground_truth_outcome_label"]].to_string(index=False))
        print("\nKemungkinan penyebab: (1) jumlah dokumen per kelas outcome masih kecil (data sampel,")
        print("bukan dataset skala penuh dari MA RI), (2) TF-IDF kurang menangkap nuansa 'berat barang")
        print("bukti' secara numerik karena angka tidak dibobotkan khusus, (3) kelas minoritas")
        print("(Rehabilitasi) rawan salah klasifikasi karena jumlah contoh sedikit (class imbalance).")
        print("Rekomendasi perbaikan: tambah volume data riil dari Direktori Putusan MA RI (≥100 dok/")
        print("kelas), tambahkan fitur numerik eksplisit (berat barang bukti), atau gunakan IndoBERT")
        print("embedding untuk menangkap makna semantik yang lebih kaya daripada TF-IDF n-gram.")

    # ---- Visualisasi bar chart ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(["Precision@5", "Recall@5", "F1@5", "Top-1 Acc"],
                [overall["mean_precision_at_k"], overall["mean_recall_at_k"],
                 overall["mean_f1_at_k"], overall["top1_label_accuracy"]],
                color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Performa Retrieval (TF-IDF + Cosine Similarity)")
    axes[0].set_ylabel("Skor")

    x = np.arange(len(pred_metrics_df))
    width = 0.2
    metrik_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    for i, m in enumerate(metrik_names):
        axes[1].bar(x + i * width, pred_metrics_df[m], width, label=m)
    axes[1].set_xticks(x + width * 1.5)
    axes[1].set_xticklabels(pred_metrics_df["metode_prediksi"], rotation=10)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Perbandingan Metode Prediksi Solusi")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    chart_path = os.path.join(EVAL_DIR, "evaluation_chart.png")
    plt.savefig(chart_path, dpi=150)
    print(f"\n[OK] Chart evaluasi disimpan -> {chart_path}")


if __name__ == "__main__":
    main()
