import json
import time
import csv
import os
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from fastembed import SparseTextEmbedding

BASE_DIR = Path(__file__).parent          # ~/rag312/ingesta
PROJECT_DIR = BASE_DIR.parent              # ~/rag312

CHUNKS_JSON = PROJECT_DIR / "datos" / "chunks_data.json"
METRICS_CSV = PROJECT_DIR / "datos" / "metricas_embeddings.csv"
VECTORS_OUT = PROJECT_DIR / "datos" / "embeddings_data.json"

VLLM_BASE_URL = "http://localhost:8001/v1"
MODEL_NAME = "harrier-embed"
BATCH_SIZE = 16  # cuántos chunks se envían juntos al servidor en cada llamada

# --- Config sparse (BM25) ---
SPARSE_MODEL_NAME = "Qdrant/bm25"
SPARSE_LANGUAGE = "spanish"


def build_embedder():
    return OpenAIEmbeddings(
        base_url=VLLM_BASE_URL,
        api_key="no-necesaria",
        model=MODEL_NAME,
        check_embedding_ctx_length=False,
    )


def build_sparse_embedder():
    return SparseTextEmbedding(
        model_name=SPARSE_MODEL_NAME,
        cache_dir=os.environ.get("FASTEMBED_CACHE_DIR", str(Path.home() / "rag/modelos/fastembed_cache")),
        language=SPARSE_LANGUAGE,
    )


def cargar_chunks() -> list[Document]:
    with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]


def formatear_tiempo(segundos: float) -> str:
    minutos = int(segundos // 60)
    segs = segundos % 60
    return f"{minutos}m {segs:.1f}s"


def main():
    docs = cargar_chunks()
    print(f"Cargados {len(docs)} chunks desde {CHUNKS_JSON}\n")

    embedder = build_embedder()
    sparse_embedder = build_sparse_embedder()

    textos = [d.page_content for d in docs]
    metricas_filas = []
    vectores_totales = []
    vectores_sparse_totales = []

    t_inicio_global = time.time()

    for i in range(0, len(textos), BATCH_SIZE):
        batch_textos = textos[i:i + BATCH_SIZE]
        batch_docs = docs[i:i + BATCH_SIZE]

        t0 = time.time()
        try:
            batch_vectores = embedder.embed_documents(batch_textos)
            batch_sparse = list(sparse_embedder.embed(batch_textos))
            elapsed = time.time() - t0
            vectores_totales.extend(batch_vectores)
            vectores_sparse_totales.extend(batch_sparse)

            for doc, vec, sparse_vec in zip(batch_docs, batch_vectores, batch_sparse):
                metricas_filas.append([
                    doc.metadata.get("source"),
                    doc.metadata.get("chunk_index"),
                    len(doc.page_content),          # caracteres
                    len(doc.page_content.split()),  # palabras aprox.
                    len(vec),                        # dimensión del vector denso
                    round(float(np.linalg.norm(vec)), 4),  # norma (sanity check)
                    len(sparse_vec.indices),         # términos no-cero en sparse
                ])

            print(f"Batch {i // BATCH_SIZE + 1}/{(len(textos) - 1) // BATCH_SIZE + 1} "
                  f"({len(batch_textos)} chunks) en {formatear_tiempo(elapsed)}")

        except Exception as e:
            print(f"ERROR en batch {i // BATCH_SIZE + 1}: {e}")
            for doc in batch_docs:
                metricas_filas.append([
                    doc.metadata.get("source"),
                    doc.metadata.get("chunk_index"),
                    len(doc.page_content),
                    len(doc.page_content.split()),
                    "ERROR", "ERROR", "ERROR",
                ])

    tiempo_total = time.time() - t_inicio_global

    # --- Guardar métricas en CSV ---
    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Archivo", "Chunk index", "Caracteres", "Palabras", "Dimensión vector", "Norma", "Términos sparse"])
        writer.writerows(metricas_filas)

    # --- Guardar vectores (dense + sparse) + metadata para el siguiente paso (Qdrant) ---
    with open(VECTORS_OUT, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "page_content": d.page_content,
                    "metadata": d.metadata,
                    "embedding": vec,
                    "sparse_indices": sparse_vec.indices.tolist(),
                    "sparse_values": sparse_vec.values.tolist(),
                }
                for d, vec, sparse_vec in zip(docs, vectores_totales, vectores_sparse_totales)
            ],
            f,
        )

    # --- Resumen en consola ---
    ok = sum(1 for fila in metricas_filas if fila[4] != "ERROR")
    errores = len(metricas_filas) - ok
    dim = metricas_filas[0][4] if metricas_filas and metricas_filas[0][4] != "ERROR" else "N/A"

    print("\n" + "=" * 50)
    print("RESUMEN")
    print("=" * 50)
    print(f"Total chunks procesados : {len(docs)}")
    print(f"Exitosos                : {ok}")
    print(f"Errores                 : {errores}")
    print(f"Dimensión del vector    : {dim}")
    print(f"Tiempo total            : {formatear_tiempo(tiempo_total)}")
    print(f"Promedio por chunk      : {tiempo_total / max(len(docs), 1):.3f}s")
    print(f"Chunks/segundo          : {len(docs) / tiempo_total:.2f}")
    print(f"\nMétricas guardadas en  : {METRICS_CSV}")
    print(f"Vectores guardados en  : {VECTORS_OUT}")

    return docs, vectores_totales, vectores_sparse_totales


if __name__ == "__main__":
    main()