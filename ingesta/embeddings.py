import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import json
import threading
import time

import numpy as np
from langchain_core.documents import Document

from rag312.clients import build_embedder, build_sparse_embedder
from rag312.config import settings
from rag312.utils import formatear_tiempo

CHUNKS_JSON = settings.datos_dir / "chunks_data.json"
METRICS_CSV = settings.datos_dir / "metricas_embeddings.csv"
VECTORS_OUT = settings.datos_dir / "embeddings_data.json"

BATCH_SIZE = settings.batch_size_embeddings

_vectores_json_lock = threading.Lock()


def cargar_chunks() -> list[Document]:
    with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]


def cargar_chunks_por_ruta(ruta_biblioteca: str) -> list[Document]:
    return [d for d in cargar_chunks() if d.metadata.get("ruta_biblioteca") == ruta_biblioteca]


def embed_batch(embedder, sparse_embedder, batch_textos: list[str]):
    batch_vectores = embedder.embed_documents(batch_textos)
    batch_sparse = list(sparse_embedder.embed(batch_textos))
    return batch_vectores, batch_sparse


def guardar_metricas_csv(filas: list[list], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Archivo", "Chunk index", "Caracteres", "Palabras", "Dimensión vector", "Norma", "Términos sparse"])
        writer.writerows(filas)


def construir_registros(docs: list[Document], vectores_totales: list, vectores_sparse_totales: list) -> list[dict]:
    return [
        {
            "page_content": d.page_content,
            "metadata": d.metadata,
            "embedding": vec,
            "sparse_indices": sparse_vec.indices.tolist(),
            "sparse_values": sparse_vec.values.tolist(),
        }
        for d, vec, sparse_vec in zip(docs, vectores_totales, vectores_sparse_totales)
    ]


def fusionar_y_guardar_vectores_json(nuevos_registros: list[dict], path: Path) -> None:
    """Fusiona `nuevos_registros` con el `embeddings_data.json` existente,
    reemplazando cualquier entrada previa que comparta `ruta_biblioteca` con
    alguno de los nuevos (permite re-embeber un archivo actualizado sin
    duplicar sus vectores)."""
    rutas_nuevas = {r["metadata"]["ruta_biblioteca"] for r in nuevos_registros}
    with _vectores_json_lock:
        existentes = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                existentes = json.load(f)
        conservados = [r for r in existentes if r["metadata"].get("ruta_biblioteca") not in rutas_nuevas]
        combinados = conservados + nuevos_registros
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(combinados, f)


def embeber_ruta(ruta_biblioteca: str) -> list[dict]:
    """Embebe (dense + sparse) solo los chunks de `ruta_biblioteca` ya
    presentes en chunks_data.json. No escribe embeddings_data.json: el
    llamador decide cómo persistir (ver fusionar_y_guardar_vectores_json)."""
    docs = cargar_chunks_por_ruta(ruta_biblioteca)
    if not docs:
        raise ValueError(f"No hay chunks para ruta_biblioteca={ruta_biblioteca!r} en {CHUNKS_JSON}")

    embedder = build_embedder()
    sparse_embedder = build_sparse_embedder()

    textos = [d.page_content for d in docs]
    vectores_totales, vectores_sparse_totales = embed_batch(embedder, sparse_embedder, textos)
    return construir_registros(docs, vectores_totales, vectores_sparse_totales)


def imprimir_resumen(docs: list[Document], metricas_filas: list[list], tiempo_total: float) -> None:
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
            batch_vectores, batch_sparse = embed_batch(embedder, sparse_embedder, batch_textos)
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

    guardar_metricas_csv(metricas_filas, METRICS_CSV)
    fusionar_y_guardar_vectores_json(construir_registros(docs, vectores_totales, vectores_sparse_totales), VECTORS_OUT)
    imprimir_resumen(docs, metricas_filas, tiempo_total)

    return docs, vectores_totales, vectores_sparse_totales


if __name__ == "__main__":
    main()
