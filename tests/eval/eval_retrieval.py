"""
Evaluación de calidad de retrieval para el RAG de Procedimientos SIELSE.

Corre cada pregunta del golden_set.json contra Qdrant (vía las mismas
funciones que usa rag_query1.py) y calcula:

  - Recall@K (dense, sparse, fusión RRF): ¿el documento esperado aparece
    entre los K resultados?
  - MRR (Mean Reciprocal Rank): en qué posición aparece (1/rank), 0 si no aparece.

Por defecto la búsqueda usa la pregunta ya normalizada/reformulada
(normalizar_query.procesar_pregunta), igual que hace rag_query1.main() en
producción. Esto es importante: evaluar con la pregunta cruda del golden
set mediría un pipeline distinto al que corre en producción. Usar
--sin-normalizar para medir retrieval "puro" (útil para depurar si una
falla viene del retrieval en sí o de la reformulación).

Uso:
    # Desde ~/rag312_else/consulta (o ajustando RAG_PROJECT_DIR):
    python eval_retrieval.py
    python eval_retrieval.py --golden golden_set.json --k 7
    python eval_retrieval.py --out resultados_eval.json
    python eval_retrieval.py --sin-normalizar   # retrieval puro, sin LLM normalizador

Requiere que vLLM (8001, y 8003 si se normaliza) y Qdrant (6333) estén
corriendo, y que la colección procedimientos_sielse ya esté indexada.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import _shared
from _shared import asegurar_paths_pipeline

asegurar_paths_pipeline()

from rag312.clients import build_embedder, build_qdrant_client, build_sparse_embedder
from rag312.config import settings

try:
    from rag_query1 import recuperar_contexto
    from normalizar_query import procesar_pregunta
except ImportError as e:
    print(f"ERROR: no se pudo importar el pipeline RAG desde {_shared.CONSULTA_DIR} / {_shared.INGESTA_DIR}")
    print(f"Ajustá RAG_PROJECT_DIR o corré este script desde el entorno conda rag312.\nDetalle: {e}")
    sys.exit(1)

PRODUCTION_TOP_K = settings.top_k


def cargar_golden_set(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        datos = json.load(f)
    if not datos:
        raise ValueError(f"El golden set '{path}' está vacío.")
    return datos


def coincide(payload: dict, fuente_esperada: dict) -> bool:
    """Compara un resultado recuperado contra la fuente esperada del golden set."""
    doc_ok = payload.get("source") == fuente_esperada.get("documento")
    seccion_esperada = fuente_esperada.get("seccion_contiene")
    if seccion_esperada:
        seccion_real = (payload.get("seccion") or "")
        seccion_ok = seccion_esperada.lower() in seccion_real.lower()
        return doc_ok and seccion_ok
    return doc_ok


SNIPPET_LEN = 300  # caracteres de page_content a guardar por resultado, para revisión manual


def evaluar_pregunta(item: dict, resultados_fusion, detalle_scores, k: int) -> dict:
    fuente_esperada = item["fuente_esperada"]

    # --- Fusión RRF (lo que realmente devuelve el sistema en producción) ---
    rank_fusion = None
    for i, r in enumerate(resultados_fusion[:k], start=1):
        if coincide(r.payload, fuente_esperada):
            rank_fusion = i
            break

    # --- Rank dense / sparse puros, usando el detalle que ya calcula rag_query1 ---
    rank_dense = None
    rank_bm25 = None
    for r, d in zip(resultados_fusion, detalle_scores):
        if coincide(r.payload, fuente_esperada):
            if d["rank_dense"] is not None and rank_dense is None:
                rank_dense = d["rank_dense"]
            if d["rank_bm25"] is not None and rank_bm25 is None:
                rank_bm25 = d["rank_bm25"]

    def hit_at_k(rank):
        return rank is not None and rank <= k

    def rr(rank):
        return 1.0 / rank if rank else 0.0

    # --- Contexto para revisión manual: qué trajo realmente el top-k de fusión ---
    resultados_top_k = [
        {
            "rank": i,
            "documento": r.payload.get("source"),
            "seccion": r.payload.get("seccion"),
            "correcto": coincide(r.payload, fuente_esperada),
            "fragmento": (r.payload.get("page_content") or "")[:SNIPPET_LEN],
        }
        for i, r in enumerate(resultados_fusion[:k], start=1)
    ]

    return {
        "id": item["id"],
        "categoria": item.get("categoria"),
        "pregunta": item["pregunta"],
        "fuente_esperada": fuente_esperada,
        "rank_fusion": rank_fusion,
        "rank_dense": rank_dense,
        "rank_bm25": rank_bm25,
        "hit_fusion": hit_at_k(rank_fusion),
        "hit_dense": hit_at_k(rank_dense),
        "hit_bm25": hit_at_k(rank_bm25),
        "rr_fusion": rr(rank_fusion),
        "rr_dense": rr(rank_dense),
        "rr_bm25": rr(rank_bm25),
        "resultados_top_k": resultados_top_k,
    }


def resumen_por_categoria(filas: list[dict]) -> dict:
    por_cat = defaultdict(list)
    for f in filas:
        por_cat[f.get("categoria") or "sin_categoria"].append(f)
    return {
        cat: {
            "n_preguntas": len(items),
            "recall_fusion": sum(i["hit_fusion"] for i in items) / len(items),
            "mrr_fusion": mean(i["rr_fusion"] for i in items),
        }
        for cat, items in sorted(por_cat.items())
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluación de retrieval del RAG SIELSE")
    parser.add_argument("--golden", type=str, default=str(Path(__file__).parent / "golden_set.json"))
    parser.add_argument("--k", type=int, default=7, help="K para Recall@K (default: 7)")
    parser.add_argument("--out", type=str, default=str(Path(__file__).parent / "resultados_eval.json"))
    parser.add_argument(
        "--sin-normalizar",
        action="store_true",
        help="Buscar con la pregunta cruda del golden set, sin pasar por "
             "normalizar_query.procesar_pregunta() (salta el LLM normalizador, puerto 8003). "
             "Por defecto se normaliza, igual que en producción.",
    )
    args = parser.parse_args()

    if args.k > PRODUCTION_TOP_K:
        print(
            f"AVISO: --k={args.k} es mayor que TOP_K={PRODUCTION_TOP_K} configurado en "
            f"rag_query1.py; la fusión RRF solo devuelve {PRODUCTION_TOP_K} resultados, "
            f"así que Recall@{args.k} quedará igual a Recall@{PRODUCTION_TOP_K}."
        )

    golden_set = cargar_golden_set(Path(args.golden))
    print(f"Cargadas {len(golden_set)} preguntas desde {args.golden}")
    print(f"Modo búsqueda: {'CRUDA (sin normalizar)' if args.sin_normalizar else 'NORMALIZADA (igual que producción)'}\n")

    embedder = build_embedder()
    sparse_embedder = build_sparse_embedder()
    client = build_qdrant_client()

    filas = []
    t_inicio = time.time()
    for item in golden_set:
        pregunta = item["pregunta"]
        print(f"[{item['id']}] {pregunta}")
        try:
            if args.sin_normalizar:
                pregunta_busqueda = pregunta
            else:
                pregunta_busqueda = procesar_pregunta(pregunta)["busqueda"]
                if pregunta_busqueda != pregunta:
                    print(f"    -> búsqueda: {pregunta_busqueda}")

            resultados_fusion, detalle_scores = recuperar_contexto(
                pregunta_busqueda, embedder, sparse_embedder, client
            )
            fila = evaluar_pregunta(item, resultados_fusion, detalle_scores, args.k)
        except Exception as e:
            print(f"    ERROR: {e}")
            fila = {
                "id": item["id"], "categoria": item.get("categoria"),
                "pregunta": pregunta, "fuente_esperada": item.get("fuente_esperada"),
                "error": str(e),
                "hit_fusion": False, "hit_dense": False, "hit_bm25": False,
                "rr_fusion": 0.0, "rr_dense": 0.0, "rr_bm25": 0.0,
                "resultados_top_k": [],
            }
        filas.append(fila)
        estado = "OK" if fila.get("hit_fusion") else "MISS"
        print(f"    rank_fusion={fila.get('rank_fusion')} rank_dense={fila.get('rank_dense')} "
              f"rank_bm25={fila.get('rank_bm25')} -> {estado}\n")
    t_total = time.time() - t_inicio

    # --- Métricas agregadas ---
    n = len(filas)
    n_errores = sum(1 for f in filas if "error" in f)
    resumen = {
        "n_preguntas": n,
        "n_errores": n_errores,
        "k": args.k,
        "normalizado": not args.sin_normalizar,
        f"recall_at_{args.k}_fusion": sum(f["hit_fusion"] for f in filas) / n,
        f"recall_at_{args.k}_dense": sum(f["hit_dense"] for f in filas) / n,
        f"recall_at_{args.k}_bm25": sum(f["hit_bm25"] for f in filas) / n,
        "mrr_fusion": mean(f["rr_fusion"] for f in filas),
        "mrr_dense": mean(f["rr_dense"] for f in filas),
        "mrr_bm25": mean(f["rr_bm25"] for f in filas),
        "tiempo_total_seg": round(t_total, 2),
        "tiempo_promedio_por_pregunta_seg": round(t_total / n, 3),
        "por_categoria": resumen_por_categoria(filas),
    }

    print("=" * 60)
    print("RESUMEN")
    print("=" * 60)
    for k, v in resumen.items():
        if k == "por_categoria":
            print(f"{'por_categoria':30s}:")
            for cat, m in v.items():
                print(f"  - {cat:28s} recall={m['recall_fusion']:.2f} mrr={m['mrr_fusion']:.2f} (n={m['n_preguntas']})")
        elif isinstance(v, float):
            print(f"{k:30s}: {v:.3f}")
        else:
            print(f"{k:30s}: {v}")

    if n_errores:
        print(f"\nAVISO: {n_errores} pregunta(s) fallaron con error y cuentan como MISS. Revisá el detalle en el JSON de salida.")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"resumen": resumen, "detalle": filas}, f, ensure_ascii=False, indent=2)
    print(f"\nResultados guardados en: {args.out}")


if __name__ == "__main__":
    main()
