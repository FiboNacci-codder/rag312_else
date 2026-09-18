"""
Prueba ad-hoc del retrieval (chunking) para una pregunta suelta, comparando
las 4 combinaciones de preprocesamiento de query que soporta
normalizar_query.procesar_pregunta():

  - sin_corregir_sin_reformular  : texto tal cual (solo normalizar_basico)
  - solo_ortografia              : corregir_con_llm(texto)
  - solo_reformulacion           : reformular_query(texto)  (sobre el texto crudo)
  - ortografia_y_reformulacion   : reformular_query(corregir_con_llm(texto))
                                    (equivale a rag_query1.main() con sus
                                    defaults de producción: corregir=True,
                                    reformular=True)

No genera respuesta ni toca el LLM de generación: solo muestra qué chunks
trae recuperar_contexto() (fusión RRF dense+sparse) para cada variante, y
opcionalmente el reordenamiento del reranker.

Uso:
    python probar_chunking.py "pregunta suelta"
    python probar_chunking.py
    python probar_chunking.py "..." --top-k 5
    python probar_chunking.py "..." --modo-retrieval dense
    python probar_chunking.py "..." --rerank
    python probar_chunking.py "..." --out resultado.json

Requiere vLLM (8001 embed, 8003 normalize, y el reranker si se usa
--rerank) y Qdrant (6333) corriendo.
"""

import argparse
import json
import sys

from rag312.clients import build_embedder, build_qdrant_client, build_rerank_client, build_sparse_embedder
from rag312.config import settings

from normalizar_query import normalizar_basico, corregir_con_llm, reformular_query
from rag_query1 import recuperar_contexto, rerankear_resultados


def calcular_variantes(pregunta: str) -> list[tuple[str, str]]:
    base = normalizar_basico(pregunta)
    corregida = corregir_con_llm(base)
    reformulada_sin_corregir = reformular_query(base)
    reformulada_con_corregir = reformular_query(corregida)

    return [
        ("sin_corregir_sin_reformular", base),
        ("solo_ortografia", corregida),
        ("solo_reformulacion", reformulada_sin_corregir),
        ("ortografia_y_reformulacion", reformulada_con_corregir),
    ]


def _linea_resultado(i: int, r, d: dict) -> str:
    p = r.payload
    return (
        f"  {i}. {p.get('source')} | sección: {p.get('seccion')} "
        f"| rank_dense={d['rank_dense']} rank_bm25={d['rank_bm25']} score_fusion={d['score_fusion']:.4f}"
    )


def _linea_rerank(i: int, r, d: dict) -> str:
    p = r.payload
    return f"  {i}. {p.get('source')} | sección: {p.get('seccion')} | score_rerank={d.get('score_rerank')}"


def fuentes_a_dict(resultados, detalle_scores) -> list[dict]:
    fuentes = []
    for r, d in zip(resultados, detalle_scores):
        p = r.payload
        fuentes.append({
            "documento": p.get("source"),
            "pagina": p.get("paginas"),
            "seccion": p.get("seccion"),
            "chunk": p.get("page_content"),
            "rank_fusion": d["rank_fusion"],
            "score_fusion": d["score_fusion"],
            "rank_dense": d["rank_dense"],
            "similitud_dense": d["similitud_dense"],
            "rank_bm25": d["rank_bm25"],
            "score_bm25": d["score_bm25"],
            "rank_rerank": d.get("rank_rerank"),
            "score_rerank": d.get("score_rerank"),
        })
    return fuentes


def main():
    parser = argparse.ArgumentParser(description="Prueba ad-hoc de retrieval con variantes de normalización de query")
    parser.add_argument("pregunta", nargs="*", help="Pregunta a probar (si se omite, se pide por input())")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--modo-retrieval", choices=["hybrid", "dense", "sparse"], default="hybrid")
    parser.add_argument("--umbral-similitud", type=float, default=None)
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--rerank", action="store_true", help="También mostrar el orden tras el reranker (Qwen3-Reranker)")
    parser.add_argument("--rerank-top-n", type=int, default=None)
    parser.add_argument("--out", type=str, default=None, help="Ruta de un JSON donde volcar el detalle completo por variante")
    args = parser.parse_args()

    pregunta = " ".join(args.pregunta) if args.pregunta else input("Pregunta de prueba: ")

    print("Calculando variantes de la query (llamadas a corrección/reformulación)...")
    variantes = calcular_variantes(pregunta)

    textos = [texto for _, texto in variantes]
    if len(set(textos)) < len(textos):
        contador: dict[str, list[str]] = {}
        for nombre, texto in variantes:
            contador.setdefault(texto, []).append(nombre)
        for texto, nombres in contador.items():
            if len(nombres) > 1:
                print(f"[Nota] Variantes idénticas ({', '.join(nombres)}): \"{texto}\"")

    embedder = build_embedder()
    sparse_embedder = build_sparse_embedder()
    client = build_qdrant_client()
    rerank_client = build_rerank_client() if args.rerank else None

    salida_json: dict = {"pregunta_original": pregunta, "variantes": {}}

    for nombre, texto_busqueda in variantes:
        print(f"\n{'=' * 20} {nombre} {'=' * 20}")
        print(f'Texto de búsqueda: "{texto_busqueda}"')

        resultados, detalle_scores = recuperar_contexto(
            texto_busqueda, embedder, sparse_embedder, client,
            modo=args.modo_retrieval, top_k=args.top_k,
            umbral_similitud=args.umbral_similitud, collection_name=args.collection,
        )

        top_k = settings.top_k if args.top_k is None else args.top_k
        print(f"Top-{top_k} ({args.modo_retrieval}):")
        for i, (r, d) in enumerate(zip(resultados, detalle_scores), start=1):
            print(_linea_resultado(i, r, d))

        if args.rerank:
            resultados_rerank, detalle_rerank = rerankear_resultados(
                texto_busqueda, resultados, detalle_scores, rerank_client, top_n=args.rerank_top_n,
            )
            print("Tras reranking:")
            for i, (r, d) in enumerate(zip(resultados_rerank, detalle_rerank), start=1):
                print(_linea_rerank(i, r, d))
            resultados, detalle_scores = resultados_rerank, detalle_rerank

        salida_json["variantes"][nombre] = {
            "texto_busqueda": texto_busqueda,
            "fuentes": fuentes_a_dict(resultados, detalle_scores),
        }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(salida_json, f, ensure_ascii=False, indent=2)
        print(f"\nDetalle completo guardado en: {args.out}")


if __name__ == "__main__":
    main()
