import sys
import time
from pathlib import Path

import numpy as np
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag312.clients import build_embedder, build_llm_client, build_qdrant_client, build_sparse_embedder
from rag312.config import get_generate_config, settings

from normalizar_query import procesar_pregunta

BASE_DIR = Path(__file__).parent
SYSTEM_PROMPT_PATH = BASE_DIR / "system_prompt.txt"

COLLECTION_NAME = settings.collection_name
TOP_K = settings.top_k
PREFETCH_LIMIT = settings.prefetch_limit
UMBRAL_SIMILITUD = settings.umbral_similitud  # aplica solo sobre el score dense crudo (0-1)
DENSE_VECTOR_NAME = settings.dense_vector_name
SPARSE_VECTOR_NAME = settings.sparse_vector_name
INSTRUCTION_PREFIX = settings.instruction_prefix


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    v1, v2 = np.array(v1), np.array(v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def _embed_query(pregunta: str, embedder, sparse_embedder):
    query_con_prefijo = INSTRUCTION_PREFIX + pregunta
    vector_denso = embedder.embed_query(query_con_prefijo)
    vector_sparse_raw = next(sparse_embedder.embed([pregunta]))
    vector_sparse = SparseVector(
        indices=vector_sparse_raw.indices.tolist(),
        values=vector_sparse_raw.values.tolist(),
    )
    return vector_denso, vector_sparse


def _buscar_dense(client, vector_denso, limit: int) -> dict:
    """Rama dense sola (top-20, para rank real)."""
    resultados = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector_denso,
        using=DENSE_VECTOR_NAME,
        limit=limit,
        with_payload=False,
    ).points
    return {r.id: {"rank": i + 1, "score": r.score} for i, r in enumerate(resultados)}


def _buscar_sparse(client, vector_sparse, limit: int) -> dict:
    """Rama sparse sola (top-20, para rank real)."""
    resultados = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector_sparse,
        using=SPARSE_VECTOR_NAME,
        limit=limit,
        with_payload=False,
    ).points
    return {r.id: {"rank": i + 1, "score": r.score} for i, r in enumerate(resultados)}


def _buscar_fusionado(client, vector_denso, vector_sparse, prefetch_limit: int, top_k: int):
    """Query fusionada (RRF) — resultado final."""
    return client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=vector_denso, using=DENSE_VECTOR_NAME, limit=prefetch_limit),
            Prefetch(query=vector_sparse, using=SPARSE_VECTOR_NAME, limit=prefetch_limit),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    ).points


def _buscar_rama_simple(client, vector, using: str, limit: int):
    """Resultado directo de una sola rama (sin fusión), con payload — para modo='dense'/'sparse'."""
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        using=using,
        limit=limit,
        with_payload=True,
    ).points


def _completar_scores_dense_faltantes(client, resultados_fusionados, dense_info: dict, vector_denso) -> dict:
    """Para los que no cayeron en el top-20 de la rama dense, calcular su score real."""
    ids_faltan_dense = [r.id for r in resultados_fusionados if r.id not in dense_info]
    if not ids_faltan_dense:
        return dense_info

    puntos_dense = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=ids_faltan_dense,
        with_vectors=[DENSE_VECTOR_NAME],
    )
    for p in puntos_dense:
        vec_doc = p.vector.get(DENSE_VECTOR_NAME) if isinstance(p.vector, dict) else None
        if vec_doc:
            sim = cosine_similarity(vector_denso, vec_doc)
            dense_info[p.id] = {"rank": None, "score": sim}  # sin rank real (fuera del top-20)
    return dense_info


def _construir_detalle_scores(resultados_fusionados, dense_info: dict, sparse_info: dict) -> list[dict]:
    detalle_scores = []
    for rank_fusion, r in enumerate(resultados_fusionados, start=1):
        d = dense_info.get(r.id)
        s = sparse_info.get(r.id)
        detalle_scores.append({
            "id": r.id,
            "rank_fusion": rank_fusion,
            "score_fusion": r.score,
            "rank_dense": d["rank"] if d else None,
            "similitud_dense": round(d["score"] * 100, 1) if d else None,
            "fuera_top20_dense": d is not None and d["rank"] is None,
            "rank_bm25": s["rank"] if s else None,
            "score_bm25": round(s["score"], 4) if s else None,
        })
    return detalle_scores


def recuperar_contexto(
    pregunta: str,
    embedder,
    sparse_embedder,
    client,
    modo: str = "hybrid",
    top_k: int | None = None,
    umbral_similitud: float | None = None,
):
    top_k = TOP_K if top_k is None else top_k
    umbral_similitud = UMBRAL_SIMILITUD if umbral_similitud is None else umbral_similitud

    vector_denso, vector_sparse = _embed_query(pregunta, embedder, sparse_embedder)

    dense_info = _buscar_dense(client, vector_denso, PREFETCH_LIMIT)
    sparse_info = _buscar_sparse(client, vector_sparse, PREFETCH_LIMIT)

    if modo == "dense":
        resultados = _buscar_rama_simple(client, vector_denso, DENSE_VECTOR_NAME, top_k)
    elif modo == "sparse":
        resultados = _buscar_rama_simple(client, vector_sparse, SPARSE_VECTOR_NAME, top_k)
    else:
        resultados = _buscar_fusionado(client, vector_denso, vector_sparse, PREFETCH_LIMIT, top_k)

    dense_info = _completar_scores_dense_faltantes(client, resultados, dense_info, vector_denso)
    detalle_scores = _construir_detalle_scores(resultados, dense_info, sparse_info)

    if umbral_similitud > 0:
        pares = [
            (r, d) for r, d in zip(resultados, detalle_scores)
            if d["similitud_dense"] is not None and d["similitud_dense"] / 100 >= umbral_similitud
        ]
        resultados = [p[0] for p in pares]
        detalle_scores = [p[1] for p in pares]

    return resultados, detalle_scores


SYSTEM_PROMPT = """Eres un asistente que se llama Foquito que responde preguntas sobre procedimientos institucionales de SIELSE.
Responde usando ÚNICAMENTE la información del contexto proporcionado en cada consulta.
Si el contexto no contiene la respuesta, dilo explícitamente, no inventes información.
Cita la fuente (archivo y página) al final de tu respuesta."""


def armar_prompt(pregunta: str, resultados) -> str:
    if not resultados:
        contexto = "[No se encontró contexto relevante en los documentos para esta pregunta.]"
    else:
        contexto = ""
        for i, r in enumerate(resultados, start=1):
            p = r.payload
            contexto += f"\n[Fuente {i}: {p.get('source')}, página {p.get('paginas')}, sección: {p.get('seccion')}]\n"
            contexto += f"{p.get('page_content')}\n"

    prompt_usuario = f"""CONTEXTO:
{contexto}

PREGUNTA: {pregunta}"""
    return prompt_usuario


def cargar_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def generar_respuesta(prompt_usuario: str) -> str:
    system_prompt = cargar_system_prompt()
    role_config = get_generate_config()
    client_llm = build_llm_client(role_config)
    respuesta = client_llm.chat.completions.create(
        model=role_config.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_usuario},
        ],
        max_tokens=800,
        temperature=0.2,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return respuesta.choices[0].message.content


_embedder = None
_sparse_embedder = None
_qdrant_client = None


def _get_clients():
    """Clientes (vLLM embeddings, BM25 sparse, Qdrant) cacheados a nivel de
    módulo: reconstruirlos en cada llamada a main() era el costo pagado por
    cada request de app_web.py."""
    global _embedder, _sparse_embedder, _qdrant_client
    if _embedder is None:
        _embedder = build_embedder()
    if _sparse_embedder is None:
        _sparse_embedder = build_sparse_embedder()
    if _qdrant_client is None:
        _qdrant_client = build_qdrant_client()
    return _embedder, _sparse_embedder, _qdrant_client


def main(
    pregunta: str,
    corregir: bool = True,
    reformular: bool = True,
    modo_retrieval: str = "hybrid",
    top_k: int | None = None,
    umbral_similitud: float | None = None,
) -> dict:
    proc = procesar_pregunta(pregunta, corregir=corregir, reformular=reformular)
    pregunta_original = proc["original"]
    pregunta_busqueda = proc["busqueda"]

    embedder, sparse_embedder, client = _get_clients()

    t0 = time.time()
    resultados, detalle_scores = recuperar_contexto(
        pregunta_busqueda, embedder, sparse_embedder, client,
        modo=modo_retrieval, top_k=top_k, umbral_similitud=umbral_similitud,
    )
    t_retrieval = time.time() - t0

    prompt = armar_prompt(pregunta_original, resultados)

    t0 = time.time()
    respuesta = generar_respuesta(prompt)
    t_generacion = time.time() - t0

    # --- Armar lista de fuentes con desglose de rank/similitud ---
    fuentes = []
    for r, d in zip(resultados, detalle_scores):
        p = r.payload
        fuentes.append({
            "documento": p.get("source"),
            "pagina": p.get("paginas"),
            "seccion": p.get("seccion"),
            "ruta": p.get("ruta_biblioteca"),
            "categoria": p.get("categoria"),
            "tipo": p.get("tipo"),
            "chunk_index": p.get("chunk_index"),
            "num_chunks": p.get("num_chunks"),
            "chunk": p.get("page_content"),
            "chunk_id": d["id"],
            "rank_fusion": d["rank_fusion"],
            "score_fusion": d["score_fusion"],
            "rank_dense": d["rank_dense"],
            "similitud_dense": d["similitud_dense"],
            "fuera_top20_dense": d["fuera_top20_dense"],
            "rank_bm25": d["rank_bm25"],
            "score_bm25": d["score_bm25"],
        })

    return {
        "respuesta": respuesta,
        "fuentes": fuentes,
        "tiempo_retrieval": t_retrieval,
        "tiempo_generacion": t_generacion,
        "config": {
            "corregir": corregir,
            "reformular": reformular,
            "modo_retrieval": modo_retrieval,
            "top_k": TOP_K if top_k is None else top_k,
            "umbral_similitud": UMBRAL_SIMILITUD if umbral_similitud is None else umbral_similitud,
        },
    }


if __name__ == "__main__":
    import sys as _sys
    import json as _json
    pregunta = " ".join(_sys.argv[1:]) if len(_sys.argv) > 1 else input("Pregunta: ")
    resultado = main(pregunta)
    print(resultado["respuesta"])
    print("\n--- Fuentes ---")
    print(_json.dumps(resultado["fuentes"], ensure_ascii=False, indent=2))
    print(f"\nTiempo retrieval: {resultado['tiempo_retrieval']:.2f}s | generación: {resultado['tiempo_generacion']:.2f}s")
