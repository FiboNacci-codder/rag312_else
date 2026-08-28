import sys
import time
import numpy as np
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from normalizar_query import procesar_pregunta

BASE_DIR = Path(__file__).parent          # ~/rag312/consulta
PROJECT_DIR = BASE_DIR.parent              # ~/rag312
sys.path.insert(0, str(PROJECT_DIR / "ingesta"))
from embeddings import build_sparse_embedder

SYSTEM_PROMPT_PATH = BASE_DIR / "system_prompt.txt"

VLLM_EMBED_URL = "http://localhost:8001/v1"
EMBED_MODEL = "harrier-embed"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "procedimientos_sielse"
TOP_K = 10
PREFETCH_LIMIT = 20
UMBRAL_SIMILITUD = 0.5  # aplica solo sobre el score dense crudo (0-1)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"

INSTRUCTION_PREFIX = "Instruct: Retrieve relevant passages that answer the query\nQuery: "

VLLM_LLM_URL = "http://localhost:8002/v1"
LLM_MODEL = "qwen35-9b"

def build_embedder():
    return OpenAIEmbeddings(
        base_url=VLLM_EMBED_URL,
        api_key="no-necesaria",
        model=EMBED_MODEL,
        check_embedding_ctx_length=False,
    )

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    v1, v2 = np.array(v1), np.array(v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)

def recuperar_contexto(pregunta: str, embedder, sparse_embedder, client):
    query_con_prefijo = INSTRUCTION_PREFIX + pregunta
    vector_denso = embedder.embed_query(query_con_prefijo)
    vector_sparse_raw = next(sparse_embedder.embed([pregunta]))
    vector_sparse = SparseVector(
        indices=vector_sparse_raw.indices.tolist(),
        values=vector_sparse_raw.values.tolist(),
    )

    # --- Rama dense sola (top-20, para rank real) ---
    dense_resultados = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector_denso,
        using=DENSE_VECTOR_NAME,
        limit=PREFETCH_LIMIT,
        with_payload=False,
    ).points
    dense_info = {
        r.id: {"rank": i + 1, "score": r.score}
        for i, r in enumerate(dense_resultados)
    }

    # --- Rama sparse sola (top-20, para rank real) ---
    sparse_resultados = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector_sparse,
        using=SPARSE_VECTOR_NAME,
        limit=PREFETCH_LIMIT,
        with_payload=False,
    ).points
    sparse_info = {
        r.id: {"rank": i + 1, "score": r.score}
        for i, r in enumerate(sparse_resultados)
    }

    # --- Query fusionada (RRF) — resultado final ---
    resultados_fusionados = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=vector_denso, using=DENSE_VECTOR_NAME, limit=PREFETCH_LIMIT),
            Prefetch(query=vector_sparse, using=SPARSE_VECTOR_NAME, limit=PREFETCH_LIMIT),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=TOP_K,
        with_payload=True,
    ).points

    # --- Para los que no cayeron en el top-20 de alguna rama, calcular su score real ---
    ids_faltan_dense = [r.id for r in resultados_fusionados if r.id not in dense_info]
    if ids_faltan_dense:
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

    return resultados_fusionados, detalle_scores

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
    client_llm = OpenAI(base_url=VLLM_LLM_URL, api_key="no-necesaria")
    respuesta = client_llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_usuario},
        ],
        max_tokens=800,
        temperature=0.2,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return respuesta.choices[0].message.content


def main(pregunta: str) -> dict:
    proc = procesar_pregunta(pregunta)
    pregunta_original = proc["original"]
    pregunta_busqueda = proc["busqueda"]

    embedder = build_embedder()
    sparse_embedder = build_sparse_embedder()
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    t0 = time.time()
    resultados, detalle_scores = recuperar_contexto(pregunta_busqueda, embedder, sparse_embedder, client)
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
            "chunk": p.get("page_content"),
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