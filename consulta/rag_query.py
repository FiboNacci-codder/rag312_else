import time
from pathlib import Path
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from normalizar_query import procesar_pregunta

BASE_DIR = Path(__file__).parent
SYSTEM_PROMPT_PATH = BASE_DIR / "system_prompt.txt"

# --- Config Qdrant + Harrier (embeddings) ---
VLLM_EMBED_URL = "http://localhost:8001/v1"
EMBED_MODEL = "harrier-embed"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "procedimientos_sielse"
TOP_K = 5

INSTRUCTION_PREFIX = "Instruct: Retrieve relevant passages that answer the query\nQuery: "

# --- Config LLM generativo ---
VLLM_LLM_URL = "http://localhost:8002/v1"
LLM_MODEL = "qwen-generativo"


def build_embedder():
    return OpenAIEmbeddings(
        base_url=VLLM_EMBED_URL,
        api_key="no-necesaria",
        model=EMBED_MODEL,
        check_embedding_ctx_length=False,
    )


def recuperar_contexto(pregunta: str, embedder, client):
    query_con_prefijo = INSTRUCTION_PREFIX + pregunta
    vector_pregunta = embedder.embed_query(query_con_prefijo)

    resultados = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector_pregunta,
        limit=TOP_K,
        with_payload=True,
    ).points

    return resultados

def armar_prompt(pregunta: str, resultados) -> str:
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
        max_tokens=500,
        temperature=0.2,
    )
    return respuesta.choices[0].message.content

def main(pregunta: str):

    embedder = build_embedder()
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    t0 = time.time()
    resultados = recuperar_contexto(pregunta, embedder, client)
    t_retrieval = time.time() - t0

    prompt = armar_prompt(pregunta, resultados)

    t0 = time.time()
    respuesta = generar_respuesta(prompt)
    t_generacion = time.time() - t0

    fuentes_detalle = []
    vistos = set()
    for r in resultados:
        p = r.payload
        clave = (p.get('source'), str(p.get('paginas')))
        if clave not in vistos:
            vistos.add(clave)
            seccion = p.get('seccion')
            seccion_str = f' de la sección "{seccion}"' if seccion else ""
            fuentes_detalle.append(
                f'- "{p.get("source")}" de la página {p.get("paginas")}{seccion_str} '
                f'que se encuentra en la ruta "{p.get("ruta_biblioteca")}"'
            )
    fuentes_str = "\n".join(fuentes_detalle)

    salida = f"""{respuesta}

---
**Fuentes consultadas:**
{fuentes_str}

*Tiempo retrieval: {t_retrieval:.2f}s | Tiempo generación: {t_generacion:.2f}s*"""

    return salida


if __name__ == "__main__":
    import sys
    pregunta = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Pregunta: ")
    print(main(pregunta))