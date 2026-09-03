"""
Genera un golden_set.json a partir de los chunks reales del proyecto
(datos/chunks_data.json), usando el LLM ya desplegado en vLLM para crear
preguntas sintéticas en lenguaje coloquial.

Como la pregunta se genera A PARTIR del chunk, la fuente_esperada queda
100% alineada con lo que existe en Qdrant, sin trabajo manual de matching.

Uso:
    source ~/rag312_else/conda/activar.sh
    conda activate rag312
    export RAG_PROJECT_DIR=~/rag312_else   # si el proyecto no está en ~/rag312_else

    # Generar 2 preguntas por chunk, muestreando 60 chunks en total:
    python generar_golden_set.py --n-chunks 60 --preguntas-por-chunk 2

    # Muestrear proporcionalmente por categoría (recomendado):
    python generar_golden_set.py --n-chunks 60 --por-categoria

    # Usar todos los chunks (dataset chico) en vez de muestrear:
    python generar_golden_set.py --todos

    python generar_golden_set.py --n-chunks 60 --por-categoria --preguntas-por-chunk 1

Requiere que vLLM esté corriendo en el puerto configurado (GOLDEN_LLM_URL/GOLDEN_LLM_MODEL).
No requiere Qdrant.
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

from _shared import CHUNKS_JSON, cargar_system_prompt as _cargar_system_prompt_shared
from rag312.clients import build_llm_client
from rag312.config import get_golden_set_config

GOLDEN_CONFIG = get_golden_set_config()
LLM_MODEL = GOLDEN_CONFIG.model

# Prompt externo, editable sin tocar el código
SYSTEM_PROMPT_PATH = Path(
    os.environ.get("GOLDEN_SYSTEM_PROMPT_PATH", Path(__file__).parent / "system_prompt_golden.txt")
)

def cargar_system_prompt() -> str:
    return _cargar_system_prompt_shared(SYSTEM_PROMPT_PATH)


def cargar_chunks() -> list[dict]:
    if not CHUNKS_JSON.exists():
        print(f"ERROR: no existe {CHUNKS_JSON}. Corré primero ocr_docling.py o ajustá RAG_PROJECT_DIR.")
        sys.exit(1)
    with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_es_util(chunk: dict, min_palabras: int = 40) -> bool:
    """Filtra chunks demasiado cortos/ruidosos (tablas sueltas, headers, etc.)."""
    texto = chunk.get("page_content", "")
    return len(texto.split()) >= min_palabras


def muestrear_chunks(chunks: list[dict], n: int, por_categoria: bool, seed: int) -> list[dict]:
    random.seed(seed)
    chunks_utiles = [c for c in chunks if chunk_es_util(c)]

    if not por_categoria:
        n = min(n, len(chunks_utiles))
        return random.sample(chunks_utiles, n)

    # Muestreo proporcional por categoría (subcarpeta de biblioteca/), para
    # que el golden set cubra todas las áreas y no solo las más grandes.
    por_cat = defaultdict(list)
    for c in chunks_utiles:
        cat = c.get("metadata", {}).get("categoria", "sin_categoria")
        por_cat[cat].append(c)

    total = len(chunks_utiles)
    seleccionados = []
    for cat, lista in por_cat.items():
        cuota = max(1, round(n * len(lista) / total))
        cuota = min(cuota, len(lista))
        seleccionados.extend(random.sample(lista, cuota))

    random.shuffle(seleccionados)
    return seleccionados[:n] if len(seleccionados) > n else seleccionados


def generar_preguntas_llm(client: OpenAI, texto_chunk: str, n_preguntas: int, system_prompt: str) -> list[str]:
    user_prompt = f"Cantidad de preguntas a generar: {n_preguntas}\n\nFragmento:\n{texto_chunk}"
    respuesta = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=400,
        temperature=0.7,  # más creatividad/variedad que en rag_query1.py
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    contenido = respuesta.choices[0].message.content or ""
    lineas = [l.strip() for l in contenido.split("\n") if l.strip()]
    # Limpieza: sacar numeración/viñetas tipo "1. ", "- ", "* " si el LLM las agrega igual
    limpias = [re.sub(r"^[\-\*\d\.\)]+\s*", "", l) for l in lineas]
    # Filtrar líneas que no parezcan preguntas reales (muy cortas, o texto de relleno)
    return [l for l in limpias if len(l.split()) >= 3][:n_preguntas]


def main():
    parser = argparse.ArgumentParser(description="Generar golden_set.json desde chunks reales")
    parser.add_argument("--n-chunks", type=int, default=60, help="Cantidad de chunks a muestrear")
    parser.add_argument("--todos", action="store_true", help="Usar todos los chunks útiles (ignora --n-chunks)")
    parser.add_argument("--por-categoria", action="store_true", help="Muestreo proporcional por categoría")
    parser.add_argument("--preguntas-por-chunk", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=str(Path(__file__).parent / "golden_set.json"))
    args = parser.parse_args()

    chunks = cargar_chunks()
    print(f"Chunks totales en {CHUNKS_JSON.name}: {len(chunks)}")

    if args.todos:
        seleccionados = [c for c in chunks if chunk_es_util(c)]
    else:
        seleccionados = muestrear_chunks(chunks, args.n_chunks, args.por_categoria, args.seed)

    print(f"Chunks seleccionados para generar preguntas: {len(seleccionados)}\n")

    client = build_llm_client(GOLDEN_CONFIG)
    system_prompt = cargar_system_prompt()

    golden_set = []
    contador_id = 1
    for i, chunk in enumerate(seleccionados, start=1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "desconocido.pdf")
        print(f"[{i}/{len(seleccionados)}] {source} (chunk_index={meta.get('chunk_index')})")

        try:
            preguntas = generar_preguntas_llm(client, chunk["page_content"], args.preguntas_por_chunk, system_prompt)
        except Exception as e:
            print(f"    ERROR generando preguntas: {e}")
            continue

        for pregunta in preguntas:
            golden_set.append({
                "id": f"q{contador_id:04d}",
                "pregunta": pregunta,
                "categoria": meta.get("categoria"),
                "fuente_esperada": {
                    "documento": source,
                    "seccion_contiene": meta.get("seccion"),
                },
                "chunk_origen_index": meta.get("chunk_index"),
                "revisado": False,
            })
            contador_id += 1
            print(f"    -> {pregunta}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(golden_set, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Golden set generado: {len(golden_set)} preguntas en {args.out}")
    print(f"{'=' * 60}")
    print("IMPORTANTE: revisar manualmente antes de usarlo para evaluación.")
    print("Cada entrada tiene 'revisado': false — marcar true a medida que se")
    print("valida que la pregunta es coherente y que 'fuente_esperada' es correcta.")
    print("Descartar/editar preguntas triviales, ambiguas o mal formadas.")


if __name__ == "__main__":
    main()
