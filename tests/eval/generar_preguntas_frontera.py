"""
Genera "preguntas frontera": preguntas coherentes y plausibles para el dominio
de SIELSE, pero que NO tienen respuesta real en los documentos indexados —
para probar si el RAG admite que no tiene la información, o si alucina una
respuesta segura pero incorrecta.

Dos técnicas (ver system_prompt_preguntas_frontera.txt para el detalle):
    1. Un detalle real de la sección, distorsionado (plazo/rol/condición
       cambiada) — suena anclada al documento pero pide algo que no está.
    2. Dominio adyacente fuera de alcance (mismo tono de atención al cliente
       de una empresa de servicios, pero de un trámite que SIELSE, eléctrica,
       no cubre).

Se basa en SECCIONES de markdown (salida_md/), igual que
generar_golden_set_md.py — reusa su mismo parseo por heading y muestreo por
categoría, porque una sección da más material real para distorsionar de forma
creíble que un chunk individual.

IMPORTANTE — este archivo NO se pasa por el pipeline del golden set normal:
    - filtrar_golden_set.py rechazaría estas preguntas a propósito (su juez
      evalúa "respondible con el fragmento", que es justo lo contrario de lo
      que buscamos acá).
    - eval_retrieval.py / eval_generation.py tampoco aplican: no hay una
      fuente_esperada real que matchear ni un ground_truth legítimo.
Usar en cambio correr_preguntas_frontera.py para ver qué responde el RAG.

Uso:
    source ~/rag312_else/conda/activar.sh
    conda activate rag312
    export RAG_PROJECT_DIR=~/rag312_else
    cd ~/rag312_else/tests/eval

    python generar_preguntas_frontera.py --n-secciones 20 --por-categoria
    python generar_preguntas_frontera.py --out preguntas_frontera.json

Requiere que vLLM esté corriendo (FRONTERA_LLM_URL/FRONTERA_LLM_MODEL,
default puerto 8002 / qwen35-9b — el generador de producción, no el juez).
No requiere Qdrant.
"""

import argparse
import json
import os
from pathlib import Path

from _shared import (
    cargar_system_prompt as _cargar_system_prompt_shared,
    cargar_todas_las_secciones as _cargar_todas_las_secciones_shared,
    generar_preguntas_llm as _generar_preguntas_llm_shared,
    muestrear_secciones,
)
from rag312.clients import build_llm_client
from rag312.config import get_frontera_config, settings

SALIDA_MD_DIR = settings.salida_md_dir / "procedimientos"

FRONTERA_CONFIG = get_frontera_config()
LLM_MODEL = FRONTERA_CONFIG.model

MIN_PALABRAS_SECCION = 60
MAX_PALABRAS_SECCION = 1200

SYSTEM_PROMPT_PATH = Path(
    os.environ.get("FRONTERA_SYSTEM_PROMPT_PATH", Path(__file__).parent / "system_prompt_preguntas_frontera.txt")
)


def cargar_system_prompt() -> str:
    return _cargar_system_prompt_shared(SYSTEM_PROMPT_PATH)


def cargar_todas_las_secciones() -> list[dict]:
    return _cargar_todas_las_secciones_shared(SALIDA_MD_DIR, MIN_PALABRAS_SECCION)


def generar_preguntas_llm(client, texto_seccion: str, n_preguntas: int, system_prompt: str) -> list[str]:
    return _generar_preguntas_llm_shared(
        client, LLM_MODEL, texto_seccion, n_preguntas, system_prompt,
        temperature=0.8, max_palabras_truncado=MAX_PALABRAS_SECCION,  # un poco más creatividad que el golden set normal
    )


def main():
    parser = argparse.ArgumentParser(description="Generar preguntas frontera (coherentes pero no respondibles) desde salida_md/")
    parser.add_argument("--n-secciones", type=int, default=20)
    parser.add_argument("--todas", action="store_true")
    parser.add_argument("--por-categoria", action="store_true")
    parser.add_argument("--preguntas-por-seccion", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=str(Path(__file__).parent / "preguntas_frontera.json"))
    parser.add_argument("--id-prefix", type=str, default="f")
    args = parser.parse_args()

    secciones = cargar_todas_las_secciones()
    print(f"Secciones útiles encontradas en salida_md/: {len(secciones)}")

    if args.todas:
        seleccionadas = secciones
    else:
        seleccionadas = muestrear_secciones(secciones, args.n_secciones, args.por_categoria, args.seed)
    print(f"Secciones seleccionadas para generar preguntas: {len(seleccionadas)}\n")

    client = build_llm_client(FRONTERA_CONFIG)
    system_prompt = cargar_system_prompt()

    preguntas_frontera = []
    contador_id = 1
    for i, sec in enumerate(seleccionadas, start=1):
        print(f"[{i}/{len(seleccionadas)}] {sec['source']} | sección: {sec['seccion'][:60]}")
        try:
            preguntas = generar_preguntas_llm(client, sec["texto"], args.preguntas_por_seccion, system_prompt)
        except Exception as e:
            print(f"    ERROR generando preguntas: {e}")
            continue

        for pregunta in preguntas:
            preguntas_frontera.append({
                "id": f"{args.id_prefix}{contador_id:04d}",
                "pregunta": pregunta,
                "categoria": sec["categoria"],
                # Nombrado distinto de "fuente_esperada" a propósito: no es una
                # fuente correcta que matchear, solo de dónde salió la
                # distorsión, para depurar/leer con contexto.
                "origen_aproximado": {
                    "documento": sec["source"],
                    "seccion": sec["seccion"],
                },
                "texto_seccion": sec["texto"],
                "revisado": False,
            })
            contador_id += 1
            print(f"    -> {pregunta}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(preguntas_frontera, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Preguntas frontera generadas: {len(preguntas_frontera)} en {args.out}")
    print("Revisar 2-3 a mano: deben sonar coherentes pero NO respondibles con el")
    print("documento de origen, sin caer en preguntas absurdas o sin sentido.")
    print("Para ver qué responde el RAG: python correr_preguntas_frontera.py")


if __name__ == "__main__":
    main()
