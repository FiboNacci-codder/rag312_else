"""
Genera preguntas para el golden set a partir de los documentos markdown
completos (salida_md/), en vez de chunks individuales (chunks_data.json).

Diferencia clave con generar_golden_set.py:
    - Ese script genera preguntas ANCLADAS a un chunk específico
      (fuente_esperada incluye chunk_index).
    - Este script parte cada .md en SECCIONES por heading (##, ###, etc.)
      y genera preguntas por sección completa. Como una sección puede
      abarcar varios chunks, fuente_esperada solo especifica
      {documento, seccion_contiene} — matching a nivel de documento+sección,
      no de chunk exacto.

Esto sirve para cubrir preguntas que necesitan ver contexto más amplio que
un solo chunk (tablas largas, procedimientos con varios pasos que quedaron
en distintos chunks, etc.).

Uso:
    source ~/rag312/conda/activar.sh
    conda activate rag312
    export RAG_PROJECT_DIR=~/rag312
    cd ~/rag312/tests/eval

    python generar_golden_set_md.py --n-secciones 40 --por-categoria
    python generar_golden_set_md.py --out golden_set_md_raw.json

El output se procesa con el MISMO pipeline que golden_set.json:
    python filtrar_golden_set.py --golden golden_set_md_raw.json --outdir .
    python revisar_golden_set.py --golden golden_set_md_aceptadas.json
    ...
    python consolidar_golden_set.py --archivos golden_set_md_*.json --out golden_set_md_final.json

Y finalmente se puede unir a golden_set.json (concatenando ambos JSON) para
tener un golden set combinado chunk-level + sección-level.
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
from rag312.config import get_golden_set_config, settings

# --- Rutas del proyecto ---
SALIDA_MD_DIR = settings.salida_md_dir / "procedimientos"
BIBLIOTECA_DIR = settings.biblioteca_dir / "procedimientos"

GOLDEN_CONFIG = get_golden_set_config()
LLM_MODEL = GOLDEN_CONFIG.model

MIN_PALABRAS_SECCION = 60
MAX_PALABRAS_SECCION = 1200

# --- Prompt externo, editable sin tocar el código ---
SYSTEM_PROMPT_PATH = Path(
    os.environ.get("GOLDEN_MD_SYSTEM_PROMPT_PATH", Path(__file__).parent / "system_prompt_golden_md.txt")
)

def cargar_system_prompt() -> str:
    return _cargar_system_prompt_shared(SYSTEM_PROMPT_PATH)


def cargar_todas_las_secciones() -> list[dict]:
    return _cargar_todas_las_secciones_shared(SALIDA_MD_DIR, MIN_PALABRAS_SECCION)


def generar_preguntas_llm(client, texto_seccion: str, n_preguntas: int, system_prompt: str) -> list[str]:
    return _generar_preguntas_llm_shared(
        client, LLM_MODEL, texto_seccion, n_preguntas, system_prompt,
        temperature=0.7, max_palabras_truncado=MAX_PALABRAS_SECCION,
    )


def main():
    parser = argparse.ArgumentParser(description="Generar golden set desde salida_md/ (nivel sección)")
    parser.add_argument("--n-secciones", type=int, default=40)
    parser.add_argument("--todas", action="store_true")
    parser.add_argument("--por-categoria", action="store_true")
    parser.add_argument("--preguntas-por-seccion", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=str(Path(__file__).parent / "golden_set_md_raw.json"))
    parser.add_argument("--id-prefix", type=str, default="m")  # ids tipo m0001 para no chocar con los q0001 de chunks
    args = parser.parse_args()

    secciones = cargar_todas_las_secciones()
    print(f"Secciones útiles encontradas en salida_md/: {len(secciones)}")

    if args.todas:
        seleccionadas = secciones
    else:
        seleccionadas = muestrear_secciones(secciones, args.n_secciones, args.por_categoria, args.seed)
    print(f"Secciones seleccionadas para generar preguntas: {len(seleccionadas)}\n")

    client = build_llm_client(GOLDEN_CONFIG)
    system_prompt = cargar_system_prompt()

    golden_set = []
    contador_id = 1
    for i, sec in enumerate(seleccionadas, start=1):
        print(f"[{i}/{len(seleccionadas)}] {sec['source']} | sección: {sec['seccion'][:60]}")
        try:
            preguntas = generar_preguntas_llm(client, sec["texto"], args.preguntas_por_seccion, system_prompt)
        except Exception as e:
            print(f"    ERROR generando preguntas: {e}")
            continue

        for pregunta in preguntas:
            golden_set.append({
                "id": f"{args.id_prefix}{contador_id:04d}",
                "pregunta": pregunta,
                "categoria": sec["categoria"],
                "fuente_esperada": {
                    "documento": sec["source"],
                    "seccion_contiene": sec["seccion"],
                },
                "origen_generacion": "markdown_seccion",  # para diferenciar del origen chunk
                # Texto completo de la sección que originó la pregunta. Se guarda acá
                # (y no solo el heading) porque filtrar_golden_set.py / generar_ground_truth.py
                # necesitan el fragmento de origen y no pueden resolverlo vía chunk_origen_index
                # (este golden set es a nivel sección, no a nivel chunk).
                "texto_seccion": sec["texto"],
                "revisado": False,
            })
            contador_id += 1
            print(f"    -> {pregunta}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(golden_set, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Golden set (markdown) generado: {len(golden_set)} preguntas en {args.out}")
    print("Pasar por el mismo pipeline de filtrado/revisión que golden_set.json")
    print("antes de consolidarlo o unirlo al golden set principal.")


if __name__ == "__main__":
    main()