"""
Genera respuestas de referencia ("ground truth") para el golden_set.json ya
consolidado, usando el LLM ya desplegado en vLLM.

Con ground truth se pueden calcular las métricas de RAGAS que requieren una
respuesta de referencia — context_precision y context_recall (variantes
"WithReference") — además de faithfulness/answer_relevancy que ya no la
necesitan.

Para cada pregunta reconstruye el fragmento de origen (datos/chunks_data.json,
vía chunk_origen_index) y le pide al LLM que responda usando ÚNICAMENTE ese
fragmento. El resultado NO es confiable hasta que se revisa manualmente con
revisar_ground_truth.py — por eso cada item queda marcado con
"ground_truth_revisado": false.

Uso:
    source ~/rag312/conda/activar.sh
    conda activate rag312
    export RAG_PROJECT_DIR=~/rag312
    cd ~/rag312/tests/eval

    python generar_ground_truth.py
    python generar_ground_truth.py --golden golden_set.json --out golden_set.json

Requiere que vLLM esté corriendo (GROUND_TRUTH_LLM_URL/GROUND_TRUTH_LLM_MODEL,
default puerto 8004 / qwen35-9b — instancia dedicada en GPU 7; override por
variable de entorno para volver al 8002 compartido). No requiere Qdrant.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from _shared import CHUNKS_JSON, mapear_chunks_por_indice, obtener_contexto
from rag312.clients import build_llm_client
from rag312.config import get_ground_truth_config

# Mismo modelo (qwen35-9b) que usa eval_generation.py como juez: instancia
# dedicada en GPU 7, aislada del qwen35-9b de producción (puerto 8002).
GROUND_TRUTH_CONFIG = get_ground_truth_config()
LLM_MODEL = GROUND_TRUTH_CONFIG.model

SYSTEM_PROMPT_PATH = Path(
    os.environ.get("GROUND_TRUTH_SYSTEM_PROMPT_PATH", Path(__file__).parent / "system_prompt_ground_truth.txt")
)


def cargar_system_prompt() -> str:
    if not SYSTEM_PROMPT_PATH.exists():
        print(f"ERROR: no existe el archivo de prompt {SYSTEM_PROMPT_PATH}")
        sys.exit(1)
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def cargar_chunks_por_indice() -> dict:
    """Preserva el comportamiento original (distinto de _shared.cargar_chunks_por_indice):
    sys.exit(1) con mensaje propio si falta CHUNKS_JSON."""
    if not CHUNKS_JSON.exists():
        print(f"ERROR: no existe {CHUNKS_JSON}. Ajustá RAG_PROJECT_DIR.")
        sys.exit(1)
    with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return mapear_chunks_por_indice(data)


def generar_ground_truth_llm(client, system_prompt: str, pregunta: str, chunk_texto: str) -> str:
    user_prompt = f"FRAGMENTO:\n{chunk_texto}\n\nPREGUNTA:\n{pregunta}"
    respuesta = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=800,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    contenido = respuesta.choices[0].message.content
    return contenido.strip() if contenido else ""


def main():
    parser = argparse.ArgumentParser(description="Generar ground truth para golden_set.json")
    parser.add_argument("--golden", type=str, default=str(Path(__file__).parent / "golden_set.json"))
    parser.add_argument("--out", type=str, default=None, help="Por defecto sobreescribe --golden")
    parser.add_argument("--auto-aprobar", action="store_true",
                         help="Marca ground_truth_revisado=true directamente, sin pasar por "
                              "revisar_ground_truth.py (se confía 100% en el LLM generador). "
                              "Ahorra la revisión manual, a costa de no tener un chequeo humano "
                              "sobre las respuestas de referencia que usa RAGAS.")
    args = parser.parse_args()
    out_path = Path(args.out) if args.out else Path(args.golden)

    with open(args.golden, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    chunks_map = cargar_chunks_por_indice()
    client = build_llm_client(GROUND_TRUTH_CONFIG)
    system_prompt = cargar_system_prompt()

    print(f"Generando ground truth para {len(golden_set)} preguntas con {LLM_MODEL}...\n")

    n_generadas = 0
    n_sin_chunk = 0
    n_errores = 0

    for i, item in enumerate(golden_set, start=1):
        pregunta = item["pregunta"]
        print(f"[{i}/{len(golden_set)}] {pregunta[:70]}")

        chunk_texto = obtener_contexto(item, chunks_map)

        if not chunk_texto:
            print("    AVISO: no se encontró el fragmento de origen (texto_seccion o chunk), se omite (sin ground_truth)")
            n_sin_chunk += 1
            continue

        try:
            ground_truth = generar_ground_truth_llm(client, system_prompt, pregunta, chunk_texto)
        except Exception as e:
            print(f"    ERROR generando ground truth: {e}")
            n_errores += 1
            continue

        if not ground_truth:
            print("    AVISO: el LLM no devolvió respuesta, se omite (sin ground_truth)")
            n_errores += 1
            continue

        item["ground_truth"] = ground_truth
        item["ground_truth_revisado"] = bool(args.auto_aprobar)
        n_generadas += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(golden_set, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Ground truth generado : {n_generadas}/{len(golden_set)}")
    print(f"Sin chunk resoluble    : {n_sin_chunk}")
    print(f"Errores del LLM        : {n_errores}")
    print(f"\nGuardado en: {out_path}")
    if args.auto_aprobar:
        print("\nAVISO: --auto-aprobar activo, se marcó 'ground_truth_revisado': true sin revisión")
        print("humana. Las respuestas de referencia no fueron chequeadas por una persona.")
    else:
        print("\nIMPORTANTE: revisar manualmente antes de usarlo para evaluación con RAGAS.")
        print("Cada entrada generada tiene 'ground_truth_revisado': false — corré:")
        print("  python revisar_ground_truth.py")


if __name__ == "__main__":
    main()
