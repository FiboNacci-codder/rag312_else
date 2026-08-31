"""
Filtro automático (LLM-as-judge) para el golden_set.json generado por
generar_golden_set.py.

Evalúa cada pregunta con el chunk que la originó según 3 criterios:
  - autocontenida     : ¿tiene sentido sin contexto adicional?
  - respondible_con_chunk : ¿el chunk contiene realmente la respuesta?
  - suena_natural     : ¿suena a pregunta real de cliente, no a resumen del documento?

Con eso arma un veredicto por pregunta (aceptada / dudosa / rechazada) y separa
el golden set en 3 archivos de salida para que la revisión manual se concentre
solo donde hace falta.

Uso:
    source ~/rag312/conda/activar.sh
    conda activate rag312
    export RAG_PROJECT_DIR=~/rag312
    cd ~/rag312/tests/eval

    python filtrar_golden_set.py
    python filtrar_golden_set.py --golden golden_set.json --muestra-aceptadas 0.2

Salidas:
    golden_set_aceptadas.json   -> pasaron el filtro (revisar solo una muestra)
    golden_set_dudosas.json     -> el juez tuvo dudas (revisar el 100%)
    golden_set_rechazadas.json  -> no pasaron ningún criterio (revisar el 100%, por si el juez se equivocó)
    filtrado_reporte.json       -> veredicto completo de las 3 dimensiones por pregunta

Requiere que vLLM esté corriendo (LLM_URL/LLM_MODEL). No requiere Qdrant.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

from openai import OpenAI

RAG_PROJECT_DIR = Path(os.environ.get("RAG_PROJECT_DIR", Path.home() / "rag312"))
CHUNKS_JSON = RAG_PROJECT_DIR / "datos" / "chunks_data.json"

# Usamos el 4B (más rápido/barato) para el juez por defecto; se puede
# apuntar al 9B si se prefiere más criterio a costa de velocidad.
LLM_URL = os.environ.get("JUDGE_LLM_URL", "http://localhost:8003/v1")
LLM_MODEL = os.environ.get("JUDGE_LLM_MODEL", "qwen35-4b")

SYSTEM_PROMPT = """Eres un evaluador de calidad de preguntas para un dataset de prueba de un \
sistema de búsqueda (RAG) sobre procedimientos institucionales de una empresa de servicios (SIELSE).

Te dan un FRAGMENTO de un procedimiento y una PREGUNTA que supuestamente un cliente haría \
sobre ese fragmento. Evalúa la pregunta según 3 criterios, cada uno true/false:

1. "autocontenida": la pregunta se entiende por sí sola, sin necesitar contexto adicional \
(no usa referencias vagas tipo "eso", "lo anterior", ni está incompleta o cortada).

2. "respondible_con_chunk": el FRAGMENTO dado contiene efectivamente la información para \
responder la pregunta (no es una pregunta sobre un tema relacionado pero no cubierto ahí).

3. "suena_natural": la pregunta suena a como realmente preguntaría un cliente común, en \
lenguaje coloquial — NO suena a un resumen o paráfrasis técnica del documento, ni repite \
frases textuales del procedimiento de forma forzada.

Responde ÚNICAMENTE con un JSON válido, sin texto adicional, con este formato exacto:
{"autocontenida": true, "respondible_con_chunk": true, "suena_natural": false}"""


def cargar_chunks_por_indice() -> dict:
    with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapa = {}
    for c in data:
        meta = c.get("metadata", {})
        clave = (meta.get("source"), meta.get("chunk_index"))
        mapa[clave] = c.get("page_content", "")
    return mapa


def obtener_contexto(item: dict, chunks_map: dict) -> str | None:
    """Fragmento de origen para juzgar la pregunta.

    Los items generados por generar_golden_set_md.py (nivel sección) traen su
    propio "texto_seccion" y no tienen chunk_origen_index; los generados por
    generar_golden_set.py (nivel chunk) se resuelven vía chunks_data.json.
    """
    if item.get("texto_seccion"):
        return item["texto_seccion"]
    clave = (item["fuente_esperada"].get("documento"), item.get("chunk_origen_index"))
    return chunks_map.get(clave)


def evaluar_pregunta(client: OpenAI, pregunta: str, chunk_texto: str) -> dict:
    user_prompt = f"FRAGMENTO:\n{chunk_texto[:1500]}\n\nPREGUNTA:\n{pregunta}"
    respuesta = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=100,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    contenido = (respuesta.choices[0].message.content or "").strip()
    # Limpieza básica por si el modelo envuelve el JSON en ```json ... ```
    if contenido.startswith("```"):
        contenido = contenido.strip("`")
        contenido = contenido.replace("json", "", 1).strip()
    try:
        veredicto = json.loads(contenido)
        return {
            "autocontenida": bool(veredicto.get("autocontenida", False)),
            "respondible_con_chunk": bool(veredicto.get("respondible_con_chunk", False)),
            "suena_natural": bool(veredicto.get("suena_natural", False)),
            "error": None,
        }
    except (json.JSONDecodeError, AttributeError) as e:
        return {
            "autocontenida": False,
            "respondible_con_chunk": False,
            "suena_natural": False,
            "error": f"No se pudo parsear respuesta del juez: {contenido!r} ({e})",
        }


def clasificar(veredicto: dict) -> str:
    """
    aceptada  : pasa los 3 criterios
    rechazada : falla el criterio crítico (respondible_con_chunk) o falla 2+ criterios
    dudosa    : falla exactamente 1 criterio no crítico
    """
    if veredicto.get("error"):
        return "dudosa"  # error del juez -> requiere ojo humano, no descartar directo

    criterios = ["autocontenida", "respondible_con_chunk", "suena_natural"]
    fallos = [c for c in criterios if not veredicto.get(c, False)]

    if not fallos:
        return "aceptada"
    if "respondible_con_chunk" in fallos or len(fallos) >= 2:
        return "rechazada"
    return "dudosa"


def main():
    parser = argparse.ArgumentParser(description="Filtro LLM-as-judge para golden_set.json")
    parser.add_argument("--golden", type=str, default=str(Path(__file__).parent / "golden_set.json"))
    parser.add_argument("--outdir", type=str, default=str(Path(__file__).parent))
    parser.add_argument("--prefijo", type=str, default="golden_set",
                         help="Prefijo de los archivos de salida (aceptadas/dudosas/rechazadas). "
                              "Usar uno distinto (p.ej. golden_set_md) para no pisar los del otro pipeline.")
    parser.add_argument("--muestra-aceptadas", type=float, default=0.2,
                         help="Fracción de las aceptadas a marcar para revisión manual de control (0-1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    prefijo = args.prefijo
    nombre_aceptadas = f"{prefijo}_aceptadas.json"
    nombre_dudosas = f"{prefijo}_dudosas.json"
    nombre_rechazadas = f"{prefijo}_rechazadas.json"
    nombre_reporte = "filtrado_reporte.json" if prefijo == "golden_set" else f"filtrado_reporte_{prefijo}.json"

    with open(args.golden, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    chunks_map = cargar_chunks_por_indice()
    client = OpenAI(base_url=LLM_URL, api_key="no-necesaria")

    print(f"Evaluando {len(golden_set)} preguntas con el juez ({LLM_MODEL})...\n")

    reporte = []
    aceptadas, dudosas, rechazadas = [], [], []

    for i, item in enumerate(golden_set, start=1):
        chunk_texto = obtener_contexto(item, chunks_map)

        if not chunk_texto:
            veredicto = {"autocontenida": False, "respondible_con_chunk": False,
                         "suena_natural": False, "error": "No se encontró el fragmento de origen (texto_seccion o chunk)"}
        else:
            veredicto = evaluar_pregunta(client, item["pregunta"], chunk_texto)

        estado = clasificar(veredicto)
        item["veredicto_juez"] = veredicto
        item["estado_filtro"] = estado

        reporte.append({
            "id": item["id"], "pregunta": item["pregunta"],
            "estado": estado, **veredicto,
        })

        if estado == "aceptada":
            aceptadas.append(item)
        elif estado == "dudosa":
            dudosas.append(item)
        else:
            rechazadas.append(item)

        print(f"[{i}/{len(golden_set)}] {estado.upper():10s} | {item['pregunta'][:70]}")

    # --- Marcar una submuestra de las aceptadas para control manual ---
    # (con --muestra-aceptadas 0 no se fuerza ninguna revisión manual: se
    # confía 100% en el filtro automático para las aceptadas)
    random.seed(args.seed)
    if args.muestra_aceptadas <= 0 or not aceptadas:
        n_muestra = 0
    else:
        n_muestra = max(1, round(len(aceptadas) * args.muestra_aceptadas))
    muestra_control = set(random.sample(range(len(aceptadas)), n_muestra)) if aceptadas else set()
    for idx in muestra_control:
        aceptadas[idx]["revisado"] = False  # forzar que pase por revisión manual de control
    for idx, item in enumerate(aceptadas):
        if idx not in muestra_control:
            item["revisado"] = True  # se confía en el filtro automático

    # --- Guardar salidas ---
    with open(outdir / nombre_aceptadas, "w", encoding="utf-8") as f:
        json.dump(aceptadas, f, ensure_ascii=False, indent=2)
    with open(outdir / nombre_dudosas, "w", encoding="utf-8") as f:
        json.dump(dudosas, f, ensure_ascii=False, indent=2)
    with open(outdir / nombre_rechazadas, "w", encoding="utf-8") as f:
        json.dump(rechazadas, f, ensure_ascii=False, indent=2)
    with open(outdir / nombre_reporte, "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("RESUMEN DEL FILTRO")
    print("=" * 60)
    print(f"Total evaluadas   : {len(golden_set)}")
    print(f"Aceptadas         : {len(aceptadas)}  (muestra para revisión manual: {n_muestra})")
    print(f"Dudosas           : {len(dudosas)}  (revisar el 100%)")
    print(f"Rechazadas        : {len(rechazadas)}  (revisar el 100%, por si el juez se equivocó)")
    print(f"\nArchivos generados en {outdir}:")
    print(f"  {nombre_aceptadas}")
    print(f"  {nombre_dudosas}")
    print(f"  {nombre_rechazadas}")
    print(f"  {nombre_reporte}")
    print("\nSiguiente paso sugerido:")
    print(f"  python revisar_golden_set.py --golden {nombre_dudosas}")
    print(f"  python revisar_golden_set.py --golden {nombre_rechazadas}")
    print(f"  python revisar_golden_set.py --golden {nombre_aceptadas} --solo-pendientes")


if __name__ == "__main__":
    main()