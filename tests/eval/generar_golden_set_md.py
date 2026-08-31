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
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

# --- Rutas del proyecto (esto es lo que faltaba) ---
RAG_PROJECT_DIR = Path(os.environ.get("RAG_PROJECT_DIR", Path.home() / "rag312"))
SALIDA_MD_DIR = RAG_PROJECT_DIR / "salida_md" / "procedimientos"
BIBLIOTECA_DIR = RAG_PROJECT_DIR / "biblioteca" / "procedimientos"

LLM_URL = os.environ.get("GOLDEN_LLM_URL", "http://localhost:8002/v1")
LLM_MODEL = os.environ.get("GOLDEN_LLM_MODEL", "qwen35-9b")

MIN_PALABRAS_SECCION = 60
MAX_PALABRAS_SECCION = 1200

# --- Prompt externo, editable sin tocar el código ---
SYSTEM_PROMPT_PATH = Path(
    os.environ.get("GOLDEN_MD_SYSTEM_PROMPT_PATH", Path(__file__).parent / "system_prompt_golden_md.txt")
)

def cargar_system_prompt() -> str:
    if not SYSTEM_PROMPT_PATH.exists():
        print(f"ERROR: no existe el archivo de prompt {SYSTEM_PROMPT_PATH}")
        sys.exit(1)
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

def parsear_secciones(md_texto: str, source_pdf: str, categoria: str) -> list[dict]:
    """
    Divide un markdown en secciones usando los headings (#, ##, ###...).
    Cada sección incluye su heading + todo el texto hasta el próximo heading
    del mismo nivel o superior.
    """
    lineas = md_texto.split("\n")
    secciones = []
    heading_actual = None
    buffer = []

    def cerrar_seccion():
        if buffer:
            texto = "\n".join(buffer).strip()
            if len(texto.split()) >= MIN_PALABRAS_SECCION:
                secciones.append({
                    "seccion": heading_actual or "(sin título)",
                    "texto": texto,
                    "source": source_pdf,
                    "categoria": categoria,
                })

    for linea in lineas:
        m = re.match(r"^(#{1,4})\s+(.*)", linea)
        if m:
            cerrar_seccion()
            heading_actual = m.group(2).strip()
            buffer = []
        else:
            buffer.append(linea)
    cerrar_seccion()
    return secciones


def cargar_todas_las_secciones() -> list[dict]:
    if not SALIDA_MD_DIR.exists():
        print(f"ERROR: no existe {SALIDA_MD_DIR}. Ajustá RAG_PROJECT_DIR.")
        sys.exit(1)

    todas = []
    for md_path in sorted(SALIDA_MD_DIR.rglob("*.md")):
        categoria = md_path.parent.name
        # Nombre del PDF original: mismo stem, extensión .pdf, tal como lo
        # guarda ocr_docling.py en metadata["source"] (pdf_path.name)
        source_pdf = md_path.stem + ".pdf"
        texto = md_path.read_text(encoding="utf-8")
        todas.extend(parsear_secciones(texto, source_pdf, categoria))
    return todas


def muestrear_secciones(secciones: list[dict], n: int, por_categoria: bool, seed: int) -> list[dict]:
    random.seed(seed)
    if not por_categoria:
        n = min(n, len(secciones))
        return random.sample(secciones, n)

    por_cat = defaultdict(list)
    for s in secciones:
        por_cat[s["categoria"]].append(s)

    total = len(secciones)
    seleccionadas = []
    for cat, lista in por_cat.items():
        cuota = max(1, round(n * len(lista) / total))
        cuota = min(cuota, len(lista))
        seleccionadas.extend(random.sample(lista, cuota))

    random.shuffle(seleccionadas)
    return seleccionadas[:n] if len(seleccionadas) > n else seleccionadas


def generar_preguntas_llm(client: OpenAI, texto_seccion: str, n_preguntas: int, system_prompt: str) -> list[str]:
    texto_truncado = " ".join(texto_seccion.split()[:MAX_PALABRAS_SECCION])
    user_prompt = f"Cantidad de preguntas a generar: {n_preguntas}\n\nSECCIÓN:\n{texto_truncado}"
    respuesta = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=400,
        temperature=0.7,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    contenido = respuesta.choices[0].message.content or ""
    lineas = [l.strip() for l in contenido.split("\n") if l.strip()]
    limpias = [re.sub(r"^[\-\*\d\.\)]+\s*", "", l) for l in lineas]
    return [l for l in limpias if len(l.split()) >= 3][:n_preguntas]


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

    client = OpenAI(base_url=LLM_URL, api_key="no-necesaria")
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