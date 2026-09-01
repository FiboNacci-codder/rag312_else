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
    source ~/rag312/conda/activar.sh
    conda activate rag312
    export RAG_PROJECT_DIR=~/rag312
    cd ~/rag312/tests/eval

    python generar_preguntas_frontera.py --n-secciones 20 --por-categoria
    python generar_preguntas_frontera.py --out preguntas_frontera.json

Requiere que vLLM esté corriendo (FRONTERA_LLM_URL/FRONTERA_LLM_MODEL,
default puerto 8002 / qwen35-9b — el generador de producción, no el juez).
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

RAG_PROJECT_DIR = Path(os.environ.get("RAG_PROJECT_DIR", Path.home() / "rag312"))
SALIDA_MD_DIR = RAG_PROJECT_DIR / "salida_md" / "procedimientos"

LLM_URL = os.environ.get("FRONTERA_LLM_URL", "http://localhost:8002/v1")
LLM_MODEL = os.environ.get("FRONTERA_LLM_MODEL", "qwen35-9b")

MIN_PALABRAS_SECCION = 60
MAX_PALABRAS_SECCION = 1200

SYSTEM_PROMPT_PATH = Path(
    os.environ.get("FRONTERA_SYSTEM_PROMPT_PATH", Path(__file__).parent / "system_prompt_preguntas_frontera.txt")
)


def cargar_system_prompt() -> str:
    if not SYSTEM_PROMPT_PATH.exists():
        print(f"ERROR: no existe el archivo de prompt {SYSTEM_PROMPT_PATH}")
        sys.exit(1)
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def parsear_secciones(md_texto: str, source_pdf: str, categoria: str) -> list[dict]:
    """Divide un markdown en secciones usando los headings (#, ##, ###...)."""
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
        temperature=0.8,  # un poco más creatividad que el golden set normal
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    contenido = respuesta.choices[0].message.content or ""
    lineas = [l.strip() for l in contenido.split("\n") if l.strip()]
    limpias = [re.sub(r"^[\-\*\d\.\)]+\s*", "", l) for l in lineas]
    return [l for l in limpias if len(l.split()) >= 3][:n_preguntas]


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

    client = OpenAI(base_url=LLM_URL, api_key="no-necesaria")
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
