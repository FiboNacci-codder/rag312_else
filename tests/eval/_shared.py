"""Helpers compartidos por los scripts de tests/eval/.

Módulo local (NO parte del paquete rag312) — cada script eval se ejecuta con
tests/eval/ como sys.path[0], así que esto se importa sin ningún hack, y
mantiene la lógica eval-only fuera del paquete de producción distribuible.

Antes de este módulo, RAG_PROJECT_DIR y varias funciones (cargar_chunks_por_
indice, obtener_contexto, parsear_secciones, ...) estaban copy-pasteadas en
hasta 4 scripts distintos. Se centralizan acá preservando exactamente el
comportamiento de cada variante existente (ver comentarios puntuales).
"""

import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

RAG_PROJECT_DIR = Path(os.environ.get("RAG_PROJECT_DIR", Path.home() / "rag312"))
CONSULTA_DIR = RAG_PROJECT_DIR / "consulta"
INGESTA_DIR = RAG_PROJECT_DIR / "ingesta"
CHUNKS_JSON = RAG_PROJECT_DIR / "datos" / "chunks_data.json"


def asegurar_paths_pipeline() -> None:
    """Agrega consulta/ e ingesta/ a sys.path para poder importar el pipeline real."""
    for p in (CONSULTA_DIR, INGESTA_DIR):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def mapear_chunks_por_indice(data: list[dict]) -> dict:
    """Núcleo puro: (source, chunk_index) -> page_content. Sin I/O."""
    mapa = {}
    for c in data:
        meta = c.get("metadata", {})
        clave = (meta.get("source"), meta.get("chunk_index"))
        mapa[clave] = c.get("page_content", "")
    return mapa


def cargar_chunks_por_indice(chunks_json: Path = CHUNKS_JSON) -> dict:
    """Variante "return {} si falta el archivo" (revisar_golden_set.py / revisar_ground_truth.py)."""
    if not chunks_json.exists():
        return {}
    with open(chunks_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return mapear_chunks_por_indice(data)


def obtener_contexto(item: dict, chunks_map: dict) -> str | None:
    """Fragmento de origen de una pregunta del golden set.

    Los items generados por generar_golden_set_md.py / generar_preguntas_frontera.py
    (nivel sección) traen su propio "texto_seccion" y no tienen chunk_origen_index;
    los generados por generar_golden_set.py (nivel chunk) se resuelven vía
    chunks_data.json.
    """
    if item.get("texto_seccion"):
        return item["texto_seccion"]
    clave = (item.get("fuente_esperada", {}).get("documento"), item.get("chunk_origen_index"))
    return chunks_map.get(clave)


def guardar(data, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def cargar_system_prompt(path: Path) -> str:
    if not path.exists():
        print(f"ERROR: no existe el archivo de prompt {path}")
        sys.exit(1)
    return path.read_text(encoding="utf-8")


def parsear_secciones(md_texto: str, source_pdf: str, categoria: str, min_palabras: int = 60) -> list[dict]:
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
            if len(texto.split()) >= min_palabras:
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


def cargar_todas_las_secciones(salida_md_dir: Path, min_palabras: int = 60) -> list[dict]:
    if not salida_md_dir.exists():
        print(f"ERROR: no existe {salida_md_dir}. Ajustá RAG_PROJECT_DIR.")
        sys.exit(1)

    todas = []
    for md_path in sorted(salida_md_dir.rglob("*.md")):
        categoria = md_path.parent.name
        # Nombre del PDF original: mismo stem, extensión .pdf, tal como lo
        # guarda ocr_docling.py en metadata["source"] (pdf_path.name)
        source_pdf = md_path.stem + ".pdf"
        texto = md_path.read_text(encoding="utf-8")
        todas.extend(parsear_secciones(texto, source_pdf, categoria, min_palabras))
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


def generar_preguntas_llm(
    client,
    model: str,
    texto_seccion: str,
    n_preguntas: int,
    system_prompt: str,
    *,
    temperature: float,
    max_tokens: int = 400,
    max_palabras_truncado: int = 1200,
) -> list[str]:
    texto_truncado = " ".join(texto_seccion.split()[:max_palabras_truncado])
    user_prompt = f"Cantidad de preguntas a generar: {n_preguntas}\n\nSECCIÓN:\n{texto_truncado}"
    respuesta = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    contenido = respuesta.choices[0].message.content or ""
    lineas = [l.strip() for l in contenido.split("\n") if l.strip()]
    # Limpieza: sacar numeración/viñetas tipo "1. ", "- ", "* " si el LLM las agrega igual
    limpias = [re.sub(r"^[\-\*\d\.\)]+\s*", "", l) for l in lineas]
    # Filtrar líneas que no parezcan preguntas reales (muy cortas, o texto de relleno)
    return [l for l in limpias if len(l.split()) >= 3][:n_preguntas]


def agrupar_por_categoria(items: list[dict], categoria_key: str = "categoria") -> dict:
    por_cat = defaultdict(list)
    for item in items:
        por_cat[item.get(categoria_key) or "sin_categoria"].append(item)
    return por_cat
