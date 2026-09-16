"""Prueba y compara dos formas de chunkear con LangChain el markdown que
produce TeleOCR (ingesta/ocr_teleocr_batch.py), y deja un reporte de métricas
por splitter para poder elegir uno.

Por qué este script existe: `ocr_docling.py` chunkea con
`docling.chunking.HybridChunker`, que opera sobre el árbol de documento
interno de Docling (jerarquía de headings, `doc_items`, páginas por item).
TeleOCR no produce ese árbol — solo vuelca markdown plano en
`salida_md_teleocr/` — así que `HybridChunker` no es reutilizable ahí.
LangChain sí tiene splitters que operan directo sobre texto/markdown, que es
lo que hace falta acá.

Splitters comparados:
- "recursive_simple": `RecursiveCharacterTextSplitter` con separadores
  genéricos (sin conciencia de estructura markdown) — baseline.
- "markdown_header": `MarkdownHeaderTextSplitter` (split jerárquico por
  `#`/`##`/`###`, preserva los headers como metadata → campo `seccion`) con
  una segunda pasada de `RecursiveCharacterTextSplitter` para las secciones
  que superan el presupuesto de tokens del embedder. Es el equivalente más
  cercano al comportamiento actual de `ocr_docling.py` (jerárquico + acotado
  por tokens), aplicado a markdown plano en vez del árbol de Docling.

Ambos usan como `length_function` el mismo tokenizer que ya usa
`HybridChunker` en ocr_docling.py (`microsoft/harrier-oss-v1-0.6b`), para que
los tamaños de chunk sean directamente comparables entre splitters y contra
los chunks que ya existen en `datos/chunks_data.json`.

Limitación conocida: la salida cruda de `infer.py` (TeleOCR) no está
documentada (ver docstring de ocr_teleocr_batch.py) y no hay marcador de
página confirmado en el markdown resultante, así que acá el campo `paginas`
queda en `None` para todos los chunks — no se inventa un valor.

Este script NO es parte del pipeline de producción: no toca
`datos/chunks_data.json`. Por cada splitter escribe un archivo de chunks por
PDF (no un único JSON con todo junto), espejando la jerarquía de
`biblioteca/` — mismo patrón que `salida_chunks/` en ocr_docling.py — bajo
`salida_chunks_teleocr_recursive/` y `salida_chunks_teleocr_markdown_header/`.

Uso típico (desde el env "rag312", después de correr
ocr_teleocr_batch.py):

    python ingesta/probar_chunking_teleocr.py --limite 10   # prueba chica primero
    python ingesta/probar_chunking_teleocr.py                # toda salida_md_teleocr/
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag312.config import settings

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

INPUT_DIR = settings.salida_md_teleocr_dir
DATOS_DIR = settings.datos_dir
REPORT_CSV_PATH = DATOS_DIR / "reporte_chunking_teleocr.csv"
OUT_DIR_RECURSIVE = settings.rag_project_dir / "salida_chunks_teleocr_recursive"
OUT_DIR_MARKDOWN_HEADER = settings.rag_project_dir / "salida_chunks_teleocr_markdown_header"

MAX_TOKENS = 2048  # mismo presupuesto que HybridChunker en ocr_docling.py (harrier-embed)
CHUNK_OVERLAP_TOKENS = 200

HEADERS_TO_SPLIT_ON = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]


def build_tokenizer():
    return AutoTokenizer.from_pretrained("microsoft/harrier-oss-v1-0.6b")


def contar_tokens(tokenizer, texto: str) -> int:
    return len(tokenizer.encode(texto, add_special_tokens=False))


def encontrar_markdowns(base_dir: Path) -> list[Path]:
    return sorted(base_dir.rglob("*.md"))


def metadata_base(md_path: Path, input_dir: Path) -> dict:
    """infer.py (TeleOCR) escribe, por cada PDF, un subdirectorio nombrado con
    el stem del PDF (ver `reorganizar_resultados` en ocr_teleocr_batch.py),
    con el .md adentro junto a `*_layout.pdf`/`*_middle.json`/`images/` — no
    un .md suelto en el lugar del PDF como hace ocr_docling.py. El nombre del
    .md en sí conserva el nombre "codeado" que usa infer.py internamente
    (p.ej. "procedimientos_..._CVA-IN-001_Ingreso_de_reclamos.md"), así que
    la metadata se arma a partir del directorio padre del .md (que representa
    al PDF original), no del nombre del .md."""
    pdf_dir = md_path.parent
    return {
        "source": pdf_dir.name + ".pdf",
        "categoria": pdf_dir.parent.name,
        "ruta_biblioteca": str(pdf_dir.relative_to(input_dir)) + ".pdf",
    }


def chunkear_recursive_simple(texto: str, meta_base: dict, tokenizer) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        length_function=lambda t: contar_tokens(tokenizer, t),
    )
    fragmentos = splitter.split_text(texto)
    return [
        Document(
            page_content=frag,
            metadata={**meta_base, "chunk_index": j, "num_chunks": len(fragmentos), "paginas": None, "seccion": None},
        )
        for j, frag in enumerate(fragmentos)
    ]


def chunkear_markdown_header(texto: str, meta_base: dict, tokenizer) -> list[Document]:
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False)
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        length_function=lambda t: contar_tokens(tokenizer, t),
    )

    fragmentos = []  # (texto, seccion)
    for seccion_doc in md_splitter.split_text(texto):
        seccion = " > ".join(v for v in seccion_doc.metadata.values() if v) or None
        if contar_tokens(tokenizer, seccion_doc.page_content) > MAX_TOKENS:
            fragmentos.extend((sub, seccion) for sub in sub_splitter.split_text(seccion_doc.page_content))
        else:
            fragmentos.append((seccion_doc.page_content, seccion))

    return [
        Document(
            page_content=frag,
            metadata={**meta_base, "chunk_index": j, "num_chunks": len(fragmentos), "paginas": None, "seccion": seccion},
        )
        for j, (frag, seccion) in enumerate(fragmentos)
    ]


def guardar_chunks_pdf(docs: list[Document], pdf_dir: Path, input_dir: Path, output_dir: Path) -> Path:
    """Escribe los chunks de un único PDF en su propio archivo, espejando la
    jerarquía de biblioteca/ — mismo patrón que guardar_chunks_txt() en
    ocr_docling.py, pero en JSON en vez de .txt (no hay texto de debug extra
    que aportar más allá del page_content/metadata ya estructurado)."""
    destino = output_dir / pdf_dir.relative_to(input_dir).parent / (pdf_dir.name + "_chunks.json")
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(
            [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
            f, ensure_ascii=False, indent=2,
        )
    return destino


def calcular_metricas(nombre_splitter: str, docs: list[Document], tokenizer) -> dict:
    longitudes_chars = [len(d.page_content) for d in docs]
    longitudes_tokens = [contar_tokens(tokenizer, d.page_content) for d in docs]
    excedidos = sum(1 for t in longitudes_tokens if t > MAX_TOKENS)
    return {
        "splitter": nombre_splitter,
        "total_chunks": len(docs),
        "chars_min": min(longitudes_chars) if longitudes_chars else 0,
        "chars_prom": round(sum(longitudes_chars) / len(longitudes_chars), 1) if longitudes_chars else 0,
        "chars_max": max(longitudes_chars) if longitudes_chars else 0,
        "tokens_min": min(longitudes_tokens) if longitudes_tokens else 0,
        "tokens_prom": round(sum(longitudes_tokens) / len(longitudes_tokens), 1) if longitudes_tokens else 0,
        "tokens_max": max(longitudes_tokens) if longitudes_tokens else 0,
        "chunks_exceden_presupuesto": excedidos,
    }


def guardar_reporte_csv(filas: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columnas = [
        "splitter", "total_chunks",
        "chars_min", "chars_prom", "chars_max",
        "tokens_min", "tokens_prom", "tokens_max",
        "chunks_exceden_presupuesto",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columnas)
        writer.writeheader()
        writer.writerows(filas)


def imprimir_reporte(filas: list[dict]) -> None:
    print(f"\n{'Splitter':<18} {'Chunks':>7} {'Chars (min/prom/max)':>24} {'Tokens (min/prom/max)':>24} {'>2048 tok':>10}")
    for fila in filas:
        chars = f"{fila['chars_min']}/{fila['chars_prom']}/{fila['chars_max']}"
        tokens = f"{fila['tokens_min']}/{fila['tokens_prom']}/{fila['tokens_max']}"
        print(f"{fila['splitter']:<18} {fila['total_chunks']:>7} {chars:>24} {tokens:>24} {fila['chunks_exceden_presupuesto']:>10}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compara RecursiveCharacterTextSplitter vs MarkdownHeaderTextSplitter+RecursiveCharacterTextSplitter (LangChain) sobre el markdown de salida de TeleOCR"
    )
    parser.add_argument("--carpeta", type=str, default=None, help="Subcarpeta relativa dentro de salida_md_teleocr/ a procesar (default: toda la carpeta)")
    parser.add_argument("--limite", type=int, default=None, help="Procesar solo los primeros N .md encontrados (para probar rápido)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not INPUT_DIR.exists():
        print(f"No existe la carpeta: {INPUT_DIR}")
        print("Corré primero ingesta/ocr_teleocr_batch.py para generarla.")
        sys.exit(1)

    base_dir = (INPUT_DIR / args.carpeta) if args.carpeta else INPUT_DIR
    mds = encontrar_markdowns(base_dir)
    if args.limite:
        mds = mds[: args.limite]
    if not mds:
        print(f"No se encontraron .md en {base_dir}")
        sys.exit(1)

    print(f"Encontrados {len(mds)} archivos .md. Cargando tokenizer (microsoft/harrier-oss-v1-0.6b)...")
    tokenizer = build_tokenizer()

    docs_recursive = []
    docs_markdown_header = []

    for i, md_path in enumerate(mds, start=1):
        print(f"[{i}/{len(mds)}] Procesando: {md_path.relative_to(INPUT_DIR)}")
        texto = md_path.read_text(encoding="utf-8")
        pdf_dir = md_path.parent
        meta_base = metadata_base(md_path, INPUT_DIR)

        docs_pdf_recursive = chunkear_recursive_simple(texto, meta_base, tokenizer)
        docs_pdf_markdown_header = chunkear_markdown_header(texto, meta_base, tokenizer)
        guardar_chunks_pdf(docs_pdf_recursive, pdf_dir, INPUT_DIR, OUT_DIR_RECURSIVE)
        guardar_chunks_pdf(docs_pdf_markdown_header, pdf_dir, INPUT_DIR, OUT_DIR_MARKDOWN_HEADER)

        docs_recursive.extend(docs_pdf_recursive)
        docs_markdown_header.extend(docs_pdf_markdown_header)

    filas = [
        calcular_metricas("recursive_simple", docs_recursive, tokenizer),
        calcular_metricas("markdown_header", docs_markdown_header, tokenizer),
    ]
    guardar_reporte_csv(filas, REPORT_CSV_PATH)
    imprimir_reporte(filas)

    print(f"\nrecursive_simple: {len(docs_recursive)} chunks -> {OUT_DIR_RECURSIVE}")
    print(f"markdown_header: {len(docs_markdown_header)} chunks -> {OUT_DIR_MARKDOWN_HEADER}")
    print(f"Reporte comparativo: {REPORT_CSV_PATH}")
    print("Nota: 'paginas' queda en None para ambos splitters — TeleOCR no expone marcadores de página en su markdown.")


if __name__ == "__main__":
    main()
