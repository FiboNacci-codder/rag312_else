"""Variante de ocr_docling.py que usa el modelo VLM granite-docling
(ibm-granite/granite-docling-258M) en vez de PdfPipelineOptions+EasyOCR, para
comparar la calidad de los markdown/chunks resultantes con la pipeline
actual. Escribe en directorios y archivos separados (salida_md_granite/,
salida_chunks_granite/, datos/chunks_data_granite.json,
datos/metricas_ocr_granite.csv) para no mezclarse con los artefactos de
producción. Reusa el resto de la lógica (chunking, guardado, fusión de
chunks_data) directamente desde ocr_docling.py.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag312.config import settings

os.environ["CUDA_VISIBLE_DEVICES"] = settings.cuda_visible_devices_ocr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ocr_docling

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions, AcceleratorOptions, AcceleratorDevice
from docling.datamodel import vlm_model_specs
from docling.pipeline.vlm_pipeline import VlmPipeline

from rag312.utils import formatear_tiempo

INPUT_DIR = settings.biblioteca_dir
OUTPUT_DIR = settings.salida_md_granite_dir
CHUNKS_DIR = settings.salida_chunks_granite_dir
CSV_PATH = settings.datos_dir / "metricas_ocr_granite.csv"
CHUNKS_JSON_PATH = settings.datos_dir / "chunks_data_granite.json"


def build_converter_granite() -> DocumentConverter:
    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS,
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options, pipeline_cls=VlmPipeline)
        }
    )


def parse_args():
    parser = argparse.ArgumentParser(description="OCR con granite-docling, para comparar contra ocr_docling.py")
    parser.add_argument("--carpeta", type=str, default=None, help="Subcarpeta relativa dentro de biblioteca/ a procesar (default: toda biblioteca/)")
    parser.add_argument("--limite", type=int, default=None, help="Procesar solo los primeros N PDFs encontrados")
    parser.add_argument("--pdf", type=str, default=None, help="Ruta a un único PDF a procesar (ignora --carpeta/--limite)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not INPUT_DIR.exists():
        print(f"No existe la carpeta: {INPUT_DIR}")
        sys.exit(1)

    if args.pdf:
        pdfs = [Path(args.pdf)]
    else:
        base_dir = (INPUT_DIR / args.carpeta) if args.carpeta else INPUT_DIR
        if not base_dir.exists():
            print(f"No existe la carpeta: {base_dir}")
            sys.exit(1)
        pdfs = sorted(base_dir.rglob("*.pdf"))
        if args.limite:
            pdfs = pdfs[: args.limite]

    if not pdfs:
        print(f"No se encontraron PDFs para procesar.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    settings.datos_dir.mkdir(exist_ok=True)

    converter = build_converter_granite()
    chunker = ocr_docling.build_chunker()

    filas = []
    langchain_docs = []

    print(f"Encontrados {len(pdfs)} PDFs. Iniciando procesamiento con granite-docling...\n")

    for i, pdf_path in enumerate(pdfs, start=1):
        print(f"[{i}/{len(pdfs)}] Procesando: {pdf_path.name}")
        t0 = time.time()
        try:
            docs_del_pdf = ocr_docling.procesar_pdf(pdf_path, converter, chunker, INPUT_DIR, OUTPUT_DIR, CHUNKS_DIR)
            elapsed = time.time() - t0
            langchain_docs.extend(docs_del_pdf)

            todas_paginas = {p for d in docs_del_pdf for p in d.metadata["paginas"]}
            num_paginas = max(todas_paginas) if todas_paginas else 0
            tiempo_str = formatear_tiempo(elapsed)
            print(f"    OK en {tiempo_str} ({len(docs_del_pdf)} chunks)\n")
            filas.append([pdf_path.name, tiempo_str, num_paginas])

        except Exception as e:
            elapsed = time.time() - t0
            tiempo_str = formatear_tiempo(elapsed)
            print(f"    ERROR tras {tiempo_str}: {e}\n")
            filas.append([pdf_path.name, f"ERROR ({tiempo_str})", "N/A"])

    ocr_docling.guardar_metricas_csv(filas, CSV_PATH)
    print(f"Listo. Métricas guardadas en: {CSV_PATH}")
    print(f"Total de Document (LangChain) generados: {len(langchain_docs)}")

    ocr_docling.fusionar_y_guardar_chunks_json(langchain_docs, CHUNKS_JSON_PATH)
    print(f"Chunks (JSON) guardados en: {CHUNKS_JSON_PATH}")
    return langchain_docs


if __name__ == "__main__":
    main()
