import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag312.config import settings

os.environ["CUDA_VISIBLE_DEVICES"] = settings.cuda_visible_devices_ocr

import csv
import json
import time

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AcceleratorDevice
from docling.chunking import HybridChunker

from langchain_core.documents import Document

from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

from rag312.utils import formatear_tiempo

INPUT_DIR = settings.biblioteca_dir
OUTPUT_DIR = settings.salida_md_dir
CHUNKS_DIR = settings.salida_chunks_dir
CSV_PATH = settings.datos_dir / "metricas_ocr.csv"
CHUNKS_JSON_PATH = settings.datos_dir / "chunks_data.json"


def build_converter():
    accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.CUDA,
    )

    ocr_options = EasyOcrOptions(
        lang=["es", "en"],
        force_full_page_ocr=False,
    )

    pipeline_options = PdfPipelineOptions(
        accelerator_options=accelerator_options,
        do_ocr=True,
        do_table_structure=True,
        ocr_options=ocr_options,
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def build_chunker() -> HybridChunker:
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained("microsoft/harrier-oss-v1-0.6b"),
        max_tokens=2048,
    )
    return HybridChunker(tokenizer=tokenizer)


def convertir_pdf(pdf_path: Path, converter: DocumentConverter):
    return converter.convert(pdf_path)


def chunkear_documento(doc, chunks, pdf_path: Path, input_dir: Path) -> list[Document]:
    langchain_docs = []
    for j, chunk in enumerate(chunks):
        paginas = sorted({
            prov.page_no
            for item in chunk.meta.doc_items
            for prov in item.prov
        })
        seccion = " > ".join(chunk.meta.headings) if chunk.meta.headings else None

        tipos = sorted({item.label.value for item in chunk.meta.doc_items})
        tipo_principal = tipos[0] if len(tipos) == 1 else "mixto: " + ", ".join(tipos)

        langchain_docs.append(
            Document(
                page_content=chunk.text,
                metadata={
                    "source": pdf_path.name,
                    "chunk_index": j,
                    "num_chunks": len(chunks),
                    "paginas": paginas,
                    "seccion": seccion,
                    "tipo": tipo_principal,
                    "categoria": pdf_path.parent.name,
                    "ruta_biblioteca": str(pdf_path.relative_to(input_dir)),
                    #"ruta_completa": str(pdf_path.resolve()),  # path absoluto en el servidor
                },
            )
        )
    return langchain_docs


def guardar_markdown(doc, pdf_path: Path, input_dir: Path, output_dir: Path) -> Path:
    markdown_out = output_dir / pdf_path.relative_to(input_dir).parent / (pdf_path.stem + ".md")
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text(doc.export_to_markdown(), encoding="utf-8")
    return markdown_out


def guardar_chunks_txt(chunks, docs_del_pdf: list[Document], pdf_path: Path, input_dir: Path, chunks_dir: Path) -> Path:
    chunk_file = chunks_dir / pdf_path.relative_to(input_dir).parent / (pdf_path.stem + "_chunks.txt")
    chunk_file.parent.mkdir(parents=True, exist_ok=True)
    with open(chunk_file, "w", encoding="utf-8") as f:
        for j, chunk in enumerate(chunks):
            doc_meta = docs_del_pdf[j].metadata
            f.write("=" * 80 + "\n")
            f.write(f"Chunk: {j} / {len(chunks)}\n")
            f.write(f"Páginas: {doc_meta.get('paginas')}\n")
            f.write(f"Sección: {doc_meta.get('seccion')}\n")
            f.write(f"Tipo: {doc_meta.get('tipo')}\n")
            f.write("-" * 80 + "\n")
            f.write(chunk.text + "\n\n")
    return chunk_file


def guardar_metricas_csv(filas: list[list], csv_path: Path) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Nombre del archivo", "Tiempo en minutos y seg", "Páginas"])
        writer.writerows(filas)


def guardar_chunks_json(langchain_docs: list[Document], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            [{"page_content": d.page_content, "metadata": d.metadata} for d in langchain_docs],
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():
    if not INPUT_DIR.exists():
        print(f"No existe la carpeta: {INPUT_DIR}")
        sys.exit(1)

    pdfs = sorted(INPUT_DIR.rglob("*.pdf"))
    if not pdfs:
        print(f"No se encontraron PDFs en {INPUT_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    CHUNKS_DIR.mkdir(exist_ok=True)
    settings.datos_dir.mkdir(exist_ok=True)

    converter = build_converter()
    chunker = build_chunker()

    filas = []
    langchain_docs = []  # aquí se acumulan los Document listos para embeddings/Qdrant

    print(f"Encontrados {len(pdfs)} PDFs. Iniciando procesamiento...\n")

    for i, pdf_path in enumerate(pdfs, start=1):
        print(f"[{i}/{len(pdfs)}] Procesando: {pdf_path.name}")
        t0 = time.time()
        try:
            result = convertir_pdf(pdf_path, converter)
            elapsed = time.time() - t0

            doc = result.document
            markdown_out = guardar_markdown(doc, pdf_path, INPUT_DIR, OUTPUT_DIR)

            # --- Chunking + conversión a Document de LangChain ---
            chunks = list(chunker.chunk(doc))
            docs_del_pdf = chunkear_documento(doc, chunks, pdf_path, INPUT_DIR)
            langchain_docs.extend(docs_del_pdf)

            # --- Guardar chunks de este PDF en su propia carpeta espejo ---
            guardar_chunks_txt(chunks, docs_del_pdf, pdf_path, INPUT_DIR, CHUNKS_DIR)

            num_paginas = len(doc.pages)
            tiempo_str = formatear_tiempo(elapsed)
            print(f"    OK en {tiempo_str} ({num_paginas} pág., {len(chunks)} chunks) -> {markdown_out.name}\n")
            filas.append([pdf_path.name, tiempo_str, num_paginas])

        except Exception as e:
            elapsed = time.time() - t0
            tiempo_str = formatear_tiempo(elapsed)
            print(f"    ERROR tras {tiempo_str}: {e}\n")
            filas.append([pdf_path.name, f"ERROR ({tiempo_str})", "N/A"])

    guardar_metricas_csv(filas, CSV_PATH)

    print(f"Listo. Métricas guardadas en: {CSV_PATH}")
    print(f"Total de Document (LangChain) generados: {len(langchain_docs)}")

    guardar_chunks_json(langchain_docs, CHUNKS_JSON_PATH)
    print(f"Chunks (JSON) guardados en: {CHUNKS_JSON_PATH}")
    return langchain_docs  # listos para pasar a tu embedder


if __name__ == "__main__":
    main()
