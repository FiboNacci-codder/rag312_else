import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import csv
import sys
import time
import json
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AcceleratorDevice
from docling.chunking import HybridChunker

from langchain_core.documents import Document

from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

# BASE_DIR ahora apunta a la raiz del proyecto (~/rag), no a esta subcarpeta
BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / "biblioteca"
OUTPUT_DIR = BASE_DIR / "salida_md"
CHUNKS_DIR = BASE_DIR / "salida_chunks"
CSV_PATH = BASE_DIR / "datos" / "metricas_ocr.csv"

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


def formatear_tiempo(segundos: float) -> str:
    minutos = int(segundos // 60)
    segs = segundos % 60
    return f"{minutos}m {segs:.1f}s"


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

    converter = build_converter()
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained("microsoft/harrier-oss-v1-0.6b"),
        max_tokens=2048,
    )
    chunker = HybridChunker(tokenizer=tokenizer)

    filas = []
    langchain_docs = []  # aquí se acumulan los Document listos para embeddings/Qdrant

    print(f"Encontrados {len(pdfs)} PDFs. Iniciando procesamiento...\n")

    for i, pdf_path in enumerate(pdfs, start=1):
        print(f"[{i}/{len(pdfs)}] Procesando: {pdf_path.name}")
        t0 = time.time()
        try:
            result = converter.convert(pdf_path)
            elapsed = time.time() - t0

            doc = result.document
            markdown_out = OUTPUT_DIR / pdf_path.relative_to(INPUT_DIR).parent / (pdf_path.stem + ".md")
            markdown_out.parent.mkdir(parents=True, exist_ok=True)
            markdown_out.write_text(doc.export_to_markdown(), encoding="utf-8")

            # --- Chunking + conversión a Document de LangChain ---
            chunks = list(chunker.chunk(doc))
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
                            "ruta_biblioteca": str(pdf_path.relative_to(INPUT_DIR)),
                            #"ruta_completa": str(pdf_path.resolve()),  # path absoluto en el servidor
                        },
                    )
                )

            # --- Guardar chunks de este PDF en su propia carpeta espejo ---
            chunk_file = CHUNKS_DIR / pdf_path.relative_to(INPUT_DIR).parent / (pdf_path.stem + "_chunks.txt")
            chunk_file.parent.mkdir(parents=True, exist_ok=True)
            with open(chunk_file, "w", encoding="utf-8") as f:
                for j, chunk in enumerate(chunks):
                    doc_meta = langchain_docs[-len(chunks) + j].metadata 
                    f.write("=" * 80 + "\n")
                    f.write(f"Chunk: {j} / {len(chunks)}\n")
                    f.write(f"Páginas: {doc_meta.get('paginas')}\n")
                    f.write(f"Sección: {doc_meta.get('seccion')}\n")
                    f.write(f"Tipo: {doc_meta.get('tipo')}\n")
                    f.write("-" * 80 + "\n")
                    f.write(chunk.text + "\n\n")

            num_paginas = len(doc.pages)
            tiempo_str = formatear_tiempo(elapsed)
            print(f"    OK en {tiempo_str} ({num_paginas} pág., {len(chunks)} chunks) -> {markdown_out.name}\n")
            filas.append([pdf_path.name, tiempo_str, num_paginas])

        except Exception as e:
            elapsed = time.time() - t0
            tiempo_str = formatear_tiempo(elapsed)
            print(f"    ERROR tras {tiempo_str}: {e}\n")
            filas.append([pdf_path.name, f"ERROR ({tiempo_str})", "N/A"])

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Nombre del archivo", "Tiempo en minutos y seg", "Páginas"])
        writer.writerows(filas)

    print(f"Listo. Métricas guardadas en: {CSV_PATH}")
    print(f"Total de Document (LangChain) generados: {len(langchain_docs)}")

    chunks_json_path = BASE_DIR / "datos" / "chunks_data.json"
    with open(chunks_json_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"page_content": d.page_content, "metadata": d.metadata} for d in langchain_docs],
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Chunks (JSON) guardados en: {chunks_json_path}")
    return langchain_docs  # listos para pasar a tu embedder


if __name__ == "__main__":
    main()