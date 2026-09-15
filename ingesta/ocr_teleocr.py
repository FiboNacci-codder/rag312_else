"""Test de OCR con el VLM TeleOCR (StarDoc-AI/TeleOCR, HuggingFace) para comparar
tiempo de procesamiento vs. cantidad de páginas contra ocr_docling.py (EasyOCR) y
ocr_docling_granite.py (granite-docling).

TeleOCR no tiene integración con Docling (no está en docling.datamodel.vlm_model_specs)
y se carga directo con transformers (trust_remote_code=True), recibiendo una imagen de
página por llamada — por eso este script rasteriza cada PDF con pypdfium2 en vez de
pasar por DocumentConverter/VlmPipeline como hace la variante granite.

Solo genera markdown + métricas de tiempo/páginas (datos/metricas_ocr_teleocr.csv,
salida_md_teleocr/) para evaluar el candidato. No chunkea ni toca chunks_data.json
ni Qdrant — eso es responsabilidad de ocr_docling.py si TeleOCR llegara a adoptarse.
"""

import argparse
import os
import sys
import time
from pathlib import Path

_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--gpu", type=str, default=None)
_pre_args, _ = _pre_parser.parse_known_args()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag312.config import settings

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocr_docling import guardar_metricas_csv  # OJO: este import fija CUDA_VISIBLE_DEVICES al default (settings.cuda_visible_devices_ocr) — lo pisamos abajo con --gpu

os.environ["CUDA_VISIBLE_DEVICES"] = _pre_args.gpu or settings.cuda_visible_devices_ocr

import pypdfium2 as pdfium
import torch
from transformers import AutoModel, AutoProcessor

from rag312.utils import formatear_tiempo

INPUT_DIR = settings.biblioteca_dir
OUTPUT_DIR = settings.salida_md_teleocr_dir

MODEL_ID = "StarDoc-AI/TeleOCR"
PROMPT = "Convert this document page to markdown, preserving structure, tables and formulas."
RENDER_SCALE = 200 / 72  # ~200 DPI


def cargar_modelo():
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16)
        .cuda()
        .eval()
    )
    return processor, model


def renderizar_paginas(pdf_path: Path):
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for page in pdf:
            yield page.render(scale=RENDER_SCALE).to_pil()
    finally:
        pdf.close()


def ocr_pagina(imagen, processor, model) -> str:
    mensajes = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]},
    ]
    prompt = processor.apply_chat_template(mensajes, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, images=imagen, return_tensors="pt").to(model.device, model.dtype)
    with torch.no_grad():
        salida = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
    texto = processor.batch_decode(
        salida[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )[0]
    return texto.strip()


def procesar_pdf_teleocr(pdf_path: Path, processor, model, input_dir: Path, output_dir: Path) -> int:
    """Rasteriza y OCR-ea un PDF página por página, concatena el markdown y lo
    guarda. Devuelve la cantidad de páginas procesadas."""
    paginas_md = []
    num_paginas = 0
    for num_paginas, imagen in enumerate(renderizar_paginas(pdf_path), start=1):
        paginas_md.append(ocr_pagina(imagen, processor, model))

    markdown_out = output_dir / pdf_path.relative_to(input_dir).parent / (pdf_path.stem + ".md")
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text("\n\n".join(paginas_md), encoding="utf-8")
    return num_paginas


def parse_args():
    parser = argparse.ArgumentParser(description="Test de OCR con TeleOCR, para comparar tiempo/páginas contra ocr_docling.py y ocr_docling_granite.py")
    parser.add_argument("--carpeta", type=str, default=None, help="Subcarpeta relativa dentro de biblioteca/ a procesar (default: toda biblioteca/)")
    parser.add_argument("--limite", type=int, default=None, help="Procesar solo los primeros N PDFs encontrados")
    parser.add_argument("--pdf", type=str, default=None, help="Ruta a un único PDF a procesar (ignora --carpeta/--limite)")
    parser.add_argument("--gpu", type=str, default=None, help="GPU(s) a usar (valor de CUDA_VISIBLE_DEVICES para este proceso), ya resuelto antes de los imports")
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
        print("No se encontraron PDFs para procesar.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    settings.datos_dir.mkdir(exist_ok=True)
    csv_path = settings.datos_dir / "metricas_ocr_teleocr.csv"

    print(f"Cargando {MODEL_ID}...")
    processor, model = cargar_modelo()

    filas = []
    print(f"Encontrados {len(pdfs)} PDFs. Iniciando procesamiento con TeleOCR...\n")

    for i, pdf_path in enumerate(pdfs, start=1):
        print(f"[{i}/{len(pdfs)}] Procesando: {pdf_path.name}")
        t0 = time.time()
        try:
            num_paginas = procesar_pdf_teleocr(pdf_path, processor, model, INPUT_DIR, OUTPUT_DIR)
            elapsed = time.time() - t0
            tiempo_str = formatear_tiempo(elapsed)
            print(f"    OK en {tiempo_str} ({num_paginas} páginas)\n")
            filas.append([pdf_path.name, tiempo_str, num_paginas])

        except Exception as e:
            elapsed = time.time() - t0
            tiempo_str = formatear_tiempo(elapsed)
            print(f"    ERROR tras {tiempo_str}: {e}\n")
            filas.append([pdf_path.name, f"ERROR ({tiempo_str})", "N/A"])

    guardar_metricas_csv(filas, csv_path)
    print(f"Listo. Métricas guardadas en: {csv_path}")
    print(f"Markdown guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
