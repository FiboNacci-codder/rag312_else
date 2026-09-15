"""Corre TeleOCR (repo oficial caipeng328/TeleOCR, vía su infer.py + vLLM) sobre
los PDFs de biblioteca/, preservando la jerarquía de carpetas en la salida, y
mide tiempo/páginas por archivo — para comparar contra ocr_docling.py (EasyOCR)
y ocr_docling_granite.py (granite-docling).

Por qué este script no llama a TeleOCR directamente como los otros dos motores:
- infer.py de TeleOCR corre en su propio conda env ("teleocr", python 3.10 +
  vllm==0.11.0): el env "rag312" tiene transformers==5.16.1, que rompe el
  código custom del modelo (KeyError en ROPE_INIT_FUNCTIONS) si se lo carga
  directo con `transformers` — ver commit que borró el intento anterior
  (ingesta/ocr_teleocr.py). Vía vLLM, en cambio, se usa la implementación
  nativa de vLLM para la arquitectura Qwen2_5_VLForConditionalGeneration
  (la base de TeleOCR) y el bug no aparece.
- infer.py espera un directorio de entrada PLANO (no recursivo). biblioteca/
  tiene subcarpetas, así que acá se arma un directorio de staging con symlinks
  cuyo nombre codifica la ruta relativa (p.ej.
  "procedimientos__Procesos_de_Gestion_Comercial__...__CVA-IN-001.pdf"), y al
  terminar se reorganizan los resultados de vuelta a una jerarquía que espeja
  biblioteca/.
- infer.py no loguea tiempo por archivo. Se usa como proxy la marca de tiempo
  de cada línea "Processing: <archivo>" que imprime por stdout: el tiempo de
  un archivo es el delta hasta la próxima línea "Processing:" (o hasta que
  termina el proceso, para el último). El primer archivo incluye el arranque
  del motor de vLLM, así que su tiempo no es comparable 1:1 con el resto —
  queda marcado en el CSV.

Uso típico (desde el env "rag312", que ya tiene pypdfium2 — no hace falta
correr este script dentro del env "teleocr", el subproceso de infer.py se
lanza con el intérprete de ese env):

    python ingesta/ocr_teleocr_batch.py --limite 10   # prueba chica primero
    python ingesta/ocr_teleocr_batch.py                # biblioteca/ completa
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag312.config import settings
from rag312.utils import formatear_tiempo

import pypdfium2 as pdfium

BIBLIOTECA_DIR = settings.biblioteca_dir
OUTPUT_DIR = settings.salida_md_teleocr_dir
STAGING_DIR = settings.datos_dir / "teleocr_staging_input"
RESULT_DIR = settings.datos_dir / "teleocr_raw_output"
CSV_PATH = settings.datos_dir / "metricas_ocr_teleocr.csv"

PROCESSING_RE = re.compile(r"^Processing:\s*(.+)$")
FAILED_RE = re.compile(r"^Failed:\s*(.+)$")


def guardar_metricas_csv(filas: list[list], csv_path: Path) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Nombre del archivo", "Ruta relativa (dentro de biblioteca/)", "Tiempo en minutos y seg", "Páginas"])
        writer.writerows(filas)


def normalizar_encoding(nombre: str) -> str:
    """Reemplaza separadores de path y caracteres no-ASCII para que el nombre
    codificado sea un filename válido y estable en cualquier filesystem."""
    sin_acentos = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", sin_acentos)


def preparar_staging(pdfs: list[Path], biblioteca_dir: Path, staging_dir: Path) -> dict[str, Path]:
    """Symlinkea cada PDF a staging_dir con un nombre único que codifica su
    ruta relativa. Devuelve {stem_codificado: ruta_relativa_original}."""
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    mapeo = {}
    for pdf_path in pdfs:
        rel = pdf_path.relative_to(biblioteca_dir)
        stem_codificado = normalizar_encoding(str(rel.with_suffix("")))
        (staging_dir / f"{stem_codificado}.pdf").symlink_to(pdf_path.resolve())
        mapeo[stem_codificado] = rel
    return mapeo


def contar_paginas(pdf_path: Path) -> int:
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        return len(pdf)
    finally:
        pdf.close()


def correr_infer(teleocr_python: Path, teleocr_repo: Path, staging_dir: Path, result_dir: Path,
                  gpu: str | None, gpu_memory_utilization: float | None) -> list[str]:
    """Lanza infer.py como subproceso, imprimiendo su stdout en vivo y
    devolviendo las líneas crudas (para parsear timing después)."""
    override = ["model_path=StarDoc-AI/TeleOCR", "BACKEND=vllm-engine"]
    if gpu_memory_utilization is not None:
        override.append(f"GPU_MEMORY_UTILIZATION={gpu_memory_utilization}")

    cmd = [
        str(teleocr_python), "infer.py",
        "--image_sub_path", str(staging_dir),
        "--result_save_path", str(result_dir),
        "--override", *override,
    ]

    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu

    proceso = subprocess.Popen(
        cmd, cwd=teleocr_repo, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    lineas = []
    for linea in proceso.stdout:
        print(linea, end="")
        lineas.append((time.time(), linea.rstrip("\n")))
    proceso.wait()
    if proceso.returncode != 0:
        print(f"\n[AVISO] infer.py terminó con código {proceso.returncode} — revisá el log de arriba.\n")
    return lineas


def calcular_tiempos_por_archivo(lineas_con_ts: list[tuple[float, str]], fin_proceso: float) -> dict[str, tuple[float, bool, bool]]:
    """A partir de las líneas 'Processing: ...' / 'Failed: ...' de stdout,
    arma {stem_codificado: (segundos, hubo_error, es_el_primero)}."""
    marcas = []  # (timestamp, stem_codificado)
    fallidos = set()

    for ts, linea in lineas_con_ts:
        m = PROCESSING_RE.match(linea)
        if m:
            stem = Path(m.group(1).strip()).stem
            marcas.append((ts, stem))
            continue
        m = FAILED_RE.match(linea)
        if m:
            stem = Path(m.group(1).strip()).stem
            fallidos.add(stem)

    tiempos = {}
    for i, (ts, stem) in enumerate(marcas):
        siguiente_ts = marcas[i + 1][0] if i + 1 < len(marcas) else fin_proceso
        tiempos[stem] = (siguiente_ts - ts, stem in fallidos, i == 0)
    return tiempos


def reorganizar_resultados(mapeo: dict[str, Path], result_dir: Path, output_dir: Path, biblioteca_dir: Path) -> None:
    """Mueve cada resultado (cualquiera sea su extensión/estructura — infer.py
    no la documenta) de result_dir a output_dir, espejando la jerarquía de
    biblioteca/ y devolviéndole al archivo su nombre original."""
    for stem_codificado, rel_path in mapeo.items():
        encontrados = sorted(result_dir.glob(f"{stem_codificado}*"))
        if not encontrados:
            continue
        destino_dir = output_dir / rel_path.parent
        destino_dir.mkdir(parents=True, exist_ok=True)
        for encontrado in encontrados:
            sufijo = encontrado.name[len(stem_codificado):]  # extensión o "" si es carpeta
            destino = destino_dir / f"{rel_path.stem}{sufijo}"
            if destino.exists():
                shutil.rmtree(destino) if destino.is_dir() else destino.unlink()
            shutil.move(str(encontrado), str(destino))


def parse_args():
    parser = argparse.ArgumentParser(description="Corre TeleOCR (infer.py oficial) sobre biblioteca/ y arma métricas de tiempo/páginas")
    parser.add_argument("--carpeta", type=str, default=None, help="Subcarpeta relativa dentro de biblioteca/ a procesar (default: toda biblioteca/)")
    parser.add_argument("--limite", type=int, default=None, help="Procesar solo los primeros N PDFs encontrados (para probar antes de correr todo)")
    parser.add_argument("--teleocr-repo", type=str, default=str(settings.rag_project_dir / "teleocr_repo"), help="Ruta al clon de github.com/caipeng328/TeleOCR")
    parser.add_argument("--teleocr-env", type=str, default="teleocr", help="Nombre del conda env con vllm==0.11.0 + transformers==4.57.1")
    parser.add_argument("--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES para el subproceso infer.py (default: la del shell actual)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help="Override de GPU_MEMORY_UTILIZATION (default: el 0.95 de infer.py — usar solo si la GPU está compartida)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not BIBLIOTECA_DIR.exists():
        print(f"No existe la carpeta: {BIBLIOTECA_DIR}")
        sys.exit(1)

    base_dir = (BIBLIOTECA_DIR / args.carpeta) if args.carpeta else BIBLIOTECA_DIR
    pdfs = sorted(base_dir.rglob("*.pdf"))
    if args.limite:
        pdfs = pdfs[: args.limite]
    if not pdfs:
        print("No se encontraron PDFs para procesar.")
        sys.exit(1)

    teleocr_repo = Path(args.teleocr_repo).expanduser()
    teleocr_python = settings.rag_project_dir / "conda" / "miniconda3" / "envs" / args.teleocr_env / "bin" / "python"
    if not teleocr_python.exists():
        print(f"No encuentro el intérprete del env '{args.teleocr_env}' en {teleocr_python}")
        sys.exit(1)
    if not (teleocr_repo / "infer.py").exists():
        print(f"No encuentro infer.py en {teleocr_repo}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    settings.datos_dir.mkdir(exist_ok=True)

    print(f"Encontrados {len(pdfs)} PDFs. Armando staging...")
    mapeo = preparar_staging(pdfs, BIBLIOTECA_DIR, STAGING_DIR)

    print("Contando páginas...")
    paginas_por_stem = {}
    for stem, rel_path in mapeo.items():
        paginas_por_stem[stem] = contar_paginas(BIBLIOTECA_DIR / rel_path.with_suffix(".pdf"))

    print(f"Corriendo infer.py sobre {len(pdfs)} PDFs (un solo arranque de motor vLLM)...\n")
    t0 = time.time()
    lineas_con_ts = correr_infer(teleocr_python, teleocr_repo, STAGING_DIR, RESULT_DIR, args.gpu, args.gpu_memory_utilization)
    tiempos = calcular_tiempos_por_archivo(lineas_con_ts, time.time())

    print("\nReorganizando resultados según la jerarquía de biblioteca/...")
    reorganizar_resultados(mapeo, RESULT_DIR, OUTPUT_DIR, BIBLIOTECA_DIR)

    filas = []
    for stem, rel_path in sorted(mapeo.items(), key=lambda kv: str(kv[1])):
        num_paginas = paginas_por_stem[stem]
        if stem in tiempos:
            segundos, hubo_error, es_primero = tiempos[stem]
            tiempo_str = formatear_tiempo(segundos)
            if hubo_error:
                tiempo_str = f"ERROR ({tiempo_str})"
            if es_primero:
                tiempo_str += " (incluye arranque del motor vLLM)"
        else:
            tiempo_str = "N/A (no procesado)"
        filas.append([rel_path.name, str(rel_path), tiempo_str, num_paginas])

    guardar_metricas_csv(filas, CSV_PATH)
    print(f"\nListo en {formatear_tiempo(time.time() - t0)} totales.")
    print(f"Métricas guardadas en: {CSV_PATH}")
    print(f"Markdown/resultados guardados en: {OUTPUT_DIR} (espejando la jerarquía de biblioteca/)")
    print("OJO: el tiempo del primer archivo de la corrida incluye el arranque del motor vLLM, no es comparable 1:1 con el resto.")


if __name__ == "__main__":
    main()
