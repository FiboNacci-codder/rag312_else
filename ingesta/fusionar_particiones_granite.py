"""Combina los archivos por partición generados por corridas paralelas de
ocr_docling_granite.py (una por GPU, ver lanzar_granite_paralelo.sh) en los
archivos finales datos/metricas_ocr_granite.csv y datos/chunks_data_granite.json.

No importa docling/torch: es pura manipulación de CSV/JSON, así que corre
rápido y no necesita GPU.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag312.config import settings

CSV_FINAL = settings.datos_dir / "metricas_ocr_granite.csv"
JSON_FINAL = settings.datos_dir / "chunks_data_granite.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Fusiona las particiones de ocr_docling_granite.py en los archivos finales")
    parser.add_argument("--particiones", type=int, required=True, help="Cantidad de particiones a fusionar (0..N-1)")
    return parser.parse_args()


def fusionar_csv(n: int) -> None:
    filas = []
    for i in range(n):
        path = settings.datos_dir / f"metricas_ocr_granite_part{i}.csv"
        if not path.exists():
            print(f"    (aviso) no existe {path}, se omite")
            continue
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # descarta encabezado de cada parte
            filas.extend(reader)

    with open(CSV_FINAL, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Nombre del archivo", "Tiempo en minutos y seg", "Páginas"])
        writer.writerows(filas)
    print(f"CSV fusionado ({len(filas)} filas): {CSV_FINAL}")


def fusionar_json(n: int) -> None:
    combinados = []
    for i in range(n):
        path = settings.datos_dir / f"chunks_data_granite_part{i}.json"
        if not path.exists():
            print(f"    (aviso) no existe {path}, se omite")
            continue
        with open(path, "r", encoding="utf-8") as f:
            combinados.extend(json.load(f))

    with open(JSON_FINAL, "w", encoding="utf-8") as f:
        json.dump(combinados, f, ensure_ascii=False, indent=2)
    print(f"JSON fusionado ({len(combinados)} chunks): {JSON_FINAL}")


def main():
    args = parse_args()
    fusionar_csv(args.particiones)
    fusionar_json(args.particiones)


if __name__ == "__main__":
    main()
