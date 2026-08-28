"""
Revisión manual interactiva del ground truth generado por
generar_ground_truth.py.

Muestra cada pregunta junto con el fragmento del chunk que la originó y la
respuesta de referencia ("ground_truth") generada por el LLM, y permite:
aceptar, editar el texto, o saltar (revisar después).

Uso:
    python revisar_ground_truth.py
    python revisar_ground_truth.py --golden golden_set.json --out golden_set.json
    python revisar_ground_truth.py --solo-pendientes

Controles:
    [a] aceptar (marca ground_truth_revisado=true)
    [e] editar el texto del ground_truth (y luego marca ground_truth_revisado=true)
    [s] saltar (la deja como está, para volver después)
    [q] guardar y salir
"""

import argparse
import json
import os
import sys
from pathlib import Path

RAG_PROJECT_DIR = Path(os.environ.get("RAG_PROJECT_DIR", Path.home() / "rag312"))
CHUNKS_JSON = RAG_PROJECT_DIR / "datos" / "chunks_data.json"


def cargar_chunks_por_indice() -> dict:
    """Mapea (source, chunk_index) -> page_content, para mostrar contexto al revisar."""
    if not CHUNKS_JSON.exists():
        return {}
    with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapa = {}
    for c in data:
        meta = c.get("metadata", {})
        clave = (meta.get("source"), meta.get("chunk_index"))
        mapa[clave] = c.get("page_content", "")
    return mapa


def guardar(golden_set: list, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(golden_set, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Revisión manual interactiva del ground truth")
    parser.add_argument("--golden", type=str, default=str(Path(__file__).parent / "golden_set.json"))
    parser.add_argument("--out", type=str, default=None, help="Por defecto sobreescribe --golden")
    parser.add_argument("--solo-pendientes", action="store_true",
                         help="Mostrar solo las que tienen ground_truth_revisado=false (default: todas las que tienen ground_truth)")
    args = parser.parse_args()
    out_path = Path(args.out) if args.out else Path(args.golden)

    with open(args.golden, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    chunks_map = cargar_chunks_por_indice()

    con_ground_truth_idx = [i for i, item in enumerate(golden_set) if item.get("ground_truth")]
    if not con_ground_truth_idx:
        print("No hay preguntas con 'ground_truth' generado. Corré primero:")
        print("  python generar_ground_truth.py")
        sys.exit(1)

    pendientes_idx = [
        i for i in con_ground_truth_idx
        if not args.solo_pendientes or not golden_set[i].get("ground_truth_revisado", False)
    ]

    print(f"Preguntas con ground_truth: {len(con_ground_truth_idx)} | A revisar en esta pasada: {len(pendientes_idx)}\n")

    for n, i in enumerate(pendientes_idx, start=1):
        item = golden_set[i]
        print("=" * 70)
        print(f"[{n}/{len(pendientes_idx)}] id={item['id']}  categoria={item.get('categoria')}")
        print(f"PREGUNTA: {item['pregunta']}")

        clave = (item.get('fuente_esperada', {}).get('documento'), item.get('chunk_origen_index'))
        contexto = chunks_map.get(clave)
        if contexto:
            print("-" * 70)
            print("Fragmento origen (primeros 500 caracteres):")
            print(contexto[:500] + ("..." if len(contexto) > 500 else ""))

        print("-" * 70)
        print(f"GROUND TRUTH: {item.get('ground_truth')}")
        print("-" * 70)

        while True:
            resp = input("[a]ceptar / [e]ditar / [s]altar / [q]uardar y salir: ").strip().lower()
            if resp == "a":
                item["ground_truth_revisado"] = True
                break
            elif resp == "e":
                nueva = input("Nuevo ground truth: ").strip()
                if nueva:
                    item["ground_truth"] = nueva
                    item["ground_truth_revisado"] = True
                break
            elif resp == "s":
                break
            elif resp == "q":
                guardar(golden_set, out_path)
                n_revisadas = sum(1 for it in golden_set if it.get("ground_truth_revisado"))
                print(f"\nGuardado en {out_path}. Ground truth revisados: {n_revisadas}/{len(con_ground_truth_idx)}. Saliendo.")
                sys.exit(0)
            else:
                print("Opción inválida.")
        print()

    guardar(golden_set, out_path)

    n_revisadas = sum(1 for it in golden_set if it.get("ground_truth_revisado"))
    print("=" * 70)
    print(f"Listo. Guardado en {out_path}")
    print(f"Ground truth revisados: {n_revisadas}/{len(con_ground_truth_idx)}")
    print(f"Pendientes: {len(con_ground_truth_idx) - n_revisadas}")


if __name__ == "__main__":
    main()
