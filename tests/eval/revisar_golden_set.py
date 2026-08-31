"""
Revisión manual interactiva del golden_set.json generado por
generar_golden_set.py.

Muestra cada pregunta junto con el fragmento del chunk que la originó,
y permite: aceptar, descartar, editar el texto, o saltar (revisar después).

Uso:
    python revisar_golden_set.py
    python revisar_golden_set.py --golden golden_set.json --out golden_set.json

Controles:
    [a] aceptar (marca revisado=true)
    [d] descartar (elimina la pregunta del set)
    [e] editar el texto de la pregunta (y luego marca revisado=true)
    [s] saltar (la deja como está, revisado=false, para volver después)
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
    parser = argparse.ArgumentParser(description="Revisión manual interactiva del golden set")
    parser.add_argument("--golden", type=str, default=str(Path(__file__).parent / "golden_set.json"))
    parser.add_argument("--out", type=str, default=None, help="Por defecto sobreescribe --golden")
    parser.add_argument("--solo-pendientes", action="store_true",
                         help="Mostrar solo las que tienen revisado=false (default: todas)")
    args = parser.parse_args()
    out_path = Path(args.out) if args.out else Path(args.golden)

    with open(args.golden, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    chunks_map = cargar_chunks_por_indice()

    pendientes_idx = [
        i for i, item in enumerate(golden_set)
        if not args.solo_pendientes or not item.get("revisado", False)
    ]

    print(f"Total preguntas: {len(golden_set)} | A revisar en esta pasada: {len(pendientes_idx)}\n")

    a_eliminar = set()

    for n, i in enumerate(pendientes_idx, start=1):
        item = golden_set[i]
        print("=" * 70)
        print(f"[{n}/{len(pendientes_idx)}] id={item['id']}  categoria={item.get('categoria')}")
        print(f"Documento esperado : {item['fuente_esperada'].get('documento')}")
        print(f"Sección esperada   : {item['fuente_esperada'].get('seccion_contiene')}")
        print("-" * 70)
        print(f"PREGUNTA: {item['pregunta']}")

        if item.get("texto_seccion"):
            contexto = item["texto_seccion"]
        else:
            clave = (item['fuente_esperada'].get('documento'), item.get('chunk_origen_index'))
            contexto = chunks_map.get(clave)
        if contexto:
            print("-" * 70)
            print("Fragmento origen (primeros 500 caracteres):")
            print(contexto[:500] + ("..." if len(contexto) > 500 else ""))

        print("-" * 70)
        while True:
            resp = input("[a]ceptar / [d]escartar / [e]ditar / [s]altar / [q]uardar y salir: ").strip().lower()
            if resp == "a":
                item["revisado"] = True
                break
            elif resp == "d":
                a_eliminar.add(i)
                break
            elif resp == "e":
                nueva = input("Nueva pregunta: ").strip()
                if nueva:
                    item["pregunta"] = nueva
                    item["revisado"] = True
                break
            elif resp == "s":
                break
            elif resp == "q":
                golden_set_final = [it for j, it in enumerate(golden_set) if j not in a_eliminar]
                guardar(golden_set_final, out_path)
                print(f"\nGuardado en {out_path} ({len(golden_set_final)} preguntas). Saliendo.")
                sys.exit(0)
            else:
                print("Opción inválida.")
        print()

    golden_set_final = [it for j, it in enumerate(golden_set) if j not in a_eliminar]
    guardar(golden_set_final, out_path)

    n_revisadas = sum(1 for it in golden_set_final if it.get("revisado"))
    print("=" * 70)
    print(f"Listo. {len(golden_set_final)} preguntas guardadas en {out_path}")
    print(f"Revisadas (aceptadas/editadas): {n_revisadas}")
    print(f"Pendientes: {len(golden_set_final) - n_revisadas}")


if __name__ == "__main__":
    main()