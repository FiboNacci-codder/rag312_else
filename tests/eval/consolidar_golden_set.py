"""
Consolida los archivos generados por filtrar_golden_set.py + revisar_golden_set.py
(golden_set_aceptadas.json, golden_set_dudosas.json, golden_set_rechazadas.json)
en un único golden_set.json final, listo para eval_retrieval.py.

Solo incluye preguntas con "revisado": true. Limpia campos internos del
pipeline de filtrado (veredicto_juez, estado_filtro) que ya no hacen falta
para la evaluación.

Uso:
    python consolidar_golden_set.py
    python consolidar_golden_set.py --out golden_set_final.json
"""

import argparse
import json
from pathlib import Path

ARCHIVOS_DEFAULT = [
    "golden_set_aceptadas.json",
    "golden_set_dudosas.json",
    "golden_set_rechazadas.json",
]

CAMPOS_A_LIMPIAR = ["veredicto_juez", "estado_filtro"]


def main():
    parser = argparse.ArgumentParser(description="Consolidar golden set final")
    parser.add_argument("--dir", type=str, default=str(Path(__file__).parent))
    parser.add_argument("--archivos", nargs="+", default=ARCHIVOS_DEFAULT)
    parser.add_argument("--out", type=str, default="golden_set.json")
    parser.add_argument("--solo-revisadas", action="store_true", default=True)
    args = parser.parse_args()
    base = Path(args.dir)

    consolidado = []
    resumen_por_archivo = {}

    for nombre in args.archivos:
        path = base / nombre
        if not path.exists():
            print(f"Aviso: no existe {path}, se omite.")
            continue
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)

        incluidas = [it for it in items if it.get("revisado", False)]
        resumen_por_archivo[nombre] = {"total": len(items), "incluidas": len(incluidas)}

        for it in incluidas:
            for campo in CAMPOS_A_LIMPIAR:
                it.pop(campo, None)
            consolidado.append(it)

    # Renumerar ids para que queden consecutivos y sin duplicados
    for i, item in enumerate(consolidado, start=1):
        item["id"] = f"q{i:04d}"

    out_path = base / args.out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(consolidado, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("CONSOLIDACIÓN DEL GOLDEN SET")
    print("=" * 60)
    for nombre, info in resumen_por_archivo.items():
        print(f"{nombre:30s}: {info['incluidas']}/{info['total']} incluidas (revisado=true)")
    print("-" * 60)
    print(f"TOTAL preguntas en golden set final: {len(consolidado)}")
    print(f"Guardado en: {out_path}")

    if len(consolidado) < 20:
        print("\nAviso: el golden set final tiene menos de 20 preguntas.")
        print("Es usable para una primera corrida, pero para conclusiones más")
        print("confiables conviene generar más chunks con generar_golden_set.py")
        print("(otra categoría, otro --seed) y repetir filtrado + revisión.")


if __name__ == "__main__":
    main()