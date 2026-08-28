"""
Inspecciona en detalle los casos MISS (o cualquier caso) de una corrida de
eval_retrieval.py, mostrando qué trajo realmente el sistema en el top-K
comparado con lo que se esperaba en el golden set.

Útil para distinguir:
  - falla real de retrieval (el chunk esperado existe y es específico, pero
    no se recuperó)
  - problema del golden set (pregunta demasiado genérica/ambigua, con
    múltiples respuestas "correctas" posibles — p.ej. "¿quién aprueba este
    procedimiento?" que se repite casi igual en todos los documentos)
  - fuente_esperada mal etiquetada

Por defecto busca con la pregunta normalizada/reformulada (igual que
rag_query1.main() y eval_retrieval.py en producción), para que lo que se
inspecciona acá coincida con lo que realmente se evaluó. Usar
--sin-normalizar si corriste eval_retrieval.py con esa misma opción.

Uso:
    python inspeccionar_miss.py
    python inspeccionar_miss.py --resultados resultados_eval.json --solo-miss
    python inspeccionar_miss.py --id q0010
    python inspeccionar_miss.py --sin-normalizar

Requiere que vLLM (8001, y 8003 si se normaliza) y Qdrant (6333) estén
corriendo, porque vuelve a ejecutar recuperar_contexto() para esas
preguntas puntuales (resultados_eval.json no guarda el detalle de qué
chunks trajo, solo el resumen de ranks).
"""

import argparse
import json
import os
import sys
from pathlib import Path

RAG_PROJECT_DIR = Path(os.environ.get("RAG_PROJECT_DIR", Path.home() / "rag312"))
CONSULTA_DIR = RAG_PROJECT_DIR / "consulta"
INGESTA_DIR = RAG_PROJECT_DIR / "ingesta"
for p in (CONSULTA_DIR, INGESTA_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from qdrant_client import QdrantClient
from rag_query1 import build_embedder, recuperar_contexto, QDRANT_HOST, QDRANT_PORT, TOP_K
from embeddings import build_sparse_embedder
from normalizar_query import procesar_pregunta


def main():
    parser = argparse.ArgumentParser(description="Inspección detallada de casos del golden set")
    parser.add_argument("--golden", type=str, default=str(Path(__file__).parent / "golden_set.json"))
    parser.add_argument("--resultados", type=str, default=str(Path(__file__).parent / "resultados_eval.json"))
    parser.add_argument("--solo-miss", action="store_true", default=True)
    parser.add_argument("--id", type=str, default=None, help="Inspeccionar solo un id puntual, p.ej. q0010")
    parser.add_argument(
        "--sin-normalizar",
        action="store_true",
        help="Buscar con la pregunta cruda, sin pasar por normalizar_query.procesar_pregunta().",
    )
    args = parser.parse_args()

    with open(args.golden, "r", encoding="utf-8") as f:
        golden_set = {item["id"]: item for item in json.load(f)}

    with open(args.resultados, "r", encoding="utf-8") as f:
        detalle = json.load(f)["detalle"]

    if args.id:
        objetivo = [d for d in detalle if d["id"] == args.id]
    elif args.solo_miss:
        objetivo = [d for d in detalle if not d.get("hit_fusion")]
    else:
        objetivo = detalle

    print(f"Inspeccionando {len(objetivo)} caso(s)...\n")

    embedder = build_embedder()
    sparse_embedder = build_sparse_embedder()
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    for fila in objetivo:
        item = golden_set.get(fila["id"])
        if not item:
            continue

        print("=" * 80)
        print(f"[{fila['id']}] {item['pregunta']}")
        print(f"Esperado -> documento: {item['fuente_esperada'].get('documento')}")
        print(f"            sección  : {item['fuente_esperada'].get('seccion_contiene')}")
        print("-" * 80)

        origen = item.get("origen_generacion", "chunk")
        if origen == "markdown_seccion":
            print(f"(origen: sección de markdown completa, sin chunk_index puntual)")

        if args.sin_normalizar:
            pregunta_busqueda = item["pregunta"]
        else:
            pregunta_busqueda = procesar_pregunta(item["pregunta"])["busqueda"]
            if pregunta_busqueda != item["pregunta"]:
                print(f"Búsqueda (normalizada): {pregunta_busqueda}")

        resultados, _ = recuperar_contexto(pregunta_busqueda, embedder, sparse_embedder, client)

        print(f"Top-{TOP_K} recuperado (fusión RRF):")
        for i, r in enumerate(resultados, start=1):
            p = r.payload
            print(f"  {i}. {p.get('source')}  |  sección: {p.get('seccion')}")
        print()


if __name__ == "__main__":
    main()