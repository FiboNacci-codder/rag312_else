"""
Corre las "preguntas frontera" (generadas con generar_preguntas_frontera.py)
contra el pipeline RAG real (rag_query1.main()) y vuelca pregunta + respuesta
+ fuentes recuperadas a un JSON, para leer directo qué contestó el sistema.

No hace ningún scoring ni usa RAGAS — estas preguntas son a propósito NO
respondibles, así que no hay ground_truth ni fuente_esperada real contra la
que comparar. Es una herramienta de lectura humana: el objetivo es ver si el
RAG admite que no tiene la información, o si alucina una respuesta segura
pero incorrecta.

Uso:
    source ~/rag312/conda/activar.sh
    conda activate rag312
    export RAG_PROJECT_DIR=~/rag312
    cd ~/rag312/tests/eval

    python generar_preguntas_frontera.py --n-secciones 20 --por-categoria
    python correr_preguntas_frontera.py --n-muestra 5   # prueba rápida
    python correr_preguntas_frontera.py                 # todas

Requiere: vLLM embeddings (8001), normalizador (8003), generador (8002) y
Qdrant (6333) — el pipeline RAG completo, igual que rag_query1.py en
producción. No requiere el juez de RAGAS ni ninguna instancia dedicada.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

RAG_PROJECT_DIR = Path(os.environ.get("RAG_PROJECT_DIR", Path.home() / "rag312"))
CONSULTA_DIR = RAG_PROJECT_DIR / "consulta"
if str(CONSULTA_DIR) not in sys.path:
    sys.path.insert(0, str(CONSULTA_DIR))

try:
    from rag_query1 import main as rag_main
except ImportError as e:
    print(f"ERROR: no se pudo importar rag_query1 desde {CONSULTA_DIR}. Detalle: {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Correr preguntas frontera contra el RAG real, para lectura manual")
    parser.add_argument("--preguntas", type=str, default=str(Path(__file__).parent / "preguntas_frontera.json"))
    parser.add_argument("--n-muestra", type=int, default=None,
                         help="Correr solo una muestra aleatoria (por defecto: todas)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=str(Path(__file__).parent / "respuestas_preguntas_frontera.json"))
    args = parser.parse_args()

    with open(args.preguntas, "r", encoding="utf-8") as f:
        preguntas_frontera = json.load(f)
    if not preguntas_frontera:
        print(f"ERROR: '{args.preguntas}' está vacío. Corré primero generar_preguntas_frontera.py")
        sys.exit(1)

    if args.n_muestra:
        random.seed(args.seed)
        preguntas_frontera = random.sample(preguntas_frontera, min(args.n_muestra, len(preguntas_frontera)))

    print(f"Corriendo {len(preguntas_frontera)} preguntas frontera contra el RAG real...\n")

    resultados = []
    errores = []
    t_inicio = time.time()
    for i, item in enumerate(preguntas_frontera, start=1):
        pregunta = item["pregunta"]
        print(f"[{i}/{len(preguntas_frontera)}] {pregunta}")
        try:
            resultado = rag_main(pregunta)
        except Exception as e:
            print(f"    ERROR ejecutando el pipeline: {e}")
            errores.append({"id": item.get("id"), "pregunta": pregunta, "error": str(e)})
            continue

        respuesta = resultado.get("respuesta") or ""
        print(f"    -> {respuesta[:200]}{'...' if len(respuesta) > 200 else ''}\n")

        resultados.append({
            "id": item.get("id"),
            "categoria": item.get("categoria"),
            "pregunta": pregunta,
            "origen_aproximado": item.get("origen_aproximado"),
            "respuesta": respuesta,
            "documentos_recuperados": [
                {"documento": f.get("documento"), "seccion": f.get("seccion"), "rank_fusion": f.get("rank_fusion")}
                for f in resultado.get("fuentes", [])
            ],
        })
    t_total = time.time() - t_inicio

    print("=" * 60)
    print(f"Preguntas corridas: {len(resultados)}/{len(preguntas_frontera)}")
    print(f"Errores de pipeline: {len(errores)}")
    print(f"Tiempo total: {round(t_total, 2)}s")
    if errores:
        print("AVISO: revisá 'errores' en el JSON de salida.")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"resultados": resultados, "errores": errores}, f, ensure_ascii=False, indent=2)
    print(f"\nResultados guardados en: {args.out}")
    print("Revisar 'respuesta' de cada item: ¿admite que no tiene la información,")
    print("o inventa algo con apariencia de segura?")


if __name__ == "__main__":
    main()
