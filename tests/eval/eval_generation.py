"""
Evaluación de calidad de GENERACIÓN (no solo retrieval) usando RAGAS,
apuntado a los servidores vLLM locales (sin depender de OpenAI/Anthropic).

Corre el pipeline completo (rag_query1.main()) sobre cada pregunta del
golden set, y evalúa la respuesta generada con 4 métricas de RAGAS:

  - faithfulness         : ¿la respuesta se sostiene únicamente en el
                            contexto recuperado, sin alucinar información?
  - answer_relevancy     : ¿la respuesta contesta realmente lo que se
                            preguntó?
  - context_precision    : ¿los chunks recuperados relevantes están
                            rankeados arriba, comparado contra la
                            respuesta de referencia (ground_truth)?
  - context_recall       : ¿el contexto recuperado contiene lo necesario
                            para derivar la respuesta de referencia?

Las dos últimas requieren una respuesta de referencia curada
("ground_truth") por pregunta — generarla con:

    python generar_ground_truth.py
    python revisar_ground_truth.py    # revisión manual, obligatoria

Solo se evalúan preguntas con "ground_truth" presente y
"ground_truth_revisado": true. Las que no lo tienen se excluyen y se
reportan aparte (no frenan la corrida).

Uso:
    source ~/rag312/conda/activar.sh
    conda activate rag312
    pip install ragas --break-system-packages   # una sola vez
    export RAG_PROJECT_DIR=~/rag312
    cd ~/rag312/tests/eval

    python eval_generation.py
    python eval_generation.py --golden golden_set.json --n-muestra 15

Requiere: vLLM embeddings (8001), vLLM normalizador (8003), vLLM generador
(8002), Qdrant (6333) arriba, con la colección procedimientos_sielse ya
indexada.
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

RAG_PROJECT_DIR = Path(os.environ.get("RAG_PROJECT_DIR", Path.home() / "rag312"))
CONSULTA_DIR = RAG_PROJECT_DIR / "consulta"
INGESTA_DIR = RAG_PROJECT_DIR / "ingesta"
for p in (CONSULTA_DIR, INGESTA_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from rag_query1 import main as rag_main
except ImportError as e:
    print(f"ERROR: no se pudo importar rag_query1 desde {CONSULTA_DIR}. Detalle: {e}")
    sys.exit(1)

# --- Parche: ragas importa ChatVertexAI desde una ruta que ya no existe en
# versiones recientes de langchain_community. No usamos Vertex AI, así que
# registramos un módulo "stub" para que ese import no rompa la carga de ragas.
import types as _types
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = _types.ModuleType("langchain_community.chat_models.vertexai")
    class ChatVertexAI:
        pass
    _stub.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.run_config import RunConfig
    from ragas.metrics import faithfulness, answer_relevancy, LLMContextPrecisionWithReference, LLMContextRecall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:
    print("ERROR: falta instalar ragas. Corré:")
    print("  pip install ragas --break-system-packages")
    sys.exit(1)

# --- Modelos usados para el juez de RAGAS: los mismos vLLM del proyecto ---
JUDGE_LLM_URL = os.environ.get("JUDGE_LLM_URL", "http://localhost:8002/v1")  # qwen35-9b
JUDGE_LLM_MODEL = os.environ.get("JUDGE_LLM_MODEL", "qwen35-9b")
EMBED_URL = os.environ.get("EMBED_URL", "http://localhost:8001/v1")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "harrier-embed")

METRICAS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def build_ragas_llm_embeddings():
    chat = ChatOpenAI(
        base_url=JUDGE_LLM_URL,
        api_key="no-necesaria",
        model=JUDGE_LLM_MODEL,
        temperature=0.0,
        max_tokens=4096,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},  # <-- clave
    )
    embed = OpenAIEmbeddings(
        base_url=EMBED_URL,
        api_key="no-necesaria",
        model=EMBED_MODEL,
        check_embedding_ctx_length=False,
    )
    return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(embed)


def promedio_valido(valores: list[float]):
    validos = [v for v in valores if v is not None and not math.isnan(v)]
    return (sum(validos) / len(validos)) if validos else None


def resumen_por_categoria(detalle: list[dict]) -> dict:
    por_cat = defaultdict(list)
    for d in detalle:
        por_cat[d.get("categoria") or "sin_categoria"].append(d)
    resumen = {}
    for cat, items in sorted(por_cat.items()):
        resumen[cat] = {"n_preguntas": len(items)}
        for metrica in METRICAS:
            resumen[cat][f"{metrica}_promedio"] = promedio_valido([i.get(metrica) for i in items])
    return resumen


def main():
    parser = argparse.ArgumentParser(description="Evaluación de generación (RAGAS) sobre el golden set")
    parser.add_argument("--golden", type=str, default=str(Path(__file__).parent / "golden_set.json"))
    parser.add_argument("--n-muestra", type=int, default=None,
                         help="Evaluar solo una muestra aleatoria (por defecto: todas)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=str(Path(__file__).parent / "resultados_generation.json"))
    args = parser.parse_args()

    with open(args.golden, "r", encoding="utf-8") as f:
        golden_set = json.load(f)
    if not golden_set:
        print(f"ERROR: el golden set '{args.golden}' está vacío.")
        sys.exit(1)

    n_total = len(golden_set)
    golden_set = [
        item for item in golden_set
        if item.get("ground_truth") and item.get("ground_truth_revisado")
    ]
    n_sin_ground_truth = n_total - len(golden_set)
    if not golden_set:
        print(f"ERROR: ninguna de las {n_total} preguntas tiene 'ground_truth' revisado. Corré primero:")
        print("  python generar_ground_truth.py")
        print("  python revisar_ground_truth.py")
        sys.exit(1)
    if n_sin_ground_truth:
        print(f"AVISO: se excluyen {n_sin_ground_truth}/{n_total} preguntas sin 'ground_truth' revisado.\n")

    if args.n_muestra:
        random.seed(args.seed)
        golden_set = random.sample(golden_set, min(args.n_muestra, len(golden_set)))

    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logging.getLogger("ragas").setLevel(logging.DEBUG)

    print(f"Corriendo pipeline RAG completo sobre {len(golden_set)} preguntas...\n")

    preguntas, respuestas, contextos, referencias = [], [], [], []
    detalle_por_pregunta = []
    errores = []

    t_inicio = time.time()
    for i, item in enumerate(golden_set, start=1):
        pregunta = item["pregunta"]
        print(f"[{i}/{len(golden_set)}] {pregunta[:70]}")
        try:
            resultado = rag_main(pregunta)
        except Exception as e:
            print(f"    ERROR ejecutando el pipeline: {e}")
            errores.append({"id": item.get("id"), "pregunta": pregunta, "error": str(e)})
            continue

        respuesta = resultado.get("respuesta")
        chunks_texto = [f["chunk"] for f in resultado["fuentes"] if f.get("chunk")]
        if not respuesta or not chunks_texto:
            print("    AVISO: respuesta vacía o sin contexto recuperado, se excluye de RAGAS")
            errores.append({
                "id": item.get("id"), "pregunta": pregunta,
                "error": "respuesta vacía o sin chunks de contexto",
            })
            continue

        preguntas.append(pregunta)
        respuestas.append(respuesta)
        # RAGAS espera "retrieved_contexts" como lista de strings por pregunta
        contextos.append(chunks_texto)
        referencias.append(item["ground_truth"])

        detalle_por_pregunta.append({
            "id": item.get("id"),
            "categoria": item.get("categoria"),
            "pregunta": pregunta,
            "respuesta": respuesta,
            "ground_truth": item["ground_truth"],
            "documentos_recuperados": [f["documento"] for f in resultado["fuentes"]],
        })
    t_total = time.time() - t_inicio

    if not preguntas:
        print("No se pudo generar ninguna respuesta válida. Verificá que Qdrant/vLLM estén arriba.")
        sys.exit(1)

    dataset = Dataset.from_dict({
        "user_input": preguntas,
        "response": respuestas,
        "retrieved_contexts": contextos,
        "reference": referencias,
    })

    print(f"\nEvaluando con RAGAS ({', '.join(METRICAS)}) usando {JUDGE_LLM_MODEL} como juez...")
    ragas_llm, ragas_embeddings = build_ragas_llm_embeddings()
    context_precision = LLMContextPrecisionWithReference(llm=ragas_llm)
    context_recall = LLMContextRecall(llm=ragas_llm)

    run_config = RunConfig(
        max_workers=1,
        timeout=600,
        max_retries=3,
        max_wait=60,
    )

    resultado_ragas = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
        raise_exceptions=True,   # <-- agregar temporalmente
    )

    df = resultado_ragas.to_pandas()
    for i, fila in enumerate(detalle_por_pregunta):
        for metrica in METRICAS:
            if metrica in df.columns:
                valor = df.iloc[i][metrica]
                es_nan = isinstance(valor, float) and math.isnan(valor)
                fila[metrica] = float(valor) if valor is not None and not es_nan else None
            else:
                fila[metrica] = None

    resumen = {
        "n_preguntas_golden": n_total,
        "n_sin_ground_truth": n_sin_ground_truth,
        "n_preguntas_evaluadas": len(preguntas),
        "n_errores_pipeline": len(errores),
        "tiempo_total_seg": round(t_total, 2),
        "tiempo_promedio_por_pregunta_seg": round(t_total / len(golden_set), 3),
        "por_categoria": resumen_por_categoria(detalle_por_pregunta),
    }
    for metrica in METRICAS:
        if metrica in df.columns:
            valores = df[metrica].tolist()
            resumen[f"{metrica}_promedio"] = promedio_valido(valores)
            resumen[f"{metrica}_n_validas"] = sum(1 for v in valores if v is not None and not math.isnan(v))
        else:
            resumen[f"{metrica}_promedio"] = None
            resumen[f"{metrica}_n_validas"] = 0
            print(f"AVISO: la métrica '{metrica}' no se pudo calcular (falló toda la columna).")
    print("\n" + "=" * 60)
    print("RESUMEN — CALIDAD DE GENERACIÓN")
    print("=" * 60)
    for k, v in resumen.items():
        if k == "por_categoria":
            print(f"{'por_categoria':30s}:")
            for cat, m in v.items():
                partes = [f"{met}={m[f'{met}_promedio']:.2f}" if m[f"{met}_promedio"] is not None else f"{met}=N/A"
                          for met in METRICAS]
                print(f"  - {cat:28s} {' '.join(partes)} (n={m['n_preguntas']})")
        else:
            print(f"{k:30s}: {v}")

    if errores:
        print(f"\nAVISO: {len(errores)} pregunta(s) no se pudieron evaluar (pipeline falló o respuesta vacía).")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"resumen": resumen, "detalle": detalle_por_pregunta, "errores": errores}, f, ensure_ascii=False, indent=2)
    print(f"\nResultados guardados en: {args.out}")


if __name__ == "__main__":
    main()
