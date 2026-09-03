# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A production RAG (Retrieval-Augmented Generation) system over internal procedure documents (PDFs) for ELSE/SIELSE, a utility company. Users ask questions in natural language and get answers citing the source document/page/section. Everything runs on local vLLM servers (no OpenAI/Anthropic API calls) plus a local Qdrant vector DB. See `docs/readme.md` for the full narrative writeup (hardware, per-script deep dives) — this file only covers what's needed to be productive quickly.

The project's canonical layout lives at `~/rag312_else` on a remote GPU server (`elsecluster5`, RHEL 9.4, 8x L40S, Apptainer for GPU containers). This repo is the code; large/generated artifacts (`conda/`, `containers/*.sif`, `qdrant_storage/`) are gitignored and only exist on that server.

## Pipeline architecture (read this before touching any script)

Two independent pipelines share the same data:

**1. Ingesta (build the index)** — run once, or whenever `biblioteca/` (source PDFs) changes:
```
biblioteca/*.pdf → ingesta/ocr_docling.py → salida_md/, salida_chunks/, datos/chunks_data.json
                 → ingesta/embeddings.py   → datos/embeddings_data.json (dense via vLLM + sparse BM25 via FastEmbed)
                 → ingesta/ingest_qdrant.py → Qdrant collection "procedimientos_sielse" (dense + sparse hybrid)
```
`ingest_qdrant.py` **drops and recreates** the collection every run — it is destructive, not incremental.

**2. Consulta (answer a question)**, used by both interfaces and by the eval scripts:
```
question → consulta/normalizar_query.py (spelling fix + colloquial→formal reformulation via vLLM)
         → consulta/rag_query1.py: dense (vLLM harrier-embed) + sparse (BM25) hybrid search in Qdrant, RRF fusion
         → answer generation via vLLM (qwen35-9b), citing sources
```
`consulta/rag_query1.py` (hybrid dense+sparse) is the sole query engine — it's what both interfaces and every eval script call. (An older dense-only `rag_query.py` existed previously but was removed as dead code; it was never actually wired into either interface despite earlier docs claiming otherwise.)

Three vLLM servers back everything, started via `containers/env.sh` + `vllm serve` (see `docs/readme.md` §4 for exact invocations):
| Port | Model | Role |
|------|-------|------|
| 8001 | harrier-embed (`microsoft/harrier-oss-v1-0.6b`) | dense embeddings (ingest + query) |
| 8002 | qwen35-9b | answer generation, golden-set/ground-truth generation, RAGAS judge |
| 8003 | qwen35-4b | query normalization/reformulation, golden-set filter judge |

Qdrant runs on port 6333 (Podman, CPU-only). Collection name `procedimientos_sielse` is hardcoded across `ingest_qdrant.py`, `rag_query1.py`, and the eval scripts.

## Environment setup

```bash
source ~/rag312_else/conda/activar.sh
conda activate rag312          # Python 3.12
source ~/rag312_else/containers/env.sh   # sets VLLM_SIF, HF_HOME, FASTEMBED_CACHE_DIR
```
`RAG_PROJECT_DIR` (default `~/rag312_else`) is read by every `tests/eval/*.py` script to locate `consulta/`, `ingesta/`, `datos/`, `salida_md/`. Set it if the checkout isn't at `~/rag312_else`.

Dependencies: `pip install -r requirements.txt` (pinned, includes `docling`, `langchain*`, `qdrant-client`, `fastapi`/`uvicorn`, `gradio`, `ragas`). No separate lint/format tooling is configured in this repo.

Then `pip install -e .` (from the repo root) to install the local `rag312/` package in editable mode — it's the only way `ingesta/`, `consulta/`, `interfaz/`, and `tests/eval/` scripts can `import rag312`. Re-run it whenever `pyproject.toml` changes; it doesn't need to be re-run when the scripts themselves change.

`rag312/` is a small shared-infrastructure package (`config.py` for settings/constants, `clients.py` for vLLM/Qdrant client factories, `utils.py` for misc helpers) — it holds no pipeline logic itself. `ingesta/`, `consulta/`, `interfaz/`, and `tests/eval/` remain plain script directories invoked the same way as always (`python ingesta/embeddings.py`, etc.), they just import shared config/clients from `rag312` instead of redeclaring constants per file.

## Evaluation (`tests/eval/`)

There is no unit test suite yet — `tests/unit/*.py` and `tests/integration/*.py` are empty placeholder files. Quality is measured instead by the eval scripts in `tests/eval/`, all run from that directory with `RAG_PROJECT_DIR` set and the vLLM/Qdrant services up. `docs/readme.md` §6 documents each one in depth; the short version:

- **`eval_retrieval.py`** — Recall@K/MRR against `golden_set.json`, no LLM judge, deterministic.
- **`eval_generation.py`** — runs the *full* `rag_query1.main()` pipeline per golden-set question and scores the answer with RAGAS (faithfulness, answer_relevancy, context_precision, context_recall), judged by the local qwen35-9b vLLM (not OpenAI/Anthropic — there's a stub-module patch at the top of the file to stop `ragas` from choking on a missing `ChatVertexAI` import). Only evaluates items with `ground_truth` + `ground_truth_revisado: true`. Generic over `--golden`/`--out`, so different golden sets can be evaluated without overwriting each other's results.
- **Golden-set generation has two independent origins that feed the same downstream pipeline**: `generar_golden_set.py` (questions anchored to one chunk from `datos/chunks_data.json`, stores `chunk_origen_index`) vs. `generar_golden_set_md.py` (questions generated from a full markdown section in `salida_md/`, stores the section text itself in `texto_seccion` since there's no single source chunk). Both then go through `filtrar_golden_set.py` (LLM-as-judge: autocontenida/respondible/natural → aceptada/dudosa/rechazada) → optionally `revisar_golden_set.py` (interactive manual review) → `consolidar_golden_set.py` (merges reviewed items into one final golden set, strips judge-internal fields). Downstream scripts that need the source text (`filtrar_golden_set.py`, `generar_ground_truth.py`, `revisar_*.py`) check `texto_seccion` first and fall back to a `chunk_origen_index` lookup — this is how the two origins stay compatible with the same tooling. `filtrar_golden_set.py --prefijo <name>` keeps a markdown-origin filter run from overwriting the chunk-origin one's `golden_set_{aceptadas,dudosas,rechazadas}.json`.
- **`generar_ground_truth.py` + `revisar_ground_truth.py`** — required before `eval_generation.py` can compute `context_precision`/`context_recall`, since those need a reference answer. `--auto-aprobar` on `generar_ground_truth.py` and `--muestra-aceptadas 0` on `filtrar_golden_set.py` skip the manual review steps entirely (trusting the LLM judge/generator) — useful for large batches, at the cost of no human check on what RAGAS treats as ground truth.
- **`inspeccionar_miss.py`** — re-runs retrieval for MISS cases from an `eval_retrieval.py` run to distinguish a real retrieval bug from a mislabeled golden-set entry.

## Interfaces (`interfaz/`)

- **`app_gradio.py`** (port 7861) — `gr.Chatbot` over `rag_query1.main()`, no real memory across turns.
- **`app_web.py`** (port 8080, FastAPI, in development) — custom HTML/CSS/JS frontend (`interfaz/static/`) over `rag_query1.main()` (the current hybrid engine). Routes: `GET /` serves `static/index.html`, `POST /api/chat` calls the RAG pipeline, `POST /api/upload` is a stub (file ingestion via the UI isn't implemented yet). Conversation history is kept client-side in `localStorage`, not server-side.

Both require the vLLM services (8001/8002/8003) and Qdrant (6333) running, same as any `rag_query*.py` call.
