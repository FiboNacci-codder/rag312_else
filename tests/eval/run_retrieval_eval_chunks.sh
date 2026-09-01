#!/usr/bin/env bash
# Test de recuperación — golden set a nivel chunk (nombres por defecto: golden_set*.json)
# Uso: ./run_retrieval_eval_chunks.sh [n_chunks]   (default: 60)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

N_CHUNKS="${1:-60}"

python generar_golden_set.py --n-chunks "$N_CHUNKS" --por-categoria
python filtrar_golden_set.py --muestra-aceptadas 0
python consolidar_golden_set.py --archivos golden_set_aceptadas.json --out golden_set.json
python eval_retrieval.py --golden golden_set.json --out resultados_eval.json
