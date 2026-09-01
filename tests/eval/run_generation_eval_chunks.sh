#!/usr/bin/env bash
# Test de generación (RAGAS) — golden set a nivel chunk (golden_set.json)
# Requiere golden_set.json ya consolidado (ver run_retrieval_eval_chunks.sh)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ ! -f golden_set.json ]]; then
    echo "ERROR: no existe golden_set.json. Corré primero ./run_retrieval_eval_chunks.sh" >&2
    exit 1
fi

python generar_ground_truth.py --golden golden_set.json --out golden_set.json --auto-aprobar
python eval_generation.py --golden golden_set.json --out resultados_generation.json
