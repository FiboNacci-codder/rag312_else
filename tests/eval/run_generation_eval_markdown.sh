#!/usr/bin/env bash
# Test de generación (RAGAS) — golden set a nivel sección markdown (golden_set_md_final.json)
# Requiere golden_set_md_final.json ya consolidado (ver run_retrieval_eval_markdown.sh)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ ! -f golden_set_md_final.json ]]; then
    echo "ERROR: no existe golden_set_md_final.json. Corré primero ./run_retrieval_eval_markdown.sh" >&2
    exit 1
fi

python generar_ground_truth.py --golden golden_set_md_final.json --out golden_set_md_final.json --auto-aprobar
python eval_generation.py --golden golden_set_md_final.json --out resultados_generation_md.json
