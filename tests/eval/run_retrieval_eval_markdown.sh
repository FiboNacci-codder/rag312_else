#!/usr/bin/env bash
# Test de recuperación — golden set a nivel sección markdown (sufijo _md en todo)
# Uso: ./run_retrieval_eval_markdown.sh [n_secciones]   (default: 40)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

N_SECCIONES="${1:-40}"

python generar_golden_set_md.py --n-secciones "$N_SECCIONES" --por-categoria --out golden_set_md_raw.json
python filtrar_golden_set.py --golden golden_set_md_raw.json --prefijo golden_set_md --muestra-aceptadas 0
python consolidar_golden_set.py --archivos golden_set_md_aceptadas.json --out golden_set_md_final.json
python eval_retrieval.py --golden golden_set_md_final.json --out resultados_eval_md.json
