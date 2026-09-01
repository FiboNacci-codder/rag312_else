#!/usr/bin/env bash
# Test de recuperación — golden set a nivel chunk (nombres por defecto: golden_set*.json)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python generar_golden_set.py --n-chunks 60 --por-categoria
python filtrar_golden_set.py --muestra-aceptadas 0
python consolidar_golden_set.py --archivos golden_set_aceptadas.json --out golden_set.json
python eval_retrieval.py --golden golden_set.json --out resultados_eval.json
