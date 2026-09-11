#!/usr/bin/env bash
# Lanza ocr_docling_granite.py en paralelo, un proceso por GPU (una partición
# de la lista de PDFs por proceso), y al terminar fusiona los resultados en
# datos/metricas_ocr_granite.csv y datos/chunks_data_granite.json.
#
# Uso: bash ingesta/lanzar_granite_paralelo.sh [args extra para ocr_docling_granite.py]
# Ejemplo (prueba chica): bash ingesta/lanzar_granite_paralelo.sh --limite 8
#
# Las GPUs vienen de rag312.config.settings.cuda_visible_devices_ocr
# ("4,5,6,7" por defecto). Override: GRANITE_GPUS="4,5" bash ingesta/lanzar_granite_paralelo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS_STR="${GRANITE_GPUS:-$(python3 -c 'from rag312.config import settings; print(settings.cuda_visible_devices_ocr)')}"
IFS=',' read -ra GPUS <<< "$GPUS_STR"
N=${#GPUS[@]}

mkdir -p datos
echo "Lanzando $N procesos en paralelo (GPUs: ${GPUS[*]})..."

pids=()
for i in "${!GPUS[@]}"; do
    gpu="${GPUS[$i]}"
    log="datos/log_granite_gpu${gpu}.log"
    echo "  partición $i -> GPU $gpu (log: $log)"
    python ingesta/ocr_docling_granite.py --gpu "$gpu" --particiones "$N" --particion "$i" "$@" \
        > "$log" 2>&1 &
    pids+=($!)
done

fallo=0
for pid in "${pids[@]}"; do
    wait "$pid" || fallo=1
done

if [ "$fallo" -ne 0 ]; then
    echo "Al menos un proceso terminó con error — revisá los logs en datos/log_granite_gpu*.log antes de fusionar."
    exit 1
fi

echo "Todas las particiones terminaron OK. Fusionando..."
python ingesta/fusionar_particiones_granite.py --particiones "$N"
