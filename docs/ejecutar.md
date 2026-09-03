# Ejecutar: reconstruir la base vectorial desde cero

Guía paso a paso para levantar el entorno y correr el pipeline de **ingesta**
completo (`biblioteca/*.pdf` → Qdrant) desde cero, en el servidor
`elsecluster5`, ahora que el proyecto vive en `~/rag312_else` (antes
`~/rag312`).

## 1. Activar conda

```bash
source ~/rag312_else/conda/activar.sh
conda activate rag312          # Python 3.12
```

`conda/activar.sh` está en `.gitignore` (solo existe en el servidor, no en
este repo) — si adentro tiene alguna ruta hardcodeada a `~/rag312` (por
ejemplo la ubicación del propio entorno conda), hay que editarla ahí a mano;
no es algo que se pueda corregir desde este checkout.

## 2. Variables de entorno (vLLM, modelos, proyecto)

```bash
source ~/rag312_else/containers/env.sh   # VLLM_SIF, HF_HOME, FASTEMBED_CACHE_DIR
export RAG_PROJECT_DIR=~/rag312_else     # opcional: ya es el default en rag312/config.py, pero no está de más ser explícito
```

## 3. Instalar el paquete `rag312` en modo editable

Solo hace falta la primera vez, o cuando cambie `pyproject.toml`:

```bash
cd ~/rag312_else
pip install -e .
```

## 4. Levantar Qdrant (Podman)

```bash
mkdir -p ~/rag312_else/qdrant_storage   # si no existe todavía

# si ya hay un contenedor viejo apuntando a la ruta anterior:
podman stop qdrant && podman rm qdrant

podman run -d --name qdrant -p 6333:6333 \
  -v ~/rag312_else/qdrant_storage:/qdrant/storage:Z \
  docker.io/qdrant/qdrant
```

El flag `:Z` es para SELinux (RHEL 9.4) — sin él, Podman puede no poder
escribir en el volumen montado.

## 5. Levantar el vLLM de embeddings (harrier-embed, puerto 8001)

Es el único de los tres servidores vLLM que hace falta para la ingesta
(`embeddings.py`). Los otros dos (`qwen35-9b`/8002, `qwen35-4b`/8003) solo
hacen falta después, para consultar (ver `docs/readme.md` §4 para los tres
comandos completos):

```bash
CUDA_VISIBLE_DEVICES=6 apptainer exec --nv \
  --env HF_HOME="$HF_HOME" \
  "$VLLM_SIF" \
  vllm serve microsoft/harrier-oss-v1-0.6b \
    --runner pooling \
    --convert embed \
    --served-model-name harrier-embed \
    --host 0.0.0.0 \
    --port 8001 \
    --gpu-memory-utilization 0.20
```

## 6. Correr el pipeline de ingesta, en orden

Desde `ingesta/` (o con rutas completas):

```bash
cd ~/rag312_else/ingesta

python ocr_docling.py     # biblioteca/*.pdf -> salida_md/, salida_chunks/, datos/chunks_data.json
                           # (no necesita Qdrant ni vLLM arriba, solo GPU + HF_HOME)

python embeddings.py      # datos/chunks_data.json -> datos/embeddings_data.json
                           # (necesita el vLLM de embeddings del paso 5, puerto 8001)

python ingest_qdrant.py   # datos/embeddings_data.json -> colección "procedimientos_sielse" en Qdrant
                           # (necesita Qdrant del paso 4, puerto 6333)
```

`ingest_qdrant.py` **borra y recrea** la colección en cada corrida — es
justo el comportamiento que hace falta para reconstruir "desde cero", no un
efecto secundario a evitar.

## 7. Verificar

Al final de `ingest_qdrant.py` ya se imprime el conteo de puntos insertados.
Para confirmarlo aparte:

```bash
python -c "
from rag312.clients import build_qdrant_client
from rag312.config import settings
c = build_qdrant_client()
print(c.count(settings.collection_name))
"
```

Si da un `count` mayor a 0, la base vectorial quedó reconstruida. A partir
de ahí ya se puede levantar el vLLM generador (8002) y el normalizador
(8003) y probar una consulta real con `python consulta/rag_query1.py "pregunta de prueba"`.
