# Ejecutar: reconstruir la base vectorial desde cero

Guía paso a paso para levantar el entorno y correr el pipeline de **ingesta**
completo (`biblioteca/*.pdf` → Qdrant) desde cero, en el servidor
`elsecluster5`, ahora que el proyecto vive en `~/rag312_else` (antes
`~/rag312`).

## 0. Antes de nada: actualizar el repo con `git pull`

`salida_md/`, `salida_chunks/`, `datos/` y los outputs de eval
(`tests/eval/resultados_*.json`, `filtrado_reporte.json`) dejaron de
trackearse en git (ahora están en `.gitignore`) porque son puramente
regenerables y causaban que `git pull` fallara cada vez que se corría el
pipeline en el servidor (`error: Your local changes ... would be
overwritten by merge`).

Si en el servidor esos archivos ya existen con cambios locales (de correr
`ocr_docling.py`/`embeddings.py`/algún eval antes de este cambio), hay que
descartarlos **antes** de tirar del pull — es seguro porque son regenerables
con el pipeline de este mismo documento:

```bash
git checkout -- salida_md/ salida_chunks/ datos/ \
  tests/eval/resultados_eval.json tests/eval/resultados_eval_md.json \
  tests/eval/resultados_generation.json tests/eval/resultados_generation_md.json \
  tests/eval/filtrado_reporte.json 2>/dev/null

git pull
```

Después del pull, esos archivos siguen físicamente en disco tal cual
estaban — solo dejan de estar bajo control de git. `git status` los va a
mostrar como ignorados, no como modificados. (`tests/eval/golden_set*.json`
— el dataset de preguntas curado a mano — sigue trackeado normalmente, no
se toca.)

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

## 3. Instalar el paquete `rag312` y las dependencias

Un `pip install -e .` a secas intenta resolver de una las ~300 dependencias
fijadas en `requirements.txt`, y ahí se choca con que `torch==2.6.0+cu124` /
`torchaudio==2.6.0+cu124` / `torchvision==0.21.0+cu124` no existen en PyPI
(solo están en `https://download.pytorch.org/whl/cu124`) — y encima el conda
base del servidor trae torch con CUDA 13, no la 12.4 que necesita este
proyecto. En vez de pelear con eso, conviene instalar de forma incremental:

**3.1. Registrar `rag312` en modo editable sin arrastrar sus dependencias**
(solo hace falta la primera vez, o cuando cambie `pyproject.toml`):

```bash
cd ~/rag312_else
pip install -e . --no-deps
```

**3.2. Arreglar torch/CUDA primero**, porque el conda base trae CUDA 13 y acá
se necesita CUDA 12.4:

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verificar:

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python3 -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

**3.3. Correr los scripts de la ingesta en orden** (`ingesta/ocr_docling.py`
→ `ingesta/embeddings.py` → `ingesta/ingest_qdrant.py`) e ir instalando lo
que falte a medida que aparezcan `ModuleNotFoundError` (`pip install
<paquete>` y volver a correr). Según lo que se ha necesitado hasta ahora:
`docling`, `langchain_core`, `easyocr`, `langchain_openai`, `fastembed`,
`qdrant_client`, `gradio`, `langchain_community`, `ragas` — pero esto es una
guía, no una checklist fija: instalar lo que pida cada traceback, que varía
según qué scripts se corran primero.

`requirements.txt` sigue existiendo como referencia de versiones conocidas
que funcionan, pero ya no se instala de una con `pip install -r
requirements.txt` / `pip install -e .` a secas.

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
