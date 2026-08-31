# Proyecto RAG — Procedimientos SIELSE — Resumen de contexto (v5)

Documento de contexto para retomar el proyecto sin código. 

---

## 1. Objetivo del proyecto

RAG de producción sobre procedimientos internos de ELSE. Los usuarios preguntan en lenguaje natural y el sistema responde citando el documento/página/sección fuente real.

---

## 2. Hardware y entorno

- **Servidor:** `elsecluster5` — Lenovo ThinkSystem SR675 V3, RHEL 9.4, gestionado con LICO.
- **8x NVIDIA L40S (46GB c/u)**, driver 550.90.07 (soporta hasta CUDA 12.4).
- Usuario `u23u`, **sin sudo real** (restricción LICO).
- **Apptainer con `--nv`** para todo lo que necesita GPU (no requiere sudo). Podman solo para Qdrant (sin GPU).
- Imagen usada: `vllm-v0271.sif` 
- Python del sistema **3.12**, con `Conda`.
- Servicios corren en terminales SSH manuales (no tmux) → se caen si se cierra la sesión. **Pendiente:** persistencia con systemd/tmux.
- Regla clave vLLM: `--gpu-memory-utilization` es fracción de la memoria **total** de la GPU, no de la libre.

---

## 3. Estructura de directorios (confirmada)

```
~/rag312/
├── biblioteca/                               ← Lugar donde se encuentra los pdf's
├── conda/                                    ← Se encuentra el activador de conda
├── consulta/                                 ← Se encuentran los archivos para realizar la parte de la consulta en el rag (normalizar_query.py y rag_query1.py) 
├── containers/                               ← Se encuentra el .sh para activar las variables de entorno de los vllm y tambien el archivo vllm 
├── datos/                                    ← Se encuentra los archivos de los resultados como chunks o metricas
├── docs/                                     ← Se encuentra 2 archovs readme.md (Documentacion breve del proyecto) y agregar donde se encuentra las tareas pendientes
├── ingesta/                                  ← Se encuentran 3 archivos para la creacion de la base de datos vectorial  
├── interfaz/                                 ← Se encuentra el codigo de interfaz en gradio para las consultas  
├── modelos/                                  ← Se encuentran almacenado los modelos que se utilizan para embedding y para los llm's
├── salida_chunks/                            ← Se encuentran los chunks de la ejecucion del archivo ocr_docling.py  en el mismo orden que biblioteca
├── salida_md/                                ← Se encuentranl los archivos de resultado de la ejecucion de ocr_docling.py  en el mismo orden que biblioteca
├── qdrant_storage/                           ← Se encuentra los archivos almacenados de la base de datos vectorial  
└── requirements.txt                          ← Se encuentran las librerias actuales para correr el proyecto
```

## Entorno de CONDA
Para activar el entorno de CONDA se debe de activar con los siguientes comandos el cual hace que se tenga la version de python 3.12
- source ~/rag312/conda/activar.sh 
- conda activate rag312

## 4. Servicios vLLM 

Para iniciar los servicios vLLM se debe de iniciar con la variables de entorno se utiliza el siguiente comando 

- source ~/rag312/containers/env.sh



```bash
# Embeddings — Harrier (GPU 6, puerto 8001)
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

# Qwen 3.5 9B - generador de las respuestas (GPU 5, puerto 8002)
CUDA_VISIBLE_DEVICES=5 apptainer exec --nv \
  --env HF_HOME="$HF_HOME" \
  "$VLLM_SIF" \
  vllm serve Qwen/Qwen3.5-9B \
    --served-model-name qwen35-9b \
    --host 0.0.0.0 \
    --port 8002 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 16384 \
    --reasoning-parser qwen3 \
    --language-model-only

# Qwen 3.5 4B — normalizador de preguntas (GPU 4, puerto 8003)
CUDA_VISIBLE_DEVICES=4 apptainer exec --nv \
  --env HF_HOME="$HF_HOME" \
  "$VLLM_SIF" \
  vllm serve Qwen/Qwen3.5-4B \
    --served-model-name qwen35-4b \
    --host 0.0.0.0 \
    --port 8003 \
    --gpu-memory-utilization 0.5 \
    --max-model-len 8192 \
    --reasoning-parser qwen3 \
    --language-model-only
```

---

## 5. Estructura de datos 
biblioteca/
└── procedimientos
    ├── Atencion al Cliente
    ├── Clientes Mayores
    ├── Cobranzas
    ├── Facturacion
    ├── Instalaciones
    ├── Marketing
    ├── Perdidas Comerciales
    ├── Relaciones Públicas
    └── Supervision Comercial

conda
├── activar.sh
├── miniconda3
└── Miniconda3-latest-Linux-x86_64.sh

consulta/
├── normalizar_query.py
├── rag_query1.py
└── system_prompt.txt

containers/
├── env.sh
└── vllm-v0271.sif

datos/
├── chunks_data.json
├── embeddings_data.json
├── metricas_embeddings.csv
└── metricas_ocr.csv

docs
├── agregar.md
└── readme.md

ingesta/
├── embeddings.py
├── ingest_qdrant.py
└── ocr_docling.py
    
interfaz/
└── app_gradio.py

modelos
├── fastembed_cache
└── hf_cache

qdrant_storage/
├── aliases
├── collections
└── raft_state.json

/home/u23u/rag312/tests/
├── eval
    ├── consolidar_golden_set.py
    ├── eval_generation.py
    ├── eval_retrieval.py
    ├── filtrado_reporte.json
    ├── filtrar_golden_set.py
    ├── generar_golden_set_md.py
    ├── generar_golden_set.py
    ├── golden_set_aceptadas.json
    ├── golden_set_dudosas.json
    ├── golden_set.json
    ├── golden_set_raw_59.json
    ├── golden_set_rechazadas.json
    ├── inspeccionar_miss.py
    ├── resultados_eval.json
    └── revisar_golden_set.py

salida_chunks/
└── procedimientos
    ├── Atencion al Cliente
    ├── Clientes Mayores
    ├── Cobranzas
    ├── Facturacion
    ├── Instalaciones
    ├── Marketing
    ├── Perdidas Comerciales
    ├── Relaciones Públicas
    └── Supervision Comercial

salida_md/
└── procedimientos
    ├── Atencion al Cliente
    ├── Clientes Mayores
    ├── Cobranzas
    ├── Facturacion
    ├── Instalaciones
    ├── Marketing
    ├── Perdidas Comerciales
    ├── Relaciones Públicas
    └── Supervision Comercial

---
## Funcionamiento de los programas

### Resumen de `ocr_docling.py`

Descripción General
Script de procesamiento OCR y extracción de documentos PDF que utiliza **Docling** para convertir archivos PDF en markdown estructurado, generar chunks semánticos y preparar documentos para sistemas RAG (Retrieval-Augmented Generation).

Flujo de Procesamiento

### **Input**
- **PDFs**: Archivos en la carpeta `biblioteca/` (búsqueda recursiva de `.pdf`)
- **Configuración**: GPU CUDA (dispositivos 4-7), OCR multilingüe (español/inglés)

### **Procesamiento por PDF**
1. **Conversión Docling**: 
   - OCR completo (EasyOCR)
   - Extracción de estructura de tablas
   - Exportación a formato markdown

2. **Chunking Inteligente**:
   - Tokenizador: `microsoft/harrier-oss-v1-0.6b`
   - Límite: 2048 tokens por chunk
   - Preserva jerarquía de secciones y metadatos

3. **Generación de Metadatos**:
   - Número de páginas
   - Sección jerárquica (` > ` separador)
   - Tipo de contenido (texto, tabla, mixto)
   - Categoría (nombre de subcarpeta)
   - Ruta relativa en biblioteca

### **Outputs**

| Archivo | Ubicación | Contenido |
|---------|-----------|-----------|
| **Markdown** | `salida_md/` (estructura espejo) | Documento completo en formato markdown con estructura preservada |
| **Chunks TXT** | `salida_chunks/` (estructura espejo) | Archivo por PDF con chunks numerados y metadatos |
| **CSV Métricas** | `datos/metricas_ocr.csv` | Tiempo de procesamiento y páginas por PDF |
| **JSON Chunks** | `datos/chunks_data.json` | Lista de objetos Document (page_content + metadata) |

### **Estructura de Output**
```
rag/
├── biblioteca/           # Input PDFs (con subcarpetas)
├── salida_md/           # Markdowns espejo
├── salida_chunks/       # Chunks TXT espejo  
└── datos/
    ├── metricas_ocr.csv
    └── chunks_data.json  # Datos preparados para embeddings/Qdrant
```

### **Formato Document (LangChain)**
Cada chunk se almacena como:
```python
Document(
    page_content=chunk.text,
    metadata={
        "source": "nombre.pdf",
        "chunk_index": 0,
        "num_chunks": 10,
        "paginas": [1, 2],
        "seccion": "Introducción > Objetivos",
        "tipo": "texto" | "tabla" | "mixto",
        "categoria": "ciencia_datos",
        "ruta_biblioteca": "subcarpeta/nombre.pdf"
    }
)
```

### **Manejo de Errores**
- Captura excepciones por PDF
- Registra tiempo de fallo en CSV
- Continúa con siguiente documento

### **Requisitos** (no explicitados en código)
- `docling` con dependencias OCR
- `langchain_core`
- `transformers`
- GPU CUDA para aceleración

### **Salida Final**
Retorna lista de objetos `Document` lista para:
- Ingesta en Qdrant/vectorstore
- Embeddings (el script externo)
- Sistema RAG completo

## Nota para el Chatbot
Este script es el **primer paso del pipeline RAG**: convierte PDFs raw en chunks semánticos enriquecidos con metadatos jerárquicos, listos para indexación vectorial. La estructura espejo permite trazabilidad del documento original y sus secciones.


# Resumen de `embeddings.py`

## Descripción General
Script de generación de **embeddings híbridos** (densos + dispersos) para chunks de documentos previamente procesados. Conecta con un servidor vLLM para obtener embeddings densos y utiliza FastEmbed para generar representaciones sparse (BM25) para búsqueda híbrida en Qdrant.

## Flujo de Procesamiento

### **Input**
- **Chunks**: Archivo JSON `datos/chunks_data.json` generado por `ocr_docling.py`
- **Servidor vLLM**: Endpoint local `http://localhost:8001/v1` con modelo `harrier-embed`
- **Modelo Sparse**: `Qdrant/bm25` con soporte para español

### **Procesamiento por Lotes**
1. **Carga de Chunks**:
   - Lee JSON con objetos Document (page_content + metadata)
   - Convierte a lista de objetos `Document` de LangChain

2. **Embedding Denso (vLLM)**:
   - Envía lotes de texto al servidor vLLM
   - Tamaño de batch configurable (16 por defecto)
   - Utiliza API compatible con OpenAI

3. **Embedding Sparse (FastEmbed)**:
   - Procesa los mismos lotes
   - Genera representaciones BM25 con índices y valores
   - Cachea modelos en `~/rag/modelos/fastembed_cache`

4. **Métricas por Chunk**:
   - Caracteres y palabras
   - Dimensión del vector denso
   - Norma del vector (sanity check)
   - Términos no-cero en vector sparse

### **Outputs**

| Archivo | Ubicación | Contenido |
|---------|-----------|-----------|
| **CSV Métricas** | `datos/metricas_embeddings.csv` | Métricas por chunk: tamaño, dim vector, norma, términos sparse |
| **JSON Vectores** | `datos/embeddings_data.json` | Datos completos: texto, metadata, embedding denso, sparse (índices+valores) |

### **Estructura de Output JSON**
```json
[
  {
    "page_content": "texto del chunk",
    "metadata": {
      "source": "doc.pdf",
      "chunk_index": 0,
      "paginas": [1, 2],
      "seccion": "Introducción",
      ...
    },
    "embedding": [0.123, -0.456, ...],        // vector denso
    "sparse_indices": [5, 23, 67, ...],       // índices BM25
    "sparse_values": [0.789, 0.234, ...]      // valores BM25
  }
]
```

### **Manejo de Errores**
- Captura excepciones por batch
- Registra errores en métricas (campo "ERROR")
- Continúa con siguientes lotes

### **Estadísticas Generadas**
- Total chunks procesados
- Tasa de éxito/error
- Tiempo total y promedio por chunk
- Throughput (chunks/segundo)

## Relación con Pipeline RAG

**Posición en el flujo**:
1. `ocr_docling.py` → Genera chunks semánticos
2. **`embeddings.py`** → Convierte chunks a embeddings
3. (Siguiente paso) → Indexación en Qdrant

**Preparado para**:
- Búsqueda híbrida (dense + sparse)
- Almacenamiento en Qdrant (vectores densos y sparse)
- Métricas de calidad de embeddings

## Requisitos Técnicos
- **Servidor vLLM**: Puerto 8001 con modelo de embeddings
- **Dependencias**: `langchain-openai`, `fastembed`, `numpy`
- **GPU**: Recomendada (vLLM en CUDA)

## Nota para el Chatbot
Este script es el **segundo paso crítico** en el pipeline RAG híbrido. Convierte texto en representaciones vectoriales duales (densas y dispersas) que permiten búsquedas semánticas y de palabras clave simultáneamente. El JSON final contiene todo lo necesario para indexar en Qdrant sin reprocesar los chunks.

# Resumen de `ingest_qdrant.py`

## Descripción General
Script de **indexación híbrida en Qdrant** que almacena vectores densos y dispersos (BM25) en una colección configurada para búsqueda híbrida. Toma los datos generados por `embeddings.py` y los inserta en la base de datos vectorial.

## Flujo de Procesamiento

### **Input**
- **Datos vectoriales**: Archivo JSON `datos/embeddings_data.json` generado por `embeddings.py`
- **Qdrant**: Servidor local en `localhost:6333`

### **Procesamiento**
1. **Carga de Datos**:
   - Lee JSON con embeddings densos y sparse
   - Verifica integridad de los datos

2. **Gestión de Colección**:
   - Verifica si la colección existe
   - **Elimina y recrea** la colección (destructivo)
   - Configura colección con dos vector spaces:
     - **Dense**: 1024 dimensiones, distancia COSINE
     - **Sparse**: BM25 automático

3. **Preparación de Puntos**:
   - Mapea cada registro a `PointStruct`
   - ID autoincremental (posición en lista)
   - Vector híbrido: {dense + sparse}
   - Payload: texto completo + metadatos

4. **Inserción en Qdrant**:
   - Upsert batch de todos los puntos
   - Registra tiempo de inserción
   - Verifica conteo final en colección

### **Output**
- **Colección Qdrant**: `procedimientos_sielse` con puntos indexados
- **Métricas en consola**: 
  - Tiempo de inserción
  - Cantidad de puntos insertados
  - Conteo final en colección

### **Estructura en Qdrant**

| Componente | Tipo | Configuración |
|------------|------|---------------|
| **Colección** | `procedimientos_sielse` | Nombre fijo |
| **Vector Dense** | `dense` | 1024 dim, COSINE |
| **Vector Sparse** | `bm25` | BM25 automático |
| **Payload** | JSON | `page_content` + todos los metadatos |

### **Payload Almacenado**
```json
{
  "page_content": "texto del chunk",
  "source": "doc.pdf",
  "chunk_index": 0,
  "num_chunks": 10,
  "paginas": [1, 2],
  "seccion": "Introducción > Objetivos",
  "tipo": "texto",
  "categoria": "ciencia_datos",
  "ruta_biblioteca": "subcarpeta/nombre.pdf"
}
```

## Características Importantes

### ⚠️ **Comportamiento Destructivo**
- **Elimina la colección existente** antes de recrearla
- No conserva datos previos
- Ideal para procesos de recarga completa

### **Búsqueda Híbrida**
- Permite búsquedas que combinan:
  - **Semántica**: Vector denso (similaridad COSINE)
  - **Palabras clave**: Vector sparse BM25
  - **Fusión**: Qdrant puede combinar ambos resultados

## Relación con Pipeline RAG

**Flujo completo**:
1. `ocr_docling.py` → Extrae texto y genera chunks
2. `embeddings.py` → Crea embeddings híbridos
3. **`ingest_qdrant.py`** → Indexa en Qdrant
4. (Siguiente paso) → Búsqueda y generación de respuestas

**Preparado para**:
- Consultas híbridas (dense + sparse)
- Filtrado por metadatos (fuente, categoría, sección)
- Retrieval de chunks relevantes para RAG

## Requisitos Técnicos
- **Qdrant**: Servidor ejecutándose en localhost:6333
- **Dependencias**: `qdrant-client`
- **Memoria**: Suficiente para cargar todos los embeddings

## Nota para el Chatbot
Este script completa la **fase de ingesta del pipeline RAG**. La colección resultante permite búsquedas híbridas que aprovechan lo mejor de la semántica (embeddings densos) y la precisión léxica (BM25). El payload enriquecido con metadatos jerárquicos permite filtrado avanzado durante la recuperación, mejorando la relevancia de los chunks recuperados para el sistema RAG.

## Fase de Consulta
# Resumen de `normalizarquery.py`

## Descripción General
Script de **preprocesamiento de consultas** para el sistema RAG. Mejora la calidad de las preguntas de los usuarios mediante corrección ortográfica y reformulación a lenguaje formal/técnico, optimizando así la búsqueda en Qdrant.

## Flujo de Procesamiento

### **Input**
- **Texto del usuario**: Pregunta en lenguaje natural (coloquial, con posibles errores)
- **Servidor LLM**: `http://localhost:8003/v1` con modelo `qwen35-4b`

### **Pipeline de Procesamiento**

1. **Normalización Básica**:
   - Elimina espacios extra (strip y compresión de espacios múltiples)
   - Preparación para etapas posteriores

2. **Corrección Ortográfica** (opcional):
   - Usa LLM para corregir errores de tipeo y ortografía
   - Preserva tildes, mayúsculas al inicio de oración
   - **No cambia** significado ni estructura
   - Temperatura 0.0 (determinístico)

3. **Reformulación a Lenguaje Formal** (opcional):
   - Convierte lenguaje coloquial a técnico/institucional
   - Ejemplo: "me llegó carísimo el recibo" → "reclamo por consumo excesivo"
   - Temperatura 0.2 (flexibilidad controlada)
   - **Mantiene** el significado exacto

### **Output**
Diccionario con tres versiones del texto:
```python
{
  "original": "me llego carisimo el recibo",      # Texto original
  "corregida": "Me llegó carísimo el recibo",    # Solo ortografía corregida
  "busqueda": "reclamo por consumo excesivo"     # Versión final para búsqueda
}
```

## Configuración de LLM

| Parámetro | Corrección | Reformulación |
|-----------|------------|---------------|
| **Temperatura** | 0.0 | 0.2 |
| **Max Tokens** | 1024 | 1024 |
| **System Prompt** | Corrección ortográfica | Reformulación formal |
| **enable_thinking** | False | False |

## Características Clave

### 🎯 **Propósito Dual**
- **Corrección**: Mejora la precisión léxica
- **Reformulación**: Alinea la consulta con el lenguaje de los documentos

### 🔄 **Fallback Seguro**
- Si el LLM no devuelve respuesta, usa el texto original
- Evita fallos completos en el pipeline

### 📝 **Transparencia**
- Imprime en consola los cambios aplicados
- Facilita depuración y seguimiento

## Relación con Pipeline RAG

**Posición en el flujo de consulta**:
1. Usuario escribe pregunta
2. **`normalizar_query.py`** → Limpia y optimiza la consulta

**Beneficios**:
- **Corrección**: Evita fallos en búsqueda por errores ortográficos
- **Reformulación**: Mejora el matching semántico con los documentos técnicos
- **Versiones separadas**: Permite comparar y ajustar estrategias

## Ejemplo de Uso

```python
# Entrada: pregunta informal con error
pregunta = "como ago para reclamar mi recivo de luz?"

# Salida del procesamiento
{
  "original": "como ago para reclamar mi recivo de luz?",
  "corregida": "Cómo hago para reclamar mi recibo de luz?",
  "busqueda": "procedimiento para reclamar factura de servicio eléctrico"
}
```

## Requisitos Técnicos
- **Servidor vLLM**: Puerto 8003 con modelo Qwen 3.5 4B
- **Dependencias**: `openai`, `re`
- **Modo CLI**: Puede ejecutarse con argumento o entrada interactiva

## Nota para el Chatbot
Este script es **crítico para la calidad del sistema RAG**. La reformulación a lenguaje técnico alinea las consultas de los usuarios con el vocabulario de los documentos institucionales, mejorando significativamente la recuperación. La separación entre versión corregida y reformulada permite elegir la estrategia óptima según el caso de uso. La corrección ortográfica previene que errores comunes en la escritura de usuarios afecten la búsqueda semántica.

# Resumen de `rag_query1.py`

## Descripción General
Script principal de **consulta RAG híbrida** que procesa preguntas de usuarios, recupera chunks relevantes usando búsqueda híbrida (dense + sparse) en Qdrant, y genera respuestas usando un LLM. Es el punto de entrada para el sistema de preguntas y respuestas.

## Flujo de Procesamiento

### **Input**
- **Pregunta del usuario**: Texto en lenguaje natural
- **Preprocesamiento**: Usa `normalizar_query.procesar_pregunta()` para corrección y reformulación
- **Servidores**:
  - Embeddings: `http://localhost:8001/v1` (harrier-embed)
  - LLM: `http://localhost:8002/v1` (qwen35-9b)
  - Qdrant: `localhost:6333`

### **Pipeline de Consulta**

1. **Preprocesamiento de Query**:
   - Normaliza espacios
   - Corrige ortografía (opcional)
   - Reformula a lenguaje técnico/formal

2. **Generación de Embeddings Híbridos**:
   - **Denso**: Añade prefijo instructivo: `"Instruct: Retrieve relevant passages that answer the query\nQuery: "`
   - **Sparse**: Genera vector BM25 de la pregunta original

3. **Búsqueda en Qdrant (Híbrida)**:
   - **Prefetch separado**: Busca top-20 con dense y top-20 con sparse
   - **Fusión RRF**: Combina resultados para obtener top-10 final
   - **Cálculo de scores**: 
     - Registra rank y score en cada rama
     - Calcula similitud COSINE para puntos fuera del top-20 denso

4. **Generación de Respuesta**:
   - Construye prompt con contexto recuperado
   - Usa system prompt personalizado (cargado desde `system_prompt.txt`)
   - Genera respuesta con LLM (temperatura 0.2)

### **Output**
Diccionario completo con:
```python
{
  "respuesta": "Texto generado por el LLM",
  "fuentes": [
    {
      "documento": "nombre.pdf",
      "pagina": [1, 2],
      "seccion": "Introducción",
      "ruta": "subcarpeta/nombre.pdf",
      "chunk": "texto del chunk",
      "rank_fusion": 1,
      "rank_dense": 2,
      "similitud_dense": 85.7,
      "fuera_top20_dense": false,
      "rank_bm25": 1,
      "score_bm25": 0.8765
    }
  ],
  "tiempo_retrieval": 0.45,
  "tiempo_generacion": 1.23
}
```

## Configuración Clave

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| TOP_K | 10 | Número final de chunks recuperados |
| PREFETCH_LIMIT | 20 | Top-K por cada rama (dense/sparse) |
| UMBRAL_SIMILITUD | 0.5 | Filtro opcional por score dense |
| Temperatura LLM | 0.2 | Balance entre creatividad y determinismo |
| Max tokens LLM | 800 | Longitud máxima de respuesta |

## Características Destacadas

### 🔍 **Búsqueda Híbrida Avanzada**
- **Dos ramas paralelas**: Dense (semántica) + Sparse (BM25 léxico)
- **Fusión RRF**: Reciproca Rank Fusion para combinar resultados
- **Métrica completa**: Registra scores y ranks de cada rama

### 📊 **Métricas Detalladas por Fuente**
- Rank en fusión final
- Rank en búsqueda densa y sparse
- Similitud COSINE del vector denso
- Score BM25
- Indica si el chunk estaba fuera del top-20 denso

### 🧠 **Sistema Prompt Personalizado**
- Carga system prompt desde archivo externo (`system_prompt.txt`)
- Instrucciones claras: solo usar contexto, citar fuentes
- Personalidad definida ("Foquito")

### 🔄 **Preprocesamiento Integrado**
- Usa el módulo `normalizar_query` para:
  - Corrección ortográfica
  - Reformulación a lenguaje técnico
- Versión original vs reformulada para diferentes usos

## Relación con Pipeline RAG

**Flujo completo**:
1. **Ingesta**: `ocr_docling.py` → `embeddings.py` → `ingest_qdrant.py`
2. **Consulta**: 
   - Usuario pregunta
   - **`rag_query1.py`** → Procesa, recupera y responde
   - Retorna respuesta + fuentes + métricas

**Diferencias con normalizar_query.py**:
- `normalizar_query.py`: Solo preprocesa
- `rag_query1.py`: Pipeline completo (preprocesa + recupera + genera)

## Ejemplo de Flujo

```bash
# Entrada
python rag_query1.py "como ago para reclamar mi recivo de luz?"

# Procesamiento:
1. Normaliza: "como ago para reclamar mi recivo de luz?"
2. Corrige: "Cómo hago para reclamar mi recibo de luz?"
3. Reformula: "procedimiento para reclamo de factura de servicio eléctrico"
4. Genera embeddings de la reformulada
5. Busca en Qdrant (híbrido)
6. Genera respuesta con LLM

# Salida:
Respuesta: "Para reclamar su factura de servicio eléctrico, debe..."
Fuentes: [documento1.pdf, documento2.pdf, ...]
```

## Requisitos Técnicos
- **Servidores vLLM**: Puerto 8001 (embeddings) y 8002 (LLM)
- **Qdrant**: Puerto 6333 con colección `procedimientos_sielse`
- **Archivo externo**: `system_prompt.txt` en carpeta `consulta/`
- **Dependencias**: `qdrant-client`, `langchain-openai`, `openai`, `numpy`

## Nota para el Chatbot
Este es el **script central del sistema RAG**. Combina todas las piezas:
- **Preprocesamiento** inteligente de queries
- **Búsqueda híbrida** con fusión RRF
- **Generación** contextualizada con citas

La riqueza de métricas en las fuentes permite:
- Auditar la calidad de la recuperación
- Depurar problemas de búsqueda
- Entender qué tipo de búsqueda (dense vs sparse) contribuye más

La separación entre pregunta original y reformulada permite elegir estrategias diferentes:
- **Original**: Para mostrar al usuario
- **Reformulada**: Para búsqueda semántica


## 6. Evaluación (`tests/eval/`)

Los scripts de esta carpeta miden la calidad del sistema una vez indexado
en Qdrant. Todos se ejecutan desde `tests/eval/` con el entorno conda
`rag312` activo, y respetan la variable `RAG_PROJECT_DIR` (default
`~/rag312`) para ubicar `consulta/` e `ingesta/`.

### `eval_retrieval.py` — calidad de retrieval (Recall@K, MRR)

Corre cada pregunta del `golden_set.json` contra Qdrant usando
`recuperar_contexto()` (la misma función que `rag_query1.py`) y calcula
Recall@K y MRR para dense, sparse (BM25) y la fusión RRF final. **No usa
RAGAS** — es una métrica determinística de "¿apareció el documento
esperado en el top-K?", sin LLM juez de por medio.

Por defecto normaliza/reformula cada pregunta con
`normalizar_query.procesar_pregunta()` antes de buscar, igual que hace
`rag_query1.main()` en producción — así el eval mide el pipeline real, no
uno distinto. `--sin-normalizar` permite medir retrieval "puro" para
distinguir si una falla viene del retrieval en sí o de la reformulación.

Suma respecto a la versión inicial: recall/MRR desglosados por
`categoria`, conteo de errores por pregunta (ya no desaparecen del
resumen), tiempos de ejecución, y un aviso si `--k` supera el `TOP_K` real
configurado en `rag_query1.py`.

```bash
python eval_retrieval.py                    # normalizado (igual que producción)
python eval_retrieval.py --sin-normalizar    # retrieval puro, sin LLM normalizador
python eval_retrieval.py --k 7 --out resultados_eval.json
```

Requiere: vLLM embeddings (8001), vLLM normalizador (8003, salvo con
`--sin-normalizar`) y Qdrant (6333), con la colección
`procedimientos_sielse` ya indexada.

### Generación del golden set — nivel chunk vs. nivel sección markdown

Hay dos formas de generar las **preguntas** del golden set, que después
pasan por el mismo pipeline de filtrado/revisión:

- **`generar_golden_set.py`** (original, nivel chunk): genera preguntas
  ancladas a un chunk puntual de `datos/chunks_data.json`. `fuente_esperada`
  queda 100% alineada con lo indexado en Qdrant porque la pregunta se generó
  a partir de ese chunk exacto. Guarda `chunk_origen_index` para poder
  reconstruir el fragmento de origen en los pasos siguientes.
- **`generar_golden_set_md.py`** (nivel sección): parte cada `.md` de
  `salida_md/` en secciones por heading y genera preguntas por sección
  **completa**, no por chunk. Útil para preguntas que necesitan ver más
  contexto que un solo chunk (procedimientos con pasos repartidos en varios
  chunks, tablas largas). Como no hay un chunk único de origen, cada item
  guarda directamente el texto de la sección en el campo `texto_seccion`
  (en vez de `chunk_origen_index`) — así los pasos siguientes no dependen de
  volver a parsear `salida_md/` ni de un matching por heading.

Ambos alimentan el **mismo** pipeline de filtrado y revisión:

```
generar_golden_set.py / generar_golden_set_md.py
    → filtrar_golden_set.py       (juez LLM: autocontenida / respondible / natural)
    → revisar_golden_set.py       (revisión manual de dudosas/rechazadas, opcional)
    → consolidar_golden_set.py    (une los 3 archivos en un golden set final)
```

`filtrar_golden_set.py` reconoce automáticamente si un item tiene
`texto_seccion` (nivel sección) o `chunk_origen_index` (nivel chunk) para
resolver el fragmento a evaluar; `revisar_golden_set.py` hace lo mismo al
mostrar el contexto. Tiene un flag `--prefijo` para que las salidas
(`{prefijo}_aceptadas.json`, `{prefijo}_dudosas.json`,
`{prefijo}_rechazadas.json`) no pisen las del golden set de chunks:

```bash
# Golden set a nivel chunk (comportamiento original)
python generar_golden_set.py --n-chunks 60 --por-categoria
python filtrar_golden_set.py                       # usa golden_set.json y prefijo "golden_set" por defecto
python revisar_golden_set.py --golden golden_set_dudosas.json
python revisar_golden_set.py --golden golden_set_rechazadas.json
python revisar_golden_set.py --golden golden_set_aceptadas.json --solo-pendientes
python consolidar_golden_set.py --out golden_set.json

# Golden set a nivel sección markdown (archivos separados, no pisa lo anterior)
python generar_golden_set_md.py --n-secciones 40 --por-categoria --out golden_set_md_raw.json
python filtrar_golden_set.py --golden golden_set_md_raw.json --prefijo golden_set_md
python revisar_golden_set.py --golden golden_set_md_dudosas.json
python revisar_golden_set.py --golden golden_set_md_rechazadas.json
python revisar_golden_set.py --golden golden_set_md_aceptadas.json --solo-pendientes
python consolidar_golden_set.py \
    --archivos golden_set_md_aceptadas.json golden_set_md_dudosas.json golden_set_md_rechazadas.json \
    --out golden_set_md_final.json
```

**Para saltar la revisión manual de preguntas** (tandas grandes, 40+
secciones): `filtrar_golden_set.py --muestra-aceptadas 0` confía 100% en el
juez automático para las "aceptadas" (no fuerza ninguna revisión de
control) y directamente no se corre `revisar_golden_set.py` — las "dudosas"
quedan con `revisado: false` y `consolidar_golden_set.py` las excluye solas
al consolidar solo el archivo de aceptadas:

```bash
python filtrar_golden_set.py --golden golden_set_md_raw.json --prefijo golden_set_md --muestra-aceptadas 0
python consolidar_golden_set.py --archivos golden_set_md_aceptadas.json --out golden_set_md_final.json
```

Requiere: vLLM generador (8002) para generar preguntas; vLLM 4B (8003) para
el juez de `filtrar_golden_set.py` (configurable con `JUDGE_LLM_URL`/`JUDGE_LLM_MODEL`).
No requiere Qdrant.

### `generar_ground_truth.py` + `revisar_ground_truth.py` — ground truth del golden set

Paso previo requerido por `eval_generation.py`. El golden set (chunk o
sección markdown, ver arriba) solo genera y valida **preguntas**, nunca
respuestas de referencia. Sin eso, RAGAS no puede calcular
`context_precision`/`context_recall` (exigen comparar contra una respuesta
"ideal").

- **`generar_ground_truth.py`**: para cada pregunta resuelve el fragmento de
  origen — `texto_seccion` si el item viene de `generar_golden_set_md.py`,
  o `chunk_origen_index` + `fuente_esperada.documento` →
  `datos/chunks_data.json` si viene de `generar_golden_set.py` — y le pide
  al LLM (`qwen35-9b`, igual que `generar_golden_set.py`) una respuesta
  basada ÚNICAMENTE en ese fragmento. Guarda el resultado en el campo
  `ground_truth`, marcado con `"ground_truth_revisado": false` — **no es
  confiable hasta revisarlo**. Con `--auto-aprobar` marca
  `ground_truth_revisado: true` directamente, sin pasar por
  `revisar_ground_truth.py` (confía 100% en el LLM; ahorra la revisión
  manual a costa de no tener chequeo humano sobre las respuestas que usa
  RAGAS como referencia — usar con criterio).
- **`revisar_ground_truth.py`**: revisión manual interactiva (aceptar /
  editar / saltar), igual patrón que `revisar_golden_set.py`. Marca
  `ground_truth_revisado: true` por pregunta.

```bash
python generar_ground_truth.py --golden golden_set.json --out golden_set.json
python revisar_ground_truth.py --golden golden_set.json --out golden_set.json   # obligatorio antes de confiar en el ground truth
python revisar_ground_truth.py --golden golden_set.json --solo-pendientes

# o, para saltar la revisión manual del ground truth también:
python generar_ground_truth.py --golden golden_set_md_final.json --out golden_set_md_final.json --auto-aprobar
```

Requiere: vLLM generador (8002). No requiere Qdrant.

### `eval_generation.py` — calidad de generación (RAGAS)

Corre el pipeline **completo** (`rag_query1.main()`: normalización +
retrieval + generación) sobre cada pregunta del golden set y evalúa la
respuesta con **4 métricas de RAGAS**:

- **faithfulness**: si la respuesta se sostiene solo en el contexto
  recuperado, sin alucinar.
- **answer_relevancy**: si la respuesta contesta lo que se preguntó.
- **context_precision**: si los chunks recuperados relevantes están
  rankeados arriba, comparado contra el `ground_truth`.
- **context_recall**: si el contexto recuperado alcanza para derivar el
  `ground_truth`.

El juez de RAGAS es el propio `qwen35-9b` servido localmente (no OpenAI ni
Anthropic), vía `ChatOpenAI`/`OpenAIEmbeddings` apuntando a los vLLM del
proyecto. Solo se evalúan preguntas con `ground_truth` presente y
`ground_truth_revisado: true` (ver arriba); las que no lo tienen se
excluyen y se cuentan en `n_sin_ground_truth` del resumen.

Preguntas cuyo pipeline falla, o que devuelven respuesta/contexto vacío,
se excluyen del cálculo de RAGAS pero quedan registradas en `errores`
dentro del JSON de salida (antes desaparecían sin dejar rastro). También
se agregan promedios por `categoria` y tiempos de ejecución.

```bash
pip install ragas --break-system-packages   # una sola vez (ya está en requirements.txt)
python generar_ground_truth.py && python revisar_ground_truth.py   # una sola vez por golden set

# golden set de chunks (default)
python eval_generation.py
python eval_generation.py --n-muestra 15

# golden set de secciones markdown — nombres de salida distintos para no
# pisar resultados_generation.json del golden set de chunks
python eval_generation.py --golden golden_set_md_final.json --out resultados_generation_md.json
```

`--golden`/`--out` son genéricos: `eval_generation.py` no distingue si el
golden set viene de chunks o de secciones markdown, solo necesita que cada
item tenga `pregunta`, `ground_truth` y `ground_truth_revisado: true`.

Requiere: vLLM embeddings (8001), normalizador (8003), generador (8002) y
Qdrant (6333) — las cuatro dependencias, porque corre el pipeline
completo por cada pregunta.

### `inspeccionar_miss.py` — depuración caso por caso

Toma el `detalle` de una corrida de `eval_retrieval.py`
(`resultados_eval.json`) y, para los casos MISS (o un `--id` puntual),
vuelve a ejecutar `recuperar_contexto()` mostrando qué trajo realmente el
sistema en el top-K, para distinguir falla real de retrieval vs. golden
set ambiguo o mal etiquetado. Igual que `eval_retrieval.py`, ahora busca
por defecto con la pregunta normalizada (usar `--sin-normalizar` si la
corrida que estás inspeccionando se hizo con esa opción).

```bash
python inspeccionar_miss.py
python inspeccionar_miss.py --id q0010
python inspeccionar_miss.py --sin-normalizar
```

---

## 7. Interfaces

### 7.1. Gradio

**`app_gradio.py`** (producción, puerto 7861): chat (`gr.Chatbot`) sobre `rag_query.main()`, sin memoria real (visualmente muestra historial, cada pregunta se procesa aislada). Acceso vía túnel SSH o forwarding automático de VS Code Remote-SSH.

```bash
cd ~/rag312/interfaz
python app_gradio.py
```

### 7.2. Web (HTML/CSS/JS + FastAPI)

**`app_web.py`** (en desarrollo, puerto 8080): interfaz HTML propia (`interfaz/static/`) sobre `rag_query1.main()`, con estilo azul/amarillo (Electro Sur Este), modo día/noche, preguntas sugeridas, historial de conversaciones (guardado en `localStorage` del navegador, no en el servidor) y visualización de fuentes/tiempo de generación. El botón de subir archivo está visible pero deshabilitado ("Función en desarrollo") — todavía no hay ingesta de archivos vía la interfaz.

Requiere el entorno conda `rag312` activo (`fastapi`, `uvicorn` y `python-multipart` ya están en `requirements.txt`) y los servicios vLLM (8001, 8002, 8003) + Qdrant (6333) corriendo, igual que `rag_query1.py`.

```bash
source ~/rag312/conda/activar.sh
conda activate rag312
source ~/rag312/containers/env.sh   # variables de entorno vLLM, si aplica

cd ~/rag312/interfaz
python app_web.py
```

Luego abrir `http://localhost:8080/` (o el host/túnel correspondiente si se accede remoto). También se puede levantar con recarga automática durante desarrollo:

```bash
cd ~/rag312/interfaz
uvicorn app_web:app --reload --host 0.0.0.0 --port 8080
```

---
