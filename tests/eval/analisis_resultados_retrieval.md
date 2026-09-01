# Análisis de resultados — `eval_retrieval.py`

Este documento interpreta las corridas de `eval_retrieval.py`. Se actualizó tras
reescribir `system_prompt_golden.txt` y `system_prompt_golden_md.txt` para generar
preguntas desde la perspectiva de **personal interno de SIELSE** (técnico y no
técnico) en vez de un cliente externo — ver §4 para la comparación antes/después.

| Archivo | Golden set | Origen de las preguntas |
|---|---|---|
| `resultados_eval.json` | `golden_set.json` | Ancladas a **un chunk** puntual (`generar_golden_set.py`) |
| `resultados_eval_md.json` | `golden_set_md_final.json` | Generadas desde **una sección completa** de markdown (`generar_golden_set_md.py`) |

Ambas corridas usan `k=7` y normalización activa (`normalizado: true`).

## 1. Qué mide cada métrica

`eval_retrieval.py` no usa un LLM juez — es una medición determinística de si el
documento/sección correcto aparece en lo que trae Qdrant.

- **`fuente_esperada`**: lo que se espera encontrar (`documento`, y opcionalmente
  `seccion_contiene`). El matching es siempre por `documento` + substring de sección,
  nunca por `chunk_origen_index`.
- **`rank_fusion` / `rank_dense` / `rank_bm25`**: posición (1..k) del primer resultado
  que hace match, para cada uno de los tres canales de búsqueda. `null` = no apareció
  en el top-k.
  - **fusion**: ranking híbrido final (RRF combinando dense + BM25) — **es el que usa
    producción** (`rag_query1.py`).
  - **dense**: solo el canal de embeddings (harrier-embed).
  - **bm25**: solo el canal léxico (FastEmbed BM25).
- **`hit_*`** (bool): `true` si `rank_* <= k`.
- **`rr_*`** (reciprocal rank): `1/rank_*` si hubo hit, `0.0` si no. Es el valor que se
  promedia para obtener el MRR.
- **`recall_at_7_*`**: proporción de preguntas con `hit_* = true`. Responde "¿el
  documento correcto apareció en algún lugar del top-7?" — no le importa en qué
  posición.
- **`mrr_*`** (Mean Reciprocal Rank): promedio de `rr_*` sobre todas las preguntas.
  Premia que el resultado correcto quede arriba (rank 1 vale 1.0, rank 7 vale 0.14).
- **`por_categoria`**: mismo recall/MRR por `categoria` (subcarpeta de `biblioteca/`).
  Ojo con categorías de `n_preguntas` bajo (1-2): el número ahí es ruidoso.
- **`n_errores`**: preguntas donde el pipeline tiró una excepción (0 en ambas
  corridas).
- **`tiempo_total_seg` / `tiempo_promedio_por_pregunta_seg`**: latencia del propio
  proceso de evaluación, no es una métrica de calidad.

## 2. Resumen actual (con el prompt de personal interno)

| Métrica | Chunks (n=35) | Markdown (n=16) | Δ |
|---|---:|---:|---:|
| Recall@7 fusion | 0.743 | **0.813** | +0.070 |
| Recall@7 dense | 0.686 | 0.688 | +0.002 |
| Recall@7 bm25 | 0.657 | 0.625 | -0.032 |
| MRR fusion | 0.501 | **0.669** | +0.168 |
| MRR dense | 0.446 | 0.500 | +0.054 |
| MRR bm25 | 0.450 | **0.642** | +0.192 |
| Tiempo prom./pregunta | 0.738s | 0.783s | — |

Con preguntas de personal interno, **markdown queda ahora por encima de chunks** en
recall y MRR de fusion — se invirtió respecto a la corrida anterior (ver §4). El
n de markdown es chico (16) así que conviene tomar esto como tendencia, no como
número definitivo — próxima corrida con más secciones (`--n-secciones 40` completo,
sin que el filtro descarte tantas) daría una base más sólida.

### BM25 supera a dense en MRR (solo en markdown)

En markdown, `mrr_bm25` (0.642) casi empata con `mrr_fusion` (0.669) y supera a
`mrr_dense` (0.500) — lo contrario de chunks, donde dense y bm25 quedan parejos y
ambos por debajo de fusion. Tiene sentido con el nuevo estilo de pregunta: el perfil
"técnico/operativo" pide códigos y nombres exactos de sistemas (`"FACTURACIÓN SIELSE
2.0"`, `"SEE RSP"`, `"IVR"`, `"Corte tipo [II]"`) que BM25 matchea directo por
coincidencia léxica, mientras que dense depende de que el embedding capture bien un
término técnico poco frecuente. Ejemplos con `hit_bm25=true` y `hit_dense=false`:
`q0007` (rank_bm25=1 vs. dense sin hit) y `q0016` (rank_bm25=1 vs. dense sin hit) en
`resultados_eval_md.json`.

## 3. Desglose por categoría

### Chunks (n=35)

| Categoría | n | Recall fusion | MRR fusion |
|---|---:|---:|---:|
| Atención al Cliente | 12 | **0.417** | **0.264** |
| Marketing | 3 | 0.667 | 0.178 |
| Relaciones Públicas | 1 | 0.000 | 0.000 |
| Clientes Mayores | 2 | 1.000 | 0.625 |
| Cobranzas | 4 | 1.000 | 0.661 |
| Facturación | 3 | 1.000 | 0.733 |
| Instalaciones | 8 | 1.000 | 0.719 |
| Pérdidas Comerciales | 2 | 1.000 | 1.000 |

### Markdown (n=16)

| Categoría | n | Recall fusion | MRR fusion |
|---|---:|---:|---:|
| Marketing | 2 | 0.500 | 0.500 |
| Facturación | 4 | 0.750 | 0.625 |
| Instalaciones | 4 | 0.750 | 0.425 |
| Atención al Cliente | 2 | 1.000 | 1.000 |
| Cobranzas | 3 | 1.000 | 0.833 |
| Pérdidas Comerciales | 1 | 1.000 | 1.000 |

**Hallazgo que se repite en las tres corridas hechas hasta ahora (chunks original,
markdown original, chunks con prompt nuevo): *Atención al Cliente* es sistemáticamente
débil o inestable.** En esta corrida concentra 7 de los 9 misses de chunks
(`recall_fusion=0.417` con `n=12`, la categoría más grande del set). Con tres corridas
distintas mostrando el mismo patrón, ya no parece ruido de una muestra chica — vale la
pena investigar si los documentos de esa categoría tienen un problema de chunking o
indexación (ver §5).

## 4. Antes / después del cambio de prompt

| Métrica | Chunks antes¹ | Chunks ahora | Markdown antes¹ | Markdown ahora |
|---|---:|---:|---:|---:|
| n_preguntas | 31 | 35 | 25 | 16 |
| Recall@7 fusion | 0.903 | 0.743 | 0.560 | 0.813 |
| MRR fusion | 0.700 | 0.501 | 0.378 | 0.669 |

¹ Corrida previa a reescribir `system_prompt_golden.txt`/`system_prompt_golden_md.txt`
(preguntas de cliente/coloquiales en markdown; chunk-set ya mixto pero sin pedir
detalles técnicos específicos).

**Chunks bajó, markdown subió.** No es una regresión del retrieval — el sistema no
cambió, cambió el tipo de pregunta:
- Las preguntas de chunk ahora piden más **detalles puntuales** (códigos exactos,
  fechas de homologación, nombres de responsables, números de actividad) en vez de
  preguntas más generales — son más difíciles de matchear por `seccion_contiene`
  aunque el chunk de origen sea el correcto, y varias (`q0003`, `q0007`, `q0010`,
  `q0015`, `q0021`, `q0023`, `q0024`, `q0029`, `q0033`) piden un dato que puede estar
  en la sección pero no en el chunk puntual que se usó para generarlas — un riesgo
  conocido del enfoque "un chunk = una pregunta".
- Las preguntas de markdown dejaron de ser conversación de WhatsApp con errores de
  tipeo y pasaron a preguntas operativas concretas sobre la sección completa, que
  matchean mejor porque usan vocabulario del propio documento.

Esto confirma que el golden set anterior no era representativo del uso real (personal
interno) — el nuevo es más duro para chunks pero más realista para ambos.

## 5. Preguntas MISS (candidatas para `inspeccionar_miss.py`)

### Chunks — 9 de 35 (`resultados_eval.json`)

| id | Categoría | Pregunta |
|---|---|---|
| q0003 | Atención al Cliente | ¿Qué documentos de salida se generan en la actividad 18 y cuáles son sus códigos CVA? |
| q0007 | Atención al Cliente | ¿Quién es el responsable de aprobar este procedimiento y supervisar su implementación? |
| q0010 | Atención al Cliente | ¿Qué documentos normativos y precedentes específicos se listan como base para la gestión de reclamos de usuarios? *(hit_dense rank 10, hit_bm25 rank 8 — casi hit)* |
| q0015 | Atención al Cliente | Si al verificar la vida útil del medidor no se cumple, ¿cuál es el subproceso al que se debe derivar inmediatamente? |
| q0021 | Atención al Cliente | ¿Qué normativa específica regula el contraste del sistema de medición de energía eléctrica en SIELSE? |
| q0023 | Marketing | ¿Quién es el responsable de aprobar el procedimiento del Sistema Integrado de Gestión? |
| q0024 | Atención al Cliente | ¿Quién es el responsable de aprobar el procedimiento y supervisar su implementación? |
| q0029 | Relaciones Públicas | ¿Cuál es la actitud recomendada frente a las solicitudes informativas cuando se desconoce la respuesta? |
| q0033 | Atención al Cliente | ¿Qué pasos debe seguir un Supervisor de Instalaciones y mediciones cuando un Contratista indica que no tiene sistema de medición? |

```bash
python inspeccionar_miss.py --resultados resultados_eval.json --golden golden_set.json --id q0010
python inspeccionar_miss.py --resultados resultados_eval.json --golden golden_set.json --id q0007
python inspeccionar_miss.py --resultados resultados_eval.json --golden golden_set.json --id q0024
# repetir para el resto de la lista según haga falta
```

### Markdown — 3 de 16 (`resultados_eval_md.json`)

| id | Categoría | Pregunta |
|---|---|---|
| q0002 | Instalaciones | ¿Cómo se clasifica un cliente que se encuentra fuera del área de concesión de SIELSE y cuál es el rol de ELSE en su atención? |
| q0005 | Marketing | ¿Cuál es el responsable de aprobar este procedimiento y en qué fecha se formalizó su homologación? |
| q0008 | Facturación | ¿Quiénes tienen las firmas de validación y aprobación de este documento? |

```bash
python inspeccionar_miss.py --resultados resultados_eval_md.json --golden golden_set_md_final.json --id q0002
python inspeccionar_miss.py --resultados resultados_eval_md.json --golden golden_set_md_final.json --id q0005
python inspeccionar_miss.py --resultados resultados_eval_md.json --golden golden_set_md_final.json --id q0008
```

Nota: `q0005`/`q0008`/`q0023`/`q0024`/`q0007`(chunks) comparten un patrón — preguntan
"quién aprueba/homologa este procedimiento" sin mencionar el nombre del documento.
Esa info suele vivir en el encabezado/metadata del PDF, no en el cuerpo de la sección
que generó la pregunta — vale la pena confirmar con `inspeccionar_miss.py` si es una
limitación real de chunking/indexado o si la pregunta es ambigua per se (aplica a
varios documentos, no solo al de origen).

## 6. Próximos pasos sugeridos

1. **Priorizar *Atención al Cliente***: es la categoría más débil en las tres corridas
   hechas hasta ahora. Revisar con `inspeccionar_miss.py` los 7 misses de esa
   categoría en chunks antes que el resto — si el patrón se sostiene, mirar el
   chunking de esos PDFs específicos en `ingesta/ocr_docling.py`.
2. Investigar el patrón "quién aprueba el procedimiento" (5 misses lo comparten,
   ver nota de §5) — puede requerir indexar mejor los metadatos de aprobación/
   homologación, no un problema de embeddings.
3. Ampliar el golden set de markdown (subió el score pero con solo 16 preguntas
   después del filtro) corriendo `run_retrieval_eval_markdown.sh` de nuevo con más
   secciones o revisando por qué el juez descartó tantas de las 40 generadas.
4. Repetir con `--sin-normalizar` para aislar si `normalizar_query.py` ayuda o
   entorpece con este estilo de pregunta técnica (antes solo se probó con el estilo
   coloquial).
