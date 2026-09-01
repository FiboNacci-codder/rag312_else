# Análisis de resultados — `eval_retrieval.py`

Este documento interpreta las dos corridas de `eval_retrieval.py` ya generadas:

| Archivo | Golden set | Origen de las preguntas |
|---|---|---|
| `resultados_eval.json` | `golden_set.json` | Ancladas a **un chunk** puntual (`generar_golden_set.py`) |
| `resultados_eval_md.json` | `golden_set_md_final.json` | Generadas desde **una sección completa** de markdown, en tono coloquial (`generar_golden_set_md.py`) |

Ambas corridas usan `k=7` y normalización activa (`normalizado: true`, es decir, cada
pregunta pasó por `normalizar_query.procesar_pregunta()` antes de buscar, igual que en
producción).

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
  A diferencia de recall, **premia que el resultado correcto quede arriba** (rank 1
  vale 1.0, rank 7 vale 0.14) — dos corridas pueden tener el mismo recall y un MRR muy
  distinto si una devuelve los hits en rank 1 y la otra en rank 6-7.
- **`por_categoria`**: mismo recall/MRR pero desglosado por `categoria` (subcarpeta de
  `biblioteca/`) — sirve para detectar si el problema es transversal o está
  concentrado en un área. Ojo con categorías de `n_preguntas` bajo (1-2): el
  recall/MRR ahí es ruidoso, no una tendencia confiable.
- **`n_errores`**: preguntas donde `recuperar_contexto()` u otra parte del pipeline
  tiró una excepción (0 en ambas corridas — no hay nada que depurar por ese lado).
- **`tiempo_total_seg` / `tiempo_promedio_por_pregunta_seg`**: latencia del propio
  proceso de evaluación (normalización + búsqueda por pregunta), no es una métrica de
  calidad.

## 2. Resumen comparativo

| Métrica | Chunks (`golden_set.json`) | Markdown (`golden_set_md_final.json`) | Δ |
|---|---:|---:|---:|
| n_preguntas | 31 | 25 | — |
| n_errores | 0 | 0 | — |
| Recall@7 fusion | **0.903** | **0.560** | -0.343 |
| Recall@7 dense | 0.871 | 0.600 | -0.271 |
| Recall@7 bm25 | 0.710 | 0.480 | -0.230 |
| MRR fusion | **0.700** | **0.378** | -0.322 |
| MRR dense | 0.733 | 0.390 | -0.343 |
| MRR bm25 | 0.584 | 0.333 | -0.251 |
| Tiempo prom./pregunta | 0.748s | 0.678s | — |

**El retrieval se comporta claramente peor sobre el golden set de markdown.** La caída
no es un problema de una sola métrica — recall y MRR bajan juntos en los tres canales,
así que no es solo "el documento correcto queda más abajo en el ranking", sino que en
buena parte de los casos **no aparece en absoluto** dentro del top-7.

### Por qué son comparables con reservas

Los dos golden sets no miden exactamente lo mismo:
- **Chunks**: la pregunta se generó a partir del chunk exacto indexado en Qdrant → el
  match es casi por diseño más fácil (mismo vocabulario, mismo nivel de granularidad).
- **Markdown**: la pregunta se generó desde una sección completa y en tono **coloquial
  deliberado** (ver ejemplos abajo: "Hola, ...", errores de tipeo como
  "interrruptores", mezcla de inglés "approving") — es una prueba de estrés más
  realista de cómo pregunta un usuario real, pero también más difícil de matchear
  porque el vocabulario de la pregunta se aleja más del texto indexado.

Es decir: la brecha no prueba por sí sola que el retrieval "empeoró" — prueba que el
sistema **se degrada frente a preguntas coloquiales/parafraseadas**, que es
información igual de valiosa (posiblemente más representativa del uso real).

### Anomalía a investigar: fusion por debajo de dense en markdown

En chunks, fusion (0.903) queda entre dense (0.871) y por encima — como se espera de
una fusión RRF que combina dos señales razonables. En **markdown, fusion (0.56) queda
por *debajo* de dense solo (0.60)** — el canal BM25 (0.48 recall) está arrastrando
hacia abajo al híbrido en vez de sumar. Ejemplo concreto: `q0022` del set de markdown
tiene `hit_bm25=true` (rank 7) pero `hit_fusion=false` (`rank_fusion=null`) — BM25 sí
encontraba el documento correcto dentro del top-7 propio, pero al fusionarse con un
dense más débil en esa pregunta, RRF lo empujó fuera del top-7 final. Vale la pena
correr `eval_retrieval.py --golden golden_set_md_final.json --sin-normalizar` para ver
si la normalización está ayudando o entorpeciendo estos casos coloquiales, y revisar
si el `TOP_K` interno de cada canal antes de fusionar (en `rag_query1.py`) es
suficiente para preguntas menos literales.

## 3. Desglose por categoría

### Chunks

| Categoría | n | Recall fusion | MRR fusion |
|---|---:|---:|---:|
| Facturación | 3 | 0.667 | 0.417 |
| Atención al Cliente | 5 | 0.800 | 0.540 |
| Instalaciones | 9 | 0.889 | 0.704 |
| Marketing | 5 | 1.000 | 0.767 |
| Cobranzas | 5 | 1.000 | 0.867 |
| Pérdidas Comerciales | 2 | 1.000 | 1.000 |
| Relaciones Públicas | 1 | 1.000 | 1.000 |
| Supervisión Comercial | 1 | 1.000 | 0.250 |

### Markdown

| Categoría | n | Recall fusion | MRR fusion |
|---|---:|---:|---:|
| Marketing | 1 | 0.000 | 0.000 |
| Cobranzas | 4 | 0.250 | 0.250 |
| Atención al Cliente | 6 | 0.333 | 0.194 |
| Instalaciones | 7 | 0.714 | 0.255 |
| Facturación | 3 | 0.667 | 0.667 |
| Clientes Mayores | 1 | 1.000 | 1.000 |
| Pérdidas Comerciales | 3 | 1.000 | 0.833 |

**Patrón consistente en ambos sets**: *Facturación* y *Atención al Cliente* son las
categorías más débiles; *Pérdidas Comerciales* es sólida en ambos. En markdown,
*Cobranzas* también cae fuerte (0.25 recall) — no se ve en chunks, lo que sugiere que
el problema ahí es específico de cómo se formulan las preguntas coloquiales sobre esos
documentos, no del contenido indexado en sí. *Marketing* e *Instalaciones* en markdown
tienen `n` bajo o son ruidosas — no sacar conclusiones fuertes de una sola categoría
con 1 pregunta.

## 4. Preguntas MISS (candidatas para `inspeccionar_miss.py`)

`inspeccionar_miss.py` vuelve a correr `recuperar_contexto()` para estos casos y
muestra qué trajo realmente el sistema, para distinguir falla real de retrieval vs.
golden set mal etiquetado.

### Chunks — 3 de 31 (`resultados_eval.json`)

| id | Categoría | Pregunta |
|---|---|---|
| q0007 | Atención al Cliente | ¿Quién es el responsable de aprobar el procedimiento CSIG y supervisar su implementación? |
| q0011 | Instalaciones | ¿Qué procedimientos se deben seguir si el cliente es mayor? |
| q0012 | Facturación | ¿Qué procedimiento administrativo debo seguir para reclamar sobre mi facturación según la Resolución N° 269-2014 OS/CD? |

```bash
python inspeccionar_miss.py --resultados resultados_eval.json --golden golden_set.json --id q0007
python inspeccionar_miss.py --resultados resultados_eval.json --golden golden_set.json --id q0011
python inspeccionar_miss.py --resultados resultados_eval.json --golden golden_set.json --id q0012
```

### Markdown — 11 de 25 (`resultados_eval_md.json`)

| id | Categoría | Pregunta |
|---|---|---|
| q0001 | Atención al Cliente | Se incluye en el documento el procedimiento para reclamar por el cobro de la factura? |
| q0002 | Instalaciones | Hola, ¿qué pasa si mi factura es muy alta, hay alguna ley que me permita reclamarla o pedir que la revisen? |
| q0003 | Atención al Cliente | Hola, si me pasa un reclamo sobre mi consumo de luz, ¿cómo me van a verificar si la lectura del medidor es correcta? |
| q0007 | Cobranzas | Hola, ¿quién es el encargado de approving las reglas de cobro y quién es el de cuidar los datos de los clientes? |
| q0009 | Cobranzas | ¿Por qué me cortaron la luz si no debo dinero? *(hit_dense=true, rank 5 — casi entra)* |
| q0011 | Instalaciones | ¿Qué hago si me piden los interrruptores, el medidor y el tubo bastón al mismo tiempo? |
| q0012 | Marketing | Hola, ¿cómo hago para dejar mi opinión sobre el servicio que me prestaron? |
| q0013 | Atención al Cliente | Hola, me llegó un reclamo que dice que tengo que firmar una carta de confirmación en dos días, ¿qué pasa si no logro hacerlo a tiempo? |
| q0015 | Facturación | ¿Autoriza la dirección para que yo haga uso de este documento? |
| q0019 | Atención al Cliente | ¿Qué pasa si queremos cancelar el trámite directo y pedirnos el reintegro de lo que pagamos? |
| q0022 | Cobranzas | Hola, quiero pagar mi recibo de luz, ¿qué me pueden decir sobre esos talones desglosables? *(hit_bm25=true rank 7 — ver anomalía §2)* |

```bash
python inspeccionar_miss.py --resultados resultados_eval_md.json --golden golden_set_md_final.json --id q0009
python inspeccionar_miss.py --resultados resultados_eval_md.json --golden golden_set_md_final.json --id q0022
# repetir para el resto de la lista según haga falta
```

Prioridad sugerida: `q0009` y `q0022` primero (son "casi hits" — algún canal sí los
encontró, lo que sugiere ajuste fino más que falla estructural); después las de
*Cobranzas* y *Atención al Cliente*, que son las categorías más débiles en markdown.

## 5. Próximos pasos sugeridos

1. Correr `inspeccionar_miss.py` sobre los casos "casi hit" (q0009, q0022) para
   entender si es un problema de `TOP_K` insuficiente antes de fusionar, de la
   reformulación de `normalizar_query.py`, o de que la sección esperada realmente no
   contiene la respuesta literal.
2. Repetir la corrida de markdown con `--sin-normalizar` y comparar recall/MRR contra
   esta corrida normalizada, para aislar si la reformulación coloquial→formal está
   ayudando o hace falta ajustar su prompt.
3. Si el patrón de *Cobranzas*/*Atención al Cliente* se confirma con más preguntas,
   revisar si esos documentos necesitan mejor chunking (`ingesta/ocr_docling.py`) o si
   el golden set de markdown está generando preguntas demasiado alejadas del texto
   real de esas secciones.
