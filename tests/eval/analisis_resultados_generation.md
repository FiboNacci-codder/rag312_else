# Análisis de resultados — `eval_generation.py` (RAGAS)

Interpreta las dos corridas de `eval_generation.py` ya generadas:

| Archivo | Golden set | Origen de las preguntas |
|---|---|---|
| `resultados_generation.json` | `golden_set.json` | Ancladas a **un chunk** puntual (`generar_golden_set.py`) |
| `resultados_generation_md.json` | `golden_set_md_final.json` | Generadas desde **una sección completa** de markdown (`generar_golden_set_md.py`) |

**Ambas corridas son de solo 5 preguntas** (`n_preguntas_evaluadas: 5`, de 25/18
disponibles con `ground_truth` revisado) — probablemente la prueba rápida con
`--n-muestra 5`. Sirven para ver patrones cualitativos, pero son una muestra
demasiado chica para conclusiones estadísticas firmes — ver §5.

## 1. Qué mide cada métrica

A diferencia de `eval_retrieval.py` (determinístico, sin LLM), acá el pipeline
**completo** (`rag_query1.main()`: normalización + retrieval + generación) corre
sobre cada pregunta, y la respuesta se evalúa con un LLM juez (`qwen35-9b`) usando 4
métricas de RAGAS:

- **`faithfulness`**: descompone la respuesta en afirmaciones atómicas y verifica
  cada una contra el contexto recuperado (`retrieved_contexts`) — **no** contra la
  respuesta de referencia. Mide "¿lo que dice la respuesta se sostiene en lo que se
  recuperó?", sin importar si lo recuperado era lo correcto.
- **`answer_relevancy`**: genera preguntas sintéticas a partir de la respuesta y las
  compara (por embedding) contra la pregunta original. Mide "¿la respuesta contesta
  lo que se preguntó?", sin mirar si es objetivamente correcta.
- **`context_precision`** (`ContextPrecisionWithReference`): para cada chunk
  recuperado, compara contra el `ground_truth` y juzga si es relevante, ponderando
  por posición — chunks relevantes arriba en el ranking puntúan más.
- **`context_recall`**: descompone el `ground_truth` en afirmaciones y verifica si el
  contexto recuperado alcanza para derivarlas. Es la métrica más cercana a "¿el
  retrieval trajo lo que hacía falta?".

**Punto clave para leer estos resultados**: `faithfulness` y `answer_relevancy` NO
usan el `ground_truth` — solo miran consistencia interna (respuesta vs. contexto
recuperado, respuesta vs. pregunta). Por eso un caso puede tener `faithfulness=1.0` y
aun así estar completamente equivocado, si el contexto recuperado ya era el
incorrecto (ver §3). Solo `context_precision`/`context_recall` comparan contra la
referencia real.

Otros campos del resumen:
- **`n_preguntas_golden`** / **`n_sin_ground_truth`** / **`n_preguntas_evaluadas`**:
  cuántas preguntas tiene el golden set, cuántas se excluyeron por no tener
  `ground_truth_revisado: true`, y cuántas entraron finalmente a RAGAS (afectado
  también por `--n-muestra`).
- **`n_errores_pipeline`** / `errores`: preguntas donde `rag_query1.main()` falló o
  devolvió respuesta/contexto vacío — no llegan a evaluarse con RAGAS.
- **`documentos_recuperados`** (por pregunta, en `detalle`): lista de los documentos
  que trajo la fusión RRF para esa pregunta — es lo que hay que mirar primero cuando
  un score sale mal, para saber si el problema es de retrieval o de generación.

## 2. Resumen comparativo

| Métrica | Chunks (n=5) | Markdown (n=5) |
|---|---:|---:|
| faithfulness | 0.657 | 0.637 |
| answer_relevancy | 0.783 | 0.838 |
| context_precision | 0.423 | 0.709 |
| context_recall | 0.800 | 0.800 |
| tiempo prom/pregunta | 10.1s | 10.4s |

`faithfulness` es la métrica más débil en ambas corridas — y la más reveladora del
comportamiento real del sistema (ver §3). `context_recall` da exactamente 0.8 en las
dos, con la misma causa concreta: en cada corrida, 1 de las 5 preguntas tuvo una
falla total de retrieval (`recall=0`, `precision=0`) — ver §4.

## 3. Hallazgo principal: respuestas largas y "seguras" que se alejan del contexto real

Los casos con `faithfulness` más bajo son justamente las respuestas más elaboradas —
el modelo arma listas con múltiples "Fuentes", notas al margen y detalles extra que
RAGAS no puede verificar contra los chunks recuperados:

- **q0021 (chunks, faithfulness 0.33)**: el dato central de la respuesta (caja porta
  medidor ELSE 2) es correcto, pero agrega una atribución específica a un *segundo*
  documento ("Fuente 2... sección Requerimientos para la Reinstalación de
  medidores") con un nivel de precisión que probablemente no está respaldado — el
  modelo "decora" con detalles no verificados en vez de limitarse a lo que puede
  sostener.
- **q0018 (markdown, faithfulness 0.25)**: la respuesta cubre 4 unidades distintas
  con responsables para 8+ procedimientos, mucho más de lo que la pregunta pedía —
  más afirmaciones puntuales significan más superficie para que alguna no esté
  sostenida por el contexto.

Esto no es "el modelo inventa contenido al azar" — es sobre-elaboración: el prompt
de generación en `rag_query1.py` empuja a respuestas muy exhaustivas/estructuradas, y
esa exhaustividad juega en contra de `faithfulness`.

## 4. Hallazgo más importante: `faithfulness` alto no significa respuesta correcta

**q0001 del set de markdown** es el caso a mirar con más cuidado: `context_recall=0`
y `context_precision=0` (falla total de retrieval — no trajo ningún documento
relacionado con la pregunta real, sobre confirmación de recepción de notificación) —
pero **`faithfulness=1.0`** y `answer_relevancy=0.91`. El sistema recuperó contexto
de un procedimiento distinto (sistema "Clectro") y armó una respuesta larga, bien
formateada y **totalmente segura de sí misma**, sin ninguna señal de que está
contestando con información equivocada. `faithfulness` da 1.0 porque la respuesta es
consistente con lo que se recuperó — el problema es que lo recuperado ya estaba mal,
y esa métrica no lo detecta.

Comparar con **q0004 del set de chunks**, que tuvo la misma falla de retrieval
(`recall=0`, `precision=0`) pero ahí el modelo sí hedgeó ("No se especifican salidas
documentales específicas... nota: en la actividad 15...") — mucho más seguro para
quien lo lea, aunque la respuesta final también termine siendo incorrecta respecto al
`ground_truth`.

**Con solo 5 preguntas por corrida ya apareció una falla total de retrieval en cada
una (1 de 5, 20%)** — consistente con las fallas ya vistas en
`analisis_resultados_retrieval.md`. Vale la pena cruzar `q0004` (chunks) y `q0001`
(markdown) contra `resultados_eval.json`/`resultados_eval_md.json` para confirmar si
son los mismos MISS ya identificados ahí o casos nuevos.

## 5. Casos limpios (para contraste)

- **q0009 markdown** ("clave SEE RSP Empresarial") — 1.0 / 0.83 / 1.0 / 1.0 en las 4
  métricas, respuesta corta y precisa (clave: SOL). Cuando el retrieval trae
  exactamente lo que corresponde y la pregunta es puntual, el sistema responde bien y
  sin ruido — el problema no es capacidad de generación, es cuándo se le da
  suficiente cuerda (respuestas largas) o mal contexto.
- **q0024 chunks** (responsable de indicadores de calidad de Contact Center) — buen
  balance en las 4 métricas (0.82 / 0.87 / 0.67 / 1.0), con una nota aclaratoria
  correctamente marcada como matiz ("aunque otros procedimientos mencionan...") en
  vez de presentada como hecho.

## 6. Próximos pasos sugeridos

1. **Correr con muestra más grande** (`--n-muestra 15-20` o el set completo) antes de
   sacar conclusiones firmes — con n=5 cualquier promedio se mueve mucho con un solo
   caso extremo, y ya cambió bastante entre esta corrida y el análisis de retrieval
   anterior (golden sets regenerados con el prompt de personal interno).
2. **Cruzar `q0004` (chunks) y `q0001` (markdown)** contra
   `resultados_eval*.json` para confirmar si son fallas de retrieval ya conocidas.
3. **Si el patrón de `faithfulness` bajo por sobre-elaboración se sostiene** con más
   muestra, revisar el prompt de generación en `rag_query1.py` (`armar_prompt()`)
   para pedir respuestas más ceñidas a lo que el contexto sostiene explícitamente, en
   vez de maximizar exhaustividad citando múltiples fuentes.
4. **El caso q0001 de markdown es el más urgente de los dos**: mostrar cómo un fallo
   de retrieval se traduce en una respuesta segura pero incorrecta, sin ningún
   indicio para el usuario — priorizarlo sobre los de `faithfulness` bajo por
   sobre-elaboración, que al menos son detectables por su extensión/formato.
5. Complementa bien con **`preguntas_frontera.json`** (`generar_preguntas_frontera.py`
   + `correr_preguntas_frontera.py`): mientras este análisis mide fidelidad sobre
   preguntas respondibles, ese set mide directamente si el sistema admite no saber
   algo — el patrón de "respuesta segura sobre contexto equivocado" visto en q0001
   es exactamente el riesgo que ese set está diseñado para exponer.
