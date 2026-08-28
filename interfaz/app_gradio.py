import sys
import html
from pathlib import Path

BASE_DIR = Path(__file__).parent          # ~/rag/interfaz
PROJECT_DIR = BASE_DIR.parent              # ~/rag
sys.path.insert(0, str(PROJECT_DIR / "consulta"))

import gradio as gr
from rag_query1 import main as consultar_rag

EJEMPLOS = [
    "¿Cuál es el procedimiento para solicitar vacaciones?",
    "¿Cómo se tramita un cambio de razón social?",
    "¿Qué pasos sigue una atención de reclamo por consumo excesivo?",
]


def formatear_fuentes(fuentes: list[dict]) -> str:
    if not fuentes:
        return "_No se encontraron fuentes relevantes._"

    bloques = []
    for f in fuentes:
        seccion_str = f' — sección "{f["seccion"]}"' if f.get("seccion") else ""
        if f.get("similitud_dense") is not None:
            if f.get("fuera_top20_dense"):
                similitud_str = f'{f["similitud_dense"]}% (fuera del top-20, calculado aparte)'
            else:
                similitud_str = f'{f["similitud_dense"]}%'
        else:
            similitud_str = "N/D"
        bm25_str = f'{f["score_bm25"]}' if f.get("score_bm25") is not None else "no en top-20 BM25"
        rank_dense_str = f'#{f["rank_dense"]}' if f.get("rank_dense") is not None else "—"
        rank_bm25_str = f'#{f["rank_bm25"]}' if f.get("rank_bm25") is not None else "—"
        score_fusion_str = f'{f["score_fusion"]:.4f}' if f.get("score_fusion") is not None else "N/D"

        chunk_escapado = html.escape(f.get("chunk") or "")

        bloque = f"""**{f["rank_fusion"]}. {f["documento"]}** (página {f["pagina"]}{seccion_str})
   - Score RRF (fusión): {score_fusion_str}
   - Similitud embedding (dense): {similitud_str} (rank {rank_dense_str})
   - Score BM25 (léxico): {bm25_str} (rank {rank_bm25_str})
   - Ruta: `{f["ruta"]}`

<details>
<summary>Ver chunk recuperado</summary>
{chunk_escapado}
</details>"""
        bloques.append(bloque)

    return "\n\n---\n\n".join(bloques)


def responder(pregunta, historial):
    if not pregunta.strip():
        return "Por favor escribe una pregunta."
    try:
        resultado = consultar_rag(pregunta)
    except Exception as e:
        return f"⚠️ Ocurrió un error al consultar el sistema: {e}"

    fuentes_md = formatear_fuentes(resultado["fuentes"])

    salida = f"""{resultado['respuesta']}

---
**Fuentes consultadas:**

{fuentes_md}

*Tiempo retrieval: {resultado['tiempo_retrieval']:.2f}s | Tiempo generación: {resultado['tiempo_generacion']:.2f}s*"""

    return salida


with gr.Blocks(title="RAG - Procedimientos SIELSE") as demo:
    gr.Markdown("# 📋 Asistente de Procedimientos SIELSE")
    gr.Markdown("Escribe una pregunta sobre cualquier procedimiento institucional. El sistema busca en los documentos oficiales (búsqueda híbrida: semántica + léxica) y responde citando la fuente.")

    with gr.Row():
        pregunta_input = gr.Textbox(
            label="Tu pregunta",
            placeholder="Ej: ¿Cuál es el procedimiento para...?",
            lines=2,
        )

    respuesta_output = gr.Markdown(label="Respuesta")

    with gr.Row():
        boton_enviar = gr.Button("Consultar", variant="primary")
        boton_limpiar = gr.Button("Limpiar")

    gr.Examples(examples=EJEMPLOS, inputs=pregunta_input)

    boton_enviar.click(fn=responder, inputs=[pregunta_input, respuesta_output], outputs=respuesta_output)
    pregunta_input.submit(fn=responder, inputs=[pregunta_input, respuesta_output], outputs=respuesta_output)
    boton_limpiar.click(fn=lambda: ("", ""), inputs=None, outputs=[pregunta_input, respuesta_output])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)

