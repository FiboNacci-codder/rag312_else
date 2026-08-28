import re
from openai import OpenAI

VLLM_LLM_URL = "http://localhost:8003/v1"
LLM_MODEL = "qwen35-4b"

SYSTEM_PROMPT_CORRECCION = """Eres un corrector ortográfico. Tu única tarea es corregir errores ortográficos y de tipeo en el texto que te dan, sin cambiar su significado ni estructura.

Reglas:
- Corrige solo ortografía (letras mal escritas, tildes, mayúsculas al inicio de oración).
- NO cambies palabras por sinónimos.
- NO respondas la pregunta, NO agregues explicaciones.
- NO agregues ni quites signos de interrogación si ya estaban o no estaban.
- Si el texto ya está bien escrito, devuélvelo exactamente igual.
- Responde ÚNICAMENTE con el texto corregido, nada más."""

SYSTEM_PROMPT_REWRITING = """Eres un asistente que reformula preguntas de usuarios a un lenguaje formal y técnico, como el usado en manuales de procedimientos institucionales.

Reglas:
- Mantén el significado exacto de la pregunta original, no agregues ni quites información.
- Usa terminología formal/técnica en vez de coloquial (ej: "reclamo por consumo excesivo" en vez de "me llegó carísimo el recibo").
- No respondas la pregunta, solo reformúlala.
- Responde ÚNICAMENTE con la pregunta reformulada, nada más."""


def normalizar_basico(texto: str) -> str:
    texto = texto.strip()
    texto = re.sub(r"\s+", " ", texto)
    return texto

def corregir_con_llm(texto: str) -> str:
    client_llm = OpenAI(base_url=VLLM_LLM_URL, api_key="no-necesaria")
    respuesta = client_llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_CORRECCION},
            {"role": "user", "content": texto},
        ],
        max_tokens=1024,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    contenido = respuesta.choices[0].message.content
    if not contenido:
        # fallback: si el LLM no devolvió nada, usar el texto original sin corregir
        return texto
    return contenido.strip()



def reformular_query(texto: str) -> str:
    client_llm = OpenAI(base_url=VLLM_LLM_URL, api_key="no-necesaria")
    respuesta = client_llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_REWRITING},
            {"role": "user", "content": texto},
        ],
        max_tokens=1024,
        temperature=0.2,  # algo de flexibilidad, pero controlada
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    contenido = respuesta.choices[0].message.content
    if not contenido:
        # fallback: si el LLM no devolvió nada, usar el texto original sin corregir
        return texto
    return contenido.strip()

def procesar_pregunta(texto: str, corregir: bool = True, reformular: bool = True) -> dict:
    """
    Retorna un diccionario con las distintas versiones del texto:
    - original: tal como lo escribió el usuario
    - corregida: con ortografía arreglada
    - busqueda: versión final usada para el embedding (reformulada si aplica)
    """
    original = texto
    texto = normalizar_basico(texto)

    if corregir:
        texto_corregido = corregir_con_llm(texto)
        if texto_corregido != texto:
            print(f"[Corrección aplicada]: '{texto}' -> '{texto_corregido}'")
        texto = texto_corregido

    texto_busqueda = texto
    if reformular:
        texto_reformulado = reformular_query(texto)
        if texto_reformulado != texto:
            print(f"[Reformulación aplicada]: '{texto}' -> '{texto_reformulado}'")
        texto_busqueda = texto_reformulado

    return {
        "original": original,
        "corregida": texto,
        "busqueda": texto_busqueda,
    }

if __name__ == "__main__":
    import sys
    pregunta = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Pregunta de prueba: ")
    resultado = procesar_pregunta(pregunta)
    print(f"\nOriginal    : {resultado['original']}")
    print(f"Corregida   : {resultado['corregida']}")
    print(f"Para búsqueda: {resultado['busqueda']}")