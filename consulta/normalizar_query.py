import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag312.clients import build_llm_client
from rag312.config import get_normalize_config

SYSTEM_PROMPT_CORRECCION = """Eres un corrector ortográfico. Tu única tarea es corregir errores ortográficos y de tipeo en el texto que te dan, sin cambiar su significado ni estructura.

Reglas:
- Corrige solo ortografía (letras mal escritas, tildes, mayúsculas al inicio de oración).
- NO cambies palabras por sinónimos.
- NO respondas la pregunta, NO agregues explicaciones.
- NO agregues ni quites signos de interrogación si ya estaban o no estaban.
- Si el texto ya está bien escrito, devuélvelo exactamente igual.
- Responde ÚNICAMENTE con el texto corregido, nada más."""

SYSTEM_PROMPT_REWRITING = """ROL
Eres un asistente con 10 años de experiencia en sistemas RAG, encargado de normalizar \
queries antes de la etapa de retrieval.

DESCRIPCIÓN DE LA ENTRADA
Una pregunta de usuario en lenguaje coloquial.

OBJETIVO
Reformular la pregunta a un lenguaje formal y técnico, como el usado en manuales de \
procedimientos institucionales.

REGLAS
- Mantén el significado exacto de la pregunta original, no agregues ni quites información.
- Usa terminología formal/técnica en vez de coloquial (ej: "reclamo por consumo excesivo" en \
vez de "me llegó carísimo el recibo").
- No respondas la pregunta, solo reformúlala.
- Responde ÚNICAMENTE con la pregunta reformulada, nada más."""

_client_llm = None
_role_config = None


def _get_client_llm():
    global _client_llm, _role_config
    if _client_llm is None:
        _role_config = get_normalize_config()
        _client_llm = build_llm_client(_role_config)
    return _client_llm, _role_config


def normalizar_basico(texto: str) -> str:
    texto = texto.strip()
    texto = re.sub(r"\s+", " ", texto)
    return texto


def _chat_completar(system_prompt: str, texto: str, temperature: float, max_tokens: int = 1024) -> str:
    client_llm, role_config = _get_client_llm()
    respuesta = client_llm.chat.completions.create(
        model=role_config.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": texto},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    contenido = respuesta.choices[0].message.content
    if not contenido:
        # fallback: si el LLM no devolvió nada, usar el texto original sin corregir
        return texto
    return contenido.strip()


def corregir_con_llm(texto: str) -> str:
    return _chat_completar(SYSTEM_PROMPT_CORRECCION, texto, temperature=0.0)


def reformular_query(texto: str) -> str:
    return _chat_completar(SYSTEM_PROMPT_REWRITING, texto, temperature=0.2)  # algo de flexibilidad, pero controlada


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
    pregunta = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Pregunta de prueba: ")
    resultado = procesar_pregunta(pregunta)
    print(f"\nOriginal    : {resultado['original']}")
    print(f"Corregida   : {resultado['corregida']}")
    print(f"Para búsqueda: {resultado['busqueda']}")
