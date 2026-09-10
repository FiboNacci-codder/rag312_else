"""Ingesta incremental de un único PDF ya guardado en biblioteca/.

A diferencia de ocr_docling.py / embeddings.py / ingest_qdrant.py corridos
como scripts (que procesan TODA la biblioteca y, en el caso de
ingest_qdrant.py, borran y recrean la colección), este módulo encadena las
versiones "un solo archivo" de cada etapa para agregar/actualizar un
documento sin tocar el resto del índice. Pensado para ser importado desde
interfaz/app_web.py (POST /api/upload).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag312.config import settings

import embeddings
import ingest_qdrant
import ocr_docling


def ingestar_archivo(pdf_path: Path) -> dict:
    """Corre OCR+chunking, embeddings e inserción en Qdrant para un solo PDF.

    `pdf_path` debe estar dentro de settings.biblioteca_dir. Devuelve un
    resumen con la ruta relativa y cantidad de chunks generados.
    """
    pdf_path = Path(pdf_path).resolve()

    try:
        docs = ocr_docling.procesar_uno(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Fallo en OCR/chunking: {e}") from e

    ruta_biblioteca = str(pdf_path.relative_to(settings.biblioteca_dir))

    try:
        ocr_docling.fusionar_y_guardar_chunks_json(docs, ocr_docling.CHUNKS_JSON_PATH)
    except Exception as e:
        raise RuntimeError(f"Fallo al guardar chunks_data.json: {e}") from e

    try:
        vectores = embeddings.embeber_ruta(ruta_biblioteca)
        embeddings.fusionar_y_guardar_vectores_json(vectores, embeddings.VECTORS_OUT)
    except Exception as e:
        raise RuntimeError(f"Fallo en embeddings: {e}") from e

    try:
        num_puntos = ingest_qdrant.upsert_ruta(ruta_biblioteca)
    except Exception as e:
        raise RuntimeError(f"Fallo al insertar en Qdrant: {e}") from e

    return {"ruta_biblioteca": ruta_biblioteca, "num_chunks": len(docs), "num_puntos": num_puntos}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python ingestar_archivo.py <ruta_al_pdf>")
        sys.exit(1)
    resultado = ingestar_archivo(Path(sys.argv[1]))
    print(resultado)
