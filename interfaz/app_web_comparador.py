import sys
from pathlib import Path
from typing import Literal

BASE_DIR = Path(__file__).parent          # ~/rag312_else/interfaz
STATIC_DIR = BASE_DIR / "static_comparador"

sys.path.insert(0, str(BASE_DIR.parent))
from rag312.config import settings

sys.path.insert(0, str(settings.rag_project_dir / "consulta"))

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_query1 import main as consultar_rag

app = FastAPI(title="RAG - Procedimientos SIELSE (comparador)")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PreguntaRequest(BaseModel):
    pregunta: str
    corregir: bool = True
    reformular: bool = True
    modo_retrieval: Literal["hybrid", "dense", "sparse"] = "hybrid"
    top_k: int | None = None
    umbral_similitud: float | None = None
    rerank: bool = True
    rerank_top_n: int | None = None
    collection_name: str = "procedimientos_sielse"


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/chat")
def chat(request: PreguntaRequest):
    pregunta = request.pregunta.strip()
    if not pregunta:
        return JSONResponse({"error": "Por favor escribe una pregunta."})
    try:
        resultado = consultar_rag(
            pregunta,
            corregir=request.corregir,
            reformular=request.reformular,
            modo_retrieval=request.modo_retrieval,
            top_k=request.top_k,
            umbral_similitud=request.umbral_similitud,
            rerank=request.rerank,
            rerank_top_n=request.rerank_top_n,
            collection_name=request.collection_name,
        )
    except Exception as e:
        return JSONResponse({"error": f"Ocurrió un error al consultar el sistema: {e}"})
    return resultado


@app.get("/api/documento/{ruta_biblioteca:path}")
def documento(ruta_biblioteca: str):
    biblioteca_dir = settings.biblioteca_dir.resolve()
    destino = (biblioteca_dir / ruta_biblioteca).resolve()
    if destino.suffix.lower() != ".pdf" or not destino.is_relative_to(biblioteca_dir) or not destino.is_file():
        return JSONResponse(status_code=404, content={"detail": "Documento no encontrado"})
    return FileResponse(
        destino,
        media_type="application/pdf",
        filename=destino.name,
        content_disposition_type="inline",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7863)
