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
        )
    except Exception as e:
        return JSONResponse({"error": f"Ocurrió un error al consultar el sistema: {e}"})
    return resultado


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7863)
