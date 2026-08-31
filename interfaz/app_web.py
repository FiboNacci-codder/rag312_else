import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent          # ~/rag312/interfaz
PROJECT_DIR = BASE_DIR.parent              # ~/rag312
STATIC_DIR = BASE_DIR / "static"
sys.path.insert(0, str(PROJECT_DIR / "consulta"))

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_query1 import main as consultar_rag

app = FastAPI(title="RAG - Procedimientos SIELSE")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PreguntaRequest(BaseModel):
    pregunta: str


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/chat")
def chat(request: PreguntaRequest):
    pregunta = request.pregunta.strip()
    if not pregunta:
        return JSONResponse({"error": "Por favor escribe una pregunta."})
    try:
        resultado = consultar_rag(pregunta)
    except Exception as e:
        return JSONResponse({"error": f"Ocurrió un error al consultar el sistema: {e}"})
    return resultado


@app.post("/api/upload")
def upload():
    return JSONResponse(status_code=501, content={"detail": "Función en desarrollo"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
