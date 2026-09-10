import sys
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).parent          # ~/rag312_else/interfaz
STATIC_DIR = BASE_DIR / "static"

sys.path.insert(0, str(BASE_DIR.parent))
from rag312.config import settings

sys.path.insert(0, str(settings.rag_project_dir / "consulta"))

from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_query1 import main as consultar_rag

app = FastAPI(title="RAG - Procedimientos SIELSE")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Estado de las cargas de archivo en curso, en memoria (proceso único, sin cola/DB).
JOBS: dict[str, dict] = {}


@app.on_event("startup")
def crear_carpeta_cargas_manuales():
    settings.cargas_manuales_dir.mkdir(parents=True, exist_ok=True)


class PreguntaRequest(BaseModel):
    pregunta: str
    modo_retrieval: str = "hybrid"
    top_k: int | None = None
    umbral_similitud: float | None = None
    rerank: bool = True
    rerank_top_n: int | None = None


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
            modo_retrieval=request.modo_retrieval,
            top_k=request.top_k,
            umbral_similitud=request.umbral_similitud,
            rerank=request.rerank,
            rerank_top_n=request.rerank_top_n,
        )
    except Exception as e:
        return JSONResponse({"error": f"Ocurrió un error al consultar el sistema: {e}"})
    return resultado


def _guardar_archivo_sin_colision(carpeta: Path, nombre: str) -> Path:
    destino = carpeta / nombre
    if not destino.exists():
        return destino
    stem, suffix = destino.stem, destino.suffix
    contador = 1
    while destino.exists():
        destino = carpeta / f"{stem}_{contador}{suffix}"
        contador += 1
    return destino


def _procesar_carga(job_id: str, pdf_path: Path) -> None:
    # Import perezoso: ingesta/ carga docling/torch/transformers, pesados,
    # y no deben inflar el arranque del servidor web.
    sys.path.insert(0, str(settings.rag_project_dir / "ingesta"))
    from ingestar_archivo import ingestar_archivo

    try:
        resultado = ingestar_archivo(pdf_path)
        JOBS[job_id] = {
            "status": "completado",
            "mensaje": f"Archivo procesado: {resultado['num_chunks']} chunks insertados.",
            "resultado": resultado,
        }
    except Exception as e:
        JOBS[job_id] = {"status": "error", "mensaje": str(e)}


@app.post("/api/upload")
def upload(background_tasks: BackgroundTasks, archivo: UploadFile = File(...)):
    nombre_original = Path(archivo.filename or "").name
    if not nombre_original.lower().endswith(".pdf"):
        return JSONResponse(status_code=400, content={"detail": "Solo se aceptan archivos PDF."})

    cargas_dir = settings.cargas_manuales_dir
    cargas_dir.mkdir(parents=True, exist_ok=True)
    destino = _guardar_archivo_sin_colision(cargas_dir, nombre_original)
    with open(destino, "wb") as f:
        f.write(archivo.file.read())

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "en_progreso", "mensaje": "Procesando archivo..."}
    background_tasks.add_task(_procesar_carga, job_id, destino)

    return {"job_id": job_id, "archivo": destino.name}


@app.get("/api/upload/status/{job_id}")
def upload_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"detail": "job_id desconocido"})
    return job


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
    uvicorn.run(app, host="0.0.0.0", port=7862)
