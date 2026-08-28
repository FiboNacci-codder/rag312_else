import json
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    SparseVector,
)

BASE_DIR = Path(__file__).parent          # ~/rag/ingesta
PROJECT_DIR = BASE_DIR.parent              # ~/rag
VECTORS_JSON = PROJECT_DIR / "datos" / "embeddings_data.json"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "procedimientos_sielse"
VECTOR_DIM = 1024

# Nombres de los vector spaces dentro de la colección
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"


def cargar_datos():
    with open(VECTORS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def recrear_coleccion(client: QdrantClient):
    """
    IMPORTANTE: no se puede agregar un vector space sparse a una colección
    existente creada solo con vector denso. Hay que borrar y recrear.
    Esto BORRA todos los puntos actuales de la colección.
    """
    if client.collection_exists(COLLECTION_NAME):
        print(f"Colección '{COLLECTION_NAME}' ya existe. Se eliminará para recrearla con soporte hybrid (dense + sparse).")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(),
        },
    )
    print(f"Colección '{COLLECTION_NAME}' creada con vector spaces: '{DENSE_VECTOR_NAME}' (dense) y '{SPARSE_VECTOR_NAME}' (sparse).")


def main():
    datos = cargar_datos()
    print(f"Cargados {len(datos)} registros desde {VECTORS_JSON}")

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    recrear_coleccion(client)

    points = []
    for i, item in enumerate(datos):
        points.append(
            PointStruct(
                id=i,
                vector={
                    DENSE_VECTOR_NAME: item["embedding"],
                    SPARSE_VECTOR_NAME: SparseVector(
                        indices=item["sparse_indices"],
                        values=item["sparse_values"],
                    ),
                },
                payload={
                    "page_content": item["page_content"],
                    **item["metadata"],
                },
            )
        )

    t0 = time.time()
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    elapsed = time.time() - t0

    count = client.count(collection_name=COLLECTION_NAME).count
    print(f"\nInsertados {len(points)} puntos en {elapsed:.2f}s")
    print(f"Total de puntos en la colección '{COLLECTION_NAME}': {count}")


if __name__ == "__main__":
    main()