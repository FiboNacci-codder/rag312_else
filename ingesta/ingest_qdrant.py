import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    SparseVector,
)

from rag312.clients import build_qdrant_client
from rag312.config import settings

VECTORS_JSON = settings.datos_dir / "embeddings_data.json"

COLLECTION_NAME = settings.collection_name
VECTOR_DIM = settings.vector_dim
DENSE_VECTOR_NAME = settings.dense_vector_name
SPARSE_VECTOR_NAME = settings.sparse_vector_name


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

    client = build_qdrant_client()

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
