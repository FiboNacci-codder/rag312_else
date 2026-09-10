import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    SparseVector,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
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


def _crear_coleccion(client: QdrantClient):
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


def recrear_coleccion(client: QdrantClient):
    """
    IMPORTANTE: no se puede agregar un vector space sparse a una colección
    existente creada solo con vector denso. Hay que borrar y recrear.
    Esto BORRA todos los puntos actuales de la colección.
    """
    if client.collection_exists(COLLECTION_NAME):
        print(f"Colección '{COLLECTION_NAME}' ya existe. Se eliminará para recrearla con soporte hybrid (dense + sparse).")
        client.delete_collection(COLLECTION_NAME)
    _crear_coleccion(client)


def asegurar_coleccion(client: QdrantClient):
    """Crea la colección solo si no existe todavía. A diferencia de
    recrear_coleccion, nunca borra puntos existentes — es lo que usa la
    ingesta incremental de un solo archivo."""
    if not client.collection_exists(COLLECTION_NAME):
        _crear_coleccion(client)


def punto_id(ruta_biblioteca: str, chunk_index: int) -> str:
    """ID determinístico por chunk, estable entre corridas (build completo e
    incremental), para poder hacer upsert/replace sin colisionar."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{ruta_biblioteca}::{chunk_index}"))


def construir_puntos(datos: list[dict]) -> list[PointStruct]:
    return [
        PointStruct(
            id=punto_id(item["metadata"]["ruta_biblioteca"], item["metadata"]["chunk_index"]),
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
        for item in datos
    ]


def upsert_ruta(ruta_biblioteca: str) -> int:
    """Inserta/actualiza en Qdrant solo los puntos de `ruta_biblioteca`, sin
    tocar el resto de la colección. Borra primero cualquier punto previo de
    ese mismo archivo (por si se re-sube una versión actualizada con menos
    chunks que la anterior)."""
    datos = [d for d in cargar_datos() if d["metadata"].get("ruta_biblioteca") == ruta_biblioteca]
    if not datos:
        raise ValueError(f"No hay vectores para ruta_biblioteca={ruta_biblioteca!r} en {VECTORS_JSON}")

    client = build_qdrant_client()
    asegurar_coleccion(client)

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="ruta_biblioteca", match=MatchValue(value=ruta_biblioteca))])
        ),
    )

    points = construir_puntos(datos)
    client.upsert(collection_name=COLLECTION_NAME, points=points)

    count = client.count(collection_name=COLLECTION_NAME).count
    print(f"Insertados {len(points)} puntos de '{ruta_biblioteca}'. Total en la colección '{COLLECTION_NAME}': {count}")
    return len(points)


def main():
    datos = cargar_datos()
    print(f"Cargados {len(datos)} registros desde {VECTORS_JSON}")

    client = build_qdrant_client()

    recrear_coleccion(client)

    points = construir_puntos(datos)

    t0 = time.time()
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    elapsed = time.time() - t0

    count = client.count(collection_name=COLLECTION_NAME).count
    print(f"\nInsertados {len(points)} puntos en {elapsed:.2f}s")
    print(f"Total de puntos en la colección '{COLLECTION_NAME}': {count}")


if __name__ == "__main__":
    main()
