"""Configuración centralizada del pipeline RAG312.

Reúne las constantes que hoy están duplicadas en ingesta/, consulta/,
interfaz/ y tests/eval/: colección Qdrant, puertos/modelos de los distintos
roles de vLLM, rutas del proyecto y demás parámetros de retrieval/embeddings.

Los roles de LLM (embed/generate/normalize/golden-set/frontera/judge/
ground-truth) se exponen como funciones en vez de campos de `Settings`
porque dos roles distintos comparten hoy el mismo nombre de variable de
entorno con default distinto (ver `get_judge_filter_config` /
`get_judge_ragas_config`) — algo que un único modelo pydantic no puede
representar.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    rag_project_dir: Path = Path.home() / "rag312_else"

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "procedimientos_sielse"
    vector_dim: int = 1024
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "bm25"

    top_k: int = 10
    prefetch_limit: int = 20
    umbral_similitud: float = 0.5
    instruction_prefix: str = "Instruct: Retrieve relevant passages that answer the query\nQuery: "

    cuda_visible_devices_ocr: str = "4,5,6,7"

    batch_size_embeddings: int = 16
    sparse_model_name: str = "Qdrant/bm25"
    sparse_language: str = "spanish"
    # TODO: el default histórico apunta a ~/rag/... (typo de un rename previo
    # ~/rag -> ~/rag312). En la práctica no importa porque containers/env.sh
    # siempre exporta FASTEMBED_CACHE_DIR antes de correr cualquier script,
    # pero se preserva el valor tal cual en vez de "corregirlo" en silencio.
    fastembed_cache_dir: str = str(Path.home() / "rag/modelos/fastembed_cache")

    @property
    def biblioteca_dir(self) -> Path:
        return self.rag_project_dir / "biblioteca"

    @property
    def salida_md_dir(self) -> Path:
        return self.rag_project_dir / "salida_md"

    @property
    def salida_chunks_dir(self) -> Path:
        return self.rag_project_dir / "salida_chunks"

    @property
    def datos_dir(self) -> Path:
        return self.rag_project_dir / "datos"


settings = Settings()


@dataclass(frozen=True)
class LLMRoleConfig:
    url: str
    model: str


def get_embed_config() -> LLMRoleConfig:
    return LLMRoleConfig(
        url=os.environ.get("EMBED_URL", "http://localhost:8001/v1"),
        model=os.environ.get("EMBED_MODEL", "harrier-embed"),
    )


def get_generate_config() -> LLMRoleConfig:
    return LLMRoleConfig(
        url=os.environ.get("GENERATE_LLM_URL", "http://localhost:8002/v1"),
        model=os.environ.get("GENERATE_LLM_MODEL", "qwen35-9b"),
    )


def get_normalize_config() -> LLMRoleConfig:
    return LLMRoleConfig(
        url=os.environ.get("NORMALIZE_LLM_URL", "http://localhost:8003/v1"),
        model=os.environ.get("NORMALIZE_LLM_MODEL", "qwen35-4b"),
    )


def get_golden_set_config() -> LLMRoleConfig:
    return LLMRoleConfig(
        url=os.environ.get("GOLDEN_LLM_URL", "http://localhost:8002/v1"),
        model=os.environ.get("GOLDEN_LLM_MODEL", "qwen35-9b"),
    )


def get_frontera_config() -> LLMRoleConfig:
    return LLMRoleConfig(
        url=os.environ.get("FRONTERA_LLM_URL", "http://localhost:8002/v1"),
        model=os.environ.get("FRONTERA_LLM_MODEL", "qwen35-9b"),
    )


def get_judge_filter_config() -> LLMRoleConfig:
    """Juez de filtrar_golden_set.py: 4B rápido, puerto 8003 (compartido con el normalizador)."""
    return LLMRoleConfig(
        url=os.environ.get("JUDGE_LLM_URL", "http://localhost:8003/v1"),
        model=os.environ.get("JUDGE_LLM_MODEL", "qwen35-4b"),
    )


def get_judge_ragas_config() -> LLMRoleConfig:
    """Juez de eval_generation.py (RAGAS): 9B en instancia dedicada, puerto 8004.

    OJO: usa los MISMOS nombres de variable de entorno (JUDGE_LLM_URL /
    JUDGE_LLM_MODEL) que get_judge_filter_config(), pero con default
    distinto — reproduce exactamente el comportamiento actual del repo, con
    el mismo riesgo latente si una shell exporta esas variables para un
    script y luego se corre el otro.
    """
    return LLMRoleConfig(
        url=os.environ.get("JUDGE_LLM_URL", "http://localhost:8004/v1"),
        model=os.environ.get("JUDGE_LLM_MODEL", "qwen35-9b"),
    )


def get_ground_truth_config() -> LLMRoleConfig:
    return LLMRoleConfig(
        url=os.environ.get("GROUND_TRUTH_LLM_URL", "http://localhost:8004/v1"),
        model=os.environ.get("GROUND_TRUTH_LLM_MODEL", "qwen35-9b"),
    )
