"""Factories de clientes (vLLM embeddings, vLLM chat, Qdrant, sparse BM25).

Reemplaza las construcciones de OpenAIEmbeddings/OpenAI/QdrantClient que hoy
están duplicadas en ingesta/embeddings.py, consulta/rag_query1.py,
consulta/normalizar_query.py y varios scripts de tests/eval/.
"""

import httpx
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from qdrant_client import QdrantClient

from rag312.config import LLMRoleConfig, get_embed_config, get_rerank_config, settings


def build_embedder(role_config: LLMRoleConfig | None = None) -> OpenAIEmbeddings:
    role_config = role_config or get_embed_config()
    return OpenAIEmbeddings(
        base_url=role_config.url,
        api_key="no-necesaria",
        model=role_config.model,
        check_embedding_ctx_length=False,
    )


def build_sparse_embedder():
    from fastembed import SparseTextEmbedding
    import os

    return SparseTextEmbedding(
        model_name=settings.sparse_model_name,
        cache_dir=os.environ.get("FASTEMBED_CACHE_DIR", settings.fastembed_cache_dir),
        language=settings.sparse_language,
    )


def build_llm_client(role_config: LLMRoleConfig) -> OpenAI:
    return OpenAI(base_url=role_config.url, api_key="no-necesaria")


def build_qdrant_client() -> QdrantClient:
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def build_rerank_client(role_config: LLMRoleConfig | None = None) -> httpx.Client:
    """Cliente HTTP para el endpoint /rerank de vLLM (estilo Cohere), no cubierto
    por el SDK de openai."""
    role_config = role_config or get_rerank_config()
    return httpx.Client(base_url=role_config.url, timeout=settings.rerank_timeout_seg)
