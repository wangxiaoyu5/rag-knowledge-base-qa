# -*- coding: utf-8 -*-
"""
RAG 知识库问答系统 - 核心模块
"""

from .document_loader import (
    DocumentLoaderFactory,
    load_document,
    load_documents_from_directory
)
from .text_splitter import TextSplitter, create_splitter, get_chunk_strategy
from .embedding import EmbeddingManager, VectorStoreManager, get_recommended_models
from .retriever import (
    HybridRetriever,
    Reranker,
    AdvancedRetriever,
    RetrievalResult
)
from .qa_generator import QAGenerator, ContextCompressor, QAResponse

__all__ = [
    # Document Loader
    "DocumentLoaderFactory",
    "load_document",
    "load_documents_from_directory",
    # Text Splitter
    "TextSplitter",
    "create_splitter",
    "get_chunk_strategy",
    # Embedding
    "EmbeddingManager",
    "VectorStoreManager",
    "get_recommended_models",
    # Retriever
    "HybridRetriever",
    "Reranker",
    "AdvancedRetriever",
    "RetrievalResult",
    # QA Generator
    "QAGenerator",
    "ContextCompressor",
    "QAResponse",
]
