"""
RAG 知识库问答系统 - 核心模块
"""

from .document_loader import DocumentLoaderFactory, load_document, load_documents_from_directory
from .embedding import EmbeddingManager, VectorStoreManager, get_recommended_models
from .qa_generator import ContextCompressor, QAGenerator, QAResponse
from .query_transformer import QueryTransformer
from .retriever import AdvancedRetriever, HybridRetriever, Reranker, RetrievalResult
from .text_splitter import SmartTextSplitter, TextSplitter, create_splitter, get_chunk_strategy

__all__ = [
    # Document Loader
    "DocumentLoaderFactory",
    "load_document",
    "load_documents_from_directory",
    # Text Splitter
    "TextSplitter",
    "SmartTextSplitter",
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
    # Query Transformer
    "QueryTransformer",
    # QA Generator
    "QAGenerator",
    "ContextCompressor",
    "QAResponse",
]
