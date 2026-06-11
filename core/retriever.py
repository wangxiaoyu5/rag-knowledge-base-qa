# -*- coding: utf-8 -*-
"""
检索模块 - 实现混合检索和重排序
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """检索结果"""
    document: Document
    score: float
    rank: int
    source: str  # 来源：vector, bm25, hybrid


class HybridRetriever:
    """混合检索器（向量检索 + BM25）"""
    
    def __init__(
        self,
        vector_store_manager,
        documents: List[Document],
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        top_k: int = 10
    ):
        """
        初始化混合检索器
        
        Args:
            vector_store_manager: 向量数据库管理器
            documents: 原始文档列表
            vector_weight: 向量检索权重
            bm25_weight: BM25 检索权重
            top_k: 返回结果数量
        """
        self.vector_store_manager = vector_store_manager
        self.documents = documents
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.top_k = top_k
        
        # 初始化 BM25
        self._init_bm25()
    
    def _init_bm25(self):
        """初始化 BM25 检索器"""
        try:
            from rank_bm25 import BM25Okapi
            import jieba
            
            # 对文档进行分词
            self.tokenized_docs = []
            for doc in self.documents:
                tokens = list(jieba.cut(doc.page_content))
                self.tokenized_docs.append(tokens)
            
            self.bm25 = BM25Okapi(self.tokenized_docs)
            logger.info("BM25 检索器初始化完成")
        except ImportError:
            logger.warning("未安装 jieba，BM25 将使用简单分词")
            from rank_bm25 import BM25Okapi
            
            self.tokenized_docs = [doc.page_content.split() for doc in self.documents]
            self.bm25 = BM25Okapi(self.tokenized_docs)
    
    def _bm25_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """
        BM25 检索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            (文档索引, 分数) 列表
        """
        try:
            import jieba
            tokenized_query = list(jieba.cut(query))
        except ImportError:
            tokenized_query = query.split()
        
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取 top-k 结果
        top_indices = scores.argsort()[-k:][::-1]
        results = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
        
        return results
    
    def _vector_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """
        向量检索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            (文档索引, 分数) 列表
        """
        results = self.vector_store_manager.similarity_search(query, k=k)
        
        # 将结果映射回原始文档索引
        indexed_results = []
        for doc in results:
            # 查找文档在原始列表中的索引
            for i, original_doc in enumerate(self.documents):
                if original_doc.page_content == doc.page_content:
                    indexed_results.append((i, 1.0))  # 简化分数
                    break
        
        return indexed_results
    
    def _rrf_fusion(
        self,
        vector_results: List[Tuple[int, float]],
        bm25_results: List[Tuple[int, float]],
        k: int = 60
    ) -> List[Tuple[int, float]]:
        """
        RRF (Reciprocal Rank Fusion) 融合算法
        
        Args:
            vector_results: 向量检索结果 [(idx, score), ...]
            bm25_results: BM25 检索结果 [(idx, score), ...]
            k: RRF 常数
            
        Returns:
            融合后的结果 [(idx, score), ...]
        """
        scores = {}
        
        # 处理向量检索结果
        for rank, (idx, _) in enumerate(vector_results):
            if idx not in scores:
                scores[idx] = 0
            scores[idx] += self.vector_weight * (1.0 / (k + rank + 1))
        
        # 处理 BM25 结果
        for rank, (idx, _) in enumerate(bm25_results):
            if idx not in scores:
                scores[idx] = 0
            scores[idx] += self.bm25_weight * (1.0 / (k + rank + 1))
        
        # 排序
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:self.top_k]
    
    def retrieve(self, query: str) -> List[RetrievalResult]:
        """
        执行混合检索
        
        Args:
            query: 查询文本
            
        Returns:
            检索结果列表
        """
        logger.info(f"执行混合检索: {query}")
        
        # 向量检索
        vector_results = self._vector_search(query, k=self.top_k * 2)
        logger.info(f"向量检索返回 {len(vector_results)} 个结果")
        
        # BM25 检索
        bm25_results = self._bm25_search(query, k=self.top_k * 2)
        logger.info(f"BM25 检索返回 {len(bm25_results)} 个结果")
        
        # RRF 融合
        fused_results = self._rrf_fusion(vector_results, bm25_results)
        logger.info(f"融合后返回 {len(fused_results)} 个结果")
        
        # 构建结果
        results = []
        for rank, (idx, score) in enumerate(fused_results):
            doc = self.documents[idx]
            results.append(RetrievalResult(
                document=doc,
                score=score,
                rank=rank + 1,
                source="hybrid"
            ))
        
        return results


class Reranker:
    """重排序器（使用 Cross-Encoder）"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: str = "cpu"
    ):
        """
        初始化重排序器
        
        Args:
            model_name: 重排序模型名称
            device: 运行设备
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._use_fallback = False
    
    def _get_local_model_path(self, model_name: str) -> Optional[str]:
        """获取本地模型路径"""
        from pathlib import Path
        import os
        
        # 检查 reranker_config.txt（专门用于重排序模型）
        config_file = Path(__file__).parent.parent / "reranker_config.txt"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('RERANKER_MODEL_PATH='):
                        return line.strip().split('=', 1)[1]
        
        # 检查默认模型目录
        project_root = Path(__file__).parent.parent
        models_dir = project_root / "models"
        
        # 尝试不同的路径格式
        possible_paths = [
            models_dir / model_name.replace('/', os.sep),
            models_dir / "hub" / model_name.replace('/', os.sep),
        ]
        
        for path in possible_paths:
            if path.exists() and path.is_dir():
                return str(path)
        
        return None
    
    def _load_model(self):
        """懒加载模型"""
        if self._model is None:
            import os
            # 设置 HuggingFace 镜像源
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            
            # 先检查是否有本地模型可用
            local_path = self._get_local_model_path(self.model_name)
            
            if local_path:
                logger.info(f"使用本地重排序模型: {local_path}")
                try:
                    from sentence_transformers import CrossEncoder
                    self._model = CrossEncoder(
                        local_path,
                        device=self.device
                    )
                    logger.info("本地重排序模型加载完成")
                    return self._model
                except Exception as e:
                    logger.warning(f"加载本地重排序模型失败: {e}")
            
            # 尝试从 HuggingFace 下载
            logger.info(f"正在下载重排序模型: {self.model_name}")
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(
                    self.model_name,
                    device=self.device
                )
                logger.info("重排序模型加载完成")
            except Exception as e:
                logger.error(f"下载重排序模型失败: {e}")
                logger.warning("将使用无重排序模式（按向量相似度排序）")
                self._use_fallback = True
                
        return self._model
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        对检索结果进行重排序
        
        Args:
            query: 查询文本
            documents: 待重排序的文档列表
            top_k: 返回结果数量
            
        Returns:
            重排序后的结果
        """
        if not documents:
            return []
        
        model = self._load_model()
        
        # 如果模型加载失败，使用回退模式（按原始顺序返回）
        if self._use_fallback or model is None:
            logger.warning("使用回退模式：按原始顺序返回结果（无重排序）")
            results = []
            for i, doc in enumerate(documents[:top_k]):
                results.append(RetrievalResult(
                    document=doc,
                    score=1.0 - (i * 0.1),  # 简单的递减分数
                    rank=i + 1,
                    source="vector"  # 标记为向量检索结果
                ))
            return results
        
        # 构建查询-文档对
        pairs = [[query, doc.page_content] for doc in documents]
        
        # 计算相关性分数
        scores = model.predict(pairs)
        
        # 构建结果并排序
        results = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            results.append(RetrievalResult(
                document=doc,
                score=float(score),
                rank=i + 1,
                source="rerank"
            ))
        
        # 按分数排序
        results.sort(key=lambda x: x.score, reverse=True)
        
        # 重新设置排名
        for i, result in enumerate(results[:top_k]):
            result.rank = i + 1
        
        return results[:top_k]


class AdvancedRetriever:
    """高级检索器（整合混合检索和重排序）"""
    
    def __init__(
        self,
        vector_store_manager,
        documents: List[Document],
        use_hybrid: bool = True,
        use_rerank: bool = True,
        rerank_model: str = "BAAI/bge-reranker-large",
        device: str = "cpu",
        top_k: int = 5
    ):
        """
        初始化高级检索器
        
        Args:
            vector_store_manager: 向量数据库管理器
            documents: 原始文档列表
            use_hybrid: 是否使用混合检索
            use_rerank: 是否使用重排序
            rerank_model: 重排序模型
            device: 运行设备
            top_k: 最终返回结果数量
        """
        self.top_k = top_k
        
        # 初始化混合检索器
        if use_hybrid:
            self.hybrid_retriever = HybridRetriever(
                vector_store_manager,
                documents,
                top_k=top_k * 2  # 检索更多供重排序
            )
        else:
            self.hybrid_retriever = None
            self.vector_store_manager = vector_store_manager
        
        # 初始化重排序器
        if use_rerank:
            self.reranker = Reranker(model_name=rerank_model, device=device)
        else:
            self.reranker = None
    
    def retrieve(self, query: str) -> List[RetrievalResult]:
        """
        执行检索
        
        Args:
            query: 查询文本
            
        Returns:
            检索结果列表
        """
        logger.info(f"开始检索: {query}")
        
        # 第一阶段：检索
        if self.hybrid_retriever:
            initial_results = self.hybrid_retriever.retrieve(query)
        else:
            # 仅使用向量检索
            docs = self.vector_store_manager.similarity_search(query, k=self.top_k * 2)
            initial_results = [
                RetrievalResult(doc, 1.0, i + 1, "vector")
                for i, doc in enumerate(docs)
            ]
        
        logger.info(f"初始检索返回 {len(initial_results)} 个结果")
        
        # 第二阶段：重排序
        if self.reranker and initial_results:
            documents = [r.document for r in initial_results]
            final_results = self.reranker.rerank(query, documents, top_k=self.top_k)
            logger.info(f"重排序后返回 {len(final_results)} 个结果")
            return final_results
        else:
            return initial_results[:self.top_k]
