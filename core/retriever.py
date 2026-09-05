"""
检索模块 - 实现混合检索和重排序
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

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
    """混合检索器（向量检索 + BM25，支持动态权重和粗召回优化）"""

    def __init__(
        self,
        vector_store_manager,
        documents: List[Document],
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        top_k: int = 30,
        rrf_k: int = 60
    ):
        """
        初始化混合检索器
        
        Args:
            vector_store_manager: 向量数据库管理器
            documents: 原始文档列表
            vector_weight: 向量检索权重（会根据查询类型动态调整）
            bm25_weight: BM25 检索权重
            top_k: 粗召回数量（传给 reranker 精排前的候选数）
            rrf_k: RRF 融合算法的平滑常数
        """
        self.vector_store_manager = vector_store_manager
        self.documents = documents
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.top_k = top_k
        self.rrf_k = rrf_k

        # 初始化 BM25
        self._init_bm25()

    def _classify_query(self, query: str) -> str:
        """
        判断查询类型，用于动态调整 RRF 权重
        
        Returns:
            "keyword"   → 关键词密集型（数字/代码/配置项），BM25 权重 0.6
            "semantic"  → 纯自然语言描述，向量权重 0.7
            "default"   → 平衡各 0.5
        """
        import re
        
        # 关键词型：含数字编号、代码、特定术语符号
        keyword_indicators = [
            r'\d+[a-zA-Z]',         # 版本号如 v2.0, Python3
            r'[A-Z]{2,}',           # 缩写如 SQL, API, CPU
            r'[=:;]',               # 配置符号
            r'[\u4e00-\u9fff]*\d+',  # 编号如 步骤1, 方案A
            r'[a-zA-Z]+[_\-\.][a-zA-Z]+',  # 下划线/点号如 data_source
        ]
        
        for pattern in keyword_indicators:
            if re.search(pattern, query):
                return "keyword"
        
        # 语义型：含"怎么/如何/什么/为什么/介绍/理解"
        semantic_indicators = ['怎么', '如何', '什么', '为什么', '介绍', '理解', '原理', '工作', '作用', '区别']
        if any(kw in query for kw in semantic_indicators):
            return "semantic"
        
        return "default"

    def _get_dynamic_weights(self, query: str) -> Tuple[float, float]:
        """根据查询类型动态返回 (vector_weight, bm25_weight)"""
        query_type = self._classify_query(query)
        
        if query_type == "keyword":
            logger.debug("查询类型: keyword → BM25 权重提升")
            return 0.4, 0.6
        elif query_type == "semantic":
            logger.debug("查询类型: semantic → 向量权重提升")
            return 0.7, 0.3
        else:
            logger.debug("查询类型: default → 平衡权重")
            return self.vector_weight, self.bm25_weight

    def _init_bm25(self):
        """初始化 BM25 检索器"""
        try:
            import jieba
            from rank_bm25 import BM25Okapi

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

    def _bm25_search(
        self,
        query: str,
        k: int = 10,
        filter_dict: Dict[str, Any] | None = None,
    ) -> List[Tuple[int, float]]:
        """
        BM25 检索（支持 post-filter）
        """
        try:
            import jieba
            tokenized_query = list(jieba.cut(query))
        except ImportError:
            tokenized_query = query.split()

        scores = self.bm25.get_scores(tokenized_query)

        # post-filter：BM25 无原生 filter，检索后根据元数据过滤
        indexed_results = [(int(idx), float(scores[idx])) for idx in range(len(scores))]

        if filter_dict:
            filtered = []
            for idx, score in indexed_results:
                meta = self.documents[idx].metadata
                if all(meta.get(key) == val for key, val in filter_dict.items()):
                    filtered.append((idx, score))
            indexed_results = filtered

        # 取 top-k
        indexed_results.sort(key=lambda x: x[1], reverse=True)
        return indexed_results[:k]

    def _vector_search(
        self,
        query: str,
        k: int = 10,
        filter_dict: Dict[str, Any] | None = None,
    ) -> List[Tuple[int, float]]:
        """
        向量检索（直接透传 filter 给向量库）
        """
        results = self.vector_store_manager.similarity_search(
            query, k=k, filter_dict=filter_dict,
        )

        # 将结果映射回原始文档索引
        indexed_results = []
        for doc in results:
            for i, original_doc in enumerate(self.documents):
                if original_doc.page_content == doc.page_content:
                    indexed_results.append((i, 1.0))
                    break

        return indexed_results

    def _rrf_fusion(
        self,
        vector_results: List[Tuple[int, float]],
        bm25_results: List[Tuple[int, float]],
        vector_weight: float = None,
        bm25_weight: float = None,
    ) -> List[Tuple[int, float]]:
        """
        RRF (Reciprocal Rank Fusion) 融合算法
        
        Args:
            vector_results: 向量检索结果 [(idx, score), ...]
            bm25_results: BM25 检索结果 [(idx, score), ...]
            vector_weight: 向量权重（None 则用默认值）
            bm25_weight: BM25 权重
            
        Returns:
            融合后的结果 [(idx, score), ...]，返回 self.top_k 条
        """
        vw = vector_weight if vector_weight is not None else self.vector_weight
        bw = bm25_weight if bm25_weight is not None else self.bm25_weight
        k = self.rrf_k
        
        scores = {}

        # 处理向量检索结果
        for rank, (idx, _) in enumerate(vector_results):
            if idx not in scores:
                scores[idx] = 0
            scores[idx] += vw * (1.0 / (k + rank + 1))

        # 处理 BM25 结果
        for rank, (idx, _) in enumerate(bm25_results):
            if idx not in scores:
                scores[idx] = 0
            scores[idx] += bw * (1.0 / (k + rank + 1))

        # 排序并截断到粗召回量
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:self.top_k]

    def retrieve(
        self,
        query: str,
        filter: Dict[str, Any] | None = None,
    ) -> List[RetrievalResult]:
        """
        执行混合检索
        
        Args:
            query: 查询文本
            filter: 元数据过滤条件（如 {"file_type": "markdown", "source": "xxx.md"}）
            
        Returns:
            检索结果列表（粗召回 top_k 条，等 reranker 精排）
        """
        logger.info(f"执行混合检索: {query}, filter={filter}")

        # 动态权重
        vw, bw = self._get_dynamic_weights(query)

        # 粗召回：向量和 BM25 各取 self.top_k 条（默认 30 条）
        vector_results = self._vector_search(query, k=self.top_k, filter_dict=filter)
        logger.info(f"向量检索返回 {len(vector_results)} 个结果")

        bm25_results = self._bm25_search(query, k=self.top_k, filter_dict=filter)
        logger.info(f"BM25 检索返回 {len(bm25_results)} 个结果")

        # RRF 融合（带动态权重）
        fused_results = self._rrf_fusion(vector_results, bm25_results, vw, bw)
        logger.info(f"融合后返回 {len(fused_results)} 个候选")

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

    def _get_local_model_path(self, model_name: str) -> str | None:
        """获取本地模型路径"""
        import os
        from pathlib import Path

        # 检查 reranker_config.txt（专门用于重排序模型）
        config_file = Path(__file__).parent.parent / "reranker_config.txt"
        if config_file.exists():
            with open(config_file, encoding='utf-8') as f:
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
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

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
                self._model = None

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
        top_k: int = 5,
        recall_k: int = 30,
        query_transformer=None,
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
            recall_k: 粗召回数量（传 reranker 精排前的候选数）
            query_transformer: QueryTransformer 实例（可选，Multi-Query + HyDE）
        """
        self.top_k = top_k
        self.recall_k = recall_k
        self.query_transformer = query_transformer

        # 初始化混合检索器（粗召回 recall_k 条）
        if use_hybrid:
            self.hybrid_retriever = HybridRetriever(
                vector_store_manager,
                documents,
                top_k=recall_k
            )
        else:
            self.hybrid_retriever = None
            self.vector_store_manager = vector_store_manager

        # 初始化重排序器
        if use_rerank:
            self.reranker = Reranker(model_name=rerank_model, device=device)
        else:
            self.reranker = None

    def retrieve(
        self,
        query: str,
        filter: Dict[str, Any] | None = None,
    ) -> List[RetrievalResult]:
        """
        执行检索（可选 Multi-Query 扩展 + HyDE）
        
        Args:
            query: 查询文本
            filter: 元数据过滤条件
            
        Returns:
            检索结果列表（已 RRF 融合 + Rerank）
        """
        logger.info(f"开始检索: {query}, filter={filter}")

        # ========== Multi-Query 扩展 ==========
        queries_to_search = [query]
        hyde_text = None

        if self.query_transformer and self.query_transformer.is_available():
            queries_to_search = self.query_transformer.expand_queries(query)
            hyde_text = self.query_transformer.generate_hyde(query)

        logger.info(f"将执行 {len(queries_to_search)} 个查询变体")

        # ========== 每个查询变体独立检索 + 二次 RRF 融合 ==========
        all_rrf_scores: Dict[int, float] = {}

        for q_idx, q_text in enumerate(queries_to_search):
            query_label = "orig" if q_idx == 0 else f"var{q_idx}"
            try:
                variant_results = self._single_query_retrieve(q_text, filter)
                for rank, result in enumerate(variant_results):
                    doc_id = self._doc_to_id(result.document)
                    if doc_id is not None:
                        all_rrf_scores[doc_id] = all_rrf_scores.get(doc_id, 0) + (
                            1.0 / (60 + rank + 1)
                        )
                logger.info(f"  [{query_label}] {len(variant_results)} 结果")
            except Exception as e:
                logger.warning(f"  [{query_label}] 检索失败: {e}")

        # HyDE：假设性答案检索（只走向量）
        if hyde_text and self.hybrid_retriever:
            try:
                hyde_results = self.hybrid_retriever._vector_search(
                    hyde_text, k=self.recall_k, filter_dict=filter,
                )
                for rank, (idx, _) in enumerate(hyde_results):
                    all_rrf_scores[idx] = all_rrf_scores.get(idx, 0) + (
                        1.0 / (60 + rank + 1)
                    )
                logger.info(f"  [HyDE] {len(hyde_results)} 结果")
            except Exception as e:
                logger.warning(f"  [HyDE] 检索失败: {e}")

        # ========== 排序 + 截断 ==========
        if not all_rrf_scores:
            logger.warning("所有查询变体均无结果")
            return []

        sorted_docs = sorted(all_rrf_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_docs = sorted_docs[:self.recall_k]

        merged_results = []
        for rank, (doc_id, score) in enumerate(sorted_docs):
            doc = self._id_to_doc(doc_id)
            if doc is not None:
                merged_results.append(RetrievalResult(
                    document=doc,
                    score=score,
                    rank=rank + 1,
                    source="multi-query" if len(queries_to_search) > 1 else "single",
                ))

        logger.info(f"融合后共 {len(merged_results)} 个候选")

        # ========== Rerank 精排 ==========
        if self.reranker and merged_results:
            try:
                documents = [r.document for r in merged_results]
                final_results = self.reranker.rerank(query, documents, top_k=self.top_k)
                logger.info(f"重排序后返回 {len(final_results)} 个结果")
                return final_results
            except Exception as e:
                logger.warning(f"重排序失败，使用原始检索结果: {e}")
                return merged_results[:self.top_k]

        return merged_results[:self.top_k]

    # ========== 辅助方法 ==========

    def _doc_to_id(self, doc: Document) -> int | None:
        """文档 → 原始 documents 列表索引"""
        docs = self.hybrid_retriever.documents if self.hybrid_retriever else []
        for i, d in enumerate(docs):
            if d.page_content == doc.page_content:
                return i
        return None

    def _id_to_doc(self, idx: int) -> Document | None:
        """索引 → 文档"""
        docs = self.hybrid_retriever.documents if self.hybrid_retriever else []
        if 0 <= idx < len(docs):
            return docs[idx]
        return None

    def _single_query_retrieve(
        self, query: str, filter: Dict[str, Any] | None = None
    ) -> List[RetrievalResult]:
        """单个查询变体的检索（不含 Multi-Query 包装）"""
        if self.hybrid_retriever:
            return self.hybrid_retriever.retrieve(query, filter=filter)

        docs = self.vector_store_manager.similarity_search(
            query, k=self.recall_k, filter_dict=filter,
        )
        return [
            RetrievalResult(doc, 1.0, i + 1, "vector")
            for i, doc in enumerate(docs)
        ]
