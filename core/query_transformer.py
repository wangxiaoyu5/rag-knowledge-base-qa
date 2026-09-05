"""
查询变换模块 - Multi-Query 扩展 + HyDE

业界公认效果最显著的 RAG 优化之一：
- Multi-Query：LLM 把用户 query 扩展成 3-4 个变体，并行检索后 RRF 融合
- HyDE：LLM 生成假设性答案，用答案 embedding 去检索（跳过 BM25）
- 两者均有 try/except 降级：LLM 不可用时退化为单个原始 query 检索
"""
import logging
import os
from typing import List

logger = logging.getLogger(__name__)


class QueryTransformer:
    """
    查询变换器（Multi-Query 扩展 + HyDE）
    
    使用前需提供 API Key（或通过环境变量），无 API 时自动降级。
    """

    MULTI_QUERY_PROMPT = """你是一个检索查询优化助手。请将用户的问题扩展为 3-4 个不同的表述，以便更全面地检索到相关信息。
要求：
1. 保持原意不变，仅改变表述方式
2. 覆盖同义词、不同句式、可能的关键词
3. 每行一个变体，只输出变体本身，不要其他内容

原始问题：{query}

变体列表："""

    HYDE_PROMPT = """你是一个假设性文档生成助手。请针对用户的问题，生成一段"假设性答案"——内容可以不准确，但措辞和风格要像真实文档中的一段话。
要求：
1. 用 2-3 句话，像文档/维基百科的风格
2. 内容可以是猜测的，但 wording 要自然
3. 只输出这段假设性答案，不要其他内容

用户问题：{query}

假设性答案："""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        enable_multi_query: bool = True,
        enable_hyde: bool = True,
        num_variants: int = 4,
    ):
        """
        Args:
            api_key: LLM API Key（None 则尝试读环境变量）
            api_base: LLM API Base URL
            model: 模型名称
            enable_multi_query: 是否启用 Multi-Query
            enable_hyde: 是否启用 HyDE
            num_variants: Multi-Query 生成的变体数量
        """
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.api_base = api_base or os.environ.get("LLM_API_BASE")
        self.model = model
        self.enable_multi_query = enable_multi_query
        self.enable_hyde = enable_hyde
        self.num_variants = num_variants

        # 检查是否可用
        self._available = bool(self.api_key)

        if self._available:
            self._init_llm()
        else:
            logger.warning("[QueryTransformer] 无 API Key，Multi-Query/HyDE 将自动降级")

    def _init_llm(self):
        """初始化 LLM 客户端"""
        try:
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(
                api_key=self.api_key,
                base_url=self.api_base if self.api_base else None,
                model=self.model,
                temperature=0.1,  # 创造性低，保持稳定
                max_tokens=256,
            )
            logger.info("[QueryTransformer] LLM 初始化成功")
        except Exception as e:
            logger.warning(f"[QueryTransformer] LLM 初始化失败: {e}")
            self._available = False

    def expand_queries(self, query: str) -> List[str]:
        """
        Multi-Query：生成查询变体
        
        Returns:
            查询列表，第一个是原始 query，后面是 LLM 生成的变体
            如果 LLM 不可用，只返回 [原始 query]
        """
        if not self._available or not self.enable_multi_query:
            logger.info("[Multi-Query] 不可用，返回原始 query")
            return [query]

        try:
            from langchain_core.prompts import PromptTemplate

            prompt = PromptTemplate(
                template=self.MULTI_QUERY_PROMPT,
                input_variables=["query"],
            )
            chain = prompt | self.llm
            response = chain.invoke({"query": query})
            raw_text = response.content.strip()

            # 按行切分，取前 N 条非空的
            variants = [
                line.strip()
                for line in raw_text.split("\n")
                if line.strip() and not line.strip().startswith(("变体", "-", "•"))
            ]
            variants = variants[:self.num_variants]

            logger.info(f"[Multi-Query] 原始 query → {len(variants)} 个变体")
            for i, v in enumerate(variants):
                logger.debug(f"  变体{i+1}: {v}")

            # 原始 query 放第一个
            return [query] + variants

        except Exception as e:
            logger.warning(f"[Multi-Query] 生成失败，降级: {e}")
            return [query]

    def generate_hyde(self, query: str) -> str | None:
        """
        HyDE：生成假设性答案
        
        Returns:
            假设性答案文本，LLM 不可用或失败返回 None
        """
        if not self._available or not self.enable_hyde:
            return None

        try:
            from langchain_core.prompts import PromptTemplate

            prompt = PromptTemplate(
                template=self.HYDE_PROMPT,
                input_variables=["query"],
            )
            chain = prompt | self.llm
            response = chain.invoke({"query": query})
            hyde_text = response.content.strip()

            logger.info(f"[HyDE] 生成假设性答案: {hyde_text[:80]}...")
            return hyde_text

        except Exception as e:
            logger.warning(f"[HyDE] 生成失败，跳过: {e}")
            return None

    def is_available(self) -> bool:
        return self._available
