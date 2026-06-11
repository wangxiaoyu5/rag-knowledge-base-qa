# -*- coding: utf-8 -*-
"""
问答生成模块 - 基于检索结果生成答案
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class QAResponse:
    """问答响应"""
    answer: str
    sources: List[Document]
    confidence: float
    processing_time: float


class ContextCompressor:
    """上下文压缩器"""
    
    def __init__(
        self,
        max_context_length: int = 3000,
        compression_ratio: float = 0.5
    ):
        """
        初始化上下文压缩器
        
        Args:
            max_context_length: 最大上下文长度
            compression_ratio: 压缩比例
        """
        self.max_context_length = max_context_length
        self.compression_ratio = compression_ratio
    
    def compress(self, documents: List[Document], query: str) -> str:
        """
        压缩上下文
        
        Args:
            documents: 检索到的文档列表
            query: 查询文本
            
        Returns:
            压缩后的上下文字符串
        """
        if not documents:
            return ""
        
        contexts = []
        total_length = 0
        
        for i, doc in enumerate(documents, 1):
            content = doc.page_content.strip()
            source = doc.metadata.get('source', '未知来源')
            
            # 构建上下文片段
            context = f"[文档 {i}] 来源: {source}\n{content}\n"
            
            # 检查是否超过最大长度
            if total_length + len(context) > self.max_context_length:
                # 截断或跳过
                remaining = self.max_context_length - total_length
                if remaining > 100:  # 至少保留一些内容
                    context = context[:remaining] + "...\n"
                    contexts.append(context)
                break
            
            contexts.append(context)
            total_length += len(context)
        
        return "\n".join(contexts)


class QAGenerator:
    """问答生成器"""
    
    def __init__(
        self,
        model_type: str = "openai",
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        api_key: Optional[str] = None
    ):
        """
        初始化问答生成器
        
        Args:
            model_type: 模型类型（openai/local）
            model_name: 模型名称
            temperature: 生成温度
            max_tokens: 最大生成token数
            api_key: API密钥
        """
        self.model_type = model_type
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        
        # 初始化上下文压缩器
        self.context_compressor = ContextCompressor()
        
        # 初始化模型
        self._model = None
    
    def _load_model(self):
        """懒加载模型"""
        if self._model is None:
            if self.model_type == "openai":
                from langchain_openai import ChatOpenAI
                self._model = ChatOpenAI(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    api_key=self.api_key
                )
            else:
                raise ValueError(f"不支持的模型类型: {self.model_type}")
            
            logger.info(f"问答模型加载完成: {self.model_name}")
        
        return self._model
    
    def _build_prompt(self, query: str, context: str) -> str:
        """
        构建提示词
        
        Args:
            query: 查询文本
            context: 上下文
            
        Returns:
            提示词字符串
        """
        prompt = f"""基于以下参考文档回答问题。如果文档中没有相关信息，请明确说明无法回答。

参考文档：
{context}

用户问题：{query}

请提供准确、简洁的回答，并基于文档内容。如果引用了具体信息，请说明来源。"""
        
        return prompt
    
    def generate(
        self,
        query: str,
        documents: List[Document],
        stream: bool = False
    ) -> QAResponse:
        """
        生成答案
        
        Args:
            query: 查询文本
            documents: 检索到的文档列表
            stream: 是否流式输出
            
        Returns:
            问答响应
        """
        import time
        start_time = time.time()
        
        # 压缩上下文
        context = self.context_compressor.compress(documents, query)
        
        if not context:
            return QAResponse(
                answer="抱歉，未找到相关信息。",
                sources=[],
                confidence=0.0,
                processing_time=time.time() - start_time
            )
        
        # 构建提示词
        prompt = self._build_prompt(query, context)
        
        try:
            model = self._load_model()
            
            # 生成答案
            if stream:
                # 流式生成
                answer = ""
                for chunk in model.stream(prompt):
                    answer += chunk.content
            else:
                # 非流式生成
                response = model.invoke(prompt)
                answer = response.content
            
            # 计算置信度（简化版本）
            confidence = self._calculate_confidence(documents, query, answer)
            
            processing_time = time.time() - start_time
            
            return QAResponse(
                answer=answer,
                sources=documents,
                confidence=confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"生成答案失败: {str(e)}")
            return QAResponse(
                answer=f"生成答案时出错: {str(e)}",
                sources=documents,
                confidence=0.0,
                processing_time=time.time() - start_time
            )
    
    def _calculate_confidence(
        self,
        documents: List[Document],
        query: str,
        answer: str
    ) -> float:
        """
        计算置信度（简化实现）
        
        Args:
            documents: 检索文档
            query: 查询
            answer: 生成的答案
            
        Returns:
            置信度分数 (0-1)
        """
        if not documents:
            return 0.0
        
        # 基于文档数量和答案长度计算简单置信度
        doc_score = min(len(documents) / 5.0, 1.0)  # 最多5个文档得满分
        length_score = min(len(answer) / 100.0, 1.0)  # 至少100字符
        
        confidence = (doc_score * 0.6 + length_score * 0.4)
        return round(confidence, 2)
    
    def generate_with_history(
        self,
        query: str,
        documents: List[Document],
        chat_history: List[Dict[str, str]],
        stream: bool = False
    ) -> QAResponse:
        """
        基于对话历史生成答案
        
        Args:
            query: 当前查询
            documents: 检索文档
            chat_history: 对话历史 [{"role": "user/assistant", "content": "..."}, ...]
            stream: 是否流式输出
            
        Returns:
            问答响应
        """
        import time
        start_time = time.time()
        
        # 压缩上下文
        context = self.context_compressor.compress(documents, query)
        
        # 构建包含历史的提示词
        history_str = ""
        for msg in chat_history[-5:]:  # 只保留最近5轮
            role = "用户" if msg["role"] == "user" else "助手"
            history_str += f"{role}: {msg['content']}\n"
        
        prompt = f"""基于以下参考文档和对话历史回答问题。

参考文档：
{context}

对话历史：
{history_str}

当前问题：{query}

请提供准确、简洁的回答，考虑对话上下文。"""
        
        try:
            model = self._load_model()
            
            if stream:
                answer = ""
                for chunk in model.stream(prompt):
                    answer += chunk.content
            else:
                response = model.invoke(prompt)
                answer = response.content
            
            confidence = self._calculate_confidence(documents, query, answer)
            processing_time = time.time() - start_time
            
            return QAResponse(
                answer=answer,
                sources=documents,
                confidence=confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"生成答案失败: {str(e)}")
            return QAResponse(
                answer=f"生成答案时出错: {str(e)}",
                sources=documents,
                confidence=0.0,
                processing_time=time.time() - start_time
            )
