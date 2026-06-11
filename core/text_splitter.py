# -*- coding: utf-8 -*-
"""
文本分块模块 - 将长文档切分为适合检索的小块
"""
from typing import List, Optional, Dict, Any
import logging

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class TextSplitter:
    """文本分块器"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        splitter_type: str = "recursive",
        separators: Optional[List[str]] = None
    ):
        """
        初始化文本分块器
        
        Args:
            chunk_size: 块大小（字符数）
            chunk_overlap: 块重叠大小
            splitter_type: 分块类型（recursive, character, token）
            separators: 分隔符列表（用于 recursive 类型）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter_type = splitter_type
        self.separators = separators or ["\n\n", "\n", "。", ".", " ", ""]
        
        # 初始化 LangChain 分块器
        self._splitter = self._create_splitter()
    
    def _create_splitter(self):
        """创建对应类型的分块器"""
        if self.splitter_type == "recursive":
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators,
                length_function=len
            )
        elif self.splitter_type == "character":
            from langchain_text_splitters import CharacterTextSplitter
            return CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator="\n",
                length_function=len
            )
        else:
            raise ValueError(f"不支持的分块类型: {self.splitter_type}")
    
    def split_text(self, text: str) -> List[str]:
        """
        分割纯文本
        
        Args:
            text: 输入文本
            
        Returns:
            文本块列表
        """
        chunks = self._splitter.split_text(text)
        logger.info(f"文本已分割为 {len(chunks)} 个块")
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        分割文档列表
        
        Args:
            documents: Document 列表
            
        Returns:
            分割后的 Document 列表
        """
        chunks = self._splitter.split_documents(documents)
        logger.info(f"文档已分割为 {len(chunks)} 个块")
        return chunks


class SemanticTextSplitter:
    """语义分块器（基于句子边界）"""
    
    def __init__(
        self,
        max_chunk_size: int = 500,
        min_chunk_size: int = 100
    ):
        """
        初始化语义分块器
        
        Args:
            max_chunk_size: 最大块大小
            min_chunk_size: 最小块大小
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
    
    def split_text(self, text: str) -> List[str]:
        """
        按语义分割文本
        
        Args:
            text: 输入文本
            
        Returns:
            文本块列表
        """
        import re
        
        # 按句子分割
        sentences = re.split(r'(?<=[。！？.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # 如果当前句子本身就超过最大块大小，直接作为一个块
            if sentence_size > self.max_chunk_size:
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                chunks.append(sentence)
                continue
            
            # 如果加入当前句子会超过最大块大小，先保存当前块
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                chunks.append(''.join(current_chunk))
                # 保留部分句子作为重叠
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # 处理剩余的句子
        if current_chunk:
            chunk_text = ''.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size or not chunks:
                chunks.append(chunk_text)
            else:
                # 合并到最后一个块
                if chunks:
                    chunks[-1] += chunk_text
                else:
                    chunks.append(chunk_text)
        
        return chunks


# 便捷函数
def create_splitter(
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    splitter_type: str = "recursive"
) -> TextSplitter:
    """
    创建文本分块器
    
    Args:
        chunk_size: 块大小
        chunk_overlap: 重叠大小
        splitter_type: 分块类型
        
    Returns:
        TextSplitter 实例
    """
    return TextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        splitter_type=splitter_type
    )


def get_chunk_strategy() -> Dict[str, Any]:
    """
    获取推荐的分块策略
    
    Returns:
        分块策略配置字典
    """
    return {
        "default": {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "splitter_type": "recursive"
        },
        "small": {
            "chunk_size": 300,
            "chunk_overlap": 30,
            "splitter_type": "recursive"
        },
        "large": {
            "chunk_size": 1000,
            "chunk_overlap": 100,
            "splitter_type": "recursive"
        },
        "semantic": {
            "max_chunk_size": 500,
            "min_chunk_size": 100,
            "splitter_type": "semantic"
        }
    }
