"""
文本分块模块 - 将长文档切分为适合检索的小块
"""
import logging
from typing import Any, Dict, List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class TextSplitter:
    """文本分块器"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        splitter_type: str = "recursive",
        separators: List[str] | None = None
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


# ==================== 智能分块策略路由 ====================

# 文件类型到分块策略的映射
FILE_TYPE_MAP = {
    # Markdown → 按标题层级切
    ".md": "markdown",
    ".markdown": "markdown",
    # 代码文件 → 按函数/类边界切
    ".py": "code",
    ".java": "code",
    ".js": "code",
    ".ts": "code",
    ".cpp": "code",
    ".c": "code",
    ".go": "code",
    # 配置文件 → 按块切（chunk_size 更小）
    ".yaml": "config",
    ".yml": "config",
    ".json": "config",
    ".toml": "config",
    ".xml": "config",
    # 长文小说 → 按段落切
    ".txt": "paragraph",
}


class SmartTextSplitter:
    """
    智能分块器（按文件类型自动选择策略 + 元数据抽取）
    
    - Markdown 文件 → 按标题层级切，保留 section / section_path 元数据
    - 代码文件 → 按函数/类切，chunk_size 更大
    - 配置文件 → 小 chunk + 按块切
    - 其他 → Recursive 通用兜底
    - 所有文件都会补充 metadata: file_type, chunk_index, total_chunks
    """

    def __init__(
        self,
        default_chunk_size: int = 500,
        default_chunk_overlap: int = 50,
        collection: str = "default",
    ):
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap
        self.collection = collection

    def _detect_strategy(self, file_name: str) -> str:
        """根据文件名识别分块策略"""
        import os
        ext = os.path.splitext(file_name)[1].lower()
        return FILE_TYPE_MAP.get(ext, "recursive")

    def _split_markdown(self, text: str) -> List[Dict[str, Any]]:
        """Markdown 专用分块器（按标题层级）"""
        from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

        headers_to_split_on = [
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
        ]

        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
        splits = header_splitter.split_text(text)

        # 对大段内容再用 recursive 细切
        final_chunks = []
        sub_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.default_chunk_size,
            chunk_overlap=self.default_chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],
            length_function=len,
        )

        for doc in splits:
            # 合并 header 信息
            section_path_parts = []
            for i in range(1, 4):
                key = f"header_{i}"
                if key in doc.metadata and doc.metadata[key]:
                    section_path_parts.append(doc.metadata[key])
            section_path = " / ".join(section_path_parts) if section_path_parts else ""

            # 对每一段做 recursive 细切
            sub_chunks = sub_splitter.split_text(doc.page_content)
            for chunk_text in sub_chunks:
                final_chunks.append({
                    "text": chunk_text,
                    "metadata_extra": {
                        "section": section_path_parts[-1] if section_path_parts else "",
                        "section_path": section_path,
                    }
                })

        return final_chunks

    def _split_code(self, text: str) -> List[Dict[str, Any]]:
        """代码文件分块器（更大 chunk_size + 代码友好分隔符）"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        code_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            separators=[
                "\nclass ", "\ndef ", "\nfunction ",  # 函数/类边界
                "\n\n", "\n",                       # 段落/行
                " ", ""
            ],
            length_function=len,
        )
        chunks = code_splitter.split_text(text)
        return [{"text": c, "metadata_extra": {"section": "code"}} for c in chunks]

    def _split_paragraph(self, text: str) -> List[Dict[str, Any]]:
        """段落式分块（长文/小说）"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        paragraph_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n\n", "\n。", "\n", "。", " ", ""],
            length_function=len,
        )
        chunks = paragraph_splitter.split_text(text)
        return [{"text": c, "metadata_extra": {"section": "paragraph"}} for c in chunks]

    def _split_recursive(self, text: str) -> List[Dict[str, Any]]:
        """通用兜底分块"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.default_chunk_size,
            chunk_overlap=self.default_chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
            length_function=len,
        )
        chunks = splitter.split_text(text)
        return [{"text": c, "metadata_extra": {"section": "general"}} for c in chunks]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        分割文档列表，自动按文件类型选择策略 + 补充元数据
        
        Args:
            documents: Document 列表
            
        Returns:
            分割后的 Document 列表（带完整 metadata）
        """
        result = []

        for doc in documents:
            source = doc.metadata.get("source", "")
            file_name = doc.metadata.get("file_name", source)
            file_type = self._detect_strategy(file_name)

            logger.info(f"[SmartSplitter] {file_name} → 策略: {file_type}")

            # 选择分块方法
            if file_type == "markdown":
                raw_chunks = self._split_markdown(doc.page_content)
            elif file_type == "code":
                raw_chunks = self._split_code(doc.page_content)
            elif file_type == "paragraph":
                raw_chunks = self._split_paragraph(doc.page_content)
            else:
                raw_chunks = self._split_recursive(doc.page_content)

            total = len(raw_chunks)
            for idx, chunk in enumerate(raw_chunks):
                merged_meta = dict(doc.metadata)  # 保留原 metadata
                merged_meta.update(chunk["metadata_extra"])
                merged_meta.update({
                    "file_type": file_type,
                    "chunk_index": idx,
                    "total_chunks": total,
                    "collection": self.collection,
                })
                result.append(Document(
                    page_content=chunk["text"],
                    metadata=merged_meta,
                ))

        logger.info(f"[SmartSplitter] {len(documents)} 文档 → {len(result)} 块")
        return result


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
