# -*- coding: utf-8 -*-
"""
文档加载模块 - 支持多种文档格式的解析与加载
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
import logging

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class BaseDocumentLoader(ABC):
    """文档加载器基类"""
    
    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        """
        加载文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            Document 列表
        """
        pass
    
    @abstractmethod
    def supports(self, file_extension: str) -> bool:
        """
        检查是否支持该文件类型
        
        Args:
            file_extension: 文件扩展名（如 .pdf, .docx）
            
        Returns:
            是否支持
        """
        pass


class PDFLoader(BaseDocumentLoader):
    """PDF 文档加载器"""
    
    def load(self, file_path: str) -> List[Document]:
        """加载 PDF 文档"""
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # 添加文件元数据
            for doc in documents:
                doc.metadata.update({
                    'source': file_path,
                    'file_type': 'pdf',
                    'file_name': Path(file_path).name
                })
            
            logger.info(f"成功加载 PDF 文件: {file_path}, 共 {len(documents)} 页")
            return documents
            
        except Exception as e:
            logger.error(f"加载 PDF 文件失败: {file_path}, 错误: {str(e)}")
            raise
    
    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() == '.pdf'


class DocxLoader(BaseDocumentLoader):
    """Word 文档加载器"""
    
    def load(self, file_path: str) -> List[Document]:
        """加载 Word 文档"""
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            
            # 添加文件元数据
            for doc in documents:
                doc.metadata.update({
                    'source': file_path,
                    'file_type': 'docx',
                    'file_name': Path(file_path).name
                })
            
            logger.info(f"成功加载 Word 文件: {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"加载 Word 文件失败: {file_path}, 错误: {str(e)}")
            raise
    
    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() in ['.docx', '.doc']


class TextLoader(BaseDocumentLoader):
    """文本文件加载器（支持 .txt, .md, .py 等）"""
    
    def load(self, file_path: str) -> List[Document]:
        """加载文本文件"""
        try:
            from langchain_community.document_loaders import TextLoader as LangChainTextLoader
            loader = LangChainTextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            
            # 添加文件元数据
            file_ext = Path(file_path).suffix.lower()
            for doc in documents:
                doc.metadata.update({
                    'source': file_path,
                    'file_type': file_ext.lstrip('.'),
                    'file_name': Path(file_path).name
                })
            
            logger.info(f"成功加载文本文件: {file_path}")
            return documents
            
        except UnicodeDecodeError:
            # 尝试使用其他编码
            try:
                loader = LangChainTextLoader(file_path, encoding='gbk')
                documents = loader.load()
                file_ext = Path(file_path).suffix.lower()
                for doc in documents:
                    doc.metadata.update({
                        'source': file_path,
                        'file_type': file_ext.lstrip('.'),
                        'file_name': Path(file_path).name
                    })
                return documents
            except Exception as e:
                logger.error(f"加载文本文件失败: {file_path}, 错误: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"加载文本文件失败: {file_path}, 错误: {str(e)}")
            raise
    
    def supports(self, file_extension: str) -> bool:
        text_extensions = ['.txt', '.md', '.markdown', '.py', '.java', '.cpp', 
                          '.c', '.h', '.js', '.ts', '.html', '.css', '.json', 
                          '.xml', '.yaml', '.yml', '.ini', '.conf', '.sh', '.bat']
        return file_extension.lower() in text_extensions


class DocumentLoaderFactory:
    """文档加载器工厂"""
    
    _loaders: List[BaseDocumentLoader] = [
        PDFLoader(),
        DocxLoader(),
        TextLoader(),
    ]
    
    @classmethod
    def get_loader(cls, file_path: str) -> BaseDocumentLoader:
        """
        根据文件路径获取对应的加载器
        
        Args:
            file_path: 文件路径
            
        Returns:
            文档加载器
            
        Raises:
            ValueError: 不支持的文件类型
        """
        file_extension = Path(file_path).suffix.lower()
        
        for loader in cls._loaders:
            if loader.supports(file_extension):
                return loader
        
        raise ValueError(f"不支持的文件类型: {file_extension}")
    
    @classmethod
    def load_document(cls, file_path: str) -> List[Document]:
        """
        加载文档（便捷方法）
        
        Args:
            file_path: 文件路径
            
        Returns:
            Document 列表
        """
        loader = cls.get_loader(file_path)
        return loader.load(file_path)
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """获取支持的文件扩展名列表"""
        extensions = []
        for loader in cls._loaders:
            if isinstance(loader, PDFLoader):
                extensions.append('.pdf')
            elif isinstance(loader, DocxLoader):
                extensions.extend(['.docx', '.doc'])
            elif isinstance(loader, TextLoader):
                extensions.extend(['.txt', '.md', '.markdown', '.py', '.java', 
                                 '.cpp', '.c', '.h', '.js', '.ts', '.html', 
                                 '.css', '.json', '.xml', '.yaml', '.yml'])
        return extensions


# 便捷函数
def load_document(file_path: str) -> List[Document]:
    """
    加载单个文档
    
    Args:
        file_path: 文件路径
        
    Returns:
        Document 列表
    """
    return DocumentLoaderFactory.load_document(file_path)


def load_documents_from_directory(directory: str, recursive: bool = True) -> List[Document]:
    """
    从目录加载所有支持的文档
    
    Args:
        directory: 目录路径
        recursive: 是否递归子目录
        
    Returns:
        Document 列表
    """
    from pathlib import Path
    
    documents = []
    dir_path = Path(directory)
    
    if not dir_path.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    
    supported_extensions = DocumentLoaderFactory.get_supported_extensions()
    
    # 构建搜索模式
    if recursive:
        files = dir_path.rglob("*")
    else:
        files = dir_path.glob("*")
    
    for file_path in files:
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                docs = load_document(str(file_path))
                documents.extend(docs)
                logger.info(f"已加载: {file_path}")
            except Exception as e:
                logger.error(f"加载文件失败: {file_path}, 错误: {str(e)}")
    
    logger.info(f"总共加载了 {len(documents)} 个文档片段")
    return documents
