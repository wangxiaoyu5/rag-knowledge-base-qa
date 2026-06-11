# -*- coding: utf-8 -*-
"""
Embedding 模块 - 文本向量化和向量数据库管理
"""
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
import pickle
import os

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Embedding 管理器"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh",
        device: str = "cpu",
        normalize_embeddings: bool = True,
        use_openai: bool = False,
        openai_api_key: Optional[str] = None
    ):
        """
        初始化 Embedding 管理器
        
        Args:
            model_name: 模型名称
            device: 运行设备（cpu/cuda）
            normalize_embeddings: 是否归一化向量
            use_openai: 是否使用 OpenAI Embedding
            openai_api_key: OpenAI API Key
        """
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.use_openai = use_openai
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self._model = None
        self._openai_client = None
    
    def _get_local_model_path(self, model_name: str) -> Optional[str]:
        """获取本地模型路径"""
        project_root = Path(__file__).parent.parent
        
        # 检查 model_config.txt
        config_file = project_root / "model_config.txt"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('LOCAL_MODEL_PATH='):
                        path_str = line.strip().split('=', 1)[1]
                        # 转换为绝对路径
                        local_path = project_root / path_str
                        if local_path.exists() and local_path.is_dir():
                            return str(local_path)
        
        # 检查默认模型目录
        models_dir = project_root / "models"
        
        # 尝试不同的路径格式
        possible_paths = [
            models_dir / model_name.replace('/', os.sep),
            models_dir / "hub" / model_name.replace('/', os.sep),
            models_dir / "._____temp" / model_name.replace('/', os.sep),
        ]
        
        for path in possible_paths:
            if path.exists() and path.is_dir():
                return str(path)
        
        return None
    
    def _load_model(self):
        """懒加载模型"""
        if self._model is None:
            if self.use_openai:
                logger.info("使用 OpenAI Embedding API")
                try:
                    from openai import OpenAI
                    self._openai_client = OpenAI(api_key=self.openai_api_key)
                except ImportError:
                    logger.error("请先安装 openai: pip install openai")
                    raise
            else:
                # 先检查是否有本地模型
                local_path = self._get_local_model_path(self.model_name)
                
                if local_path:
                    logger.info(f"使用本地模型: {local_path}")
                    try:
                        from sentence_transformers import SentenceTransformer
                        self._model = SentenceTransformer(
                            local_path,
                            device=self.device,
                            trust_remote_code=True
                        )
                        logger.info("本地模型加载完成")
                    except Exception as e:
                        logger.error(f"加载本地模型失败: {e}")
                        raise
                else:
                    logger.info(f"正在下载 Embedding 模型: {self.model_name}")
                    # 设置 HuggingFace 镜像源
                    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                    
                    try:
                        from sentence_transformers import SentenceTransformer
                        self._model = SentenceTransformer(
                            self.model_name,
                            device=self.device,
                            trust_remote_code=True
                        )
                        logger.info("Embedding 模型加载完成")
                    except Exception as e:
                        logger.error(f"下载模型失败: {e}")
                        logger.info("尝试使用 OpenAI Embedding API...")
                        try:
                            from openai import OpenAI
                            self._openai_client = OpenAI(api_key=self.openai_api_key)
                            self.use_openai = True
                            logger.info("已切换到 OpenAI Embedding API")
                        except Exception as e2:
                            logger.error(f"OpenAI API 也失败: {e2}")
                            raise e
        return self._model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        将文档编码为向量
        
        Args:
            texts: 文本列表
            
        Returns:
            向量列表
        """
        if self.use_openai or self._openai_client:
            # 使用 OpenAI API
            try:
                from openai import OpenAI
                client = self._openai_client or OpenAI(api_key=self.openai_api_key)
                response = client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=texts
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                logger.error(f"OpenAI Embedding 失败: {e}")
                raise
        else:
            model = self._load_model()
            embeddings = model.encode(
                texts,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=True
            )
            return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        将查询编码为向量
        
        Args:
            text: 查询文本
            
        Returns:
            向量
        """
        if self.use_openai or self._openai_client:
            # 使用 OpenAI API
            try:
                from openai import OpenAI
                client = self._openai_client or OpenAI(api_key=self.openai_api_key)
                response = client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=[text]
                )
                return response.data[0].embedding
            except Exception as e:
                logger.error(f"OpenAI Embedding 失败: {e}")
                raise
        else:
            model = self._load_model()
            embedding = model.encode(
                text,
                normalize_embeddings=self.normalize_embeddings
            )
            return embedding.tolist()


class VectorStoreManager:
    """向量数据库管理器"""
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        vector_store_type: str = "faiss",
        persist_directory: Optional[str] = None
    ):
        """
        初始化向量数据库管理器
        
        Args:
            embedding_manager: Embedding 管理器
            vector_store_type: 向量数据库类型（faiss/chroma）
            persist_directory: 持久化目录
        """
        self.embedding_manager = embedding_manager
        self.vector_store_type = vector_store_type.lower()
        self.persist_directory = persist_directory or "./data/vector_db"
        self._vector_store = None
    
    def create_vector_store(
        self,
        documents: List[Document],
        save_path: Optional[str] = None
    ) -> Any:
        """
        从文档创建向量数据库
        
        Args:
            documents: 文档列表
            save_path: 保存路径
            
        Returns:
            向量数据库实例
        """
        logger.info(f"正在创建向量数据库，文档数: {len(documents)}")
        
        # 设置 HuggingFace 镜像源
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        if self.vector_store_type == "faiss":
            from langchain_community.vectorstores import FAISS
            
            if self.embedding_manager.use_openai:
                # 使用 OpenAI Embedding
                from langchain_openai import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings(
                    api_key=self.embedding_manager.openai_api_key,
                    model="text-embedding-ada-002"
                )
            else:
                # 检查是否有本地模型
                local_path = self.embedding_manager._get_local_model_path(self.embedding_manager.model_name)
                
                from langchain_community.embeddings import HuggingFaceEmbeddings
                
                if local_path:
                    logger.info(f"使用本地模型路径: {local_path}")
                    # 使用绝对路径并确保路径格式正确
                    abs_path = str(Path(local_path).resolve())
                    logger.info(f"绝对路径: {abs_path}")
                    embeddings = HuggingFaceEmbeddings(
                        model_name=abs_path,
                        model_kwargs={'device': self.embedding_manager.device, 'trust_remote_code': True},
                        encode_kwargs={'normalize_embeddings': self.embedding_manager.normalize_embeddings}
                    )
                else:
                    logger.info(f"使用在线模型: {self.embedding_manager.model_name}")
                    embeddings = HuggingFaceEmbeddings(
                        model_name=self.embedding_manager.model_name,
                        model_kwargs={'device': self.embedding_manager.device, 'trust_remote_code': True},
                        encode_kwargs={'normalize_embeddings': self.embedding_manager.normalize_embeddings}
                    )
            
            vector_store = FAISS.from_documents(documents, embeddings)
            
            # 保存到本地
            if save_path:
                vector_store.save_local(save_path)
                logger.info(f"向量数据库已保存到: {save_path}")
            
            self._vector_store = vector_store
            return vector_store
            
        elif self.vector_store_type == "chroma":
            from langchain_community.vectorstores import Chroma
            
            if self.embedding_manager.use_openai:
                from langchain_openai import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings(
                    api_key=self.embedding_manager.openai_api_key,
                    model="text-embedding-ada-002"
                )
            else:
                # 检查是否有本地模型
                local_path = self.embedding_manager._get_local_model_path(self.embedding_manager.model_name)
                
                from langchain_community.embeddings import HuggingFaceEmbeddings
                
                if local_path:
                    logger.info(f"使用本地模型路径: {local_path}")
                    # 使用绝对路径并确保路径格式正确
                    abs_path = str(Path(local_path).resolve())
                    logger.info(f"绝对路径: {abs_path}")
                    embeddings = HuggingFaceEmbeddings(
                        model_name=abs_path,
                        model_kwargs={'device': self.embedding_manager.device, 'trust_remote_code': True},
                        encode_kwargs={'normalize_embeddings': self.embedding_manager.normalize_embeddings}
                    )
                else:
                    logger.info(f"使用在线模型: {self.embedding_manager.model_name}")
                    embeddings = HuggingFaceEmbeddings(
                        model_name=self.embedding_manager.model_name,
                        model_kwargs={'device': self.embedding_manager.device, 'trust_remote_code': True},
                        encode_kwargs={'normalize_embeddings': self.embedding_manager.normalize_embeddings}
                    )
            
            vector_store = Chroma.from_documents(
                documents,
                embeddings,
                persist_directory=save_path or self.persist_directory
            )
            
            self._vector_store = vector_store
            return vector_store
        else:
            raise ValueError(f"不支持的向量数据库类型: {self.vector_store_type}")
    
    def load_vector_store(self, load_path: str) -> Any:
        """
        从本地加载向量数据库
        
        Args:
            load_path: 加载路径
            
        Returns:
            向量数据库实例
        """
        logger.info(f"正在加载向量数据库: {load_path}")
        
        # 设置 HuggingFace 镜像源
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        if self.embedding_manager.use_openai:
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(
                api_key=self.embedding_manager.openai_api_key,
                model="text-embedding-ada-002"
            )
        else:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_manager.model_name,
                model_kwargs={'device': self.embedding_manager.device, 'trust_remote_code': True},
                encode_kwargs={'normalize_embeddings': self.embedding_manager.normalize_embeddings}
            )
        
        if self.vector_store_type == "faiss":
            from langchain_community.vectorstores import FAISS
            vector_store = FAISS.load_local(
                load_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
        elif self.vector_store_type == "chroma":
            from langchain_community.vectorstores import Chroma
            vector_store = Chroma(
                persist_directory=load_path,
                embedding_function=embeddings
            )
        else:
            raise ValueError(f"不支持的向量数据库类型: {self.vector_store_type}")
        
        self._vector_store = vector_store
        logger.info("向量数据库加载完成")
        return vector_store
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Document]:
        """
        相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filter_dict: 过滤条件
            
        Returns:
            相关文档列表
        """
        if self._vector_store is None:
            raise ValueError("向量数据库未初始化，请先创建或加载向量数据库")
        
        results = self._vector_store.similarity_search(
            query,
            k=k,
            filter=filter_dict
        )
        return results


def get_recommended_models() -> Dict[str, Dict[str, Any]]:
    """
    获取推荐的 Embedding 模型
    
    Returns:
        模型配置字典
    """
    return {
        "text-embedding-ada-002": {
            "dimension": 1536,
            "language": "zh/en",
            "description": "OpenAI API（无需下载，需要API Key）",
            "recommended": True,
            "type": "api"
        },
        "BAAI/bge-large-zh": {
            "dimension": 1024,
            "language": "zh",
            "description": "中文大模型，效果最佳（需下载1.2GB）",
            "recommended": False,
            "type": "local"
        },
        "BAAI/bge-base-zh": {
            "dimension": 768,
            "language": "zh",
            "description": "中文基础模型（需下载400MB）",
            "recommended": False,
            "type": "local"
        },
        "moka-ai/m3e-base": {
            "dimension": 768,
            "language": "zh/en",
            "description": "多语言模型（需下载400MB）",
            "recommended": False,
            "type": "local"
        }
    }
