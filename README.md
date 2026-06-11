# RAG 知识库问答系统

基于检索增强生成（Retrieval-Augmented Generation）技术的智能问答系统，支持多格式文档上传、混合检索、重排序优化和引用溯源。

## 项目特点

- 📄 **多格式文档支持**: PDF、Word、Markdown、文本文件、代码文件等
- 🔍 **混合检索**: 结合向量检索（Embedding）和 BM25 关键词检索
- 🔄 **重排序优化**: 使用 Cross-Encoder 对检索结果进行精排
- 📚 **引用溯源**: 答案附带参考来源，可追溯原始文档
- 💾 **本地存储**: 向量数据库本地持久化，保护数据隐私
- 🌐 **Web 界面**: 基于 Streamlit 的交互式界面
- ⚙️ **灵活配置**: 支持多种 Embedding 模型和参数调整

## 技术架构

```
RAG知识库问答系统/
├── core/                    # 核心模块
│   ├── document_loader.py   # 文档加载（PDF、Word、Markdown等）
│   ├── text_splitter.py     # 文本分块（递归分块、语义分块）
│   ├── embedding.py         # Embedding 模型和向量数据库
│   ├── retriever.py         # 检索器（混合检索、重排序）
│   └── qa_generator.py      # 问答生成（RAG Chain）
├── frontend/                # Web 界面
│   └── app.py              # Streamlit 应用
├── data/                    # 数据目录
│   ├── documents/          # 上传的文档
│   └── vector_db/          # 向量数据库
├── main.py                 # 主程序入口
├── requirements.txt        # 依赖列表
└── config.yaml            # 配置文件
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
cd RAG知识库问答系统

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

复制环境变量模板并填写你的 API Key：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1  # 可选：如果使用代理
```

### 3. 准备文档

将你的文档放入 `data/documents` 目录：

```bash
mkdir -p data/documents
# 复制你的文档到该目录
```

支持的格式：
- PDF (.pdf)
- Word (.docx, .doc)
- Markdown (.md)
- 文本文件 (.txt)
- 代码文件 (.py, .java, .js 等)

### 4. 启动系统

#### 方式一：Web 界面（推荐）

```bash
python main.py --mode web
```

然后访问 http://localhost:8501

#### 方式二：命令行界面

```bash
python main.py --mode cli
```

#### 方式三：仅构建知识库

```bash
python main.py --mode build --data-dir ./data/documents
```

## 使用指南

### Web 界面

1. **配置 API**: 在左侧边栏输入 OpenAI API Key
2. **选择配置**: 选择 Embedding 模型、分块大小等参数
3. **构建知识库**: 点击"构建知识库"按钮
4. **开始问答**: 在输入框中提问

### 命令行界面

```bash
# 基本使用
python main.py --mode cli --api-key your_key

# 指定数据目录
python main.py --mode cli --data-dir ./my_docs

# 使用不同的 Embedding 模型
python main.py --mode cli --embedding-model BAAI/bge-base-zh

# 调整分块参数
python main.py --mode cli --chunk-size 1000 --chunk-overlap 100
```

## 核心功能详解

### 1. 文档加载

支持多种文档格式的自动识别和加载：

```python
from core import load_documents_from_directory

documents = load_documents_from_directory("./data/documents")
```

### 2. 文本分块

使用递归分块策略，优先按段落、句子切分：

```python
from core import TextSplitter

text_splitter = TextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    splitter_type="recursive"
)
split_documents = text_splitter.split_documents(documents)
```

### 3. Embedding 和向量数据库

使用 BGE 模型进行文本向量化，FAISS 存储向量：

```python
from core import EmbeddingManager, VectorStoreManager

embedding_manager = EmbeddingManager(model_name="BAAI/bge-large-zh")
vector_store_manager = VectorStoreManager(
    embedding_manager=embedding_manager,
    vector_store_type="faiss",
    persist_directory="./data/vector_db"
)
vector_store_manager.create_vector_store(split_documents)
```

### 4. 混合检索

结合向量检索和 BM25 关键词检索：

```python
from core import AdvancedRetriever

retriever = AdvancedRetriever(
    vector_store_manager=vector_store_manager,
    documents=split_documents,
    use_hybrid=True,      # 启用混合检索
    use_rerank=True       # 启用重排序
)

results = retriever.retrieve("你的问题")
```

### 5. 问答生成

使用 LLM 基于检索结果生成答案：

```python
from core import QAGenerator

qa_generator = QAGenerator(api_key="your_key")
response = qa_generator.generate_answer("你的问题", documents)

print(response.answer)      # 答案
print(response.sources)     # 引用来源
print(response.confidence)  # 置信度
```

## 配置说明

### Embedding 模型选择

| 模型 | 维度 | 特点 |
|------|------|------|
| BAAI/bge-large-zh | 1024 | 中文效果优秀，推荐 |
| BAAI/bge-base-zh | 768 | 速度更快，资源占用少 |
| moka-ai/m3e-base | 768 | 多语言支持 |

### 分块策略

| 策略 | chunk_size | chunk_overlap | 适用场景 |
|------|------------|---------------|----------|
| 小块 | 200 | 20 | 精细检索，短文本 |
| 中等 | 500 | 50 | 平衡精度和上下文（推荐） |
| 大块 | 1000 | 100 | 保留更多上下文 |

### 检索参数

- **vector_weight**: 向量检索权重 (0.0-1.0)
- **bm25_weight**: BM25 检索权重 (0.0-1.0)
- **top_k_retrieve**: 检索阶段返回数量
- **top_k_rerank**: 重排序后返回数量

## 性能优化

### 1. 使用 GPU 加速

```python
embedding_manager = EmbeddingManager(
    model_name="BAAI/bge-large-zh",
    device="cuda"  # 或 "cuda:0"
)
```

### 2. 调整批处理大小

```python
embedding_manager = EmbeddingManager(
    model_name="BAAI/bge-large-zh",
    batch_size=64  # 根据显存调整
)
```

### 3. 使用轻量级模型

如果资源有限，可以使用基础版模型：

```bash
python main.py --embedding-model BAAI/bge-base-zh
```

## 项目扩展

### 添加新的文档加载器

```python
from core.document_loader import BaseDocumentLoader

class MyCustomLoader(BaseDocumentLoader):
    def load(self, file_path: str):
        # 实现加载逻辑
        pass
    
    def supports(self, file_extension: str):
        return file_extension == ".custom"
```

### 自定义检索策略

```python
from core.retriever import HybridRetriever

class MyCustomRetriever(HybridRetriever):
    def retrieve(self, query: str):
        # 实现自定义检索逻辑
        pass
```

## 常见问题

### Q: 如何在没有 OpenAI API Key 的情况下使用？

可以使用本地 LLM 替代，如：
- Ollama + Llama 2
- Hugging Face Transformers
- vLLM

### Q: 向量数据库占用空间太大怎么办？

1. 使用更小的 Embedding 模型（如 bge-base-zh）
2. 增大 chunk_size 减少分块数量
3. 使用量化技术压缩向量

### Q: 检索效果不佳如何优化？

1. 调整分块大小和重叠度
2. 尝试不同的 Embedding 模型
3. 启用混合检索和重排序
4. 优化查询文本（Query Expansion）

## 技术亮点

1. **模块化设计**: 每个组件独立，易于扩展和维护
2. **工厂模式**: 文档加载器、分块器等使用工厂模式，支持灵活配置
3. **懒加载**: Embedding 模型和重排序模型按需加载，节省资源
4. **错误处理**: 完善的异常处理和日志记录
5. **类型提示**: 全程使用 Python 类型提示，提高代码可读性

## 学习资源

- [LangChain 官方文档](https://python.langchain.com/)
- [FAISS 文档](https://faiss.ai/)
- [BGE Embedding](https://github.com/FlagOpen/FlagEmbedding)
- [RAG 最佳实践](https://www.pinecone.io/learn/retrieval-augmented-generation/)

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

---

**作者**: AI 开发学习者
**版本**: 1.0.0
**日期**: 2024
