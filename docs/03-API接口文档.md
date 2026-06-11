# RAG 知识库问答系统 - API 接口文档

## 目录
1. [核心模块 API](#1-核心模块-api)
2. [Web 界面 API](#2-web-界面-api)
3. [命令行接口](#3-命令行接口)
4. [配置参数说明](#4-配置参数说明)

---

## 1. 核心模块 API

### 1.1 文档加载 API

#### `load_document(file_path: str) -> List[Document]`

加载单个文档文件。

**参数：**
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| file_path | str | 是 | 文档文件路径 |

**返回值：**
- `List[Document]`: 文档对象列表

**示例：**
```python
from core import load_document

docs = load_document("./data/document.pdf")
for doc in docs:
    print(f"内容: {doc.page_content[:100]}")
    print(f"元数据: {doc.metadata}")
```

#### `load_documents_from_directory(directory: str, recursive: bool = True) -> List[Document]`

从目录批量加载文档。

**参数：**
| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| directory | str | 是 | - | 文档目录路径 |
| recursive | bool | 否 | True | 是否递归子目录 |

**返回值：**
- `List[Document]`: 文档对象列表

**示例：**
```python
from core import load_documents_from_directory

docs = load_documents_from_directory("./data/documents", recursive=True)
print(f"共加载 {len(docs)} 个文档片段")
```

---

### 1.2 文本分块 API

#### `TextSplitter`

文本分块器类。

**构造函数：**
```python
TextSplitter(
    chunk_size: int = 500,           # 块大小
    chunk_overlap: int = 50,         # 重叠大小
    splitter_type: str = "recursive" # 分块类型
)
```

**方法：**

##### `split_documents(documents: List[Document]) -> List[Document]`

对文档列表进行分块。

**示例：**
```python
from core import TextSplitter

text_splitter = TextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    splitter_type="recursive"
)
split_docs = text_splitter.split_documents(documents)
```

##### `split_text(text: str) -> List[str]`

对纯文本进行分块。

**示例：**
```python
chunks = text_splitter.split_text("这是一段长文本...")
```

#### `create_splitter(strategy: str, **kwargs) -> TextSplitter`

工厂函数，创建分块器。

**参数：**
| 参数名 | 类型 | 说明 |
|--------|------|------|
| strategy | str | 策略名称: "recursive", "character", "token", "semantic" |
| chunk_size | int | 块大小 |
| chunk_overlap | int | 重叠大小 |

**示例：**
```python
from core import create_splitter

text_splitter = create_splitter(
    strategy="recursive",
    chunk_size=500,
    chunk_overlap=50
)
```

#### `get_chunk_strategy(strategy_name: str) -> dict`

获取预定义的分块策略。

**参数：**
| 参数名 | 类型 | 说明 |
|--------|------|------|
| strategy_name | str | "small", "medium", "large", "code" |

**返回值：**
```python
{
    "chunk_size": 500,
    "chunk_overlap": 50,
    "description": "中等块策略"
}
```

---

### 1.3 Embedding API

#### `EmbeddingManager`

Embedding 模型管理器。

**构造函数：**
```python
EmbeddingManager(
    model_name: str = "BAAI/bge-large-zh",  # 模型名称
    device: str = "cpu",                     # 运行设备
    normalize_embeddings: bool = True       # 是否归一化
)
```

**方法：**

##### `embed_documents(texts: List[str]) -> List[List[float]]`

批量文本向量化。

**参数：**
| 参数名 | 类型 | 说明 |
|--------|------|------|
| texts | List[str] | 文本列表 |

**返回值：**
- `List[List[float]]`: 向量列表

**示例：**
```python
from core import EmbeddingManager

embedding_mgr = EmbeddingManager(model_name="BAAI/bge-large-zh")
vectors = embedding_mgr.embed_documents(["文本1", "文本2"])
print(f"向量维度: {len(vectors[0])}")
```

##### `embed_query(text: str) -> List[float]`

单条查询文本向量化。

**示例：**
```python
query_vector = embedding_mgr.embed_query("查询文本")
```

##### `get_embedding_dimension() -> int`

获取向量维度。

**示例：**
```python
dimension = embedding_mgr.get_embedding_dimension()
print(f"Embedding 维度: {dimension}")  # 输出: 1024
```

---

#### `VectorStoreManager`

向量数据库管理器。

**构造函数：**
```python
VectorStoreManager(
    embedding_manager: EmbeddingManager,    # Embedding 管理器
    vector_store_type: str = "faiss",       # 向量数据库类型
    persist_directory: Optional[str] = None # 持久化目录
)
```

**方法：**

##### `create_vector_store(documents: List[Document]) -> VectorStore`

从文档创建向量数据库。

**示例：**
```python
from core import VectorStoreManager

vector_store_mgr = VectorStoreManager(
    embedding_manager=embedding_mgr,
    vector_store_type="faiss",
    persist_directory="./data/vector_db"
)
vector_store = vector_store_mgr.create_vector_store(documents)
```

##### `load_vector_store() -> VectorStore`

加载已存在的向量数据库。

**示例：**
```python
vector_store = vector_store_mgr.load_vector_store()
```

##### `similarity_search(query: str, k: int = 5, filter_dict: Optional[dict] = None) -> List[Tuple[Document, float]]`

相似度搜索。

**参数：**
| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| query | str | 是 | - | 查询文本 |
| k | int | 否 | 5 | 返回结果数量 |
| filter_dict | dict | 否 | None | 过滤条件 |

**返回值：**
- `List[Tuple[Document, float]]`: (文档, 相似度分数) 列表

**示例：**
```python
results = vector_store_mgr.similarity_search("查询文本", k=5)
for doc, score in results:
    print(f"相似度: {score:.4f}, 内容: {doc.page_content[:100]}")
```

##### `add_documents(documents: List[Document])`

添加新文档到向量数据库。

**示例：**
```python
new_docs = [Document(page_content="新内容", metadata={"source": "new"})]
vector_store_mgr.add_documents(new_docs)
```

---

### 1.4 检索 API

#### `HybridRetriever`

混合检索器。

**构造函数：**
```python
HybridRetriever(
    vector_store_manager: VectorStoreManager,  # 向量数据库管理器
    documents: List[Document],                  # 原始文档列表
    vector_weight: float = 0.7,                # 向量检索权重
    bm25_weight: float = 0.3,                  # BM25 检索权重
    top_k: int = 10                            # 返回结果数量
)
```

**方法：**

##### `retrieve(query: str) -> List[RetrievalResult]`

执行混合检索。

**返回值：**
```python
[
    RetrievalResult(
        document=Document(...),
        score=0.95,
        rank=1,
        retrieval_type="hybrid"  # "vector", "bm25", "hybrid"
    ),
    ...
]
```

**示例：**
```python
from core import HybridRetriever

hybrid_retriever = HybridRetriever(
    vector_store_manager=vector_store_mgr,
    documents=split_documents,
    vector_weight=0.7,
    bm25_weight=0.3,
    top_k=10
)

results = hybrid_retriever.retrieve("查询文本")
for result in results:
    print(f"排名: {result.rank}, 分数: {result.score:.4f}")
    print(f"类型: {result.retrieval_type}")
    print(f"内容: {result.document.page_content[:100]}")
```

---

#### `Reranker`

重排序器。

**构造函数：**
```python
Reranker(
    model_name: str = "BAAI/bge-reranker-large",  # 重排序模型
    device: str = "cpu",                           # 运行设备
    max_length: int = 512                         # 最大输入长度
)
```

**方法：**

##### `rerank(query: str, documents: List[Document], top_k: int = 5) -> List[Tuple[Document, float]]`

对文档进行重排序。

**返回值：**
- `List[Tuple[Document, float]]`: (文档, 重排序分数) 列表

**示例：**
```python
from core import Reranker

reranker = Reranker(model_name="BAAI/bge-reranker-large")

# 先进行混合检索
results = hybrid_retriever.retrieve("查询文本")
documents = [r.document for r in results]

# 重排序
reranked = reranker.rerank("查询文本", documents, top_k=5)
for doc, score in reranked:
    print(f"重排序分数: {score:.4f}, 内容: {doc.page_content[:100]}")
```

---

#### `AdvancedRetriever`

高级检索器（整合混合检索和重排序）。

**构造函数：**
```python
AdvancedRetriever(
    vector_store_manager: VectorStoreManager,
    documents: List[Document],
    use_hybrid: bool = True,                    # 是否使用混合检索
    use_rerank: bool = True,                    # 是否使用重排序
    reranker_model: str = "BAAI/bge-reranker-large",
    top_k_retrieve: int = 10,                   # 检索阶段返回数量
    top_k_rerank: int = 5                       # 重排序后返回数量
)
```

**方法：**

##### `retrieve(query: str) -> List[RetrievalResult]`

执行完整检索流程。

**示例：**
```python
from core import AdvancedRetriever

advanced_retriever = AdvancedRetriever(
    vector_store_manager=vector_store_mgr,
    documents=split_documents,
    use_hybrid=True,
    use_rerank=True,
    top_k_retrieve=10,
    top_k_rerank=5
)

results = advanced_retriever.retrieve("查询文本")
for result in results:
    print(f"排名: {result.rank}, 分数: {result.score:.4f}")
```

---

### 1.5 问答生成 API

#### `QAGenerator`

问答生成器。

**构造函数：**
```python
QAGenerator(
    llm_model: str = "gpt-3.5-turbo",    # LLM 模型
    temperature: float = 0.7,            # 生成温度
    max_tokens: int = 1000,              # 最大 token 数
    api_key: Optional[str] = None,       # API 密钥
    api_base: Optional[str] = None,      # API 基础 URL
    use_citation: bool = True            # 是否使用引用格式
)
```

**方法：**

##### `generate_answer(query: str, documents: List[Document], max_context_length: int = 4000) -> QAResponse`

生成答案。

**返回值：**
```python
QAResponse(
    answer="生成的答案...",
    sources=[
        {
            "index": 1,
            "content": "来源内容...",
            "metadata": {"file_name": "doc.pdf"}
        }
    ],
    retrieved_documents=[Document(...)],
    confidence=0.85,
    query="原始查询"
)
```

**示例：**
```python
from core import QAGenerator

qa_generator = QAGenerator(
    llm_model="gpt-3.5-turbo",
    api_key="your-api-key"
)

response = qa_generator.generate_answer(
    query="什么是 RAG？",
    documents=retrieved_documents
)

print(f"答案: {response.answer}")
print(f"置信度: {response.confidence:.2%}")
print(f"来源数量: {len(response.sources)}")
```

##### `generate_streaming_answer(query: str, documents: List[Document], max_context_length: int = 4000)`

流式生成答案。

**示例：**
```python
print("答案: ", end="", flush=True)
for chunk in qa_generator.generate_streaming_answer("什么是 RAG？", documents):
    print(chunk, end="", flush=True)
print()
```

---

#### `ContextCompressor`

上下文压缩器。

**构造函数：**
```python
ContextCompressor(
    max_tokens: int = 3000,           # 最大 token 数
    compression_ratio: float = 0.5    # 压缩比例
)
```

**方法：**

##### `compress(documents: List[Document], query: str) -> List[Document]`

压缩文档列表。

**示例：**
```python
from core import ContextCompressor

compressor = ContextCompressor(max_tokens=3000)
compressed_docs = compressor.compress(documents, query="查询文本")
```

---

## 2. Web 界面 API

### 2.1 启动 Web 服务

```bash
# 方式1: 使用 main.py
python main.py --mode web --port 8501

# 方式2: 直接使用 streamlit
streamlit run frontend/app.py --server.port 8501
```

### 2.2 Web 界面功能

#### 侧边栏配置

| 配置项 | 类型 | 说明 |
|--------|------|------|
| OpenAI API Key | password | API 密钥输入 |
| 数据目录 | text | 文档存放目录 |
| Embedding 模型 | select | bge-large-zh / bge-base-zh / m3e-base |
| 分块大小 | number | 100-2000 |
| 重叠大小 | number | 0-500 |
| 启用混合检索 | checkbox | BM25 + 向量检索 |
| 启用重排序 | checkbox | Cross-Encoder 重排序 |

#### 主界面功能

1. **知识库构建**
   - 显示构建进度条
   - 显示文档加载、分块、Embedding、向量存储状态

2. **对话问答**
   - 输入框提问
   - 显示历史对话
   - 显示答案和引用来源
   - 显示置信度

3. **对话历史管理**
   - 保存多轮对话
   - 显示用户和助手消息
   - 引用溯源展示

---

## 3. 命令行接口

### 3.1 主程序命令

```bash
python main.py [选项]
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --mode | str | web | 运行模式: web, cli, build, test |
| --api-key | str | env | OpenAI API Key |
| --api-base | str | None | API Base URL |
| --data-dir | str | ./data/documents | 数据目录 |
| --vector-db-path | str | ./data/vector_db | 向量数据库路径 |
| --embedding-model | str | BAAI/bge-large-zh | Embedding 模型 |
| --llm-model | str | gpt-3.5-turbo | LLM 模型 |
| --device | str | cpu | 运行设备 |
| --chunk-size | int | 500 | 分块大小 |
| --chunk-overlap | int | 50 | 重叠大小 |
| --vector-store | str | faiss | 向量数据库类型 |
| --use-hybrid | bool | True | 启用混合检索 |
| --use-rerank | bool | True | 启用重排序 |
| --host | str | 0.0.0.0 | Web 服务主机 |
| --port | int | 8501 | Web 服务端口 |

#### 使用示例

```bash
# Web 模式
python main.py --mode web --api-key your_key --port 8080

# CLI 模式
python main.py --mode cli --api-key your_key --data-dir ./my_docs

# 仅构建知识库
python main.py --mode build --data-dir ./my_docs --chunk-size 1000

# 使用不同的 Embedding 模型
python main.py --mode web --embedding-model BAAI/bge-base-zh

# 禁用重排序（加快检索速度）
python main.py --mode web --use-rerank False
```

---

## 4. 配置参数说明

### 4.1 环境变量 (.env)

```bash
# OpenAI API 配置
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1

# Embedding 配置
EMBEDDING_MODEL=BAAI/bge-large-zh
EMBEDDING_DEVICE=cpu

# 向量数据库配置
VECTOR_STORE_TYPE=faiss
VECTOR_DB_PATH=./data/vector_db

# 检索配置
TOP_K_RETRIEVE=10
TOP_K_RERANK=5
USE_HYBRID_RETRIEVAL=true
USE_RERANK=true

# LLM 配置
LLM_MODEL=gpt-3.5-turbo
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1000

# 文本分块配置
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# 数据目录
DATA_DIR=./data/documents

# 日志级别
LOG_LEVEL=INFO
```

### 4.2 配置文件 (config.yaml)

```yaml
# 系统配置
system:
  name: "RAG Knowledge Base QA System"
  version: "1.0.0"
  log_level: "INFO"

# 文档处理配置
document:
  supported_extensions:
    - ".pdf"
    - ".docx"
    - ".txt"
    - ".md"
  chunking:
    chunk_size: 500
    chunk_overlap: 50
    strategy: "recursive"

# Embedding 配置
embedding:
  model_name: "BAAI/bge-large-zh"
  device: "cpu"
  normalize_embeddings: true

# 向量数据库配置
vector_store:
  type: "faiss"
  persist_directory: "./data/vector_db"

# 检索配置
retrieval:
  hybrid:
    enabled: true
    vector_weight: 0.7
    bm25_weight: 0.3
  rerank:
    enabled: true
    model_name: "BAAI/bge-reranker-large"
  top_k:
    retrieve: 10
    rerank: 5

# LLM 配置
llm:
  provider: "openai"
  model: "gpt-3.5-turbo"
  temperature: 0.7
  max_tokens: 1000

# Web 界面配置
web:
  host: "0.0.0.0"
  port: 8501
  title: "RAG 知识库问答系统"
```

### 4.3 参数调优指南

#### Embedding 模型选择

| 场景 | 推荐模型 | 理由 |
|------|----------|------|
| 中文文档 | BAAI/bge-large-zh | 中文效果最佳 |
| 资源受限 | BAAI/bge-base-zh | 速度快，资源占用少 |
| 多语言 | moka-ai/m3e-base | 支持多种语言 |

#### 分块参数调优

| 文档类型 | chunk_size | chunk_overlap | 说明 |
|----------|------------|---------------|------|
| 技术文档 | 500 | 50 | 平衡精度和上下文 |
| 论文 | 800 | 100 | 保留段落完整性 |
| 代码 | 300 | 30 | 按函数/类切分 |
| 聊天记录 | 200 | 20 | 精细检索 |

#### 检索参数调优

| 场景 | vector_weight | bm25_weight | use_rerank | 说明 |
|------|---------------|-------------|------------|------|
| 语义查询为主 | 0.8 | 0.2 | true | 强调语义匹配 |
| 关键词查询为主 | 0.3 | 0.7 | true | 强调关键词匹配 |
| 快速检索 | 1.0 | 0.0 | false | 仅向量检索，最快 |
| 高精度检索 | 0.7 | 0.3 | true | 完整流程，最准 |

---

## 5. 错误处理

### 5.1 常见错误码

| 错误类型 | 错误信息 | 解决方案 |
|----------|----------|----------|
| APIError | OpenAI API 调用失败 | 检查 API Key 和网络 |
| FileNotFoundError | 文档或目录不存在 | 检查路径是否正确 |
| ValueError | 不支持的文件类型 | 检查文件扩展名 |
| MemoryError | 内存不足 | 减小 batch_size 或 chunk_size |
| TimeoutError | 请求超时 | 检查网络或减小请求量 |

### 5.2 异常处理示例

```python
from core import load_document, TextSplitter, EmbeddingManager

try:
    # 加载文档
    documents = load_document("./data/document.pdf")
    
    # 文本分块
    text_splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    
    # Embedding
    embedding_mgr = EmbeddingManager(model_name="BAAI/bge-large-zh")
    vectors = embedding_mgr.embed_documents([d.page_content for d in split_docs])
    
except FileNotFoundError as e:
    print(f"文件不存在: {e}")
except ValueError as e:
    print(f"参数错误: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

---

## 6. 性能指标

### 6.1 检索性能

| 指标 | 目标值 | 测试方法 |
|------|--------|----------|
| 检索延迟 | < 100ms | 单次检索平均时间 |
| Recall@5 | > 80% | 前5个结果的召回率 |
| MRR | > 0.7 | 平均倒数排名 |

### 6.2 生成性能

| 指标 | 目标值 | 测试方法 |
|------|--------|----------|
| 首 Token 延迟 | < 2s | 流式输出的首字时间 |
| 生成速度 | > 10 tokens/s | 文本生成速度 |
| 答案准确率 | > 85% | 人工评估 |

### 6.3 资源占用

| 资源 | 基础配置 | 推荐配置 |
|------|----------|----------|
| CPU | 2 核 | 4 核 |
| 内存 | 4 GB | 8 GB |
| 磁盘 | 10 GB | 50 GB |
| GPU | 可选 | RTX 3060+ |
