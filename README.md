# RAG 知识库问答系统

基于检索增强生成（Retrieval-Augmented Generation）技术的通用知识库问答系统，支持 **25 种文档格式**上传、混合检索 + 重排序、Multi-Query + HyDE 查询增强、多知识库隔离与引用溯源。

## 项目特点

- **多格式文档支持**: PDF、Word、Excel、CSV、PPTX、Markdown、代码文件等 25 种格式
- **智能分块**: 按文件类型自动路由分块策略（Markdown 按标题 / 代码按函数 / Excel 按表头 / 长文按段落）
- **混合检索**: 向量检索 + BM25 关键词检索，RRF 动态权重融合，粗召回 30 条候选
- **查询增强**: Multi-Query 多路查询变体 + HyDE 假设性文档嵌入，二次 RRF 融合
- **重排序优化**: BAAI/bge-reranker 对候选结果精排 Top 5
- **元数据过滤**: 按知识库集合 / 文件 / 类型三维过滤
- **引用溯源**: 答案附带参考来源，可追溯原始文档
- **本地存储**: FAISS 向量库 + documents.pkl 持久化，刷新后一键恢复
- **极简 Web 界面**: Streamlit 极简科技风，Lucide-style SVG 图标，触屏友好
- **全链路降级**: 无 API Key / LLM 失败时自动降级为检索模式，永不崩溃

## 技术架构

```
RAG知识库问答系统/
├── core/                        # 核心模块（不依赖 Streamlit）
│   ├── document_loader.py       # 多格式文档加载（PDF/DOCX/Excel/CSV/PPTX...）
│   ├── text_splitter.py         # SmartTextSplitter 智能分块策略路由
│   ├── embedding.py             # Embedding 模型 + FAISS 向量库（含中文路径修复）
│   ├── retriever.py             # 混合检索 + RRF 融合 + Reranker 精排
│   ├── query_transformer.py     # Multi-Query + HyDE 查询增强
│   └── qa_generator.py          # LLM 问答生成 + 降级模式
├── frontend/
│   └── app.py                   # Streamlit Web 界面（极简科技风）
├── data/
│   ├── documents/               # 上传的文档（本地，不入库）
│   └── vector_db/               # 向量数据库（本地，不入库）
├── main.py                      # CLI / Web 入口
├── config.yaml                  # 配置文件
├── .env.example                 # 环境变量模板
└── requirements.txt             # 依赖列表
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/wangxiaoyu5/rag-knowledge-base-qa.git
cd rag-knowledge-base-qa

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

编辑 `.env` 文件（支持任何 OpenAI 兼容 API）：

```env
LLM_API_KEY=your_api_key_here
LLM_API_BASE=https://api.siliconflow.cn/v1
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

> 无 API Key 也能使用：系统自动降级为"检索模式"，直接展示检索结果。

### 3. 准备文档

将你的文档放入 `data/documents` 目录：

```bash
mkdir -p data/documents
```

支持的格式：
- PDF / Word（.pdf, .docx, .doc）
- Excel / CSV（.xlsx, .xlsm, .csv）
- PPT（.pptx, .ppt）
- Markdown / 文本（.md, .markdown, .txt）
- 代码文件（.py, .java, .js, .html 等）

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

1. **配置数据**: 侧边栏设置数据目录、知识库名称，或直接拖拽上传文档
2. **选择配置**: 选择 Embedding 模型、分块大小、检索策略（BM25 混合 / Rerank）
3. **构建知识库**: 点击"构建知识库"按钮（或加载已有向量库）
4. **检索过滤**: 按集合 / 文件 / 类型过滤检索范围
5. **开始问答**: 在输入框中提问，支持快捷问题

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

### 1. 文档加载（25 种格式）

```python
from core import load_documents_from_directory

documents = load_documents_from_directory("./data/documents")
```

### 2. 智能分块（SmartTextSplitter）

按文件类型自动选择分块策略：Markdown 按标题层级、代码按函数/类边界、Excel/CSV 按行分组保留表头、PPTX 按页、长文按段落、其他递归分块。

```python
from core import SmartTextSplitter

splitter = SmartTextSplitter()
split_documents = splitter.split_documents(documents)
```

### 3. Embedding 和向量数据库

使用 BGE 模型向量化，FAISS 存储，`save_local` + `documents.pkl` 持久化（兼容中文路径）。

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

### 4. 混合检索 + 查询增强

向量检索与 BM25 各召回 30 条，RRF 动态权重融合后 Reranker 精排 Top 5；支持 Multi-Query 多路变体与 HyDE 假设文档嵌入。

```python
from core import AdvancedRetriever, QueryTransformer

retriever = AdvancedRetriever(
    vector_store_manager=vector_store_manager,
    documents=split_documents,
    use_hybrid=True,      # 启用混合检索
    use_rerank=True       # 启用重排序
)

# Multi-Query + HyDE 查询增强（无 API Key 时自动降级）
query_transformer = QueryTransformer()
variants = query_transformer.expand_queries("你的问题")

results = retriever.retrieve("你的问题", filter={"collection": "default"})
```

### 5. 问答生成

使用 LLM 基于检索结果生成答案，附置信度与引用来源。

```python
from core import QAGenerator

qa_generator = QAGenerator(
    model_type="openai",
    model_name="Qwen/Qwen2.5-7B-Instruct",
    api_key=api_key,
    api_base=api_base,
)
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
| paraphrase-MiniLM-L3-v2 | 384 | 轻量，适合低资源环境 |

### 智能分块策略

| 文件类型 | 策略 | 额外 metadata |
|---------|------|--------------|
| .md / .markdown | 按标题层级切 | `section`, `section_path` |
| .py / .java / .js | 按 def / class 边界切 | — |
| .yaml / .json / .xml | 通用兜底 + 小 chunk | — |
| .txt 长文 | 按段落（空行）切 | — |
| .xlsx / .csv | 按行分组保留表头 | `sheet_name`, `row_start` |
| .pptx | 按页切 | `slide_number` |
| 其他（PDF/DOCX） | 递归分块 | — |

### 检索参数

- **recall_k**: 粗召回数量（默认 30）
- **top_k**: Rerank 后返回数量（默认 5）
- **RRF 权重**: 关键词型查询偏向 BM25，语义型查询偏向向量
- **filter**: 按 `collection` / `file_name` / `file_type` 过滤

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

## 技术亮点

1. **粗召回 10 → 30**: 扩大候选池，Reranker 有更多选择空间
2. **RRF 动态权重**: 按查询类型（关键词/语义）动态调节 BM25 与向量权重
3. **Multi-Query + HyDE**: LLM 生成查询变体 + 假设文档嵌入，双通道提升召回率
4. **智能分块路由**: 6 种分块策略按文件类型自动切换
5. **全链路降级**: QueryTransformer / Reranker / LLM 任一层失败都不阻塞
6. **模块化设计**: `core/` 不依赖 Streamlit，可复用为 FastAPI / CLI 后端
7. **前端工程化**: SVG 图标替换 Emoji、prefers-reduced-motion、触屏优化

## 常见问题

### Q: 如何在没有 LLM API Key 的情况下使用？

系统自动降级为"检索模式"：直接展示最相关的检索片段，不调用 LLM。功能不中断。

### Q: 向量数据库占用空间太大怎么办？

1. 使用更小的 Embedding 模型（如 bge-base-zh）
2. 增大 chunk_size 减少分块数量
3. 清理 `data/vector_db` 后重新构建

### Q: 检索效果不佳如何优化？

1. 调整分块大小和重叠度
2. 尝试不同的 Embedding 模型
3. 确保启用了混合检索、Rerank 和 Multi-Query 增强
4. 用检索过滤缩小搜索范围

### Q: 刷新页面后知识库没了？

构建时会持久化 FAISS 索引和 `documents.pkl`。刷新后在侧边栏点击"加载已有知识库"即可恢复。

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

**版本**: 1.0.0
