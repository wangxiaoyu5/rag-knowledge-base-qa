# 贡献指南

感谢你对这个简历知识库问答系统感兴趣！本指南帮你快速上手贡献代码。

---

## 🚀 快速开始

```bash
# 1. Clone 并进入项目
cd RAG知识库问答系统

# 2. 创建虚拟环境
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
# 或（推荐，pyproject.toml 里定义了全部依赖）
pip install -e ".[dev]"

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的硅基流动 API Key

# 5. 预下载模型（可选，首次运行会自动下载）
python download_model.py

# 6. 启动项目
python main.py --mode web   # Streamlit 界面 http://localhost:8501
python main.py --mode cli   # 命令行交互
```

---

## 🧰 开发工具链

| 工具 | 用途 | 安装 |
|------|------|------|
| **Ruff** | Lint + 格式化（替代 Flake8 + isort + black） | `pip install ruff` |
| **Mypy** | 类型检查 | `pip install mypy` |
| **Pytest** | 单元测试 | `pip install pytest pytest-cov` |
| **Pre-commit** | Git hooks 自动跑检查 | `pip install pre-commit && pre-commit install` |

### IDE 推荐配置

**VS Code**：
```jsonc
// .vscode/settings.json
{
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "charliermarsh.ruff",
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": true
        }
    },
    "ruff.lineLength": 120,
    "ruff.lint.enabled": true
}
```

**PyCharm**：
- Settings → Tools → Actions on Save → 勾 Run Ruff

---

## 📐 项目架构

```
core/                       # 纯逻辑层（不依赖 Streamlit/FastAPI）
├── document_loader.py      # 文档加载
├── text_splitter.py        # 文本分块
├── embedding.py            # Embedding 模型
├── vector_store.py         # FAISS 向量存储
├── retriever.py            # 混合检索 + Rerank
└── qa_generator.py         # LLM 问答

frontend/
└── app.py                  # Streamlit UI

main.py                     # CLI 入口
config.yaml                 # 默认配置
.env                        # 敏感配置（不提交 git）
.env.example                # 敏感配置模板
docs/
├── 编码规范.md             # ← 详细编码规范
├── 01-系统架构设计.md
├── 02-核心模块详解.md
├── 03-API接口文档.md
├── 04-部署运维文档.md
└── 05-面试准备指南.md
```

**核心原则**：`core/` 不 import `streamlit`、`fastapi` 等演示层框架，保持可复用。

---

## 🔄 开发流程规范（必须遵守）

> 核心原则：**一次一个功能 → 完成即测试 → 全部完成后端到端验收**
> 禁止多个功能并行写到一半就切下一个，禁止所有功能写完才第一次启动测试。

### 流程总览

```
┌──────────────────────────────────────────────────────┐
│  开始任务：拆分为功能 A、功能 B、功能 C...             │
└──────────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────┐
│  功能 A —— 编码实现 ✅                                │
└──────────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────┐
│  功能 A —— 立刻测试 ✅                                │
│  ┌────────────────────────────────────────────────┐  │
│  │ 1. ruff check . --fix       代码质量检查       │  │
│  │ 2. 启动项目跑一遍这个功能的完整链路             │  │
│  │ 3. 输入典型输入，检查输出是否符合预期           │  │
│  │ 4. 输入边界/异常输入，看是否正常处理            │  │
│  └────────────────────────────────────────────────┘  │
│  通过 ✅ → 继续下一个功能                              │
│  不通过 ❌ → 修到通过再继续（别先写功能 B）              │
└──────────────────────────────────────────────────────┘
            │
            ▼ （循环）
┌──────────────────────────────────────────────────────┐
│  功能 B —— 编码 ✅ → 测试 ✅                          │
└──────────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────┐
│  功能 C —— 编码 ✅ → 测试 ✅                          │
└──────────────────────────────────────────────────────┘
            │
            ▼ （所有功能完成）
┌──────────────────────────────────────────────────────┐
│  🎯 端到端验收（触屏/点击完整跑一遍）                  │
│  ┌────────────────────────────────────────────────┐  │
│  │ 1. 启动 Streamlit / 浏览器打开                  │  │
│  │ 2. 从 0 开始：构建知识库 → 提问 → 看回答        │  │
│  │ 3. 测试所有新增功能的 UI 交互                    │  │
│  │ 4. 测试边界场景（空输入、长输入、特殊字符）      │  │
│  │ 5. 截图/录屏记录验收结果                         │  │
│  │ 6. 有问题 → 逐个修 → 重新验收                   │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

### 单功能测试清单（每个功能完成后必过）

```bash
# ① 代码质量
ruff check .                    # Lint 检查
ruff check . --fix              # 自动修复
ruff format --check .           # 检查格式（不改动）

# ② 项目能启动
python main.py --mode web       # Streamlit 能起来，无 import 错误

# ③ 功能能跑通（手动/CLI 至少跑一遍）
#    比如改了 embedding：
python -c "from core import EmbeddingManager; m = EmbeddingManager(); print(m.embed(['测试']))"

#    比如改了 retriever：
python main.py --mode cli       # 进 CLI 问一个问题看结果对不对

# ④ 边界场景
#    - 空输入 ""
#    - 极长输入（1000 字）
#    - 特殊字符 <script>alert(1)</script>
#    - 不存在的文档/文件
#    - API 挂了怎么办（降级机制）
```

### 端到端验收清单（全部功能完成后）

```
□ 页面正常加载（无白屏、无 traceback）
□ 侧边栏配置完整可用（修改参数能生效）
□ 知识库构建（加载文档 → 分块 → 向量化 → 显示进度）
□ 知识库状态卡片正确显示
□ 快捷提问按钮能点击且正常触发
□ 对话输入 → 发送 → 回答完整链路跑通
□ 回答卡片显示：内容 + 置信度 + 耗时 + 引用来源
□ 对话历史持久化（刷新页面后？当前 session）
□ 清除对话历史按钮正常
□ 空输入不应该触发查询
□ API 失败时显示友好降级（不崩溃）
□ 页面样式正常（无错位、溢出）
```

### 典型反模式（禁止）

| ❌ 反模式 | ✅ 正确做法 |
|-----------|-----------|
| "先把 3 个功能代码全写了，一起调" | 写完 1 个 → 测 1 个 → 通过才写下 1 个 |
| "写完直接 commit，测试放 PR 后做" | 本地先跑通再 commit |
| "改了 retriever 但没跑检索看效果就去改 qa_generator" | retriever 独立跑一遍，确认结果对再往下 |
| "前端加了新按钮，但没点过就以为能用" | 启动浏览器实际点一遍 |
| "改了 .env 但没重新加载验证就忘了" | 改完 `.env` 重启服务，打印一下读的值 |

### 为什么要这样？

1. **RAG 链路长**：文档加载 → 分块 → Embedding → 向量检索 → Rerank → LLM 生成，任何一环错了都影响最终结果。多改动叠加后很难定位。
2. **本地模型重**：bge-large-zh 加载一次几秒到几十秒，每次启动都在等。尽早发现问题省时间。
3. **LLM 调用花钱**：每跑一次 QA 都在烧 API 额度。调试阶段用 CLI / 小样本测试。
4. **Streamlit 状态管理特殊**：`st.session_state` + `st.rerun()` 容易出边界问题，必须实际点页面验证。

---

## 🛠️ 常用命令

```bash
# 代码检查
ruff check .

# 自动修复 + 格式化
ruff check . --fix
ruff format .

# 类型检查
mypy core/ frontend/ --ignore-missing-imports

# 跑测试
pytest tests/ -v
pytest tests/ --cov=core --cov-report=term

# 启动不同模式
python main.py --mode web           # Streamlit UI
python main.py --mode cli           # 命令行
python main.py --mode api           # FastAPI REST API（待实现）

# 清除缓存和向量库
rm -rf data/vector_db* .ruff_cache .mypy_cache
```

---

## 📝 代码规范摘要

完整规范见 [docs/编码规范.md](./docs/编码规范.md)，重点摘录：

1. **行宽 120**，4 空格缩进（`.editorconfig` 已统一）
2. **Ruff 自动格式化**，不需要手动调 PEP8
3. **导入顺序**：标准库 → 第三方 → 项目内部（isort 自动处理）
4. **不要裸 except**，具体异常 + 日志 + `from` 链式保留堆栈
5. **不要 print() 替代日志**，用 `import logging; logger = logging.getLogger(__name__)`
6. **公开函数必须标注类型**（`def retrieve(query: str) -> List[Result]`）
7. **Prompt 集中在 `_build_prompt()` 方法**，不要散落
8. **敏感信息放 `.env`**，`.env.example` 只留占位
9. **Git commit 写清楚改了什么**，推荐 `feat/fix/refactor:` 前缀

---

## 🐛 问题排查

| 症状 | 可能原因 | 解决 |
|------|---------|------|
| API 401 invalid_key | `.env` 的 base_url 和 Key 不匹配 | 硅基流动 Key 配了 OpenAI 地址 → 改 `OPENAI_API_BASE=https://api.siliconflow.cn/v1` |
| HF model download 卡住 | HuggingFace 国内网络慢 | `.env` 加 `HF_ENDPOINT=https://hf-mirror.com` |
| FAISS index 找不到 | 向量库路径被改了 | `python core/vector_store.py` 重建，或检查 `.env` 里的 `VECTOR_DB_PATH` |
| Streamlit 白屏 | 前端 import 报错 | 终端日志看具体堆栈，常见是 `.env` 加载失败 |
| 检索结果全是同一个文档 | 分块太小 / filter 没生效 | 检查 `chunk_size` 从 300 → 500+，并看 `retriever.retrieve` 的 filter 参数 |

---

## 📄 License

MIT — 随便用，但别赖我 😄
