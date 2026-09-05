"""
Streamlit Web 界面 - 通用 RAG 知识库问答系统
极简科技风：白底深灰字 / 细线边框 / 无渐变无阴影 / 类 Notion / Linear
"""
import os

# 设置 HuggingFace 镜像源（必须在导入 transformers 之前）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'

import sys
from pathlib import Path

import streamlit as st

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv(project_root / '.env')

from core import (
    AdvancedRetriever,
    EmbeddingManager,
    QAGenerator,
    QueryTransformer,
    SmartTextSplitter,
    VectorStoreManager,
    load_documents_from_directory,
)

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="RAG 知识库问答系统",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== SVG 图标常量 ====================
# 用 URL-encoded SVG 作为内联图标，统一尺寸，支持 CSS color 控制
def _svg(icon_path, color="#6b7280", size=16):
    """生成一个 inline SVG 的 <span class="ic">图标</span>，通过 CSS 变量控制"""
    import urllib.parse
    svg_str = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">{icon_path}</svg>'''
    b64 = urllib.parse.quote(svg_str, safe='')
    return f'<span class="ic" style="--src:url(\'data:image/svg+xml,{b64}\')"></span>'

# 预定义常用图标（纯 stroke，无 fill）
IC = {
    # Hero / 欢迎页
    'brain':     _svg('<path d="M12 2a4 4 0 0 1 4 4v1a4 4 0 0 1 2 7.5A4 4 0 0 1 12 22a4 4 0 0 1-6-7.5A4 4 0 0 1 8 7V6a4 4 0 0 1 4-4Z"/>', '#111827', 20),
    # 侧边栏 section header
    'folder':    _svg('<path d="M4 7a2 2 0 0 1 2-2h4l2 2h6a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7Z"/>'),
    'upload':    _svg('<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>'),
    'chip':      _svg('<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>'),
    'bolt':      _svg('<path d="M13 2 3 14h9l-1 8 10-12h-9l1-8Z"/>'),
    'plug':      _svg('<path d="M9 2v6M15 2v6"/><path d="M7 8h10v4a5 5 0 0 1-10 0V8Z"/><path d="M12 17v5"/>'),
    'bar-chart': _svg('<line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/>'),
    'search':    _svg('<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>'),
    # 统计卡 icon
    'file':      _svg('<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><polyline points="14 2 14 8 20 8"/>', '#6b7280', 20),
    'scissors':  _svg('<circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><line x1="20" y1="4" x2="8.12" y2="15.88"/><line x1="14.47" y1="14.48" x2="20" y2="20"/><line x1="8.12" y1="8.12" x2="12" y2="12"/>', '#6b7280', 20),
    'puzzle':    _svg('<path d="M19.439 7.85c-.049.322.059.648.289.878l1.568 1.568c.47.47.706 1.087.706 1.704s-.235 1.233-.706 1.704l-1.611 1.611a.98.98 0 0 1-.837.276c-.47-.07-.802-.48-.968-.925a2.501 2.501 0 1 0-3.214 3.214c.446.166.855.497.925.968a.979.979 0 0 1-.276.837l-1.61 1.61a2.404 2.404 0 0 1-1.705.707 2.402 2.402 0 0 1-1.704-.706l-1.568-1.568a1.026 1.026 0 0 0-.877-.29c-.493.074-.84.504-1.02.968a2.5 2.5 0 1 1-3.237-3.237c.464-.18.894-.527.967-1.02a1.026 1.026 0 0 0-.289-.877l-1.568-1.568A2.402 2.402 0 0 1 1.998 12c0-.617.236-1.234.706-1.704L4.23 8.77c.24-.24.581-.353.917-.303.515.077.877.528 1.073 1.01a2.5 2.5 0 1 0 3.259-3.259c-.482-.196-.933-.558-1.01-1.073-.05-.336.062-.676.303-.917l1.525-1.525A2.402 2.402 0 0 1 12 2c.617 0 1.234.236 1.704.706l1.568 1.568c.23.23.556.338.877.29.493-.074.84-.504 1.02-.968a2.5 2.5 0 1 1 3.237 3.237c-.464.18-.894.527-.967 1.02Z"/>', '#6b7280', 20),
    # 回答卡标签
    'sparkles':  _svg('<path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M18.4 5.6l-2.8 2.8M8.4 15.6l-2.8 2.8"/>'),
    'search2':   _svg('<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>'),
    'user':      _svg('<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>'),
    'brain2':    _svg('<path d="M12 2a4 4 0 0 1 4 4v1a4 4 0 0 1 2 7.5A4 4 0 0 1 12 22a4 4 0 0 1-6-7.5A4 4 0 0 1 8 7V6a4 4 0 0 1 4-4Z"/>'),
    'clock':     _svg('<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>'),
    'hammer':    _svg('<path d="M15 12 8 5l-5 5 7 7c4-4 5-5 5-5Z"/><path d="M22 2 16 8"/><path d="m2 22 4.5-4.5"/><path d="m18 6 4 4"/>'),
    'rocket':    _svg('<path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09Z"/><path d="M12 15l-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2Z"/><path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/><path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/>'),
    'trash':     _svg('<polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>'),
    'folder-open': _svg('<path d="M3 7a2 2 0 0 1 2-2h4l2 2h6a2 2 0 0 1 2 2v2"/><path d="M3 13h18l-1.5 8a2 2 0 0 1-2 1.7h-13a2 2 0 0 1-2-1.7L3 13Z"/>'),
    'tag':       _svg('<path d="M20.59 13.41 13.42 20.58a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82Z"/><line x1="7" y1="7" x2="7.01" y2="7"/>'),
    'file-text': _svg('<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>'),
    'help':      _svg('<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/>'),
    'inbox':     _svg('<polyline points="22 12 16 12 14 15 10 15 8 12 2 12"/><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11Z"/>'),
}

# ==================== 全局样式（极简科技风） ====================
st.markdown("""
<style>
    /* === Web Interface Guidelines (Vercel) === */
    :root { color-scheme: light; }
    * { -webkit-tap-highlight-color: transparent; }
    html, body { touch-action: manipulation; }
    .stat-value { font-variant-numeric: tabular-nums; }
    .ic { -webkit-user-select: none; user-select: none; }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }

    /* === 基础 === */
    .stApp { background: #ffffff; color: #1f2937; }
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
    h1 { color: #111827; font-weight: 700; letter-spacing: -0.03em; }
    h2, h3 { color: #1f2937; font-weight: 600; }

    /* === inline SVG 图标类 === */
    .ic {
        display: inline-flex;
        width: 1em; height: 1em;
        vertical-align: -0.125em;
        background-image: var(--src);
        background-size: contain;
        background-repeat: no-repeat;
        background-position: center;
        flex-shrink: 0;
    }

    /* === 侧边栏：白底 + 左侧细线 === */
    section[data-testid="stSidebar"] {
        background: #fafafa;
        border-right: 1px solid #e5e7eb;
    }
    section[data-testid="stSidebar"] * { color: #374151 !important; }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] caption { color: #6b7280 !important; }
    section[data-testid="stSidebar"] hr { border-top-color: #e5e7eb !important; }

    /* === 统一边框 + 圆角 === */
    .stTextInput input, .stSelectbox div[data-baseweb],
    .stNumberInput input, .stCheckbox label {
        border-radius: 6px !important;
        border-color: #e5e7eb !important;
    }
    div[data-testid="stTextInput"] input:focus,
    div[data-baseweb]:has(input:focus) {
        border-color: #111827 !important;
        box-shadow: 0 0 0 2px rgba(17,24,39,0.1) !important;
    }

    /* === 按钮 === */
    button[kind="primary"] {
        background: #111827 !important;
        border: 1px solid #111827 !important;
        border-radius: 6px !important;
        color: #ffffff !important;
        font-weight: 500 !important;
        box-shadow: none !important;
    }
    button[kind="primary"]:hover { background: #000000 !important; }

    button[kind="secondary"] {
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 6px !important;
        color: #374151 !important;
        font-weight: 500 !important;
        box-shadow: none !important;
    }
    button[kind="secondary"]:hover {
        background: #f3f4f6 !important;
        border-color: #d1d5db !important;
    }

    /* === Hero === */
    .hero {
        border-bottom: 1px solid #e5e7eb;
        padding: 1.5rem 0;
        margin-bottom: 1.5rem;
    }
    .hero-title {
        font-size: 1.75rem; font-weight: 700;
        color: #111827; letter-spacing: -0.02em;
    }
    .hero-sub {
        font-size: 0.9rem; color: #6b7280; margin-top: 0.25rem;
    }
    .hero-badges { margin-top: 0.75rem; display: flex; gap: 0.4rem; flex-wrap: wrap; }
    .hero-badge {
        background: #f3f4f6; border: 1px solid #e5e7eb;
        color: #374151;
        padding: 2px 10px; border-radius: 4px;
        font-size: 0.72rem; font-weight: 500; letter-spacing: 0.02em;
    }

    /* === 统计卡片：纯线框 === */
    .stat-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0;
               border: 1px solid #e5e7eb; border-radius: 6px; margin-bottom: 1.5rem; }
    .stat-row > .stat-card + .stat-card { border-left: 1px solid #e5e7eb; }
    .stat-card {
        padding: 1rem 1.2rem;
        background: #ffffff;
        border-radius: 0;
        box-shadow: none !important;
    }
    .stat-card:hover { transform: none; box-shadow: none; }
    .stat-icon {
        display: flex; align-items: center; justify-content: center;
        width: 28px; height: 28px; border-radius: 6px;
        background: #f3f4f6; margin-bottom: 0.4rem;
        color: #111827;
    }
    .stat-icon .ic { width: 16px; height: 16px; }
    .stat-value { font-size: 1.5rem; font-weight: 700; color: #111827; }
    .stat-label { font-size: 0.72rem; color: #6b7280; margin-top: 0.1rem; font-weight: 500; }

    /* === 回答卡片 === */
    .answer-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        box-shadow: none;
    }
    .answer-header {
        display: flex; gap: 0.5rem; align-items: center;
        font-size: 0.8rem; color: #6b7280;
        margin-bottom: 0.6rem; flex-wrap: wrap;
        padding-bottom: 0.6rem; border-bottom: 1px solid #f3f4f6;
    }
    .tag {
        display: inline-flex; align-items: center; gap: 4px;
        padding: 1px 8px; border-radius: 4px;
        font-size: 0.7rem; font-weight: 500;
        border: 1px solid transparent;
    }
    .tag-llm { background: #f3f4f6; color: #111827; border-color: #e5e7eb; }
    .tag-hybrid { background: #ffffff; color: #374151; border-color: #d1d5db; }
    .tag-conf-high { background: #ecfdf5; color: #047857; border-color: #a7f3d0; }
    .tag-conf-mid  { background: #fffbeb; color: #b45309; border-color: #fde68a; }
    .tag-conf-low  { background: #fef2f2; color: #b91c1c; border-color: #fecaca; }

    /* === 来源卡片 === */
    .source-card {
        background: #fafafa;
        border: 1px solid #e5e7eb;
        border-left: 2px solid #111827;
        border-radius: 4px;
        padding: 0.6rem 0.85rem;
        margin: 0.4rem 0;
        font-size: 0.82rem;
    }
    .source-header {
        font-weight: 600; color: #1f2937;
        margin-bottom: 0.2rem; font-size: 0.78rem;
        display: flex; justify-content: space-between; align-items: center;
    }
    .source-content { color: #6b7280; font-size: 0.8rem; line-height: 1.55; }

    /* === 用户气泡 === */
    .user-bubble {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 0.6rem 0.9rem; margin: 0.4rem 0;
        color: #111827; font-weight: 500;
        box-shadow: none;
        max-width: 80%;
        margin-left: auto;
    }

    /* === 空状态欢迎页 === */
    .welcome {
        background: #ffffff;
        border-radius: 6px;
        border: 1px solid #e5e7eb;
        padding: 2.5rem 2rem;
        text-align: center;
        margin: 3rem auto;
        max-width: 640px;
    }
    .welcome-icon {
        display: inline-flex; align-items: center; justify-content: center;
        width: 64px; height: 64px; border-radius: 14px;
        background: #f3f4f6; margin-bottom: 1rem;
    }
    .welcome-icon .ic { width: 32px; height: 32px; }
    .welcome-title { font-size: 1.25rem; font-weight: 700; color: #111827; margin-bottom: 0.4rem; }
    .welcome-desc { color: #6b7280; font-size: 0.9rem; line-height: 1.7; }
    .welcome-steps {
        display: flex; gap: 0.5rem; justify-content: center;
        margin-top: 1.25rem; flex-wrap: wrap;
    }
    .step-badge {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 4px;
        padding: 0.5rem 0.85rem;
        font-size: 0.8rem; color: #374151;
    }
    .step-num {
        display: inline-block;
        width: 18px; height: 18px; line-height: 17px;
        background: #111827; color: #ffffff;
        border-radius: 50%;
        font-size: 0.68rem; font-weight: 700;
        margin-right: 5px; text-align: center;
    }

    /* === Sidebar section 标题 === */
    .sidebar-section {
        font-size: 0.68rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.12em;
        color: #9ca3af !important;
        margin: 1rem 0 0.4rem;
        display: flex; align-items: center; gap: 6px;
    }
    .sidebar-section .ic { width: 14px; height: 14px; }
</style>
""", unsafe_allow_html=True)


# ==================== 状态管理 ====================
def init_session():
    defaults = {
        'chat_history': [],
        'vector_store_manager': None,
        'advanced_retriever': None,
        'qa_generator': None,
        'documents': [],
        'is_initialized': False,
        'last_build_info': {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_api_config():
    return {
        'api_key': os.getenv('LLM_API_KEY'),
        'api_base': os.getenv('LLM_API_BASE'),
        'llm_model': os.getenv('LLM_MODEL', 'Qwen/Qwen2.5-7B-Instruct'),
    }


# ==================== 知识库构建 ====================
def build_knowledge_base(data_dir, embedding_model, chunk_size, chunk_overlap,
                        use_hybrid, use_rerank, collection_name="default"):
    try:
        progress_bar = st.progress(0)
        api_cfg = get_api_config()

        st.caption("加载文档...")
        documents = load_documents_from_directory(data_dir)
        if not documents:
            progress_bar.empty()
            return False, "未找到任何文档"

        progress_bar.progress(20)
        st.caption(f"加载 {len(documents)} 个文档片段")

        smart_splitter = SmartTextSplitter(
            default_chunk_size=chunk_size,
            default_chunk_overlap=chunk_overlap,
            collection=collection_name or "default",
        )
        split_docs = smart_splitter.split_documents(documents)

        progress_bar.progress(50)
        st.caption(f"智能分块完成: {len(split_docs)} 个片段")

        actual_model = embedding_model.split(" (")[0].strip()
        use_openai_emb = "openai" in embedding_model.lower()

        embedding_mgr = EmbeddingManager(
            model_name=actual_model,
            device="cpu",
            use_openai=use_openai_emb,
            openai_api_key=api_cfg['api_key'],
        )

        progress_bar.progress(70)
        st.caption("创建 FAISS 向量库...")
        vector_store = VectorStoreManager(
            embedding_manager=embedding_mgr,
            vector_store_type="faiss",
            persist_directory="./data/vector_db",
        )
        vector_store.create_vector_store(split_docs)

        progress_bar.progress(90)
        st.caption("初始化检索器...")

        query_transformer = None
        try:
            query_transformer = QueryTransformer(
                api_key=api_cfg['api_key'],
                api_base=api_cfg['api_base'] if api_cfg['api_base'] else None,
            )
        except Exception as e:
            st.warning(f"QueryTransformer 初始化失败: {e}")
            query_transformer = None

        retriever = AdvancedRetriever(
            vector_store_manager=vector_store,
            documents=split_docs,
            use_hybrid=use_hybrid,
            use_rerank=use_rerank,
            top_k=5,
            recall_k=30,
            query_transformer=query_transformer,
        )

        file_names = sorted({d.metadata.get('file_name', '未知') for d in documents})
        file_types = sorted({d.metadata.get('file_type', 'recursive') for d in split_docs})

        progress_bar.progress(100)
        st.caption("知识库构建完成")

        st.session_state.vector_store_manager = vector_store
        st.session_state.advanced_retriever = retriever
        st.session_state.documents = split_docs
        st.session_state.is_initialized = True
        st.session_state.last_build_info = {
            'doc_count': len(documents),
            'chunk_count': len(split_docs),
            'file_list': file_names,
            'file_types': file_types,
            'collection': collection_name or "default",
            'embedding_model': actual_model,
            'query_transformer': query_transformer.is_available() if query_transformer else False,
        }

        return True, f"成功：{len(documents)} 文档 → {len(split_docs)} 分块"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"构建失败: {str(e)}"


# ==================== 问答执行 ====================
def ask_question(query):
    import time
    start = time.time()

    retriever = st.session_state.advanced_retriever
    api_cfg = get_api_config()

    metadata_filter = {}
    cf = st.session_state.get("filter_collection", "（全部集合）")
    ff = st.session_state.get("filter_file", "（全部文档）")
    tf = st.session_state.get("filter_type", "（全部类型）")
    if cf and cf != "（全部集合）":
        metadata_filter["collection"] = cf
    if ff and ff != "（全部文档）":
        metadata_filter["file_name"] = ff
    if tf and tf != "（全部类型）":
        metadata_filter["file_type"] = tf

    retrieval_results = retriever.retrieve(query, filter=metadata_filter or None)
    docs = [r.document for r in retrieval_results]
    retrieval_time = round(time.time() - start, 3)

    qa = st.session_state.qa_generator
    answer_text = ""
    confidence = 0.0
    answer_time = 0.0
    used_llm = False
    llm_error = None

    if qa is not None and api_cfg['api_key']:
        try:
            gen_start = time.time()
            response = qa.generate(query, docs)
            answer_text = response.answer
            confidence = response.confidence
            answer_time = round(time.time() - gen_start, 3)
            used_llm = True
        except Exception as e:
            llm_error = str(e)
            st.warning(f"LLM 失败（{llm_error[:60]}），自动降级")

    if not used_llm:
        answer_text = build_simple_answer(docs)
        # 动态置信度：基于检索到的来源数量 + rerank 分数
        confidence = _compute_retrieval_confidence(docs, retrieval_results)

    return {
        'answer': answer_text,
        'sources': docs,
        'confidence': confidence,
        'retrieval_time': retrieval_time,
        'answer_time': answer_time,
        'used_llm': used_llm,
    }


def build_simple_answer(documents):
    if not documents:
        return "未找到相关文档。请先构建知识库或调整检索参数。"
    lines = ["从文档中找到以下相关内容：\n"]
    for i, doc in enumerate(documents[:5], 1):
        fn = doc.metadata.get('file_name', '未知文件')
        preview = doc.page_content.strip()[:250]
        lines.append(f"**来源 {i}**: {fn}\n\n{preview}")
        lines.append("---")
    return "\n\n".join(lines)


def _compute_retrieval_confidence(docs, retrieval_results):
    """
    基于检索结果数量 + rerank 分数动态估算置信度
    
    计算逻辑：
    - 来源数量权重 60%（越多来源交叉验证 → 越高置信）
    - 平均分数权重 40%（rerank 分越高 → 越高置信）
    """
    # 维度 1：来源数量（最多 5 条）
    n = len(docs)
    if n == 0:
        return 0.0
    # 1→30%, 2→50%, 3→65%, 4→80%, 5+→90%
    count_score = min(0.3 + n * 0.13, 0.9)

    # 维度 2：分数分布
    if retrieval_results:
        # rerank 后的分数通常很小，做归一化
        scores = [r.score for r in retrieval_results[:5]]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores) if scores else 0
        # max_score > 0.05 → 好结果，> 0.1 → 很好
        score_score = min(max_score * 4, 1.0)  # 0.0→0, 0.25→1.0
    else:
        score_score = 0.3

    # 加权融合
    confidence = 0.6 * count_score + 0.4 * score_score
    # clamp 到 [0.0, 0.95]，上限 0.95 表示"不是 LLM 生成的但信息够全"
    return round(min(max(confidence, 0.0), 0.95), 2)


# ==================== 主函数 ====================
def main():
    init_session()
    api_cfg = get_api_config()

    # ============ 侧边栏（深色主题） ============
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:left; padding: 0.2rem 0 0.8rem; border-bottom: 1px solid #e5e7eb; margin-bottom: 0.8rem;">
            <div style="font-size:1rem; font-weight:700; color:#111827; letter-spacing:-0.02em; display:flex; align-items:center; gap:6px;">
                {IC['brain']} RAG 知识库系统
            </div>
            <div style="font-size:0.72rem; color:#9ca3af; margin-top:2px;">knowledge · qa · multi-format</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="sidebar-section">{IC["folder"]} 数据配置</div>', unsafe_allow_html=True)
        data_dir = st.text_input(
            "数据目录",
            value=str(project_root / "data" / "documents"),
            label_visibility="collapsed",
        )
        collection_name = st.text_input(
            "知识库名称", value="default",
            help="用于区分不同知识库集合",
        )

        # 文件上传（直接保存到 data_dir）
        st.markdown(f'<div class="sidebar-section">{IC["upload"]} 上传文档</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "拖放或点击上传",
            type=["pdf", "docx", "doc", "txt", "md", "markdown", "xlsx", "xlsm", "csv", "pptx", "ppt", "json", "yaml", "yml", "py", "java", "html", "jpg", "jpeg", "png"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded:
            upload_dir = Path(data_dir)
            upload_dir.mkdir(parents=True, exist_ok=True)
            saved_count = 0
            for up in uploaded:
                out_path = upload_dir / up.name
                if not out_path.exists() or out_path.stat().st_size != up.size:
                    with open(out_path, "wb") as f:
                        f.write(up.getvalue())
                    saved_count += 1
            if saved_count > 0:
                st.success(f"已保存 {saved_count} 个新文件到 {data_dir}")
            else:
                st.info(f"{len(uploaded)} 个文件已存在，无需重复保存")

        st.markdown(f'<div class="sidebar-section">{IC["chip"]} Embedding</div>', unsafe_allow_html=True)
        embedding_model = st.selectbox(
            "Embedding 模型",
            ["BAAI/bge-large-zh", "BAAI/bge-base-zh", "paraphrase-MiniLM-L3-v2"],
            index=0, label_visibility="collapsed",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            chunk_size = st.number_input("块大小", min_value=100, max_value=2000, value=500, step=50)
        with col_b:
            chunk_overlap = st.number_input("重叠", min_value=0, max_value=500, value=50, step=10)

        st.markdown(f'<div class="sidebar-section">{IC["bolt"]} 检索策略</div>', unsafe_allow_html=True)
        col_c, col_d = st.columns(2)
        with col_c:
            use_hybrid = st.checkbox("BM25 混合", value=True)
        with col_d:
            use_rerank = st.checkbox("Rerank", value=True)

        st.markdown("---")

        # 持久化状态 + 快速加载
        persist_dir = Path(project_root / "data" / "vector_db")
        has_saved = persist_dir.exists() and any(persist_dir.glob("*.faiss"))
        if has_saved:
            st.success("检测到已保存的向量库（可直接加载）")
            if st.button("加载已有知识库", use_container_width=True):
                try:
                    st.caption("加载向量库...")
                    vector_store = VectorStoreManager(
                        embedding_manager=EmbeddingManager(
                            model_name=embedding_model.split(" (")[0].strip(),
                            device="cpu",
                        ),
                        vector_store_type="faiss",
                        persist_directory=str(persist_dir),
                    )
                    vector_store.load_vector_store(str(persist_dir))
                    saved_docs = getattr(vector_store, '_cached_documents', None) or []
                    if not saved_docs:
                        st.warning("向量库已加载但无文档缓存，请重新构建")
                    else:
                        retriever = AdvancedRetriever(
                            vector_store_manager=vector_store,
                            documents=saved_docs,
                            use_hybrid=use_hybrid,
                            use_rerank=use_rerank,
                            top_k=5,
                            recall_k=30,
                        )
                        st.session_state.vector_store_manager = vector_store
                        st.session_state.advanced_retriever = retriever
                        st.session_state.documents = saved_docs
                        st.session_state.is_initialized = True
                        st.success(f"加载成功！{len(saved_docs)} 个分块")
                        st.rerun()
                except Exception as e:
                    st.error(f"加载失败: {e}")

        build_btn = st.button("构建知识库", type="primary", use_container_width=True)

        # API 状态
        st.markdown(f'<div class="sidebar-section">{IC["plug"]} API 状态</div>', unsafe_allow_html=True)
        if api_cfg['api_key']:
            st.success(f"LLM 已就绪（{api_cfg['llm_model'][:25]}）")
        else:
            st.info("无 LLM API（检索模式）")

        # ============ 知识库状态仪表盘 ============
        st.markdown(f'<div class="sidebar-section">{IC["bar-chart"]} 知识库</div>', unsafe_allow_html=True)
        if st.session_state.is_initialized and st.session_state.last_build_info:
            info = st.session_state.last_build_info
            st.markdown(f"""
            <div style="background:#ffffff; border:1px solid #e5e7eb; border-radius:6px; padding:0.75rem 1rem; margin-bottom:0.5rem;">
                <div style="display:flex; justify-content:space-between; margin-bottom:0.35rem;">
                    <span style="color:#6b7280; font-size:0.72rem;">{IC['file-text']} 文档</span>
                    <span style="color:#111827; font-weight:700; font-size:0.85rem;">{info['doc_count']}</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:0.35rem;">
                    <span style="color:#6b7280; font-size:0.72rem;">{IC['scissors']} 分块</span>
                    <span style="color:#111827; font-weight:700; font-size:0.85rem;">{info['chunk_count']}</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:0.35rem;">
                    <span style="color:#6b7280; font-size:0.72rem;">{IC['tag']} 集合</span>
                    <span style="color:#111827; font-weight:700; font-size:0.85rem;">{info.get('collection', 'default')}</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#6b7280; font-size:0.72rem;">{IC['sparkles']} 增强</span>
                    <span style="color:{'#047857' if info.get('query_transformer') else '#6b7280'}; font-weight:600; font-size:0.72rem;">
                        {'Multi-Query 已启用' if info.get('query_transformer') else '基础检索'}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("文档列表", expanded=False):
                for f in info.get('file_list', []):
                    st.markdown(f"<div style='font-size:0.75rem; color:#94a3b8; padding:2px 0;'>• {f}</div>",
                                unsafe_allow_html=True)

            # 检索过滤
            st.markdown(f'<div class="sidebar-section">{IC["search"]} 检索过滤</div>', unsafe_allow_html=True)
            all_collections = ["（全部集合）"] + sorted(
                {d.metadata.get('collection', 'default') for d in st.session_state.documents}
            )
            st.selectbox("集合", options=all_collections, index=0, key="filter_collection", label_visibility="collapsed")
            all_files = ["（全部文档）"] + sorted(info.get('file_list', []))
            st.selectbox("文件", options=all_files, index=0, key="filter_file", label_visibility="collapsed")
            all_types = ["（全部类型）"] + sorted(
                {d.metadata.get('file_type', 'recursive') for d in st.session_state.documents}
            )
            st.selectbox("类型", options=all_types, index=0, key="filter_type", label_visibility="collapsed")

            if st.button("清除对话", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        else:
            st.info("知识库尚未构建")

    # ============ 构建逻辑 ============
    if build_btn:
        if not Path(data_dir).exists():
            st.error(f"数据目录不存在: {data_dir}")
        else:
            success, msg = build_knowledge_base(
                data_dir, embedding_model, chunk_size, chunk_overlap,
                use_hybrid, use_rerank, collection_name=collection_name or "default",
            )
            st.progress(100).empty()
            if success:
                st.success(msg)
                qa = None
                if api_cfg['api_key']:
                    try:
                        qa = QAGenerator(
                            model_type="openai",
                            model_name=api_cfg['llm_model'],
                            temperature=0.3,
                            api_key=api_cfg['api_key'],
                            api_base=api_cfg['api_base'],
                        )
                    except Exception as e:
                        st.warning(f"LLM 初始化失败: {str(e)[:60]}")
                st.session_state.qa_generator = qa
            else:
                st.error(msg)
            st.rerun()

    # ============ 主界面 ============
    if not st.session_state.is_initialized:
        # ===== 欢迎页 =====
        st.markdown("""
        <div class="hero">
            <div class="hero-title">RAG 知识库问答系统</div>
            <div class="hero-sub">Retrieval-Augmented Generation · Multi-Format Knowledge Base QA</div>
            <div class="hero-badges">
                <span class="hero-badge">PDF</span>
                <span class="hero-badge">Word</span>
                <span class="hero-badge">Excel</span>
                <span class="hero-badge">CSV</span>
                <span class="hero-badge">PPT</span>
                <span class="hero-badge">Markdown</span>
                <span class="hero-badge">Code</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="welcome">
            <div class="welcome-icon">{IC['brain']}</div>
            <div class="welcome-title">开始使用</div>
            <div class="welcome-desc">将文档放入 data/documents 目录，点击左侧 {IC['hammer']} 构建知识库，即可开始智能问答</div>
            <div class="welcome-steps">
                <div class="step-badge"><span class="step-num">1</span>放入文档</div>
                <div class="step-badge"><span class="step-num">2</span>点击构建</div>
                <div class="step-badge"><span class="step-num">3</span>开始提问</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        existing = list(Path(data_dir).glob("*")) if Path(data_dir).exists() else []
        if existing:
            st.markdown(f"#### {IC['folder-open']} 检测到的文档", unsafe_allow_html=True)
            for f in existing:
                if f.is_file():
                    st.markdown(f"- {IC['file-text']} {f.name}", unsafe_allow_html=True)
        return

    # ===== Hero + 统计 =====
    info = st.session_state.last_build_info
    st.markdown("""
    <div class="hero">
        <div class="hero-title">RAG 知识库问答系统</div>
        <div class="hero-sub">Knowledge base ready · Multi-Query + HyDE retrieval enhancement</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-icon">{IC['file']}</div>
            <div class="stat-value">{info.get('doc_count', 0)}</div>
            <div class="stat-label">原始文档</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">{IC['scissors']}</div>
            <div class="stat-value">{info.get('chunk_count', 0)}</div>
            <div class="stat-label">向量分块</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">{IC['puzzle']}</div>
            <div class="stat-value">{len(info.get('file_types', []))}</div>
            <div class="stat-label">分块策略</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">{IC['sparkles'] if info.get('query_transformer') else IC['search2']}</div>
            <div class="stat-value">{'开启' if info.get('query_transformer') else '关闭'}</div>
            <div class="stat-label">查询增强</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ===== 快捷提问 =====
    st.markdown(f"#### {IC['help']} 试试这些问题", unsafe_allow_html=True)
    quick_questions = [
        "这份文档讲的是什么？",
        "有哪些主要内容？",
        "技术栈是什么？",
        "有什么关键结论？",
        "作者是谁？",
        "如何使用？",
    ]
    cols = st.columns(3)
    for i, q in enumerate(quick_questions):
        if cols[i % 3].button(q, key=f"quick_{i}", use_container_width=True, type="secondary"):
            st.session_state.pending_question = q
            st.rerun()

    st.markdown("---")

    # ===== 对话历史 =====
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-bubble">{IC["user"]} {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                show_answer_card(msg)
        st.markdown("---")

    # ===== 输入框 =====
    with st.form("question_form", clear_on_submit=True):
        user_input = st.text_input(
            "输入问题",
            placeholder="输入你的问题，然后回车发送...",
            label_visibility="collapsed",
        )
        sent = st.form_submit_button("发送", use_container_width=True)

    if 'pending_question' in st.session_state and st.session_state.pending_question:
        sent = True
        user_input = st.session_state.pending_question
        st.session_state.pending_question = ""

    # ===== 问答处理 =====
    if sent and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.markdown(f'<div class="user-bubble">{IC["user"]} {user_input}</div>', unsafe_allow_html=True)

        with st.spinner("检索中..."):
            result = ask_question(user_input)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result['answer'],
            "sources": result['sources'],
            "confidence": result['confidence'],
            "retrieval_time": result['retrieval_time'],
            "answer_time": result['answer_time'],
            "used_llm": result['used_llm'],
        })
        show_answer_card(st.session_state.chat_history[-1])
        st.rerun()


def show_answer_card(msg):
    """渲染现代化回答卡片"""
    answer = msg["content"]
    sources = msg.get("sources", [])
    confidence = msg.get("confidence", 0)
    ret_time = msg.get("retrieval_time", 0)
    ans_time = msg.get("answer_time", 0)
    used_llm = msg.get("used_llm", False)

    conf_pct = f"{confidence:.0%}"
    if confidence >= 0.7:
        conf_tag = f'<span class="tag tag-conf-high">高置信 {conf_pct}</span>'
    elif confidence >= 0.4:
        conf_tag = f'<span class="tag tag-conf-mid">中置信 {conf_pct}</span>'
    else:
        conf_tag = f'<span class="tag tag-conf-low">低置信 {conf_pct}</span>'

    mode_tag = f'<span class="tag tag-llm">{IC["brain2"]} LLM 生成</span>' if used_llm else f'<span class="tag tag-hybrid">{IC["search2"]} 检索模式</span>'

    st.markdown(f"""
    <div class="answer-card">
        <div class="answer-header">
            <span>{IC['brain2']} 助手回答</span>
            {mode_tag}
            {conf_tag}
            <span style="margin-left:auto;">{IC['clock']} 检索 {ret_time}s{f' · 生成 {ans_time}s' if used_llm else ''}</span>
        </div>
    """, unsafe_allow_html=True)
    st.markdown(answer)

    if sources:
        with st.expander(f"参考来源（{len(sources)} 条）", expanded=False):
            for i, doc in enumerate(sources, 1):
                fn = doc.metadata.get('file_name', '未知文件')
                ft = doc.metadata.get('file_type', 'recursive')
                content = doc.page_content.strip()
                st.markdown(f"""
                <div class="source-card">
                    <div class="source-header">
                        <span>来源 {i}: {IC['file-text']} {fn}</span>
                        <span style="color:#9ca3af;">{ft}</span>
                    </div>
                    <div class="source-content">{content[:300]}{'...' if len(content) > 300 else ''}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
