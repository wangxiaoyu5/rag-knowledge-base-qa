"""
Streamlit Web 界面 - RAG 知识库问答系统
"""
import os
# 设置 HuggingFace 镜像源（必须在导入 transformers 之前）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'

import streamlit as st
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core import (
    load_documents_from_directory,
    TextSplitter,
    EmbeddingManager,
    VectorStoreManager,
    AdvancedRetriever,
    QAGenerator,
    QAResponse
)

# 页面配置
st.set_page_config(
    page_title="RAG 知识库问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .source-box {
        background-color: #f5f5f5;
        padding: 0.8rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .confidence-high { color: #4caf50; }
    .confidence-medium { color: #ff9800; }
    .confidence-low { color: #f44336; }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """初始化会话状态"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'vector_store_manager' not in st.session_state:
        st.session_state.vector_store_manager = None
    
    if 'advanced_retriever' not in st.session_state:
        st.session_state.advanced_retriever = None
    
    if 'qa_generator' not in st.session_state:
        st.session_state.qa_generator = None
    
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    
    if 'is_initialized' not in st.session_state:
        st.session_state.is_initialized = False


def get_confidence_color(confidence: float) -> str:
    """根据置信度返回颜色类名"""
    if confidence >= 0.7:
        return "confidence-high"
    elif confidence >= 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"


def display_chat_message(role: str, content: str, sources=None, confidence=None):
    """显示聊天消息"""
    if role == "user":
        st.markdown(f'<div class="chat-message user-message"><strong>👤 用户:</strong><br>{content}</div>', unsafe_allow_html=True)
    else:
        confidence_html = ""
        if confidence is not None:
            color_class = get_confidence_color(confidence)
            confidence_html = f'<span class="{color_class}">置信度: {confidence:.1%}</span>'
        
        sources_html = ""
        if sources:
            sources_html = '<div style="margin-top: 10px;"><strong>📚 参考来源:</strong></div>'
            for i, source in enumerate(sources):
                # 处理 Document 对象或字典
                if hasattr(source, 'page_content'):
                    # Document 对象
                    file_name = source.metadata.get('file_name', source.metadata.get('source', '未知文件'))
                    source_content = source.page_content
                elif isinstance(source, dict):
                    # 字典格式
                    file_name = source.get('metadata', {}).get('file_name', '未知文件')
                    source_content = source.get('content', '')
                else:
                    file_name = '未知文件'
                    source_content = str(source)
                
                sources_html += f'<div class="source-box"><strong>来源 {i + 1}:</strong> {file_name}<br><small>{source_content[:150]}...</small></div>'
        
        st.markdown(f'<div class="chat-message assistant-message"><strong>🤖 助手:</strong> {confidence_html}<br><br>{content}{sources_html}</div>', unsafe_allow_html=True)


def build_knowledge_base(
    data_dir: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    use_hybrid: bool,
    use_rerank: bool,
    progress_bar
):
    """构建知识库"""
    try:
        progress_bar.progress(10, "📄 加载文档...")
        
        # 1. 加载文档
        documents = load_documents_from_directory(data_dir)
        if not documents:
            return False, "未找到任何文档，请检查数据目录"
        
        progress_bar.progress(30, f"✅ 加载 {len(documents)} 个文档片段")
        
        # 2. 文本分块
        progress_bar.progress(40, "✂️ 文本分块...")
        text_splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            splitter_type="recursive"
        )
        split_documents = text_splitter.split_documents(documents)
        
        progress_bar.progress(60, f"✅ 分块完成: {len(split_documents)} 个片段")
        
        # 3. 初始化 Embedding
        progress_bar.progress(70, "🔢 初始化 Embedding 模型...")
        
        # 检查是否使用 OpenAI Embedding
        use_openai = "openai" in embedding_model.lower() or "text-embedding-ada" in embedding_model.lower()
        
        # 提取实际的模型名称（去掉括号中的说明）
        actual_model_name = embedding_model.split(" (")[0].strip()
        
        if use_openai:
            embedding_manager = EmbeddingManager(
                use_openai=True
            )
        else:
            embedding_manager = EmbeddingManager(
                model_name=actual_model_name,
                device="cpu"
            )
        
        # 4. 创建向量数据库
        progress_bar.progress(80, "💾 创建向量数据库...")
        # 使用相对路径，避免 FAISS 的 Unicode 路径问题
        vector_db_path = "./data/vector_db"
        vector_store_manager = VectorStoreManager(
            embedding_manager=embedding_manager,
            vector_store_type="faiss",
            persist_directory=vector_db_path
        )
        vector_store_manager.create_vector_store(split_documents)
        
        progress_bar.progress(90, "✅ 向量数据库创建完成")
        
        # 5. 初始化检索器
        progress_bar.progress(95, "🔍 初始化检索器...")
        advanced_retriever = AdvancedRetriever(
            vector_store_manager=vector_store_manager,
            documents=split_documents,
            use_hybrid=use_hybrid,
            use_rerank=use_rerank,
            top_k=5
        )
        
        # 保存到会话状态
        st.session_state.vector_store_manager = vector_store_manager
        st.session_state.advanced_retriever = advanced_retriever
        st.session_state.documents = split_documents
        st.session_state.is_initialized = True
        
        progress_bar.progress(100, "✅ 知识库构建完成！")
        
        return True, f"成功构建知识库：{len(documents)} 个原始文档，{len(split_documents)} 个分块"
        
    except Exception as e:
        return False, f"构建知识库失败: {str(e)}"


def main():
    """主函数"""
    init_session_state()
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 系统配置")
        
        # API 配置
        st.subheader("API 配置")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="输入你的 OpenAI API Key"
        )
        
        api_base = st.text_input(
            "API Base URL (可选)",
            value=os.getenv("OPENAI_API_BASE", ""),
            help="如果使用代理或自定义 API，请填写"
        )
        
        # 知识库配置
        st.subheader("知识库配置")
        
        data_dir = st.text_input(
            "数据目录",
            value=os.path.join(project_root, "data", "documents"),
            help="存放文档的目录路径"
        )
        
        embedding_model = st.selectbox(
            "Embedding 模型",
            ["BAAI/bge-large-zh (本地模型 - 已下载)", "sentence-transformers/paraphrase-MiniLM-L3-v2", "text-embedding-ada-002 (OpenAI API)", "BAAI/bge-base-zh", "moka-ai/m3e-base"],
            index=0,
            help="选择文本向量化的模型（推荐使用本地模型，无需联网）"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input(
                "分块大小",
                min_value=100,
                max_value=2000,
                value=500,
                step=50
            )
        with col2:
            chunk_overlap = st.number_input(
                "重叠大小",
                min_value=0,
                max_value=500,
                value=50,
                step=10
            )
        
        use_hybrid = st.checkbox("启用混合检索 (BM25 + 向量)", value=True)
        use_rerank = st.checkbox("启用重排序 (Cross-Encoder)", value=True)
        
        # 构建知识库按钮
        if st.button("🔨 构建知识库", type="primary", use_container_width=True):
            use_openai_embedding = "openai" in embedding_model.lower() or "text-embedding-ada" in embedding_model.lower()
            
            # 检查 API Key（仅当使用 OpenAI Embedding 或 QA 生成时需要）
            if use_openai_embedding and not api_key:
                st.error("使用 OpenAI Embedding 需要输入 API Key")
            elif not os.path.exists(data_dir):
                st.error(f"数据目录不存在: {data_dir}")
            else:
                progress_bar = st.progress(0)
                try:
                    success, message = build_knowledge_base(
                        data_dir, embedding_model, chunk_size, chunk_overlap,
                        use_hybrid, use_rerank, progress_bar
                    )
                    
                    if success:
                        st.success(message)
                        # 初始化 QA 生成器
                        if api_key:
                            st.session_state.qa_generator = QAGenerator(
                                api_key=api_key,
                                api_base=api_base if api_base else None
                            )
                        else:
                            # 尝试使用本地模型，失败则使用简单检索模式
                            try:
                                qa_gen = QAGenerator(
                                    model_type="local",
                                    model_name="Qwen/Qwen2-0.5B-Instruct"
                                )
                                # 立即测试模型是否能加载
                                qa_gen._load_model()
                                st.session_state.qa_generator = qa_gen
                                st.info("使用本地模型进行问答")
                            except Exception as e:
                                st.session_state.qa_generator = None
                                st.session_state.use_simple_qa = True
                                st.info("使用简单检索模式（无需 API Key）")
                    else:
                        st.error(message)
                except Exception as e:
                    st.error(f"构建知识库时发生意外错误: {str(e)}")
                    import traceback
                    st.error(f"详细错误信息:\n{traceback.format_exc()}")
        
        # 显示状态
        st.divider()
        if st.session_state.is_initialized:
            st.success("✅ 系统已就绪")
        else:
            st.warning("⚠️ 请先构建知识库")
    
    # 主界面
    st.markdown('<div class="main-header">📚 RAG 知识库问答系统</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">基于检索增强生成技术的智能问答系统</div>', 
                unsafe_allow_html=True)
    
    # 显示聊天历史
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            display_chat_message(
                message["role"],
                message["content"],
                message.get("sources"),
                message.get("confidence")
            )
    
    # 输入框
    if st.session_state.is_initialized:
        import re

        def simple_qa_extract(question, documents):
            answer_parts = []
            seen = set()
            
            for doc in documents:
                content = doc.page_content
                file_name = doc.metadata.get('file_name', '未知文件')
                
                if "毕业时间" in question or "毕业日期" in question:
                    match = re.search(r'毕业时间[\uff1a:]?\s*(\d{4}\.\d{2})', content)
                    if match and match.group(1) not in seen:
                        answer_parts.append(f"毕业时间：{match.group(1)}")
                        seen.add(match.group(1))
                
                if "教育背景" in question or "学历" in question:
                    match = re.search(r'学历[\uff1a:]?\s*([\u4e00-\u9fa5]+)', content)
                    if match and match.group(1) not in seen:
                        answer_parts.append(f"学历：{match.group(1)}")
                        seen.add(match.group(1))
                    
                    match = re.search(r'毕业院校[\uff1a:]?\s*([\u4e00-\u9fa5]+)', content)
                    if match and match.group(1) not in seen:
                        answer_parts.append(f"毕业院校：{match.group(1)}")
                        seen.add(match.group(1))
                
                if "专业" in question:
                        match = re.search(r'专\s*业[\uff1a:]?\s*([\u4e00-\u9fa5]+(?:与[\u4e00-\u9fa5]+)*)', content)
                        if match and match.group(1) not in seen:
                            answer_parts.append(f"专业：{match.group(1)}")
                            seen.add(match.group(1))
                
                if "姓名" in question or "名字" in question:
                    match = re.search(r'姓名[\uff1a:]?\s*([\u4e00-\u9fa5]+)', content)
                    if match and match.group(1) not in seen:
                        answer_parts.append(f"姓名：{match.group(1)}")
                        seen.add(match.group(1))
                
                if "邮箱" in question or "email" in question:
                    match = re.search(r'邮箱[\uff1a:]?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', content)
                    if match and match.group(1) not in seen:
                        answer_parts.append(f"邮箱：{match.group(1)}")
                        seen.add(match.group(1))
                
                if "电话" in question or "手机" in question:
                    match = re.search(r'手机号码[\uff1a:]?\s*(\d{11})', content)
                    if match and match.group(1) not in seen:
                        answer_parts.append(f"手机号码：{match.group(1)}")
                        seen.add(match.group(1))
            
            if answer_parts:
                return "；".join(answer_parts)
            else:
                result = "从文档中找到以下相关信息：\n\n"
                for i, doc in enumerate(documents[:3], 1):
                    result += f"📄 来源 {i}: {doc.metadata.get('file_name', '未知文件')}\n"
                    result += f"内容: {doc.page_content[:200]}...\n\n"
                return result

        qa_available = st.session_state.get('qa_generator') is not None or st.session_state.get('use_simple_qa')
        
        with st.container():
            col1, col2 = st.columns([6, 1])
            with col1:
                user_input = st.text_input(
                    "输入你的问题",
                    key="user_input",
                    placeholder="例如：王晓宇的毕业时间是什么？",
                    disabled=not qa_available
                )
            with col2:
                st.write("")  # 占位
                st.write("")
                send_button = st.button("发送", use_container_width=True, disabled=not qa_available)
        
        if send_button and user_input and qa_available:
            # 添加用户消息到历史
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            # 显示用户消息
            with chat_container:
                display_chat_message("user", user_input)
            
            # 生成回答
            with st.spinner("🤔 思考中..."):
                try:
                    # 1. 检索相关文档
                    retrieval_results = st.session_state.advanced_retriever.retrieve(user_input)
                    documents = [r.document for r in retrieval_results]
                    
                    # 2. 生成答案（根据模式选择）
                    use_simple = st.session_state.get('use_simple_qa') or st.session_state.get('qa_generator') is None
                    
                    if use_simple:
                        answer = simple_qa_extract(user_input, documents)
                        response = type('obj', (object,), {
                            'answer': answer,
                            'sources': documents,
                            'confidence': 0.8
                        })
                    else:
                        try:
                            response = st.session_state.qa_generator.generate(
                                user_input, documents
                            )
                            # 检查回答是否为空
                            if not response.answer or not response.answer.strip():
                                st.warning("模型生成内容为空，使用简单检索模式")
                                answer = simple_qa_extract(user_input, documents)
                                response = type('obj', (object,), {
                                    'answer': answer,
                                    'sources': documents,
                                    'confidence': 0.8
                                })
                        except Exception as e:
                            error_str = str(e)
                            if (
                                "Can't load the model" in error_str or 
                                "pytorch_model.bin" in error_str or
                                "WinError 10060" in error_str or
                                "Connection refused" in error_str or
                                "connection attempt failed" in error_str or
                                "download" in error_str.lower() or
                                "huggingface" in error_str.lower()
                            ):
                                st.session_state.use_simple_qa = True
                                st.session_state.qa_generator = None
                                st.warning("模型加载失败（网络或文件问题），已切换到简单检索模式")
                                
                                answer = simple_qa_extract(user_input, documents)
                                response = type('obj', (object,), {
                                    'answer': answer,
                                    'sources': documents,
                                    'confidence': 0.8
                                })
                            else:
                                raise
                    
                    # 3. 添加到历史
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.sources,
                        "confidence": response.confidence
                    })
                    
                    # 4. 显示回答
                    with chat_container:
                        display_chat_message(
                            "assistant",
                            response.answer,
                            response.sources,
                            response.confidence
                        )
                    
                except Exception as e:
                    error_message = f"生成回答时出错: {str(e)}"
                    
                    if "unsupported_country_region_territory" in str(e) or "403" in str(e):
                        st.session_state.use_simple_qa = True
                        st.session_state.qa_generator = None
                        st.warning("检测到 API 访问受限，已自动切换到简单检索模式")
                        
                        answer = simple_qa_extract(user_input, documents)
                        response = type('obj', (object,), {
                            'answer': answer,
                            'sources': documents,
                            'confidence': 0.8
                        })
                        
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response.answer,
                            "sources": response.sources,
                            "confidence": response.confidence
                        })
                        with chat_container:
                            display_chat_message(
                                "assistant",
                                response.answer,
                                response.sources,
                                response.confidence
                            )
                    else:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_message,
                            "sources": [],
                            "confidence": 0.0
                        })
                        with chat_container:
                            display_chat_message("assistant", error_message)
            
            # 清空输入框
            st.rerun()
        elif not qa_available:
            st.warning("⚠️ 请在左侧配置 OpenAI API Key 以启用问答功能")
    else:
        st.info("👈 请在左侧配置系统并构建知识库")
        
        # 显示使用说明
        with st.expander("📖 使用说明"):
            st.markdown("""
            ### 快速开始
            
            1. **配置 API**: 在左侧输入你的 OpenAI API Key
            2. **准备数据**: 将文档（PDF、Word、Markdown 等）放入数据目录
            3. **构建知识库**: 点击"构建知识库"按钮
            4. **开始问答**: 在下方输入框中提问
            
            ### 支持的文档格式
            - PDF (.pdf)
            - Word (.docx, .doc)
            - Markdown (.md)
            - 文本文件 (.txt)
            - 代码文件 (.py, .java, .js 等)
            
            ### 系统特性
            - 🔍 **混合检索**: 结合向量检索和 BM25 关键词检索
            - 🔄 **重排序**: 使用 Cross-Encoder 优化检索结果
            - 📚 **引用溯源**: 答案附带参考来源
            - 💾 **本地存储**: 向量数据库本地持久化
            """)


if __name__ == "__main__":
    main()
