"""
RAG 知识库问答系统 - 主程序入口

使用方法:
    python main.py --mode web          # 启动 Web 界面
    python main.py --mode cli          # 启动命令行界面
    python main.py --mode build        # 仅构建知识库
    python main.py --mode test         # 运行测试
"""
import os

# 设置 HuggingFace 镜像源（必须在导入 transformers 之前）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'

import argparse
import logging
import sys
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_env():
    """加载环境变量"""
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"已加载环境变量: {env_path}")
    except ImportError:
        logger.warning("python-dotenv 未安装，跳过环境变量加载")


def build_knowledge_base(args):
    """构建知识库"""
    from core import (
        AdvancedRetriever,
        EmbeddingManager,
        TextSplitter,
        VectorStoreManager,
        load_documents_from_directory,
    )

    logger.info("开始构建知识库...")

    # 1. 加载文档
    logger.info(f"从目录加载文档: {args.data_dir}")
    documents = load_documents_from_directory(args.data_dir)
    logger.info(f"成功加载 {len(documents)} 个文档片段")

    # 2. 文本分块
    logger.info("进行文本分块...")
    text_splitter = TextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        splitter_type="recursive"
    )
    split_documents = text_splitter.split_documents(documents)
    logger.info(f"分块完成: {len(split_documents)} 个片段")

    # 3. 初始化 Embedding
    logger.info(f"初始化 Embedding 模型: {args.embedding_model}")
    embedding_manager = EmbeddingManager(
        model_name=args.embedding_model,
        device=args.device
    )

    # 4. 创建向量数据库
    logger.info("创建向量数据库...")
    vector_store_manager = VectorStoreManager(
        embedding_manager=embedding_manager,
        vector_store_type=args.vector_store,
        persist_directory=args.vector_db_path
    )
    vector_store_manager.create_vector_store(split_documents)
    logger.info(f"向量数据库已保存到: {args.vector_db_path}")

    # 5. 初始化检索器
    logger.info("初始化检索器...")
    advanced_retriever = AdvancedRetriever(
        vector_store_manager=vector_store_manager,
        documents=split_documents,
        use_hybrid=args.use_hybrid,
        use_rerank=args.use_rerank
    )

    logger.info("知识库构建完成！")
    return vector_store_manager, advanced_retriever


def run_cli(args):
    """运行命令行界面"""
    from core import QAGenerator

    logger.info("启动命令行问答系统...")

    # 构建知识库
    vector_store_manager, advanced_retriever = build_knowledge_base(args)

    # 初始化 QA 生成器
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("未提供 OpenAI API Key")
        return

    qa_generator = QAGenerator(
        llm_model=args.llm_model,
        api_key=api_key,
        api_base=args.api_base
    )

    print("\n" + "="*50)
    print("🤖 RAG 知识库问答系统")
    print("="*50)
    print("输入问题开始对话，输入 'quit' 或 'exit' 退出\n")

    while True:
        try:
            query = input("👤 你: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break

            if not query:
                continue

            # 检索
            print("🔍 检索中...")
            retrieval_results = advanced_retriever.retrieve(query)
            documents = [r.document for r in retrieval_results]

            # 生成答案
            print("🤔 生成答案中...")
            response = qa_generator.generate_answer(query, documents)

            # 显示结果
            print(f"\n🤖 助手: {response.answer}")
            print(f"\n📊 置信度: {response.confidence:.2%}")

            if response.sources:
                print("\n📚 参考来源:")
                for source in response.sources:
                    print(f"  [{source['index']}] {source['metadata'].get('file_name', '未知')}")

            print("-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            logger.error(f"处理查询时出错: {str(e)}")
            print(f"错误: {str(e)}")


def run_web(args):
    """运行 Web 界面"""
    import subprocess

    logger.info("启动 Web 界面...")

    web_app_path = Path(__file__).parent / "frontend" / "app.py"

    if not web_app_path.exists():
        logger.error(f"Web 应用文件不存在: {web_app_path}")
        return

    # 设置环境变量
    env = os.environ.copy()
    if args.api_key:
        env["OPENAI_API_KEY"] = args.api_key

    # 启动 Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(web_app_path),
        "--server.port", str(args.port),
        "--server.address", args.host
    ]

    logger.info(f"启动命令: {' '.join(cmd)}")
    subprocess.run(cmd, env=env)


def run_test(args):
    """运行测试"""
    import subprocess

    logger.info("运行测试...")

    test_dir = Path(__file__).parent / "tests"

    if test_dir.exists():
        subprocess.run([sys.executable, "-m", "pytest", str(test_dir), "-v"])
    else:
        logger.warning("测试目录不存在")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="RAG 知识库问答系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 启动 Web 界面
    python main.py --mode web --api-key your_key
    
    # 命令行交互
    python main.py --mode cli --data-dir ./my_docs
    
    # 仅构建知识库
    python main.py --mode build --data-dir ./my_docs
        """
    )

    # 模式选择
    parser.add_argument(
        "--mode",
        choices=["web", "cli", "build", "test"],
        default="web",
        help="运行模式 (默认: web)"
    )

    # API 配置
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API Key"
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv("OPENAI_API_BASE"),
        help="OpenAI API Base URL"
    )

    # 数据配置
    parser.add_argument(
        "--data-dir",
        default="./data/documents",
        help="文档数据目录"
    )
    parser.add_argument(
        "--vector-db-path",
        default="./data/vector_db",
        help="向量数据库存储路径"
    )

    # 模型配置
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-large-zh",
        help="Embedding 模型名称"
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-3.5-turbo",
        help="LLM 模型名称"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "cuda:0", "cuda:1"],
        help="运行设备"
    )

    # 分块配置
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="文本分块大小"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="文本分块重叠大小"
    )

    # 检索配置
    parser.add_argument(
        "--vector-store",
        default="faiss",
        choices=["faiss", "chroma"],
        help="向量数据库类型"
    )
    parser.add_argument(
        "--use-hybrid",
        action="store_true",
        default=True,
        help="启用混合检索"
    )
    parser.add_argument(
        "--use-rerank",
        action="store_true",
        default=True,
        help="启用重排序"
    )

    # Web 配置
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Web 服务主机地址"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Web 服务端口"
    )

    args = parser.parse_args()

    # 加载环境变量
    load_env()

    # 根据模式执行
    if args.mode == "web":
        run_web(args)
    elif args.mode == "cli":
        run_cli(args)
    elif args.mode == "build":
        build_knowledge_base(args)
    elif args.mode == "test":
        run_test(args)


if __name__ == "__main__":
    main()
