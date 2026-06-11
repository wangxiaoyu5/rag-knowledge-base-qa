"""
功能测试脚本 - 测试核心模块是否能正常导入和运行
"""
import sys
import os

def test_imports():
    """测试依赖导入"""
    print("=" * 60)
    print("测试依赖导入")
    print("=" * 60)
    
    tests = [
        ("langchain", "LangChain 框架"),
        ("langchain_community", "LangChain Community"),
        ("langchain_openai", "LangChain OpenAI"),
        ("faiss", "FAISS 向量库"),
        ("sentence_transformers", "Sentence Transformers"),
        ("transformers", "Transformers"),
        ("torch", "PyTorch"),
        ("pypdf", "PyPDF"),
        ("docx", "python-docx"),
        ("streamlit", "Streamlit"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("tqdm", "tqdm"),
        ("dotenv", "python-dotenv"),
        ("rank_bm25", "Rank BM25"),
    ]
    
    all_passed = True
    for module, name in tests:
        try:
            __import__(module)
            print(f"[OK] {name}")
        except ImportError as e:
            print(f"[FAIL] {name}: {e}")
            all_passed = False
    
    return all_passed

def test_core_modules():
    """测试核心模块导入"""
    print("\n" + "=" * 60)
    print("测试核心模块导入")
    print("=" * 60)
    
    try:
        from core import (
            DocumentLoaderFactory,
            load_document,
            TextSplitter,
            EmbeddingManager,
            VectorStoreManager,
            HybridRetriever,
            Reranker,
            QAGenerator,
        )
        print("[OK] 所有核心模块导入成功")
        return True
    except Exception as e:
        print(f"[FAIL] 核心模块导入失败: {e}")
        return False

def test_document_loader():
    """测试文档加载器"""
    print("\n" + "=" * 60)
    print("测试文档加载器")
    print("=" * 60)
    
    try:
        from core import DocumentLoaderFactory
        factory = DocumentLoaderFactory()
        print(f"[OK] DocumentLoaderFactory 创建成功")
        print(f"[INFO] 支持的格式: PDF, DOCX, TXT, MD")
        return True
    except Exception as e:
        print(f"[FAIL] 文档加载器测试失败: {e}")
        return False

def test_text_splitter():
    """测试文本分块器"""
    print("\n" + "=" * 60)
    print("测试文本分块器")
    print("=" * 60)
    
    try:
        from core import TextSplitter
        
        splitter = TextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            splitter_type="recursive"
        )
        print("[OK] TextSplitter 创建成功")
        
        # 测试分块
        text = "这是一段测试文本。" * 50
        chunks = splitter.split_text(text)
        print(f"[OK] 文本分块成功: {len(chunks)} 个块")
        return True
    except Exception as e:
        print(f"[FAIL] 文本分块器测试失败: {e}")
        return False

def test_embedding():
    """测试 Embedding 模块（需要模型下载）"""
    print("\n" + "=" * 60)
    print("测试 Embedding 模块")
    print("=" * 60)
    
    try:
        from core import get_recommended_models
        models = get_recommended_models()
        print("[OK] 获取推荐模型列表成功")
        print(f"[INFO] 推荐模型: {', '.join(models.keys())}")
        return True
    except Exception as e:
        print(f"[FAIL] Embedding 模块测试失败: {e}")
        return False

def test_env_config():
    """测试环境配置"""
    print("\n" + "=" * 60)
    print("测试环境配置")
    print("=" * 60)
    
    from pathlib import Path
    from dotenv import load_dotenv
    
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print("[OK] .env 文件加载成功")
        
        # 检查关键配置
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key and openai_key != "your_openai_api_key_here":
            print("[OK] OPENAI_API_KEY 已配置")
        else:
            print("[WARN] OPENAI_API_KEY 未配置或使用的是默认值")
            print("[INFO] 请编辑 .env 文件设置您的 API Key")
        
        return True
    else:
        print("[FAIL] .env 文件不存在")
        return False

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("RAG 知识库问答系统 - 功能测试")
    print("=" * 60)
    
    results = []
    
    # 测试依赖导入
    results.append(("依赖导入", test_imports()))
    
    # 测试核心模块
    results.append(("核心模块", test_core_modules()))
    
    # 测试文档加载器
    results.append(("文档加载器", test_document_loader()))
    
    # 测试文本分块器
    results.append(("文本分块器", test_text_splitter()))
    
    # 测试 Embedding
    results.append(("Embedding", test_embedding()))
    
    # 测试环境配置
    results.append(("环境配置", test_env_config()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n[SUCCESS] 所有测试通过！")
        print("\n可以运行项目:")
        print("  python main.py --mode web    # 启动 Web 界面")
        print("  python main.py --mode cli    # 启动命令行界面")
        return 0
    else:
        print("\n[FAILED] 部分测试失败")
        print("\n请检查:")
        print("1. 依赖是否完整安装: pip install -r requirements.txt")
        print("2. .env 文件是否正确配置")
        return 1

if __name__ == "__main__":
    sys.exit(main())
