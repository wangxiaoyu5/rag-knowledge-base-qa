"""
项目结构测试脚本 - 不依赖外部库，仅检查项目结构
"""
import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """检查文件是否存在"""
    if Path(filepath).exists():
        print(f"[OK] {description}: {filepath}")
        return True
    else:
        print(f"[MISSING] {description} 缺失: {filepath}")
        return False

def check_directory_exists(dirpath, description):
    """检查目录是否存在"""
    if Path(dirpath).exists() and Path(dirpath).is_dir():
        print(f"[OK] {description}: {dirpath}")
        return True
    else:
        print(f"[MISSING] {description} 缺失: {dirpath}")
        return False

def main():
    print("=" * 60)
    print("RAG 知识库问答系统 - 项目结构检查")
    print("=" * 60)
    
    base_path = Path(__file__).parent
    
    # 检查核心目录
    print("\n【1. 目录结构检查】")
    dirs_ok = True
    dirs_ok &= check_directory_exists(base_path / "core", "核心模块目录")
    dirs_ok &= check_directory_exists(base_path / "frontend", "前端目录")
    dirs_ok &= check_directory_exists(base_path / "docs", "文档目录")
    
    # 检查核心文件
    print("\n【2. 核心文件检查】")
    files_ok = True
    files_ok &= check_file_exists(base_path / "main.py", "主程序入口")
    files_ok &= check_file_exists(base_path / "requirements.txt", "依赖文件")
    files_ok &= check_file_exists(base_path / ".env.example", "环境变量模板")
    files_ok &= check_file_exists(base_path / "config.yaml", "配置文件")
    files_ok &= check_file_exists(base_path / "README.md", "项目说明")
    
    # 检查 core 模块
    print("\n【3. Core 模块检查】")
    core_ok = True
    core_ok &= check_file_exists(base_path / "core" / "__init__.py", "Core 初始化")
    core_ok &= check_file_exists(base_path / "core" / "document_loader.py", "文档加载器")
    core_ok &= check_file_exists(base_path / "core" / "text_splitter.py", "文本分块器")
    core_ok &= check_file_exists(base_path / "core" / "embedding.py", "Embedding模块")
    core_ok &= check_file_exists(base_path / "core" / "retriever.py", "检索模块")
    core_ok &= check_file_exists(base_path / "core" / "qa_generator.py", "问答生成模块")
    
    # 检查 frontend
    print("\n【4. Frontend 模块检查】")
    frontend_ok = True
    frontend_ok &= check_file_exists(base_path / "frontend" / "app.py", "Streamlit 应用")
    
    # 检查 docs
    print("\n【5. 文档检查】")
    docs_ok = True
    docs_ok &= check_file_exists(base_path / "docs" / "01-系统架构设计.md", "架构文档")
    docs_ok &= check_file_exists(base_path / "docs" / "02-核心模块详解.md", "模块文档")
    docs_ok &= check_file_exists(base_path / "docs" / "03-API接口文档.md", "API文档")
    docs_ok &= check_file_exists(base_path / "docs" / "04-部署运维文档.md", "部署文档")
    docs_ok &= check_file_exists(base_path / "docs" / "05-面试准备指南.md", "面试指南")
    
    # 检查数据目录
    print("\n【6. 数据目录检查】")
    data_dir = base_path / "data"
    vector_db_dir = data_dir / "vector_db"
    documents_dir = data_dir / "documents"
    
    if not data_dir.exists():
        print(f"[INFO] 数据目录不存在，创建中...")
        data_dir.mkdir(exist_ok=True)
        vector_db_dir.mkdir(exist_ok=True)
        documents_dir.mkdir(exist_ok=True)
        print(f"[OK] 已创建数据目录: {data_dir}")
    else:
        print(f"[OK] 数据目录已存在: {data_dir}")
    
    # 检查 .env 文件
    print("\n【7. 环境配置检查】")
    env_file = base_path / ".env"
    env_example = base_path / ".env.example"
    
    if not env_file.exists() and env_example.exists():
        print(f"[INFO] .env 文件不存在，从模板创建...")
        with open(env_example, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[OK] 已创建 .env 文件，请编辑配置您的 API Key")
    elif env_file.exists():
        print(f"[OK] .env 文件已存在")
    
    # 总结
    print("\n" + "=" * 60)
    print("检查结果汇总")
    print("=" * 60)
    
    all_ok = dirs_ok and files_ok and core_ok and frontend_ok and docs_ok
    
    if all_ok:
        print("[SUCCESS] 项目结构完整！")
        print("\n下一步操作:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 配置环境: 编辑 .env 文件设置 OPENAI_API_KEY")
        print("3. 准备数据: 将文档放入 data/documents/ 目录")
        print("4. 运行项目: python main.py --mode web")
        return 0
    else:
        print("[FAILED] 项目结构不完整，请检查缺失的文件")
        return 1

if __name__ == "__main__":
    sys.exit(main())
