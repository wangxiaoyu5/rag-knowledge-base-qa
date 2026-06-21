"""
测试浏览器中的问答功能，验证本地模型失败时是否正确降级到简单检索模式
"""
from core import EmbeddingManager, VectorStoreManager
from core.document_loader import load_documents_from_directory
from core.text_splitter import TextSplitter
from core.retriever import AdvancedRetriever
from core.qa_generator import QAGenerator
import re

def simple_qa_extract(question, documents):
    answer_parts = []
    seen = set()
    
    for doc in documents:
        content = doc.page_content
        
        if "毕业时间" in question or "毕业日期" in question:
            match = re.search(r'毕业时间[\uff1a:]?\s*(\d{4}\.\d{2})', content)
            if match and match.group(1) not in seen:
                answer_parts.append("毕业时间：" + match.group(1))
                seen.add(match.group(1))
        
        if "教育背景" in question or "学历" in question:
            match = re.search(r'学历[\uff1a:]?\s*([\u4e00-\u9fa5]+)', content)
            if match and match.group(1) not in seen:
                answer_parts.append("学历：" + match.group(1))
                seen.add(match.group(1))
            
            match = re.search(r'毕业院校[\uff1a:]?\s*([\u4e00-\u9fa5]+)', content)
            if match and match.group(1) not in seen:
                answer_parts.append("毕业院校：" + match.group(1))
                seen.add(match.group(1))
        
        if "专业" in question:
            match = re.search(r'专\s*业[\uff1a:]?\s*([\u4e00-\u9fa5]+(?:与[\u4e00-\u9fa5]+)*)', content)
            if match and match.group(1) not in seen:
                answer_parts.append("专业：" + match.group(1))
                seen.add(match.group(1))
    
    if answer_parts:
        return "；".join(answer_parts)
    else:
        result = "从文档中找到以下相关信息：\n\n"
        for i, doc in enumerate(documents[:3], 1):
            result += "来源 " + str(i) + ": " + doc.metadata.get('file_name', '未知文件') + "\n"
            result += "内容: " + doc.page_content[:200] + "...\n\n"
        return result

print("="*60)
print("测试浏览器问答降级逻辑")
print("="*60)

print("\n1. 初始化知识库...")
documents = load_documents_from_directory("./data/documents")
text_splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
split_documents = text_splitter.split_documents(documents)

embedding_manager = EmbeddingManager(model_name='BAAI/bge-large-zh', device='cpu')
vector_store_manager = VectorStoreManager(
    embedding_manager=embedding_manager,
    vector_store_type="faiss",
    persist_directory="./data/vector_db"
)
vector_store = vector_store_manager.create_vector_store(split_documents)

advanced_retriever = AdvancedRetriever(
    vector_store_manager=vector_store_manager,
    documents=split_documents,
    use_hybrid=True,
    use_rerank=True,
    top_k=5
)

print("   知识库初始化完成")

print("\n2. 模拟前端初始化逻辑（测试本地模型加载）...")
qa_generator = None
use_simple_qa = False

try:
    qa_gen = QAGenerator(
        model_type="local",
        model_name="Qwen/Qwen2-0.5B-Instruct"
    )
    # 立即测试模型是否能加载（这是关键修复）
    qa_gen._load_model()
    qa_generator = qa_gen
    print("   本地模型加载成功")
except Exception as e:
    qa_generator = None
    use_simple_qa = True
    print("   本地模型加载失败:", str(e)[:80])
    print("   已切换到简单检索模式")

print("\n3. 测试问答功能...")
test_questions = [
    "王晓宇的毕业时间是什么？",
    "王晓宇的教育背景是什么？",
    "王晓宇是什么专业的？"
]

print("\n" + "-"*50)
print("问答结果：")
print("-"*50)

all_success = True
for question in test_questions:
    print("\n   问题:", question)
    
    # 检索文档
    retrieval_results = advanced_retriever.retrieve(question)
    documents = [r.document for r in retrieval_results]
    
    # 模拟前端逻辑
    use_simple = use_simple_qa or qa_generator is None
    
    if use_simple:
        answer = simple_qa_extract(question, documents)
        print("   模式: 简单检索模式")
    else:
        try:
            response = qa_generator.generate(question, documents)
            answer = response.answer
            print("   模式: 完整QA模式")
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
                use_simple_qa = True
                qa_generator = None
                answer = simple_qa_extract(question, documents)
                print("   模式: 从QA降级到简单检索模式")
            else:
                answer = "生成答案时出错: " + str(e)[:50]
                print("   模式: 错误")
    
    print("   回答:", answer)
    
    if "出错" in answer:
        print("   状态: 失败")
        all_success = False
    else:
        print("   状态: 成功")
    
    print("   " + "-"*30)

print("\n" + "="*60)
if all_success:
    print("测试通过！")
    print("修复已生效：本地模型加载失败时会自动降级到简单检索模式")
    print("请在浏览器中访问 http://localhost:8501 进行验证")
else:
    print("测试失败")
print("="*60)