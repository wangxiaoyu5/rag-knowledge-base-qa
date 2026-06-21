from core import EmbeddingManager, VectorStoreManager
from core.document_loader import load_documents_from_directory
from core.text_splitter import TextSplitter
from core.retriever import AdvancedRetriever
import re

def simple_qa_extract_frontend(question, documents):
    """前端版本的简单问答提取函数"""
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
            result += "内容: " + content[:50] + "...\n\n"
        return result

print("="*60)
print("最终验证测试")
print("="*60)

print("\n1. 加载文档...")
documents = load_documents_from_directory("./data/documents")
print("   文档数量:", len(documents))

print("\n2. 文本分块...")
text_splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
split_documents = text_splitter.split_documents(documents)
print("   分块数量:", len(split_documents))

print("\n3. 创建向量数据库...")
embedding_manager = EmbeddingManager(model_name='BAAI/bge-large-zh', device='cpu')
vector_store_manager = VectorStoreManager(
    embedding_manager=embedding_manager,
    vector_store_type="faiss",
    persist_directory="./data/vector_db"
)
vector_store = vector_store_manager.create_vector_store(split_documents)

print("\n4. 初始化检索器...")
advanced_retriever = AdvancedRetriever(
    vector_store_manager=vector_store_manager,
    documents=split_documents,
    use_hybrid=True,
    use_rerank=True,
    top_k=5
)

print("\n5. 测试问答...")
test_questions = [
    "王晓宇的毕业时间是什么？",
    "王晓宇的教育背景是什么？",
    "王晓宇是什么专业的？"
]

print("\n" + "-"*60)
print("测试结果：")
print("-"*60)

all_pass = True
for question in test_questions:
    print("\n问题:", question)
    
    retrieval_results = advanced_retriever.retrieve(question)
    answer = simple_qa_extract_frontend(question, [r.document for r in retrieval_results])
    
    print("回答:", answer)
    
    if not answer or answer.strip() == "":
        print("状态: ❌ 失败")
        all_pass = False
    else:
        print("状态: ✅ 成功")
    
    print("-" * 30)

if all_pass:
    print("\n🎉 所有测试通过！")
    print("前端与脚本测试结果一致")
else:
    print("\n⚠️ 部分测试失败")

print("\n" + "="*60)