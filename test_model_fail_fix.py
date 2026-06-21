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
print("测试本地模型失败时的降级策略")
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
print("   向量数据库创建成功")

print("\n4. 初始化检索器...")
advanced_retriever = AdvancedRetriever(
    vector_store_manager=vector_store_manager,
    documents=split_documents,
    use_hybrid=True,
    use_rerank=True,
    top_k=5
)
print("   检索器初始化成功")

print("\n5. 模拟前端逻辑测试...")
print("\n   场景1: 尝试初始化本地模型（预计失败）")
qa_generator = None
use_simple_qa = False

try:
    qa_generator = QAGenerator(
        model_type="local",
        model_name="Qwen/Qwen2-0.5B-Instruct"
    )
    print("   OK: 本地模型加载成功")
except Exception as e:
    use_simple_qa = True
    qa_generator = None
    print("   FAIL: 本地模型加载失败:", str(e)[:80])
    print("   OK: 已切换到简单检索模式")

print("\n   场景2: 使用问答功能")
test_questions = [
    "王晓宇的毕业时间是什么？",
    "王晓宇的教育背景是什么？",
    "王晓宇是什么专业的？"
]

print("\n" + "-"*50)
print("问答测试结果：")
print("-"*50)

for question in test_questions:
    print("\n   问题:", question)
    
    retrieval_results = advanced_retriever.retrieve(question)
    documents = [r.document for r in retrieval_results]
    
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
            if "Can't load the model" in str(e):
                use_simple_qa = True
                qa_generator = None
                answer = simple_qa_extract(question, documents)
                print("   模式: 从完整QA降级到简单检索模式")
            else:
                answer = "生成答案时出错: " + str(e)[:50]
    
    print("   回答:", answer)
    print("   " + "-"*30)

print("\n" + "="*60)
print("测试完成！")
print("="*60)