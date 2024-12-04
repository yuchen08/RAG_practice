from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from functools import wraps
import os
import json
import logging
import chromadb
from chromadb.config import Settings

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定義中文語義分割符號
chinese_separators = [
    "\n\n",  # 段落分隔
    "。",    # 句號
    "！",    # 感嘆號
    "？",    # 問號
    "；",    # 分號
    "：",    # 冒號
    "\n",    # 換行
    "，",    # 逗號
    "、",    # 頓號
]

# 錯誤處理裝飾器
def handle_exceptions(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    return wrapper

# 基礎類和工具函數
class ChatRequest(BaseModel):
    message: str
    use_rag: bool
    chat_history: list = []

class KnowledgeBase(BaseModel):
    title: str
    content: str

def initialize_vectorstore():
    """初始化向量數據庫"""
    # 禁用 Chroma 遙測
    client_settings = Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
    
    if not os.path.exists("vectorstore"):
        os.makedirs("vectorstore")
        
    return Chroma(
        persist_directory="vectorstore", 
        embedding_function=embeddings,
        client_settings=client_settings
    )

def load_knowledge_base(path="knowledge_base.json"):
    """載入知識庫"""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"documents": []}

def save_knowledge_base(data, path="knowledge_base.json"):
    """保存知識庫"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 初始化全局變量
llm = Ollama(base_url="http://localhost:11434", model="llama3.2:latest")
embeddings = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 創建全局文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    separators=chinese_separators,
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
    keep_separator=True
)

# 初始化向量數據庫
vectorstore = initialize_vectorstore()

def process_document(title, content):
    """處理文檔"""
    text = f"標題：{title}\n內容：{content}"
    try:
        chunks = text_splitter.split_text(text)
        logger.info(f"Document split into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting text: {str(e)}")
        return [text]

@app.post("/chat")
@handle_exceptions
async def chat(request: ChatRequest):
    try:
        if request.use_rag:
            print(f"Searching for: {request.message}")
            
            # 獲取知識庫總文檔數
            try:
                total_docs = vectorstore._collection.count()
                print(f"Total documents in knowledge base: {total_docs}")
                
                # 動態計算檢索數量
                k = total_docs  # 直接使用總文檔數
                fetch_k = min(total_docs, k + 4)  # 確保 fetch_k 不超過總文檔數
                
                print(f"Using k={k}, fetch_k={fetch_k}")
                
                retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": k,  # 動態設置返回文檔數量
                        "fetch_k": fetch_k,  # 動態設置初始檢索數量
                        "lambda_mult": 0.9  # 相關性權重
                    }
                )
            except Exception as e:
                print(f"Error getting document count, using default values: {str(e)}")
                # 使用預設值
                retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 5,
                        "fetch_k": 8,
                        "lambda_mult": 0.9
                    }
                )
            
            # 獲取相關文檔
            try:
                # 使用 invoke 替代 get_relevant_documents
                retriever_output = retriever.invoke(request.message)
                relevant_docs = retriever_output if isinstance(retriever_output, list) else []
                print(f"Found {len(relevant_docs)} relevant documents")
            except Exception as e:
                print(f"Error retrieving documents: {str(e)}")
                relevant_docs = []
            
            # 改進文檔相關性檢查
            has_relevant_content = False
            matched_docs = []
            if relevant_docs:
                question = request.message.lower()
                # 擴充停用詞列表
                stop_words = {'的', '是', '在', '有', '和', '與', '了', '嗎', '呢', '吧', '啊', '會', '能', 
                            '什麼', '如何', '為什麼', '請問', '告訴', '說明', '想', '知道', '應該', '甚麼'}
                
                # 將問題轉換為字符列表，同時保留完整問題
                question_chars = list(question)  # 將問題拆分成單個字符
                question_words = set()
                
                # 添加完整問題
                question_words.add(question)
                
                # 添加連續的2-3個字的組合
                for i in range(len(question_chars)):
                    # 添加單個字（非停用詞）
                    if question_chars[i] not in stop_words:
                        question_words.add(question_chars[i])
                    
                    # 添加雙字組合
                    if i < len(question_chars) - 1:
                        word_2 = question_chars[i] + question_chars[i+1]
                        question_words.add(word_2)
                    
                    # 添加三字組合
                    if i < len(question_chars) - 2:
                        word_3 = question_chars[i] + question_chars[i+1] + question_chars[i+2]
                        question_words.add(word_3)
                
                print(f"Question keywords: {question_words}")
                
                # 改進相關性檢查部分
                for doc in relevant_docs:
                    doc_content = doc.page_content.lower()
                    
                    # 分別提取標題和內容
                    title_part = ""
                    content_part = ""
                    if "標題：" in doc_content and "內容：" in doc_content:
                        parts = doc_content.split("內容：")
                        title_part = parts[0].replace("標題：", "").strip()
                        content_part = parts[1].strip()
                    
                    # 分別計算標題和內容的匹配分數
                    title_words = []
                    content_words = []
                    
                    # 檢查每個關鍵詞
                    for word in question_words:
                        # 檢查標題
                        if word in title_part:
                            title_words.append(word)
                        # 檢查內容
                        if word in content_part:
                            content_words.append(word)
                    
                    # 計算匹配分數，考慮詞長度
                    match_score = 0
                    if title_words:
                        # 計算標題匹配分數，較長的詞給予更高權重
                        title_score = sum(len(word) for word in title_words) / (len(question) * 1.2)
                        match_score = max(match_score, title_score)
                    
                    if content_words:
                        # 計算內容匹配分數
                        content_score = sum(len(word) for word in content_words) / len(question)
                        match_score = max(match_score, content_score)
                    
                    match_score = min(1.0, match_score)  # 確保不超過1.0
                    
                    print(f"Document title: {title_part}")
                    print(f"Title matched words: {title_words}")
                    print(f"Content matched words: {content_words}")
                    print(f"Match score: {match_score * 100}%")
                    
                    # 調整匹配門檻
                    if match_score >= 0.3:  # 30% 相關度門檻
                        has_relevant_content = True
                        matched_docs.append({
                            "content": doc.page_content,
                            "score": match_score,
                            "matched_words": title_words + content_words
                        })
            
            if has_relevant_content:
                # 整合所有相關文檔的內容
                context = "\n\n".join([doc["content"] for doc in matched_docs])
                
                # 創建更強的提示模板
                prompt_template = """請使用以下提供的參考資料來回答問題。

參考資料內容：
{context}

用戶問題：{question}

請遵循以下規則回答：
1. 必須基於參考資料的內容來優先構建回答，如果參考資料中沒有相關信息，則使用LLM進行回答
2. 使用參考資料中的具體細節、數據和例子
3. 以自然且流暢的方式整合這些信息
4. 回答要有邏輯性和層次感
5. 如果需要補充參考資料以外的信息，請明確標註「補充說明」

回答格式建議：
1. 先直接回答核心問題
2. 然後提供具體細節和例子
3. 最後可以補充相關建議或額外信息

請開始回答："""

                PROMPT = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )
                
                # 使用 LLMChain 而不是 ConversationalRetrievalChain
                from langchain.chains import LLMChain
                
                chain = LLMChain(
                    llm=llm,
                    prompt=PROMPT
                )
                
                # 生成回答
                response = chain.invoke({
                    "context": context,
                    "question": request.message
                })
                
                return {
                    "response": response["text"],
                    "retrieved_docs": matched_docs,
                    "source": "rag"
                }
            else:
                print("No relevant content found, using pure LLM")
                response = llm.invoke(request.message)
                return {
                    "response": response,
                    "source": "llm"
                }
        else:
            response = llm.invoke(request.message)
            return {
                "response": response,
                "source": "llm"
            }
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return {"error": str(e)}

@app.post("/add_knowledge")
@handle_exceptions
async def add_knowledge(knowledge: KnowledgeBase):
    title = knowledge.title.strip()
    content = knowledge.content.strip()
    
    if not title or not content:
        raise HTTPException(status_code=400, detail="標題和內容不能為空")
    
    # 載入知識庫
    knowledge_base = load_knowledge_base()
    
    # 添加新文檔
    new_doc = {"title": title, "content": content}
    knowledge_base["documents"].append(new_doc)
    
    # 保存知識庫
    save_knowledge_base(knowledge_base)
    
    # 添加到向量數據庫
    chunks = process_document(title, content)
    vectorstore.add_texts(chunks)
    logger.info(f"Added new document: {title}")
    
    # 獲取統計信息
    collection_stats = vectorstore._collection.count()
    
    return {
        "message": "知識庫更新成功",
        "total_documents": len(knowledge_base["documents"]),
        "vector_store_count": collection_stats,
        "added_document": title
    }

@app.on_event("startup")
async def startup_event():
    global vectorstore
    logger.info("Starting initialization of vector database...")
    
    # 初始化向量數據庫
    vectorstore = initialize_vectorstore()
    
    # 載入知識庫
    knowledge_base = load_knowledge_base()
    if knowledge_base["documents"]:
        # 清空現有向量數據庫
        try:
            all_ids = vectorstore._collection.get()['ids']
            if all_ids:
                vectorstore._collection.delete(ids=all_ids)
                logger.info("Cleared existing vector database")
        except Exception as e:
            logger.warning(f"Could not clean existing documents: {str(e)}")
        
        # 處理所有文檔
        for doc in knowledge_base["documents"]:
            chunks = process_document(doc['title'], doc['content'])
            vectorstore.add_texts(chunks)
            logger.info(f"Added document: {doc['title']}")
        
        logger.info("Vector database initialization completed")

@app.get("/knowledge")
@handle_exceptions
async def get_knowledge():
    """獲取所有知識庫內容"""
    knowledge_base = load_knowledge_base()
    return knowledge_base["documents"]

@app.put("/knowledge/{doc_id}")
@handle_exceptions
async def update_knowledge(doc_id: int, knowledge: KnowledgeBase):
    """更新知識庫中的特定文檔"""
    knowledge_base = load_knowledge_base()
    
    if doc_id < 0 or doc_id >= len(knowledge_base["documents"]):
        raise HTTPException(status_code=404, detail="文檔不存在")
    
    # 更新文檔
    knowledge_base["documents"][doc_id] = {
        "title": knowledge.title.strip(),
        "content": knowledge.content.strip()
    }
    
    # 保存更新
    save_knowledge_base(knowledge_base)
    
    # 重新初始化向量數據庫
    await startup_event()
    
    return {"message": "文檔更新成功"}

@app.delete("/knowledge/{doc_id}")
@handle_exceptions
async def delete_knowledge(doc_id: int):
    """刪除知識庫中的特定文檔"""
    knowledge_base = load_knowledge_base()
    
    if doc_id < 0 or doc_id >= len(knowledge_base["documents"]):
        raise HTTPException(status_code=404, detail="文檔不存在")
    
    # 刪除文檔
    knowledge_base["documents"].pop(doc_id)
    
    # 保存更新
    save_knowledge_base(knowledge_base)
    
    # 重新初始化向量數據庫
    await startup_event()
    
    return {"message": "文檔刪除成功"}