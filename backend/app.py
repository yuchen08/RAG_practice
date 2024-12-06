from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from functools import wraps, lru_cache
import os
import json
import logging
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
    """載入knowledge_base.json"""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"documents": []}

def save_knowledge_base(data, path="knowledge_base.json"):
    """儲存knowledge_base.json"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 初始化模型
llm = Ollama(base_url="http://localhost:11434", model="llama3.2:latest")
embeddings = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 創建文件分割功能
text_splitter = RecursiveCharacterTextSplitter(
    separators=chinese_separators,
    chunk_size=500,# 每個chunk的大小
    chunk_overlap=50,# 每個chunk的重疊部分
    length_function=len,# 計算chunk長度的函數
    is_separator_regex=False,# 是否使用正則表達式
    keep_separator=True # 保留分隔符
)

# 初始化向量數據庫
vectorstore = initialize_vectorstore()

def process_document(title, content):
    """處理文件"""
    text = f"標題：{title}\n內容：{content}"
    try:
        chunks = text_splitter.split_text(text)
        logger.info(f"文件分割成 {len(chunks)} 個片段")
        return chunks
    except Exception as e:
        logger.error(f"分割文本錯誤：{str(e)}")
        return [text]

# 聊天功能
@app.post("/chat")
# 錯誤處理裝飾器
@handle_exceptions

async def chat(request: ChatRequest):
    try:
        if request.use_rag:
            print(f"搜索：{request.message}")
            
            # 獲取knowledge_base.json總文件數
            try:
                total_docs = vectorstore._collection.count()
                print(f"總文件數：{total_docs}")
                
                retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "lambda_mult": 0.9  # 相關性權重
                    }
                )
            except Exception as e:
                print(f"獲取文件數錯誤：{str(e)}")
                # 使用預設值
                retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "lambda_mult": 0.9
                    }
                )
            
            # 獲取相關文件
            try:
                retriever_output = retriever.invoke(request.message)
                relevant_docs = retriever_output if isinstance(retriever_output, list) else []
                print(f"找到 {len(relevant_docs)} 個相關文件")
            except Exception as e:
                print(f"獲取文件錯誤：{str(e)}")
                relevant_docs = []
            
            # 改進文件相關性檢查
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
                
                # 只添加雙字和三字組合，跳過單字
                for i in range(len(question_chars)):
                    # 添加雙字組合
                    if i < len(question_chars) - 1:
                        word_2 = question_chars[i] + question_chars[i+1]
                        question_words.add(word_2)
                    
                    # 添加三字組合
                    if i < len(question_chars) - 2:
                        word_3 = question_chars[i] + question_chars[i+1] + question_chars[i+2]
                        question_words.add(word_3)
                
                print(f"問題關鍵詞：{question_words}")
                
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
                    
                    # 分別計算標題和內容的相似分數
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
                    
                    # 計算相似分數，考慮詞長度
                    match_score = 0
                    if title_words:
                        # 計算標題相似分數，較長的詞給予更高權重
                        title_score = sum(len(word) for word in title_words) / (len(question) * 1.2)
                        match_score = max(match_score, title_score)
                    
                    if content_words:
                        # 計算內容相似分數
                        content_score = sum(len(word) for word in content_words) / len(question)
                        match_score = max(match_score, content_score)
                    
                    match_score = min(1.0, match_score)  # 確保不超過1.0
                    
                    print(f"文件標題：{title_part}")
                    print(f"標題匹配詞：{title_words}")
                    print(f"內容匹配詞：{content_words}")
                    print(f"相似分數：{match_score * 100}%")
                    
                    # 調整門檻
                    if match_score >= 0.3:  # 30% 相關度門檻
                        has_relevant_content = True
                        matched_docs.append({
                            "content": doc.page_content,
                            "score": match_score,
                            "matched_words": title_words + content_words
                        })
            
            if has_relevant_content:
                # 整合所有相關文件的內容
                context = "\n\n".join([doc["content"] for doc in matched_docs])
                
                # 創建提示模板
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
                
                # 創建鏈接
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
                print("未找到相關內容，使用LLM模式")
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
        print(f"聊天功能錯誤：{str(e)}")
        return {"error": str(e)}

@app.post("/add_knowledge")
@handle_exceptions
async def add_knowledge(knowledge: KnowledgeBase):
    title = knowledge.title.strip()
    content = knowledge.content.strip()
    
    if not title or not content:
        raise HTTPException(status_code=400, detail="標題和內容不能為空")
    
    # 載入knowledge_base.json
    knowledge_base = load_knowledge_base()
    
    # 添加新文件
    new_doc = {"title": title, "content": content}
    knowledge_base["documents"].append(new_doc)
    
    # 保存knowledge_base.json
    save_knowledge_base(knowledge_base)
    
    # 添加到向量數據庫
    chunks = process_document(title, content)
    vectorstore.add_texts(chunks)
    logger.info(f"添加文件：{title}")
    
    # 獲取統計信息
    collection_stats = vectorstore._collection.count()
    
    return {
        "message": "資料庫更新成功",
        "total_documents": len(knowledge_base["documents"]),
        "vector_store_count": collection_stats,
        "added_document": title
    }

@app.on_event("startup")
async def startup_event():
    global vectorstore
    logger.info("開始初始化向量數據庫...")
    
    # 初始化向量數據庫
    vectorstore = initialize_vectorstore()
    
    # 載入knowledge_base.json
    knowledge_base = load_knowledge_base()
    if knowledge_base["documents"]:
        # 清空現有向量數據庫
        try:
            all_ids = vectorstore._collection.get()['ids']
            if all_ids:
                vectorstore._collection.delete(ids=all_ids)
                logger.info("已清除現有向量數據庫")
        except Exception as e:
            logger.warning(f"無法清除現有文件：{str(e)}")
        
        # 處理所有文件
        for doc in knowledge_base["documents"]:
            chunks = process_document(doc['title'], doc['content'])
            vectorstore.add_texts(chunks)
            logger.info(f"添加文件：{doc['title']}")
        
        logger.info("向量數據庫初始化完成")

@app.get("/knowledge")
@handle_exceptions
async def get_knowledge():
    """獲取所有knowledge_base.json內容"""
    knowledge_base = load_knowledge_base()
    return knowledge_base["documents"]

@app.put("/knowledge/{doc_id}")
@handle_exceptions
async def update_knowledge(doc_id: int, knowledge: KnowledgeBase):
    """更新knowledge_base.json中的特定文件"""
    knowledge_base = load_knowledge_base()
    
    if doc_id < 0 or doc_id >= len(knowledge_base["documents"]):
        raise HTTPException(status_code=404, detail="文件不存在")
    
    # 更新文件
    knowledge_base["documents"][doc_id] = {
        "title": knowledge.title.strip(),
        "content": knowledge.content.strip()
    }
    
    # 保存更新
    save_knowledge_base(knowledge_base)
    
    # 只更新這個文件的向量
    try:
        # 清除舊的向量
        old_ids = vectorstore._collection.get()['ids']
        if old_ids and doc_id < len(old_ids):
            vectorstore._collection.delete(ids=[old_ids[doc_id]])
        
        # 添加新的向量
        chunks = process_document(knowledge.title, knowledge.content)
        vectorstore.add_texts(chunks)
    except Exception as e:
        logger.error(f"更新向量數據庫錯誤：{str(e)}")
    
    return {"message": "文件更新成功"}

@app.delete("/knowledge/{doc_id}")
@handle_exceptions
async def delete_knowledge(doc_id: int):
    """刪除knowledge_base.json中的特定文件"""
    knowledge_base = load_knowledge_base()
    
    if doc_id < 0 or doc_id >= len(knowledge_base["documents"]):
        raise HTTPException(status_code=404, detail="文件不存在")
    
    # 刪除文件
    knowledge_base["documents"].pop(doc_id)
    
    # 保存更新
    save_knowledge_base(knowledge_base)
    
    # 重新初始化向量數據庫
    await startup_event()
    
    return {"message": "文件刪除成功"}

@lru_cache(maxsize=100)
def get_document_vectors(doc_content: str):
    """緩存文件向量"""
    return embeddings.embed_query(doc_content)