from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
import os
import json

app = FastAPI()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化Ollama模型
llm = Ollama(base_url="http://localhost:11434", model="llama3.2:latest")
embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="llama3.2:latest")

# 初始化向量數據庫
if not os.path.exists("vectorstore"):
    os.makedirs("vectorstore")
    
vectorstore = Chroma(persist_directory="vectorstore", embedding_function=embeddings)

class ChatRequest(BaseModel):
    message: str
    use_rag: bool
    chat_history: list = []

class KnowledgeBase(BaseModel):
    title: str
    content: str

@app.post("/chat")
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
                            '什麼', '如何', '為什麼', '請問', '告訴', '說明', '想', '知道', '應該'}
                
                # 提取問題中的關鍵詞
                question_words = set(word for word in question.split() 
                                  if word not in stop_words and len(word) >= 2)
                
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
                    
                    # 檢查完整問題是否包含在標題或內容中
                    full_question = request.message.lower()
                    
                    # 分別計算標題和內容的匹配分數
                    title_words = []
                    content_words = []
                    
                    # 先檢查完整匹配
                    if full_question in title_part or full_question in content_part:
                        match_score = 1.0  # 完全匹配給予100%
                        title_words = [full_question] if full_question in title_part else []
                        content_words = [full_question] if full_question in content_part else []
                    else:
                        # 關鍵詞匹配
                        for word in question_words:
                            # 檢查每個關鍵詞是否出現在標題或內容中
                            if word in title_part:
                                title_words.append(word)
                            if word in content_part:
                                content_words.append(word)
                        
                        # 合併匹配的詞並去重
                        matched_words = list(set(title_words + content_words))
                        
                        # 計算基礎分數
                        match_score = len(matched_words) / len(question_words) if question_words else 0
                        
                        # 如果標題有匹配，給予額外加權，但確保不超過1.0
                        if title_words:
                            title_score = len(title_words) / len(question_words) * 1.2
                            match_score = min(1.0, max(match_score, title_score))
                    
                    # 打印調試信息
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
async def add_knowledge(knowledge: KnowledgeBase):
    try:
        # 清理和格式化輸入
        title = knowledge.title.strip()
        content = knowledge.content.strip()
        
        # 檢查標題和內容是否為空
        if not title or not content:
            return {"error": "標題和內容不能為空"}
        
        # 讀取現有的 JSON 文件
        knowledge_path = "knowledge_base.json"
        if os.path.exists(knowledge_path):
            with open(knowledge_path, "r", encoding="utf-8") as f:
                knowledge_base = json.load(f)
        else:
            knowledge_base = {"documents": []}
        
        # 添加新的文檔
        new_doc = {
            "title": title,
            "content": content
        }
        knowledge_base["documents"].append(new_doc)
        
        # 保存更新後的 JSON 文件
        with open(knowledge_path, "w", encoding="utf-8") as f:
            json.dump(knowledge_base, f, ensure_ascii=False, indent=4)
        
        # 只將新文檔添加到向量數據庫
        try:
            # 創建新文檔的文本
            text = f"標題：{title}\n內容：{content}"
            
            # 分割文本
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            texts = text_splitter.split_text(text)
            
            # 添加到向量數據庫
            vectorstore.add_texts(texts)
            print(f"Added new document: {title}")
            
            # 獲取更新後的統計信息
            collection_stats = vectorstore._collection.count()
            print(f"Total documents in vector store: {collection_stats}")
            
            return {
                "message": "知識庫更新成功",
                "total_documents": len(knowledge_base["documents"]),
                "vector_store_count": collection_stats,
                "added_document": title
            }
            
        except Exception as e:
            print(f"Error adding document to vector database: {str(e)}")
            return {"error": f"知識庫文件已更新，但向量數據庫更新失敗: {str(e)}"}
            
    except Exception as e:
        print(f"Error in add_knowledge endpoint: {str(e)}")
        return {"error": str(e)}

# 在應用啟動時初始化向量數據庫
@app.on_event("startup")
async def startup_event():
    try:
        global vectorstore
        
        # 確保向量數據庫目錄存在
        if not os.path.exists("vectorstore"):
            os.makedirs("vectorstore")
        
        # 初始化向量數據庫
        vectorstore = Chroma(persist_directory="vectorstore", embedding_function=embeddings)
        
        # 讀取JSON知識庫
        if os.path.exists("knowledge_base.json"):
            with open("knowledge_base.json", "r", encoding="utf-8") as f:
                knowledge_base = json.load(f)
            
            # 清空現有的向量數據庫
            try:
                all_ids = vectorstore._collection.get()['ids']
                if all_ids:
                    vectorstore._collection.delete(ids=all_ids)
            except Exception as e:
                print(f"Warning: Could not clean existing documents: {str(e)}")
            
            # 處理每個文檔
            for doc in knowledge_base["documents"]:
                # 為每個文檔單獨創建文本並添加到向量數據庫
                text = f"標題：{doc['title']}\n內容：{doc['content']}"
                
                # 分割文本
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,  # 減chunk大小
                    chunk_overlap=50
                )
                texts = text_splitter.split_text(text)
                
                # 添加到向量數據庫
                vectorstore.add_texts(texts)
                print(f"Added document: {doc['title']}")
            
            print(f"Vector database initialized successfully")
            
            # 驗證數據庫內容
            try:
                collection_stats = vectorstore._collection.count()
                print(f"Total documents in vector store: {collection_stats}")
            except Exception as e:
                print(f"Error checking vector store stats: {str(e)}")
        
    except Exception as e:
        print(f"Error initializing vector database: {str(e)}")
        if 'vectorstore' not in globals():
            vectorstore = Chroma(persist_directory="vectorstore", embedding_function=embeddings)