from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
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
llm = Ollama(base_url="http://localhost:11434", model="llama2:13b")
embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="llama2:13b")

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
            
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 3
                }
            )
            
            # 獲取相關文檔
            relevant_docs = retriever.get_relevant_documents(request.message)
            print(f"Found {len(relevant_docs)} relevant documents")
            
            # 檢查文檔相關性
            has_relevant_content = False
            if relevant_docs:
                # 將問題轉換為字符串進行比對
                question = request.message.lower()
                
                for doc in relevant_docs:
                    doc_content = doc.page_content.lower()
                    print(f"Checking document: {doc_content[:200]}")
                    has_relevant_content = True
                    break
            
            if has_relevant_content:
                # 格式化聊天歷史
                formatted_history = []
                if request.chat_history:
                    for entry in request.chat_history:
                        if isinstance(entry, dict) and "human" in entry and "ai" in entry:
                            formatted_history.append((entry["human"], entry["ai"]))
                
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    return_source_documents=True,
                    verbose=True
                )
                
                chain_response = qa_chain({
                    "question": request.message,
                    "chat_history": formatted_history
                })
                
                answer = chain_response.get("answer", "無法獲取回答")
                source_docs = [doc.page_content for doc in relevant_docs if hasattr(doc, 'page_content')]
                
                return {
                    "response": answer,
                    "retrieved_docs": source_docs,
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
                    chunk_size=500,  # 減小chunk大小
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