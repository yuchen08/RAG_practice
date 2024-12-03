import streamlit as st
import requests
import json

st.title("LLM + RAG 聊天機器人")

# 初始化會話狀態
if "messages" not in st.session_state:
    st.session_state.messages = []

if "use_rag" not in st.session_state:
    st.session_state.use_rag = False

if "previous_rag_state" not in st.session_state:
    st.session_state.previous_rag_state = False

# RAG開關
current_rag_state = st.toggle("啟用 RAG", st.session_state.use_rag)

# 如果 RAG 狀態改變，清空聊天歷史
if current_rag_state != st.session_state.previous_rag_state:
    st.session_state.messages = []
    st.session_state.previous_rag_state = current_rag_state

st.session_state.use_rag = current_rag_state

# 知識庫添加區域
with st.expander("新增知識庫內容"):
    st.markdown("""
    ### 知識庫添加說明
    請輸入您要添加的知識內容，系統會自動處理格式。
    
    您可以使用：
    - 條列項目（數字或破折號）
    - 具體數據說明
    - 詳細文字描述
    """)
    
    knowledge_title = st.text_input(
        "標題",
        placeholder="例如：台灣科技產業概況",
        help="請輸入簡短且具體的主題描述"
    )
    
    knowledge_content = st.text_area(
        "內容",
        height=200,
        placeholder="""請輸入詳細內容，例如：

台灣是全球半導體產業的重要基地，台積電(TSMC)作為全球最大的晶圓代工廠...

可以使用條列格式：
1. 第一點說明
2. 第二點說明

或是使用破折號：
- 重點一：具體說明
- 重點二：具體說明""",
        help="可使用條列項目或段落描述，建議包含具體數據或例子"
    )
    
    if st.button("添加到知識庫"):
        if knowledge_title and knowledge_content:
            # 清理和格式化輸入
            title = knowledge_title.strip()
            content = knowledge_content.strip()
            
            # 檢查格式
            if len(title) < 2:
                st.error("標題太短，請輸入更具體的描述")
            elif len(content) < 10:
                st.error("內容太短，請提供更詳細的資訊")
            else:
                try:
                    # 自動格式化內容
                    formatted_title = title
                    formatted_content = content
                    
                    response = requests.post(
                        "http://localhost:8000/add_knowledge",
                        json={
                            "title": formatted_title,
                            "content": formatted_content
                        }
                    )
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        st.success("知識庫更新成功！")
                        # 清空輸入框
                        st.session_state.knowledge_title = ""
                        st.session_state.knowledge_content = ""
                        # 顯示添加的內容和狀態
                        st.markdown(f"""
                        ### 已添加的內容：
                        標題：{formatted_title}
                        內容：{formatted_content}
                        
                        ### 知識庫狀態：
                        - 總文檔數：{response_data.get('total_documents', 'N/A')}
                        - 向量存儲數：{response_data.get('vector_store_count', 'N/A')}
                        - 新增文檔：{response_data.get('added_document', 'N/A')}
                        """)
                    else:
                        error_msg = response.json().get("error", "未知錯誤")
                        st.error(f"發生錯誤：{error_msg}")
                except Exception as e:
                    st.error(f"發生錯誤：{str(e)}")
        else:
            st.warning("標題和內容都不能為空")

# 顯示聊天歷史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用戶輸入
if prompt := st.chat_input("請輸入您的問題"):
    # 添加用戶消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 修改聊天歷史格式
    chat_history = []
    messages = st.session_state.messages[:-1]  # 排除最新的問題
    for i in range(0, len(messages)-1, 2):
        if i+1 < len(messages):
            if messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant":
                chat_history.append({
                    "human": messages[i]["content"],
                    "ai": messages[i+1]["content"]
                })

    # 準備請求數據
    request_data = {
        "message": prompt,
        "use_rag": st.session_state.use_rag,
        "chat_history": chat_history
    }

    # 發送請求到後端
    with st.chat_message("assistant"):
        try:
            response = requests.post(
                "http://localhost:8000/chat",
                json=request_data,
                timeout=30
            )
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    if "error" in response_data:
                        st.error(f"發生錯誤：{response_data['error']}")
                    else:
                        assistant_response = response_data.get("response", "無法獲取回答")
                        source = response_data.get("source", "unknown")
                        
                        # 添加回答來源標記
                        source_label = "🔍 RAG回答" if source == "rag" else "🤖 AI回答"
                        
                        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                        st.markdown(f"{source_label}\n\n{assistant_response}")
                        
                        # 如果有檢索到的文檔，顯示它們
                        if source == "rag" and "retrieved_docs" in response_data:
                            retrieved_docs = response_data.get("retrieved_docs", [])
                            if retrieved_docs:
                                st.markdown("### 相關參考資料")
                                for i, doc in enumerate(retrieved_docs, 1):
                                    st.markdown(f"**文檔 {i}:**\n{doc}")
                except json.JSONDecodeError:
                    st.error("回應格式錯誤")
            else:
                st.error(f"請求失敗：HTTP {response.status_code}")
        except requests.exceptions.Timeout:
            st.error("請求超時，請稍後重試")
        except Exception as e:
            st.error(f"發生錯誤：{str(e)}") 