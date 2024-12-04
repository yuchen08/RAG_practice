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

# 知識庫管理區域
with st.expander("知識庫管理"):
    st.markdown("### 知識庫管理")
    
    # 新增知識庫內容的表單
    with st.form("add_knowledge_form"):
        st.markdown("#### 新增文檔")
        new_title = st.text_input(
            "標題",
            placeholder="例如：台灣科技產業概況",
            help="請輸入簡短且具體的主題描述"
        )
        new_content = st.text_area(
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
        
        if st.form_submit_button("新增文檔"):
            if new_title and new_content:
                try:
                    response = requests.post(
                        "http://localhost:8000/add_knowledge",
                        json={
                            "title": new_title,
                            "content": new_content
                        }
                    )
                    
                    if response.status_code == 200:
                        st.success("文檔新增成功")
                        st.rerun()
                    else:
                        st.error("新增失敗")
                except Exception as e:
                    st.error(f"發生錯誤：{str(e)}")
            else:
                st.warning("標題和內容都不能為空")
    
    st.markdown("---")
    
    # 顯示和編輯現有文檔
    st.markdown("#### 現有文檔")
    try:
        response = requests.get("http://localhost:8000/knowledge")
        if response.status_code == 200:
            documents = response.json()
            
            if not documents:
                st.info("目前沒有任何文檔")
            
            for i, doc in enumerate(documents):
                with st.container():
                    st.markdown(f"##### 文檔 {i+1}")
                    
                    # 使用表單進行編輯
                    with st.form(f"edit_form_{i}"):
                        edited_title = st.text_input("標題", doc["title"], key=f"title_{i}")
                        edited_content = st.text_area("內容", doc["content"], key=f"content_{i}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.form_submit_button("更新"):
                                try:
                                    response = requests.put(
                                        f"http://localhost:8000/knowledge/{i}",
                                        json={
                                            "title": edited_title,
                                            "content": edited_content
                                        }
                                    )
                                    if response.status_code == 200:
                                        st.success("文檔更新成功")
                                        st.rerun()
                                    else:
                                        st.error("更新失敗")
                                except Exception as e:
                                    st.error(f"發生錯誤：{str(e)}")
                        
                        with col2:
                            if st.form_submit_button("刪除", type="primary"):
                                try:
                                    response = requests.delete(
                                        f"http://localhost:8000/knowledge/{i}"
                                    )
                                    if response.status_code == 200:
                                        st.success("文檔刪除成功")
                                        st.rerun()
                                    else:
                                        st.error("刪除失敗")
                                except Exception as e:
                                    st.error(f"發生錯誤：{str(e)}")
                    
                    st.markdown("---")
        else:
            st.error("無法獲取知識庫內容")
    except Exception as e:
        st.error(f"發生錯誤：{str(e)}")

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
                                    # 格式化匹配分數為百分比
                                    score_percentage = f"{doc['score']*100:.1f}%"
                                    st.markdown(f"""
                                    **文檔 {i}** (相關度: {score_percentage})
                                    {doc['content']}
                                    """)
                except json.JSONDecodeError:
                    st.error("回應格式錯誤")
            else:
                st.error(f"請求失敗：HTTP {response.status_code}")
        except requests.exceptions.Timeout:
            st.error("請求超時，請稍後重試")
        except Exception as e:
            st.error(f"發生錯誤：{str(e)}") 