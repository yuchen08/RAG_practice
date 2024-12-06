# LLM + RAG 聊天機器人

這是一個結合大型語言模型(LLM)和檢索增強生成(RAG)的智能聊天系統，使用 FastAPI 作為後端，Streamlit 作為前端，並整合了 Ollama 的 Llama 模型。系統能夠基於自定義knowledge_base.json提供準確的回答。

## 系統架構

### 後端架構
- **FastAPI**: 提供高效的 API 服務
- **LangChain**: 用於 LLM 和 RAG 的整合
- **Chroma DB**: 向量數據庫，用於存儲文件嵌入
- **HuggingFace Embeddings**: 使用 text2vec-base-chinese 模型進行中文文本嵌入

### 前端架構
- **Streamlit**: 提供直觀的用戶界面
- **即時互動**: 支持實時對話和knowledge_base.json管理

## 核心功能

### 1. 智能對話
- LLM 與 RAG 混合回答模式
- 自動判斷是否使用knowledge_base.json增強
- 支持上下文理解和多輪對話

### 2. knowledge_base.json管理
- 支持知識文件的增刪改查
- 實時更新向量數據庫
- 文件智能分割和存儲

### 3. 智能檢索
- 基於 MMR 算法的文件檢索
- 動態相關性評分
- 中文分詞和語義匹配

## 安裝要求

### 系統要求
- Python 3.8 或更高版本
- Ollama 服務環境
- 足夠的硬碟空間用於向量數據庫

### 依賴套件
```bash
# 後端依賴
fastapi==0.110.0
uvicorn==0.27.1
python-multipart==0.0.9
langchain==0.1.9
langchain-community==0.0.24
chromadb==0.4.22
pydantic==2.6.3
sentence-transformers>=2.2.2
transformers>=4.36.0
torch>=2.0.0
```

## 安裝步驟

1. **環境準備**
   ```bash
   # 克隆專案
   git clone <repository-url>
   cd <project-directory>
   
   # 創建虛擬環境（建議）
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或
   .\venv\Scripts\activate  # Windows
   ```

2. **安裝依賴**
   ```bash
   # 安裝後端依賴
   cd backend
   pip install -r requirements.txt
   ```

3. **配置 Ollama**
   - 安裝 Ollama
   - 下載 Llama 模型
   - 確保 Ollama 服務運行在 localhost:11434

4. **啟動服務**
   ```bash
   # 啟動後端（在 backend 目錄下）
   uvicorn app:app --reload
   
   # 啟動前端（在 frontend 目錄下）
   streamlit run app.py
   ```

## 使用指南

### knowledge_base.json管理
1. 打開knowledge_base.json管理界面
2. 可以進行以下操作：
   - 新增文件：輸入標題和內容
   - 編輯文件：修改現有文件
   - 刪除文件：移除不需要的文件
   - 查看所有文件

### 聊天功能
1. 使用 RAG 開關控制是否啟用knowledge_base.json增強
2. 在聊天框輸入問題
3. 系統會：
   - 自動檢索相關文件
   - 計算相關性分數
   - 生成綜合回答

## 技術細節

### 文本處理
- 使用中文專用的文本分割策略
- 基於語義的文件分塊
- 智能停用詞過濾

### 檢索策略
- MMR 算法確保結果多樣性
- 動態調整檢索文件數量
- 相關性分數計算優化

### 回答生成
- 結合knowledge_base.json內容和 LLM 能力
- 自定義提示模板
- 多層次答案結構

## 常見問題

1. **Q: 系統無法啟動？**
   A: 檢查 Ollama 服務是否正常運行，確認所有依賴都已正確安裝。

2. **Q: knowledge_base.json更新後沒有生效？**
   A: 檢查 knowledge_base.json 文件權限，確保向量數據庫正確更新。

3. **Q: 回答質量不理想？**
   A: 調整相關性閾值，優化提示模板，或改進文件分割策略。

## 維護與支持

- 定期更新依賴包版本
- 監控系統日誌
- 定期備份knowledge_base.json

## 說明

本專案僅供學習和研究使用。

## 更新日誌

### v1.0.0
- 實現基本的 RAG 功能
- 添加knowledge_base.json管理界面
- 優化中文處理
- 改進檢索算法