# RAG 聊天機器人

這是一個結合 LLM 和 RAG 的智能聊天系統，使用 FastAPI 作為後端，Streamlit 作為前端，並整合了 Ollama 的 Llama 模型。

## 功能特點

- **LLM + RAG 混合回答**：結合大語言模型和檢索增強生成技術，提供更準確的回答。
- **知識庫管理**：支持動態添加和更新知識庫內容。
- **動態相關性評分**：根據問題自動調整檢索文檔數量和相關性評分。
- **文檔智能檢索**：使用 MMR 算法提高檢索質量。

## 安裝步驟

1. **克隆專案**：
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **安裝後端依賴**：
   確保您已經安裝了 Python 3.8 或更高版本。
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **啟動後端服務**：
   確保 Ollama 已經安裝並運行，並且已下載 Llama 模型。
   ```bash
   uvicorn app:app --reload
   ```

4. **安裝前端依賴**：
   確保您已經安裝了 Node.js 和 npm。
   ```bash
   cd frontend
   npm install
   ```

5. **啟動前端服務**：
   ```bash
   streamlit run app.py
   ```

## 使用說明

- **添加知識庫內容**：在前端界面中輸入標題和內容，點擊「添加到知識庫」按鈕。
- **查詢問題**：在聊天框中輸入問題，系統會自動檢索相關文檔並生成回答。
- **切換 RAG 模式**：可以通過前端界面的開關啟用或禁用 RAG 功能。

## 注意事項

- 確保 Ollama 服務正在運行，並且 Llama 模型已正確配置。
- 如果遇到任何問題，請檢查後端和前端的日誌輸出以獲取更多信息。

## 聲明

本專案僅供學習和研究使用，不應用於商業用途。