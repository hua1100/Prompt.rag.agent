# Prompt 生成器 RAG 系統

## 系統概述

這是一個基於 LlamaIndex 和 Chroma 的智能 Prompt 檢索和生成系統，能夠根據用戶輸入自動檢索相關的 prompt 範例並生成客製化的 prompt。

## 主要功能

### 🔧 智能文檔切分
- 根據文檔長度動態選擇切分策略
- 短文檔（≤1024 tokens）完整保留
- 長文檔按語義邊界智能切分
- 支援多種邊界優先級（段落、標題、句子等）

### 🏗️ Chroma 混合架構
- **prompt_contexts**: 任務上下文與壞範例
- **prompt_examples**: 優質 prompt 範例  
- **expected_outputs**: 期望輸出示例

### 🔍 雙軌檢索策略
- **軌道A (無上下文)**: 分類引導 + 過濾建議
- **軌道B (有上下文)**: 智能匹配 + 客製化生成

### ⚙️ Response Mode 配置
- **no_text**: 直接返回原始 prompt (無上下文場景)
- **compact**: LLM 客製化 prompt (有上下文場景)

## 系統統計

- **資料集規模**: 1450 筆記錄
- **切分結果**: 2989 個 chunks
- **有效率**: 98.0% (如果 validation_results 存在)
- **平均每記錄 chunks**: 2.1 個
- **向量數據庫**: 3858 個文檔 (三個 collection)

## 安裝和使用

### 環境要求
```bash
pip install llama-index
pip install llama-index-embeddings-openai
pip install llama-index-llms-openai
pip install llama-index-vector-stores-chroma
pip install chromadb
pip install pandas numpy
```

### 快速開始
```python
# 1. 設置環境
os.environ["OPENAI_API_KEY"] = "your-api-key"

# 2. 初始化系統
from prompt_rag_system import PromptGeneratorRAGSystem
rag_system = PromptGeneratorRAGSystem.from_backup("./backup_directory")

# 3. 使用系統
# 無上下文查詢
result = rag_system.query("幫我寫郵件")

# 有上下文查詢  
result = rag_system.query("幫我回覆", context_content="原始郵件內容...")

# 過濾檢索
result = rag_system.apply_user_filter("查詢", {"prompt_type": "CONVERSATIONAL"})
```

## 檔案結構

```
prompt_rag_system_backup/
├── chroma_database/          # Chroma 向量數據庫
├── dataset/                  # 原始和處理後的資料集
├── source_code/             # 完整系統程式碼
├── processed_chunks.json    # 處理後的文檔切分
├── system_config.json       # 系統配置文件
└── README.md               # 本說明文件
```

## 技術特色

1. **智能場景檢測**: 自動判斷用戶是否提供上下文
2. **分類引導**: 無上下文時提供友好的分類選項
3. **客製化生成**: 有上下文時生成針對性 prompt
4. **過濾精準檢索**: 支援按類型、複雜度等條件過濾
5. **用戶友好顯示**: 分類整理和相似度評分

## 系統架構

```
用戶輸入 → 場景檢測 → 檢索策略選擇
    ↓
軌道A(無上下文): 語義檢索 → 分類分組 → 過濾建議
軌道B(有上下文): 上下文分析 → 智能匹配 → 客製化生成
    ↓
Response Mode → 結果格式化 → 用戶界面展示
```

## 版本信息

- **創建日期**: 2025-06-17 04:13:59
- **LlamaIndex**: 最新版本
- **Chroma**: 最新版本
- **OpenAI API**: text-embedding-ada-002, gpt-3.5-turbo

## 注意事項

1. 需要有效的 OpenAI API Key
2. 建議在有足夠記憶體的環境中運行 (推薦 8GB+)
3. 首次運行會下載 embedding 模型
4. Chroma 數據庫會在本地持久化

## 技術支援

如有問題請檢查：
1. OpenAI API Key 是否正確設置
2. 網路連接是否正常
3. 系統記憶體是否充足
4. 所有依賴套件是否正確安裝
