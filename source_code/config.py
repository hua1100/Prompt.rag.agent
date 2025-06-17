# 系統配置文件
CHROMA_PERSIST_DIR = "./chroma_db"
MAX_TOKENS = 1024
OPENAI_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"

# 分類映射
USER_FRIENDLY_FILTERS = {
    "prompt_type": {
        "CONVERSATIONAL": "💬 對話回應",
        "CREATIVE_WRITING": "✍️ 創意寫作",
        "INSTRUCTIONAL": "📚 教學指導",
        "SUMMARIZATION": "📝 內容總結",
        "ANALYSIS_CRITIQUE": "🔍 分析評論",
        "INFORMATIONAL": "📋 資訊說明",
        "QUESTION_ANSWERING": "❓ 問答回應",
        "PROGRAMMING_CODE_GENERATION": "💻 程式碼生成",
        "CODE_EXPLANATION": "🔧 程式碼解釋",
        "COMPARISON": "⚖️ 比較分析"
    },
    "complexity": {
        "low": "🟢 簡單易用",
        "medium": "🟡 功能完整",
        "high": "🔴 專業進階"
    }
}

import os
from pathlib import Path

# 基礎路徑配置
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset"
CHROMA_DIR = BASE_DIR / "chroma_database"

# 確保必要的目錄存在
DATASET_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# Chroma 配置
CHROMA_SETTINGS = {
    "chroma_db_impl": "duckdb+parquet",
    "persist_directory": str(CHROMA_DIR),
    "anonymized_telemetry": False
}

# 數據集文件
ORIGINAL_DATASET = DATASET_DIR / "original_dataset.csv"
PROCESSED_DATASET = DATASET_DIR / "processed_dataset.csv"

# 系統配置
SYSTEM_CONFIG = {
    "embedding_model": "text-embedding-ada-002",
    "llm_model": "gpt-3.5-turbo",
    "temperature": 0.1,
    "chunk_size": 1024,
    "chunk_overlap": 200
}

def get_openai_api_key():
    """獲取 OpenAI API Key"""
    # 優先從環境變量獲取
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # 如果環境變量中沒有，嘗試從 Streamlit secrets 獲取
        try:
            import streamlit as st
            api_key = st.secrets["OPENAI_API_KEY"]
        except:
            raise ValueError("未找到 OpenAI API Key，請在 Streamlit Cloud 設置中配置")
    return api_key

def initialize_environment():
    """初始化環境設置"""
    # 設置 OpenAI API Key
    os.environ["OPENAI_API_KEY"] = get_openai_api_key()
    
    # 返回初始化狀態
    return {
        "base_dir": str(BASE_DIR),
        "dataset_dir": str(DATASET_DIR),
        "chroma_dir": str(CHROMA_DIR),
        "api_key_set": bool(os.environ.get("OPENAI_API_KEY"))
    }
