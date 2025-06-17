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
