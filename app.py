import streamlit as st
import pandas as pd
from datetime import datetime
import json
import time
import os
import chromadb

# 初始化全局變數
if 'rag_system' not in st.session_state:
    st.session_state['rag_system'] = None
if 'system_stats' not in st.session_state:
    st.session_state['system_stats'] = None

# 設定頁面配置
st.set_page_config(
    page_title="Prompt 生成器 RAG 系統",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS 樣式
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .category-card {
        background: #2d3748;
        border: 1px solid #4a5568;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
        color: #e2e8f0;
    }
    
    .prompt-preview {
        background: #1a202c;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #28a745;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        color: #e2e8f0;
    }
    
    .context-box {
        background: #2d3748;
        border: 1px solid #4a5568;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #e2e8f0;
    }
    
    .success-box {
        background: #22543d;
        border: 1px solid #276749;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #e2e8f0;
    }
    
    .warning-box {
        background: #744210;
        border: 1px solid #975a16;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #e2e8f0;
    }
    
    .sidebar-metric {
        background: #2d3748;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        color: #e2e8f0;
    }
    
    /* 添加深色主題的文字顏色 */
    .stMarkdown {
        color: #e2e8f0;
    }
    
    /* 確保展開器內的文字可見 */
    .streamlit-expanderContent {
        background-color: #1a202c;
        color: #e2e8f0;
    }
    
    /* 源 Prompt 顯示樣式 */
    .source-prompt {
        background: #2d3748;
        border: 1px solid #4a5568;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #e2e8f0;
    }
    
    /* 確保所有文字在深色背景下可見 */
    div[data-testid="stExpander"] {
        background-color: #1a202c;
        color: #e2e8f0;
        border: 1px solid #4a5568;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    div[data-testid="stExpander"] > div {
        color: #e2e8f0;
    }
    
    /* 修改展開器的標題顏色 */
    .streamlit-expanderHeader {
        color: #e2e8f0 !important;
        background-color: #2d3748 !important;
    }
    
    /* 確保代碼塊文字顏色 */
    code {
        color: #e2e8f0 !important;
        background-color: #2d3748 !important;
    }
    
    /* 一般文字顏色 */
    p, h1, h2, h3, h4, h5, h6, li, span {
        color: #e2e8f0;
    }
    
    /* 確保連結顏色可見 */
    a {
        color: #63b3ed !important;
    }
    
    /* 表格文字顏色 */
    .dataframe {
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitRAGInterface:
    """
    Streamlit 前端界面類
    """
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """初始化 session state"""
        if 'system_loaded' not in st.session_state:
            st.session_state.system_loaded = False
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        if 'current_results' not in st.session_state:
            st.session_state.current_results = None
        if 'selected_category' not in st.session_state:
            st.session_state.selected_category = None
    
    def check_system_status(self):
        """檢查系統關鍵組件的狀態"""
        status = {
            "api_key": False,
            "database": False,
            "dataset": False
        }
        
        # 檢查 OpenAI API Key
        try:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key and len(api_key) > 20:  # 簡單的長度檢查
                status["api_key"] = True
        except Exception as e:
            st.error(f"API Key 檢查錯誤: {str(e)}")
            
        # 檢查 Chroma 數據庫
        try:
            if hasattr(st.session_state, 'rag_system'):
                if not isinstance(st.session_state.rag_system, MockRAGSystem):
                    # 檢查數據庫連接
                    client = chromadb.Client()
                    collections = client.list_collections()
                    status["database"] = len(collections) > 0
        except Exception as e:
            st.error(f"數據庫檢查錯誤: {str(e)}")
            
        # 檢查數據集
        try:
            dataset_path = "dataset/processed_dataset.csv"
            if os.path.exists(dataset_path):
                df = pd.read_csv(dataset_path)
                status["dataset"] = len(df) > 0
        except Exception as e:
            st.error(f"數據集檢查錯誤: {str(e)}")
            
        return status

    def render_system_status(self):
        """在側邊欄顯示系統狀態"""
        st.sidebar.markdown("### 🔧 系統狀態")
        status = self.check_system_status()
        
        # API Key 狀態
        if status["api_key"]:
            st.sidebar.success("✅ OpenAI API Key 已設置")
        else:
            st.sidebar.error("❌ OpenAI API Key 未設置或無效")
            
        # 數據庫狀態
        if status["database"]:
            st.sidebar.success("✅ Chroma 數據庫已初始化")
        else:
            st.sidebar.error("❌ Chroma 數據庫未初始化")
            
        # 數據集狀態
        if status["dataset"]:
            st.sidebar.success("✅ 數據集已載入")
        else:
            st.sidebar.error("❌ 數據集未載入")
            
        return status

    def load_system(self):
        """載入系統"""
        try:
            # 初始化環境
            from source_code.config import initialize_environment
            env_status = initialize_environment()
            
            if not env_status["api_key_set"]:
                st.error("OpenAI API Key 未設置")
                return False
                
            # 檢查數據集
            if not os.path.exists("dataset/processed_dataset.csv"):
                st.error("數據集文件不存在")
                return False
                
            # 初始化真實的 RAG 系統
            from source_code.prompt_rag_system import PromptGeneratorRAGSystem
            st.session_state.rag_system = PromptGeneratorRAGSystem()
            
            # 載入系統統計
            st.session_state.system_stats = self.load_system_stats()
            st.session_state.system_loaded = True
            
            # 顯示成功信息
            st.success("系統載入成功！")
            
            return True
            
        except Exception as e:
            st.error(f"系統載入失敗：{str(e)}")
            return False
            
    def load_system_stats(self):
        """載入系統統計信息"""
        try:
            if os.path.exists("dataset/processed_dataset.csv"):
                df = pd.read_csv("dataset/processed_dataset.csv")
                return {
                    "collections": {
                        "total_prompts": len(df),
                        "by_type": df["prompt_type"].value_counts().to_dict()
                    },
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        except Exception as e:
            st.error(f"統計信息載入失敗：{str(e)}")
            return None
    
    def render_header(self):
        """渲染頁面標題"""
        st.markdown('<h1 class="main-header">🚀 Prompt 生成器 RAG 系統</h1>', unsafe_allow_html=True)
        
        # 副標題
        st.markdown("""
        <div style="text-align: center; color: #666; margin-bottom: 2rem;">
            <h3>智能 Prompt 檢索與生成平台</h3>
            <p>基於 LlamaIndex 和 Chroma 的先進 RAG 架構</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """渲染側邊欄"""
        with st.sidebar:
            st.title("🤖 Prompt RAG")
            
            # 顯示系統狀態
            system_status = self.render_system_status()
            
            # 載入系統按鈕
            if st.button("🔄 載入系統", use_container_width=True):
                self.load_system()
            
            if st.session_state.system_loaded and self.system_stats:
                st.markdown("## 📊 系統統計")
                
                # 數據統計
                stats = self.system_stats.get("collections", {})
                total_docs = sum(stats.values()) if stats else 0
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("總文檔數", f"{total_docs:,}")
                with col2:
                    st.metric("Collections", len(stats))
                
                # Collection 詳情
                if stats:
                    st.markdown("### Collection 分佈")
                    for name, count in stats.items():
                        percentage = (count / total_docs * 100) if total_docs > 0 else 0
                        st.markdown(f"📝 **{name}**: {count:,} ({percentage:.1f}%)")
                
                # 搜尋歷史
                if st.session_state.search_history:
                    st.markdown("## 🕒 搜尋歷史")
                    for i, search in enumerate(reversed(st.session_state.search_history[-5:]), 1):
                        with st.expander(f"搜尋 {i}: {search['query'][:20]}..."):
                            st.write(f"**查詢**: {search['query']}")
                            st.write(f"**時間**: {search['timestamp']}")
                            st.write(f"**場景**: {search['scenario']}")
                            if search.get('context'):
                                st.write(f"**有上下文**: 是 ({len(search['context'])} 字符)")
                            else:
                                st.write(f"**有上下文**: 否")
                
                # 清除歷史
                if st.button("🗑️ 清除搜尋歷史"):
                    st.session_state.search_history = []
                    st.success("搜尋歷史已清除")
                    st.rerun()
    
    def render_main_interface(self):
        """渲染主界面"""
        if not st.session_state.system_loaded:
            st.markdown("""
            <div class="warning-box">
                <h3>⚠️ 系統未載入</h3>
                <p>請先在左側面板點擊「載入系統」按鈕後再使用。</p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # 主要功能標籤頁
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 智能搜尋", "🎯 過濾檢索", "📊 系統分析", "💡 使用說明"])
        
        with tab1:
            self.render_smart_search()
        
        with tab2:
            self.render_filtered_search()
        
        with tab3:
            self.render_system_analysis()
        
        with tab4:
            self.render_help_guide()
    
    def render_smart_search(self):
        """渲染智能搜尋界面"""
        st.markdown("## 🔍 智能 Prompt 搜尋")
        
        # 搜尋表單
        with st.form("search_form"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                user_query = st.text_input(
                    "請輸入您的需求",
                    placeholder="例如：幫我寫一封行銷郵件、生成 Python 排序算法、給我一些關於太空探索的創意寫作點子...",
                    help="輸入您想要的 prompt 類型或具體需求"
                )
            
            with col2:
                search_mode = st.selectbox(
                    "搜尋模式",
                    ["自動檢測", "無上下文", "有上下文"],
                    help="選擇搜尋模式。'自動檢測' 會根據是否提供上下文來決定模式。"
                )
            
            # 上下文輸入區域
            context_content = st.text_area(
                "上下文內容 (可選)",
                placeholder="如果您有具體的內容需要處理，請貼到這裡...\n例如：要回覆的郵件、需要解釋的程式碼、要摘要的文章等",
                height=120,
                help="提供上下文可以獲得更精確的客製化 prompt"
            )
            
            # 搜尋按鈕
            search_submitted = st.form_submit_button("🚀 開始搜尋", type="primary")
        
        # 處理搜尋
        if search_submitted and user_query:
            self.process_search(user_query, context_content, search_mode)
        
        # 顯示搜尋結果
        if st.session_state.current_results:
            self.display_search_results(st.session_state.current_results)
    
    def process_search(self, user_query, context_content, search_mode):
        """處理搜尋請求"""
        with st.spinner("🔍 搜尋中，請稍候..."):
            try:
                if not self.rag_system:
                    self.rag_system = st.session_state.get('rag_system')
                    if not self.rag_system:
                        st.error("系統未正確載入，請重新載入系統")
                        return
                
                # 準備上下文
                context = context_content.strip() if context_content else None
                
                # 自動檢測模式邏輯
                if search_mode == "自動檢測":
                    final_context = context
                elif search_mode == "無上下文":
                    final_context = None
                elif search_mode == "有上下文":
                    if not context:
                        st.warning("您選擇了'有上下文'模式，但未提供任何上下文內容。請在文本框中輸入內容。")
                        return
                    final_context = context
                
                # 模擬API調用延遲
                time.sleep(1.5)
                
                # 調用 RAG 系統
                result = self.rag_system.query(user_query, final_context)
                
                if "error" in result:
                    st.error(f"搜尋失敗：{result['error']}")
                    return
                
                # 保存結果
                st.session_state.current_results = result
                
                # 添加到搜尋歷史
                st.session_state.search_history.append({
                    "query": user_query,
                    "context": final_context,
                    "scenario": result.get("scenario"),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                st.success("🎉 搜尋完成！")
                
            except Exception as e:
                st.error(f"搜尋過程中發生錯誤：{str(e)}")
    
    def display_search_results(self, results):
        """顯示搜尋結果"""
        st.markdown("---")
        st.markdown("## 📋 搜尋結果")
        
        scenario = results.get("scenario")
        formatted_response = results.get("formatted_response", {})
        
        # 結果摘要
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🎯 檢測場景</h4>
                <p>{scenario}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            mode = results.get("response_mode", "unknown")
            st.markdown(f"""
            <div class="metric-card">
                <h4>⚙️ 處理模式</h4>
                <p>{mode}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if scenario == "no_context":
                count = len(formatted_response.get("categories", {}))
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📊 找到分類</h4>
                    <p>{count} 個</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                confidence = formatted_response.get("confidence", "medium")
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🎯 信心度</h4>
                    <p>{confidence.capitalize()}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # 根據場景顯示不同內容
        if scenario == "no_context":
            self.display_category_results(formatted_response)
        else:
            self.display_context_results(formatted_response)
    
    def display_category_results(self, formatted_response):
        """顯示分類結果"""
        categories = formatted_response.get("categories", {})
        filter_suggestions = formatted_response.get("filter_suggestions", [])
        
        if not categories:
            st.warning("未找到相關的 prompt，請嘗試其他關鍵詞")
            return
        
        st.markdown("### 🏷️ 按分類瀏覽 Prompt")
        
        # 分類選擇器
        category_names = list(categories.keys())
        selected_category = st.selectbox(
            "選擇分類",
            category_names,
            help="選擇一個分類查看相關的 prompt"
        )
        
        if selected_category and selected_category in categories:
            category_info = categories[selected_category]
            
            st.markdown(f"""
            <div class="category-card">
                <h4>{selected_category}</h4>
                <p><strong>類型</strong>: {category_info['prompt_type']}</p>
                <p><strong>數量</strong>: {category_info['count']} 個相關 prompt</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 顯示 prompt
            prompts = category_info.get("prompts", [])
            for i, prompt in enumerate(prompts, 1):
                unique_key = f"copy_{selected_category}_{i}"
                with st.expander(f"Prompt {i} (相似度: {prompt['score']:.3f}) - {prompt['complexity'].capitalize()}"):
                    st.markdown(f"""
                    <div class="prompt-preview">
                        {prompt['text'].replace('<', '&lt;').replace('>', '&gt;')[:500]}{'...' if len(prompt['text']) > 500 else ''}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**複雜度**: {prompt['complexity']}")
                    with col2:
                        st.write(f"**相似度**: {prompt['score']:.3f}")
                    with col3:
                        if st.button(f"📋 複製 Prompt {i}", key=unique_key):
                            st.code(prompt['text'], language="text")
                            st.success("Prompt 已複製到剪貼簿！") # Streamlit doesn't actually copy, but this provides good UX.
        
        # 過濾建議
        if filter_suggestions:
            st.markdown("### 💡 建議的過濾選項")
            st.write("以下是基於搜尋結果的過濾建議，可以幫助您在「過濾檢索」分頁中找到更精確的 prompt：")
            
            for suggestion in filter_suggestions:
                with st.expander(f"{suggestion['filter_name']} ({suggestion['count']} 個結果)"):
                    st.write(f"**類型**: {suggestion['prompt_type']}")
                    st.write(f"**複雜度分佈**: {suggestion.get('complexity_distribution', {})}")
                    if suggestion.get('sample_techniques'):
                        st.write(f"**主要技巧**: {', '.join(suggestion['sample_techniques'][:3])}")
    
    def display_context_results(self, formatted_response):
        """顯示上下文結果"""
        customized_prompt = formatted_response.get("customized_prompt", "")
        context_analysis = formatted_response.get("context_analysis", {})
        source_prompts = formatted_response.get("source_prompts", [])
        expected_outputs = formatted_response.get("expected_outputs", [])
        
        # 客製化 Prompt
        if customized_prompt:
            st.markdown("### 🎯 客製化 Prompt")
            st.markdown(f"""
            <div class="success-box">
                <h4>✨ 為您量身打造的 Prompt</h4>
                <div class="prompt-preview">
                    {customized_prompt.replace('<', '&lt;').replace('>', '&gt;')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 複製按鈕
            if st.button("📋 複製客製化 Prompt", type="primary"):
                st.code(customized_prompt, language="text")
                st.success("客製化 Prompt 已複製到剪貼簿！")
        
        with st.expander("🔍 查看生成細節 (上下文分析與源 Prompt)"):
            # 上下文分析
            if context_analysis:
                st.markdown("### 📄 上下文分析")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**內容類型**: {context_analysis.get('content_type', 'general')}")
                with col2:
                    st.write(f"**內容長度**: {context_analysis.get('length', 0)} 字符")
                
                if context_analysis.get('summary'):
                    st.write(f"**分析摘要**: {context_analysis['summary']}")
            
            # 源 Prompt 信息
            if source_prompts:
                st.markdown("### 📚 參考的來源 Prompt")
                st.write(f"本次客製化基於 {len(source_prompts)} 個高品質的來源 prompt：")
                
                for i, source in enumerate(source_prompts, 1):
                    st.markdown(f"**源 Prompt {i} (相似度: {source['score']:.3f})**")
                    st.write(f"**類型**: {source['prompt_type']}, **複雜度**: {source['complexity']}, **技巧**: {source['techniques']}")
                    st.markdown(f"""
                    <div class="source-prompt">
                        {source['original_text'].replace('<', '&lt;').replace('>', '&gt;')[:300]}...
                    </div>
                    """, unsafe_allow_html=True)
        
        # 期望輸出
        if expected_outputs:
            st.markdown("### 💡 期望輸出範例")
            for i, output in enumerate(expected_outputs, 1):
                with st.expander(f"輸出範例 {i}"):
                    st.markdown(f"""
                    <div class="context-box">
                        {output.replace('<', '&lt;').replace('>', '&gt;')[:400]}{'...' if len(output) > 400 else ''}
                    </div>
                    """, unsafe_allow_html=True)
    
    def render_filtered_search(self):
        """渲染過濾檢索界面"""
        st.markdown("## 🎯 進階過濾檢索")
        st.write("使用進階過濾功能，精確找到符合特定條件的 prompt。")
        
        # 過濾選項
        col1, col2 = st.columns(2)
        
        with col1:
            prompt_types = [
                "CONVERSATIONAL", "CREATIVE_WRITING", "INSTRUCTIONAL", 
                "SUMMARIZATION", "ANALYSIS_CRITIQUE", "INFORMATIONAL",
                "QUESTION_ANSWERING", "PROGRAMMING_CODE_GENERATION", 
                "CODE_EXPLANATION", "COMPARISON"
            ]
            
            selected_type = st.selectbox(
                "Prompt 類型",
                ["全部"] + prompt_types,
                key="filter_type",
                help="選擇特定的 prompt 類型"
            )
        
        with col2:
            complexity_levels = ["low", "medium", "high"]
            selected_complexity = st.selectbox(
                "複雜度",
                ["全部"] + complexity_levels,
                key="filter_complexity",
                help="選擇 prompt 的複雜度等級"
            )
        
        # 搜尋查詢
        filter_query = st.text_input(
            "搜尋查詢 (可選)",
            placeholder="輸入關鍵詞以進一步縮小範圍...",
            help="在已過濾的結果中進行文本搜尋"
        )
        
        # 執行過濾搜尋
        if st.button("🎯 執行過濾搜尋", type="primary"):
            self.execute_filtered_search(filter_query, selected_type, selected_complexity)

    def execute_filtered_search(self, query, prompt_type, complexity):
        """執行過濾搜索"""
        try:
            if not st.session_state.system_loaded:
                st.error("請先載入系統")
                return
                
            filters = {}
            if prompt_type != "全部":
                filters["prompt_type"] = prompt_type
            if complexity != "全部":
                filters["complexity"] = complexity
                
            results = st.session_state.rag_system.apply_user_filter(query, filters)
            
            if "error" in results:
                st.error(f"搜索錯誤：{results['error']}")
                return
                
            if results["total_found"] == 0:
                st.warning("未找到符合條件的結果")
                return
                
            # 顯示結果
            st.markdown(f"🔍 找到 {results['total_found']} 個匹配結果")
            
            for result in results["results"]:
                with st.expander(f"相似度: {result['score']:.3f}"):
                    st.markdown(f"**類型**: {result['metadata']['prompt_type']}")
                    st.markdown(f"**複雜度**: {result['metadata']['complexity']}")
                    st.text_area("Prompt 內容", result["text"], height=100)
                    
        except Exception as e:
            st.error(f"執行搜索時發生錯誤：{str(e)}")
    
    def render_system_analysis(self):
        """渲染系統分析界面"""
        st.markdown("## 📊 系統分析與統計")
        
        if not self.system_stats:
            st.warning("系統統計數據不可用")
            return
        
        # 系統概覽
        st.markdown("### 🎯 系統概覽")
        
        stats = self.system_stats.get("collections", {})
        total_docs = sum(stats.values()) if stats else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📚 總文檔數",
                value=f"{total_docs:,}",
                help="系統中所有 collection 的文檔總數"
            )
        
        with col2:
            st.metric(
                label="🗂️ Collections",
                value=len(stats),
                help="向量數據庫中的 collection 數量"
            )
        
        with col3:
            search_count = len(st.session_state.search_history)
            st.metric(
                label="🔍 搜尋次數",
                value=search_count,
                help="本次會話中的搜尋次數"
            )
        
        with col4:
            st.metric(
                label="🔄 系統狀態",
                value="運行中" if st.session_state.system_loaded else "離線",
                help="當前系統運行狀態"
            )
        
        # Collection 分佈圖表
        if stats:
            st.markdown("### 📊 Collection 文檔分佈")
            
            df_stats = pd.DataFrame(list(stats.items()), columns=['Collection', '文檔數量'])

            col1, col2 = st.columns(2)

            with col1:
                # 使用 Streamlit 原生图表显示文档数量占比
                st.subheader("各 Collection 文檔數量佔比")
                st.bar_chart(df_stats.set_index('Collection')['文檔數量'])
            
            with col2:
                # 使用 Streamlit 原生图表显示文档绝对数量
                st.subheader("各 Collection 文檔絕對數量")
                st.bar_chart(df_stats.set_index('Collection')['文檔數量'])
        
        # 搜尋歷史分析
        if st.session_state.search_history:
            st.markdown("### 📈 搜尋歷史分析")
            
            history_df = pd.DataFrame(st.session_state.search_history)
            
            # 場景分佈
            scenario_counts = history_df['scenario'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                if not scenario_counts.empty:
                    st.subheader("搜尋場景分佈")
                    st.bar_chart(scenario_counts)
                else:
                    st.info("尚無足夠的搜尋歷史來分析場景分佈。")
            
            with col2:
                # 搜尋時間趨勢
                if not history_df.empty:
                    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                    hourly_counts = history_df.groupby(history_df['timestamp'].dt.hour).size().reset_index(name='count')
                    hourly_counts.set_index('timestamp', inplace=True)
                    
                    st.subheader("每小時搜尋次數分佈")
                    st.line_chart(hourly_counts)
                else:
                    st.info("尚無足夠的搜尋歷史來分析時間趨勢。")
            
            # 詳細歷史
            st.markdown("### 📝 詳細搜尋歷史")
            st.dataframe(
                history_df[['timestamp', 'query', 'scenario']],
                use_container_width=True
            )
            
    def render_help_guide(self):
        """渲染使用說明界面"""
        st.markdown("## 💡 使用說明與指南")
        st.write("歡迎使用 Prompt 生成器 RAG 系統！這裡將引導您如何有效利用本平台。")

        with st.expander("🚀 快速入門", expanded=True):
            st.markdown("""
            1. **載入系統**: 點擊左側邊欄的 `🔄 載入系統` 按鈕。
            2. **前往智能搜尋**: 系統載入後，停留在 `🔍 智能搜尋` 標籤頁。
            3. **輸入需求**: 在輸入框中描述您想要的任務，例如「寫一封道歉信」或「解釋 Python 的 aiohttp 庫」。
            4. **提供上下文 (可選)**: 如果您的任務需要基於特定內容（如一封待回覆的郵件），請將其貼入「上下文內容」區域。
            5. **開始搜尋**: 點擊 `🚀 開始搜尋` 按鈕。
            6. **查看結果**: 系統會自動分析您的需求，並提供分類的 prompt 建議或一個為您量身打造的客製化 prompt。
            """)

        with st.expander("🔍 智能搜尋詳解"):
            st.markdown("""
            **智能搜尋** 是本系統的核心功能，它能理解您的意圖並提供最相關的結果。

            - **無上下文搜尋**:
                - **適用場景**: 當您有一個通用的想法，想尋找高品質的 prompt 模板時。
                - **例如**: 「創意寫作點子」、「總結文章的 prompt」。
                - **結果**: 系統會返回多個相關的 **分類**，每個分類下包含多個 prompt 範例，您可以從中挑選。

            - **有上下文搜尋**:
                - **適用場景**: 當您需要處理一段具體文本時。
                - **例如**: 將一封客戶投訴郵件貼入上下文，並在需求中輸入「幫我草擬一封專業的回覆」。
                - **結果**: 系統會分析您的上下文，並結合您的需求，生成一個 **獨一無二的、客製化的 prompt**。

            - **自動檢測模式**:
                - 這是**推薦模式**。您無需關心要選哪種模式。
                - 系統會自動檢查「上下文內容」區域是否為空。
            """)

        with st.expander("🎯 進階過濾檢索"):
            st.markdown("""
            當您對所需的 prompt 有非常具體的要求時，可以使用此功能。

            - **Prompt 類型**: 篩選特定用途的 prompt，例如 `PROGRAMMING_CODE_GENERATION` 只會顯示與程式碼生成相關的 prompt。
            - **複雜度**: 篩選 prompt 的複雜程度。
                - `low`: 簡單、直接的指令。
                - `medium`: 包含多個步驟或一些限制條件。
                - `high`: 複雜的、專家級的 prompt。
            - **搜尋查詢 (可選)**: 在以上過濾條件的基礎上，再進行關鍵詞搜尋。
            """)

        with st.expander("📋 解讀搜尋結果"):
            st.markdown("""
            - **檢測場景**: 系統判斷您的搜尋是 `no_context` (無上下文) 還是 `context` (有上下文)。
            - **處理模式**: 系統採用的內部處理策略。
            - **分類結果 (無上下文)**:
                - `分類`: 根據您的需求找到的相關 prompt 類別。
                - `Prompt 預覽`: 點擊展開可看到完整的 prompt 文本和其複雜度、相似度等信息。
            - **客製化結果 (有上下文)**:
                - `客製化 Prompt`: 這是系統為您量身打造的最終 prompt。
                - `上下文分析`: 系統對您提供的上下文的理解。
                - `源 Prompt`: 系統在生成客製化 prompt 時參考的基礎模板。
            - **相似度 (Score)**: 代表檢索到的 prompt 與您的查詢有多相關，分數越高越相關。
            """)