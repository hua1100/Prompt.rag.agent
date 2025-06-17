#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt 生成器 RAG 系統
基於 LlamaIndex 和 Chroma 的智能 Prompt 檢索和生成系統

功能特色：
- 智能文檔切分策略
- Chroma 混合架構 (三個專用 collection)
- 雙軌檢索策略 (無上下文 vs 有上下文)
- Response Mode 自動配置
- 用戶友好的分類顯示

作者：AI Assistant
創建日期：2025-06-17
"""

import os
import pandas as pd
import numpy as np
import re
import uuid
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional

# LlamaIndex 導入
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings as LlamaSettings

# 系統配置
def setup_environment(openai_api_key: str):
    """設置環境和 LlamaIndex 配置"""
    os.environ["OPENAI_API_KEY"] = openai_api_key
    LlamaSettings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    LlamaSettings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

# [在這裡插入所有的類定義：SmartChunkingStrategy, ChromaMixedArchitectureFixed, HybridSearchStrategy, PromptGeneratorRAGSystem]

def main():
    """主函數 - 系統初始化和測試"""
    # 設置 API Key
    api_key = input("請輸入您的 OpenAI API Key: ")
    setup_environment(api_key)
    
    # 載入資料
    dataset_path = input("請輸入資料集路徑: ")
    df = pd.read_csv(dataset_path)
    
    print(f"載入資料：{len(df)} 筆記錄")
    
    # 初始化系統
    chunking_strategy = SmartChunkingStrategy(max_tokens=1024)
    
    # 執行切分
    all_chunks, stats = process_all_records(df, chunking_strategy)
    print(f"切分完成：{stats['total_chunks']} 個 chunks")
    
    # 初始化 Chroma
    chroma_arch = ChromaMixedArchitectureFixed()
    if chroma_arch.initialize_chroma_client():
        if chroma_arch.create_collections():
            collection_chunks = chroma_arch.prepare_chunks_for_collections(all_chunks)
            if chroma_arch.add_chunks_to_collections(collection_chunks):
                print("✅ Chroma 數據庫初始化完成")
            else:
                print("❌ 數據載入失敗")
                return
        else:
            print("❌ Collection 創建失敗")
            return
    else:
        print("❌ Chroma 客戶端初始化失敗")
        return
    
    # 初始化檢索系統
    hybrid_search = HybridSearchStrategy(chroma_arch)
    rag_system = PromptGeneratorRAGSystem(chroma_arch, hybrid_search)
    
    print("🎉 系統初始化完成！")
    
    # 測試查詢
    while True:
        user_input = input("\n請輸入查詢 (輸入 'quit' 退出): ")
        if user_input.lower() == 'quit':
            break
        
        context = input("請輸入上下文 (可選，直接按 Enter 跳過): ")
        context = context if context.strip() else None
        
        result = rag_system.query(user_input, context)
        
        if "error" in result:
            print(f"❌ 查詢失敗：{result['error']}")
        else:
            print(f"✅ 查詢成功，場景：{result['scenario']}")
            # 顯示結果摘要...

if __name__ == "__main__":
    main()
