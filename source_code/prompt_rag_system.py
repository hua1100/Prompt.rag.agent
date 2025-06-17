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

class PromptGeneratorRAGSystem:
    def __init__(self):
        """初始化 RAG 系統"""
        try:
            # 初始化 Chroma 客戶端
            self.chroma_client = chromadb.Client()
            
            # 創建或獲取 collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="prompts",
                embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model_name="text-embedding-ada-002"
                )
            )
            
            # 初始化系統狀態
            self._initialize_system()
            
        except Exception as e:
            raise Exception(f"RAG 系統初始化失敗：{str(e)}")
    
    def _initialize_system(self):
        """初始化系統狀態"""
        try:
            # 檢查數據集是否已載入
            if self.collection.count() == 0:
                self.process_dataset()
        except Exception as e:
            raise Exception(f"系統狀態初始化失敗：{str(e)}")
    
    def process_dataset(self):
        """處理並載入數據集到 Chroma"""
        try:
            # 讀取數據集
            df = pd.read_csv("dataset/processed_dataset.csv")
            
            # 準備數據
            documents = df["prompt_text"].tolist()
            metadatas = df[["prompt_type", "complexity"]].to_dict('records')
            ids = [str(uuid.uuid4()) for _ in range(len(df))]
            
            # 批量添加到 collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"成功載入 {len(documents)} 條數據到 Chroma")
            return True
            
        except Exception as e:
            print(f"數據集處理錯誤：{str(e)}")
            return False
    
    def apply_user_filter(self, query: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        """執行過濾搜索
        
        Args:
            query: 用戶搜索查詢
            filters: 過濾條件，包含 prompt_type 和 complexity
            
        Returns:
            搜索結果字典
        """
        try:
            # 構建查詢條件
            where_clause = {}
            if "prompt_type" in filters and filters["prompt_type"]:
                where_clause["prompt_type"] = filters["prompt_type"]
            if "complexity" in filters and filters["complexity"]:
                where_clause["complexity"] = filters["complexity"]
            
            # 執行向量搜索
            results = self.collection.query(
                query_texts=[query],
                n_results=10,
                where=where_clause if where_clause else None
            )
            
            # 格式化結果
            formatted_results = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        "text": results['documents'][0][i],
                        "score": float(results['distances'][0][i]),
                        "metadata": {
                            "prompt_type": results['metadatas'][0][i].get('prompt_type'),
                            "complexity": results['metadatas'][0][i].get('complexity')
                        }
                    })
            
            return {
                "total_found": len(formatted_results),
                "results": formatted_results
            }
            
        except Exception as e:
            print(f"搜索錯誤：{str(e)}")
            return {
                "total_found": 0,
                "results": [],
                "error": str(e)
            }
    
    def query(self, user_query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """處理用戶查詢
        
        Args:
            user_query: 用戶查詢
            context: 可選的上下文內容
            
        Returns:
            查詢結果字典
        """
        try:
            # 根據是否有上下文選擇不同的處理邏輯
            if context:
                return self._handle_context_query(user_query, context)
            else:
                return self._handle_no_context_query(user_query)
        except Exception as e:
            return {"error": str(e)}
    
    def _handle_context_query(self, query: str, context: str) -> Dict[str, Any]:
        """處理有上下文的查詢"""
        try:
            # 使用上下文和查詢進行搜索
            results = self.collection.query(
                query_texts=[f"{query} {context}"],
                n_results=3
            )
            
            if not results['ids'] or len(results['ids'][0]) == 0:
                return {
                    "scenario": "context",
                    "response_mode": "customization",
                    "error": "未找到相關結果"
                }
            
            # 返回客製化結果
            return {
                "scenario": "context",
                "response_mode": "customization",
                "formatted_response": {
                    "customized_prompt": self._generate_custom_prompt(query, context, results),
                    "context_analysis": self._analyze_context(context),
                    "source_prompts": [
                        {
                            "score": float(results['distances'][0][i]),
                            "prompt_type": results['metadatas'][0][i].get('prompt_type'),
                            "complexity": results['metadatas'][0][i].get('complexity'),
                            "original_text": results['documents'][0][i]
                        }
                        for i in range(len(results['ids'][0]))
                    ]
                }
            }
        except Exception as e:
            return {
                "scenario": "context",
                "response_mode": "customization",
                "error": str(e)
            }
    
    def _handle_no_context_query(self, query: str) -> Dict[str, Any]:
        """處理無上下文的查詢"""
        try:
            # 執行基本搜索
            results = self.collection.query(
                query_texts=[query],
                n_results=5
            )
            
            if not results['ids'] or len(results['ids'][0]) == 0:
                return {
                    "scenario": "no_context",
                    "response_mode": "categorization",
                    "error": "未找到相關結果"
                }
            
            # 對結果進行分類
            categories = self._categorize_results(results)
            
            return {
                "scenario": "no_context",
                "response_mode": "categorization",
                "formatted_response": {
                    "categories": categories,
                    "filter_suggestions": self._generate_filter_suggestions(results)
                }
            }
        except Exception as e:
            return {
                "scenario": "no_context",
                "response_mode": "categorization",
                "error": str(e)
            }
    
    def _generate_custom_prompt(self, query: str, context: str, results: Dict) -> str:
        """生成客製化 prompt"""
        # 使用最相關的 prompt 作為模板
        template = results['documents'][0][0]
        return template.replace("[Context Placeholder]", context)
    
    def _analyze_context(self, context: str) -> Dict[str, Any]:
        """分析上下文內容"""
        return {
            "content_type": "user_input",
            "length": len(context),
            "summary": context[:200] + "..." if len(context) > 200 else context
        }
    
    def _categorize_results(self, results: Dict) -> Dict[str, Any]:
        """將搜索結果分類"""
        categories = {}
        
        for i in range(len(results['ids'][0])):
            prompt_type = results['metadatas'][0][i].get('prompt_type', 'Other')
            
            if prompt_type not in categories:
                categories[prompt_type] = {
                    "prompt_type": prompt_type,
                    "count": 0,
                    "prompts": []
                }
            
            categories[prompt_type]["count"] += 1
            categories[prompt_type]["prompts"].append({
                "text": results['documents'][0][i],
                "score": float(results['distances'][0][i]),
                "complexity": results['metadatas'][0][i].get('complexity', 'medium')
            })
        
        return categories
    
    def _generate_filter_suggestions(self, results: Dict) -> List[Dict[str, Any]]:
        """生成過濾建議"""
        suggestions = []
        prompt_types = {}
        complexities = {}
        
        for i in range(len(results['ids'][0])):
            prompt_type = results['metadatas'][0][i].get('prompt_type')
            complexity = results['metadatas'][0][i].get('complexity')
            
            if prompt_type:
                prompt_types[prompt_type] = prompt_types.get(prompt_type, 0) + 1
            if complexity:
                complexities[complexity] = complexities.get(complexity, 0) + 1
        
        for prompt_type, count in prompt_types.items():
            suggestions.append({
                "filter_name": prompt_type,
                "count": count,
                "prompt_type": prompt_type,
                "complexity_distribution": complexities
            })
        
        return suggestions

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
    rag_system = PromptGeneratorRAGSystem()
    
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
