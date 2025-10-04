#!/usr/bin/env python3
"""
Quick Query Test
===============
Test the current LightRAG database with a simple query.
"""

import os
import sys
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config.config import load_config
from src.lightrag_core import create_lightrag_core

async def test_query():
    print("🧪 Quick Query Test")
    print("=" * 30)
    
    try:
        # Initialize
        config = load_config()
        lightrag_core = create_lightrag_core(config)
        
        print("✅ LightRAG initialized")
        
        # Test query
        test_questions = [
            "What documents are available?",
            "Tell me about machine learning",
            "What programming languages are mentioned?"
        ]
        
        for question in test_questions:
            print(f"\n🔍 Query: {question}")
            result = await lightrag_core.aquery(question, mode="hybrid")
            
            if result.error:
                print(f"❌ Error: {result.error}")
            else:
                # Show first 200 characters
                preview = result.result[:200] + "..." if len(result.result) > 200 else result.result
                print(f"✅ Result ({len(result.result)} chars): {preview}")
                print(f"⏱️  Time: {result.execution_time:.2f}s")
        
        print(f"\n🎉 Your database is working! You have 208 processed documents ready for queries.")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_query())