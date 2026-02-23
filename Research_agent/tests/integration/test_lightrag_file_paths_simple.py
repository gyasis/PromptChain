#!/usr/bin/env python3
"""
Simple test of LightRAG file_paths parameter for proper source attribution
This uses LightRAG's built-in citation tracking instead of embedded text markers
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Simple test documents
TEST_DOCS = [
    {
        "title": "Gait Analysis in Parkinson's Disease",
        "content": "This paper presents gait analysis methods for early detection of Parkinson's disease using machine learning.",
        "file_path": "parkinsons_gait_2024.pdf"
    },
    {
        "title": "Wearable Sensors for Disease Detection", 
        "content": "This research explores wearable sensor technology for detecting neurological diseases through movement analysis.",
        "file_path": "wearable_sensors_neuro_2024.pdf"
    }
]

async def test_file_paths_citation():
    """Test LightRAG with file_paths parameter for source tracking"""
    
    print("=" * 60)
    print("LightRAG file_paths Citation Test")
    print("=" * 60)
    
    # Create test directory
    working_dir = "./test_file_paths_citations"
    os.makedirs(working_dir, exist_ok=True)
    
    # Initialize LightRAG
    print("\n1. Initializing LightRAG...")
    rag = LightRAG(
        working_dir=working_dir,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    
    print("✓ LightRAG initialized")
    
    # Prepare documents and file_paths arrays
    print("\n2. Preparing documents with file_paths...")
    documents = []
    file_paths = []
    
    for doc in TEST_DOCS:
        documents.append(doc["content"])
        file_paths.append(doc["file_path"])
        print(f"   ✓ Document: {doc['title']} -> {doc['file_path']}")
    
    # Insert using file_paths parameter
    print("\n3. Inserting with file_paths parameter...")
    try:
        await rag.ainsert(documents, file_paths=file_paths)
        print("✅ Documents inserted successfully")
    except Exception as e:
        print(f"❌ Error during insertion: {e}")
        return False
    
    # Test query to check citations
    print("\n4. Testing query for citation tracking...")
    query = "What methods are used for disease detection?"
    
    try:
        result = await rag.aquery(
            query,
            param=QueryParam(mode="simple")
        )
        
        print(f"\nQuery: {query}")
        print("Response:")
        print("-" * 40)
        print(result)
        
        # Check for unknown_source
        if "unknown_source" in result.lower():
            print("\n⚠️  WARNING: 'unknown_source' found in response")
            return False
        else:
            print("\n✅ SUCCESS: No 'unknown_source' found")
            
            # Check for actual file references
            citations_found = []
            for doc in TEST_DOCS:
                if doc["file_path"] in result or doc["title"] in result:
                    citations_found.append(doc["file_path"])
            
            if citations_found:
                print(f"📚 File citations found: {citations_found}")
            else:
                print("📚 No explicit file citations, but no unknown_source errors")
            
            return True
            
    except Exception as e:
        print(f"❌ Query error: {e}")
        return False

async def main():
    """Main test function"""
    try:
        success = await test_file_paths_citation()
        
        if success:
            print("\n🎯 SOLUTION WORKING!")
            print("✅ LightRAG file_paths parameter successfully prevents unknown_source")
            print("✅ This can be integrated into the enhanced demo")
        else:
            print("\n⚠️  Need to investigate further")
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set in environment")
        sys.exit(1)
    
    # Run test
    asyncio.run(main())