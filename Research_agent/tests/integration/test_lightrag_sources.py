#!/usr/bin/env python3
"""
Test script to debug why LightRAG shows 'unknown_source' for papers
Tests with just 4 papers and simple mode (choice 1)
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Test documents with clear source attribution
TEST_DOCUMENTS = [
    {
        "title": "Gait Analysis in Parkinson's Disease Using Deep Learning",
        "source": "ArXiv:2024.12345",
        "content": """
        This paper presents a comprehensive study on gait analysis for early detection of Parkinson's disease.
        We utilize deep learning models to analyze gait patterns captured through wearable sensors.
        Our approach achieves 92% accuracy in early detection, showing significant improvement over traditional methods.
        The study involved 500 participants with various stages of Parkinson's disease.
        Key findings include specific gait markers that appear 2-3 years before clinical diagnosis.
        """,
    },
    {
        "title": "Wearable Sensors for Neurological Disease Detection", 
        "source": "PubMed:PMC9876543",
        "content": """
        This research explores the use of advanced wearable sensors for detecting neurological diseases through gait analysis.
        We developed a multi-sensor system that captures detailed biomechanical data during walking.
        The system can identify subtle changes in gait patterns associated with multiple sclerosis, Parkinson's, and Alzheimer's.
        Clinical trials with 300 patients demonstrated 88% sensitivity and 91% specificity.
        The technology enables continuous monitoring in home environments.
        """,
    },
    {
        "title": "Machine Learning Approaches for Gait Pattern Recognition",
        "source": "IEEE:10.1109/TBME.2024.1234",
        "content": """
        We present novel machine learning algorithms for analyzing gait patterns in neurological disease detection.
        Our ensemble method combines CNNs, LSTMs, and transformer models to process temporal gait data.
        The approach handles variability in walking speeds and environmental conditions.
        Validation on the GaitMotion dataset shows superior performance compared to existing methods.
        Real-world deployment in 5 clinical centers confirms practical applicability.
        """,
    },
    {
        "title": "Early Biomarkers in Neurodegenerative Diseases Through Gait",
        "source": "Nature:s41598-024-5678",
        "content": """
        This study identifies early biomarkers for neurodegenerative diseases through comprehensive gait analysis.
        We discovered that specific gait parameters change up to 5 years before clinical symptoms appear.
        The research tracked 1000 individuals over a 10-year period using standardized gait assessment protocols.
        Key biomarkers include stride length variability, cadence changes, and postural sway patterns.
        These findings enable earlier intervention strategies for at-risk populations.
        """,
    }
]

async def test_lightrag_sources():
    """Test LightRAG with explicit source tracking"""
    
    print("=" * 60)
    print("LightRAG Source Attribution Test")
    print("Testing with 4 papers in Simple mode")
    print("=" * 60)
    
    # Create test directory
    working_dir = "./test_lightrag_sources"
    os.makedirs(working_dir, exist_ok=True)
    
    # Initialize LightRAG
    print("\n1. Initializing LightRAG...")
    rag = LightRAG(
        working_dir=working_dir,
        embedding_func=openai_embed,
        llm_model_func=openai_complete_if_cache,
    )
    
    # Initialize storage
    await rag.initialize_storages()
    await initialize_pipeline_status()  # CRITICAL: Initialize processing pipeline
    print("✓ LightRAG initialized")
    
    # Insert documents with source metadata
    print("\n2. Inserting test documents with sources...")
    for doc in TEST_DOCUMENTS:
        # Format document with source information
        doc_text = f"Title: {doc['title']}\nSource: {doc['source']}\n\n{doc['content']}"
        
        # Insert with metadata (use async version directly)
        await rag.ainsert(doc_text)
        print(f"   ✓ Added: {doc['title']} from {doc['source']}")
    
    print("\n3. Testing queries to check source attribution...")
    
    # Test queries
    queries = [
        "What are the research gaps in gait analysis for neurological diseases?",
        "How do wearable sensors help in early detection?",
        "What machine learning methods are used for gait analysis?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n   Query {i}: {query[:50]}...")
        
        # Query in simple mode (use async version)
        result = await rag.aquery(
            query,
            param=QueryParam(mode="simple")
        )
        
        # Check for source attribution
        print("\n   Response preview:")
        print("   " + "-" * 40)
        
        # Show first 500 chars
        preview = result[:500] if result else "No response"
        for line in preview.split('\n'):
            print(f"   {line}")
        
        # Check for unknown_source
        if "unknown_source" in result:
            print("\n   ⚠️  WARNING: 'unknown_source' found in response!")
            
            # Extract context around unknown_source
            import re
            matches = re.finditer(r'.{0,50}unknown_source.{0,50}', result)
            for match in matches:
                print(f"      Context: ...{match.group()}...")
        else:
            print("\n   ✓ No 'unknown_source' found - sources properly attributed")
    
    # Check the knowledge graph
    print("\n4. Checking knowledge graph for source information...")
    
    # Query for entities
    kg_query = "Show me all entities and their sources"
    kg_result = await rag.aquery(kg_query, param=QueryParam(mode="simple"))
    
    if "unknown_source" in kg_result:
        print("   ⚠️  Knowledge graph contains 'unknown_source' references")
    else:
        print("   ✓ Knowledge graph has proper source attribution")
    
    # Clean up
    print("\n5. Test complete!")
    print("=" * 60)
    
    return "unknown_source" not in kg_result

async def main():
    """Main test function"""
    try:
        success = await test_lightrag_sources()
        
        if success:
            print("\n✅ SUCCESS: All sources properly attributed!")
        else:
            print("\n⚠️  ISSUE: Some sources showing as 'unknown_source'")
            print("\nPossible causes:")
            print("1. LightRAG may not preserve source metadata during entity extraction")
            print("2. The knowledge graph construction might strip source information")
            print("3. Document insertion format might need adjustment")
            print("\nRecommended fix:")
            print("- Ensure source is embedded in document content")
            print("- Check LightRAG's metadata handling in entity extraction")
            print("- Consider using document IDs that include source information")
            
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set in environment")
        sys.exit(1)
    
    # Run test
    asyncio.run(main())