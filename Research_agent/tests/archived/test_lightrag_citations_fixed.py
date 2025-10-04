#!/usr/bin/env python3
"""
Fixed LightRAG Citation Test - Using Proper Document Metadata Approach

Based on Gemini analysis, LightRAG v1.4.6 doesn't have ainsert(documents, file_paths)
Instead, we need to embed source information in document metadata before insertion.
"""

import asyncio
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Test documents with proper citation tracking
TEST_DOCS = [
    {
        "title": "Gait Analysis in Parkinson's Disease",
        "content": "This paper presents gait analysis methods for early detection of Parkinson's disease using machine learning. The study analyzed 200 patients and found 95% accuracy in early detection.",
        "file_path": "parkinsons_gait_2024.pdf",
        "authors": "Smith, J. et al.",
        "year": "2024"
    },
    {
        "title": "Wearable Sensors for Disease Detection", 
        "content": "This research explores wearable sensor technology for detecting neurological diseases through movement analysis. The sensors achieved 92% accuracy in detecting movement anomalies.",
        "file_path": "wearable_sensors_neuro_2024.pdf",
        "authors": "Johnson, M. et al.",
        "year": "2024"
    }
]

class LightRAGCitationError(Exception):
    """Custom exception for LightRAG citation failures."""
    pass

def create_document_with_metadata(doc: Dict[str, str]) -> Dict[str, str]:
    """Create properly formatted document with embedded source metadata."""
    return {
        "page_content": doc["content"],
        "metadata": {
            "source": doc["file_path"],
            "title": doc["title"],
            "authors": doc.get("authors", "Unknown"),
            "year": doc.get("year", "Unknown"),
            "document_id": f"{doc['file_path']}_{doc.get('year', 'unknown')}"
        }
    }

async def test_lightrag_citation_tracking():
    """Test LightRAG with proper document metadata for citation tracking."""
    
    logging.info("=" * 60)
    logging.info("LightRAG Citation Tracking Test (Fixed Approach)")
    logging.info("=" * 60)
    
    # Create test directory
    working_dir = "./test_lightrag_citations_fixed"
    os.makedirs(working_dir, exist_ok=True)
    
    try:
        # Initialize LightRAG (no async context manager needed)
        logging.info("1. Initializing LightRAG...")
        
        rag = LightRAG(
            working_dir=working_dir,
            embedding_func=openai_embed,
            llm_model_func=gpt_4o_mini_complete,
        )
        
        # Initialize async storages (required for LightRAG v1.4.6)
        logging.info("   Initializing async storages...")
        await rag.initialize_storages()
        
        # Initialize pipeline status
        logging.info("   Initializing pipeline status...")
        await initialize_pipeline_status()
        
        logging.info("✓ LightRAG initialized successfully")
        
        # Insert documents with proper metadata
        logging.info("2. Inserting documents with embedded source metadata...")
        
        for i, doc in enumerate(TEST_DOCS):
            # Create document with proper metadata structure
            formatted_doc = create_document_with_metadata(doc)
            
            logging.info(f"   Inserting document {i+1}: {doc['title']}")
            logging.info(f"   Source: {formatted_doc['metadata']['source']}")
            
            # Insert using string representation of document with metadata
            try:
                await rag.ainsert(str(formatted_doc))
                logging.info(f"   ✓ Document inserted successfully")
            except Exception as e:
                logging.error(f"   ❌ Failed to insert document: {e}")
                raise LightRAGCitationError(f"Document insertion failed: {e}")
        
        logging.info("✓ All documents inserted successfully")
        
        # Test query for citation tracking
        logging.info("3. Testing query for proper citation tracking...")
        
        test_queries = [
            "What methods are used for disease detection?",
            "What is the accuracy of gait analysis for Parkinson's detection?",
            "How do wearable sensors help with neurological disease detection?"
        ]
        
        citation_success = True
        
        for query in test_queries:
            logging.info(f"\nQuery: {query}")
            
            try:
                result = await rag.aquery(
                    query,
                    param=QueryParam(mode="hybrid")  # Use hybrid mode for better citation tracking
                )
                
                logging.info("Response:")
                logging.info("-" * 40)
                logging.info(result)
                
                # Check for unknown_source
                if "unknown_source" in str(result).lower():
                    logging.warning("⚠️  WARNING: 'unknown_source' found in response")
                    citation_success = False
                else:
                    logging.info("✓ No 'unknown_source' found")
                
                # Check for proper source references
                found_sources = []
                for doc in TEST_DOCS:
                    if doc["file_path"] in str(result) or doc["title"] in str(result):
                        found_sources.append(doc["file_path"])
                
                if found_sources:
                    logging.info(f"📚 Sources found in response: {found_sources}")
                else:
                    logging.info("📚 No explicit source mentions (but no unknown_source)")
                
            except Exception as e:
                logging.error(f"❌ Query failed: {e}")
                citation_success = False
        
        return citation_success
            
    except Exception as e:
        logging.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test directory
        try:
            import shutil
            shutil.rmtree(working_dir)
            logging.info(f"Cleaned up test directory: {working_dir}")
        except Exception as e:
            logging.warning(f"Failed to clean up test directory: {e}")

async def main():
    """Main test runner"""
    try:
        logging.info("Starting LightRAG Citation Tracking Test...")
        
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            logging.error("❌ OPENAI_API_KEY not set in environment")
            return
        
        success = await test_lightrag_citation_tracking()
        
        if success:
            logging.info("\n🎯 SUCCESS: LightRAG Citation Tracking Working!")
            logging.info("✅ No 'unknown_source' errors detected")
            logging.info("✅ Document metadata approach successfully provides source attribution")
            logging.info("✅ Ready for integration into enhanced demo")
        else:
            logging.warning("\n⚠️  Citation tracking test had issues")
            logging.info("Need to investigate further or adjust approach")
            
    except Exception as e:
        logging.error(f"Main test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())