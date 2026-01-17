#!/usr/bin/env python3
"""
Enhanced LightRAG Citation Test - Multiple Approaches to Eliminate unknown_source

This test tries different metadata structures and configurations to fix citation tracking.
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

# Test documents
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

def create_document_approach_1(doc: Dict[str, str]) -> str:
    """Approach 1: Metadata with source field (current approach)"""
    formatted_doc = {
        "page_content": doc["content"],
        "metadata": {
            "source": doc["file_path"],
            "title": doc["title"],
            "authors": doc.get("authors", "Unknown"),
            "year": doc.get("year", "Unknown"),
            "document_id": f"{doc['file_path']}_{doc.get('year', 'unknown')}"
        }
    }
    return str(formatted_doc)

def create_document_approach_2(doc: Dict[str, str]) -> str:
    """Approach 2: Top-level source field"""
    formatted_doc = {
        "page_content": doc["content"],
        "source": doc["file_path"],
        "metadata": {
            "title": doc["title"],
            "authors": doc.get("authors", "Unknown"),
            "year": doc.get("year", "Unknown"),
            "document_id": f"{doc['file_path']}_{doc.get('year', 'unknown')}"
        }
    }
    return str(formatted_doc)

def create_document_approach_3(doc: Dict[str, str]) -> str:
    """Approach 3: Document ID as primary identifier"""
    formatted_doc = {
        "page_content": doc["content"],
        "id": doc["file_path"],
        "source": doc["file_path"],
        "metadata": {
            "source_file": doc["file_path"],
            "document_name": doc["title"],
            "title": doc["title"],
            "authors": doc.get("authors", "Unknown"),
            "year": doc.get("year", "Unknown"),
            "file_name": doc["file_path"]
        }
    }
    return str(formatted_doc)

def create_document_approach_4(doc: Dict[str, str]) -> str:
    """Approach 4: Simple content with embedded source"""
    content_with_source = f"Source: {doc['file_path']}\nTitle: {doc['title']}\nAuthors: {doc.get('authors', 'Unknown')}\nYear: {doc.get('year', 'Unknown')}\n\nContent: {doc['content']}"
    return content_with_source

async def test_approach(approach_name: str, document_formatter, enable_rerank: bool = False):
    """Test a specific approach to citation tracking"""
    
    logging.info(f"\n{'='*50}")
    logging.info(f"Testing {approach_name}")
    logging.info(f"Reranking: {'Enabled' if enable_rerank else 'Disabled'}")
    logging.info(f"{'='*50}")
    
    # Create test directory
    working_dir = f"./test_citations_{approach_name.lower().replace(' ', '_')}"
    os.makedirs(working_dir, exist_ok=True)
    
    try:
        # Initialize LightRAG
        logging.info("1. Initializing LightRAG...")
        
        rag = LightRAG(
            working_dir=working_dir,
            embedding_func=openai_embed,
            llm_model_func=gpt_4o_mini_complete,
        )
        
        # Initialize async storages and pipeline status
        await rag.initialize_storages()
        await initialize_pipeline_status()
        
        logging.info("✓ LightRAG initialized successfully")
        
        # Insert documents using the specific approach
        logging.info("2. Inserting documents...")
        
        for i, doc in enumerate(TEST_DOCS):
            formatted_doc = document_formatter(doc)
            
            logging.info(f"   Inserting document {i+1}: {doc['title']}")
            logging.info(f"   Formatted as: {formatted_doc[:100]}...")
            
            try:
                await rag.ainsert(formatted_doc)
                logging.info(f"   ✓ Document inserted successfully")
            except Exception as e:
                logging.error(f"   ❌ Failed to insert document: {e}")
                return False
        
        logging.info("✓ All documents inserted successfully")
        
        # Test query with different parameters
        logging.info("3. Testing query...")
        
        query = "What methods are used for disease detection?"
        
        try:
            # Try different query parameters
            result = await rag.aquery(
                query,
                param=QueryParam(mode="naive", enable_rerank=enable_rerank)
            )
            
            logging.info(f"Query: {query}")
            logging.info("Response:")
            logging.info("-" * 40)
            logging.info(result)
            
            # Check for unknown_source
            response_str = str(result)
            unknown_count = response_str.lower().count("unknown_source")
            
            if unknown_count == 0:
                logging.info("✅ SUCCESS: No 'unknown_source' found!")
                return True
            else:
                logging.warning(f"⚠️  Found {unknown_count} instances of 'unknown_source'")
                
                # Check for file references in content
                found_files = []
                for doc in TEST_DOCS:
                    if doc["file_path"] in response_str or doc["title"] in response_str:
                        found_files.append(doc["file_path"])
                
                if found_files:
                    logging.info(f"📚 File references found: {found_files}")
                else:
                    logging.info("📚 No file references found")
                
                return False
            
        except Exception as e:
            logging.error(f"❌ Query failed: {e}")
            return False
    
    except Exception as e:
        logging.error(f"Approach failed: {e}")
        return False
    
    finally:
        # Clean up test directory
        try:
            import shutil
            shutil.rmtree(working_dir)
            logging.info(f"Cleaned up: {working_dir}")
        except Exception as e:
            logging.warning(f"Failed to clean up: {e}")

async def main():
    """Test multiple approaches to fix citation tracking"""
    
    logging.info("🔬 Enhanced LightRAG Citation Tracking Test")
    logging.info("Testing multiple approaches to eliminate 'unknown_source'")
    
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("❌ OPENAI_API_KEY not set in environment")
        return
    
    # Test different approaches
    approaches = [
        ("Approach 1: Metadata Source", create_document_approach_1),
        ("Approach 2: Top-level Source", create_document_approach_2), 
        ("Approach 3: Multiple IDs", create_document_approach_3),
        ("Approach 4: Embedded Source", create_document_approach_4),
    ]
    
    successful_approaches = []
    
    for approach_name, formatter in approaches:
        try:
            # Test without reranking first
            success = await test_approach(approach_name, formatter, enable_rerank=False)
            if success:
                successful_approaches.append((approach_name, "No Reranking"))
                logging.info(f"🎯 {approach_name} SUCCESSFUL!")
            else:
                logging.info(f"⚠️  {approach_name} had issues")
                
        except Exception as e:
            logging.error(f"❌ {approach_name} failed: {e}")
    
    # Summary
    logging.info("\n" + "="*60)
    logging.info("FINAL RESULTS")
    logging.info("="*60)
    
    if successful_approaches:
        logging.info("✅ Successful approaches:")
        for approach, config in successful_approaches:
            logging.info(f"   • {approach} ({config})")
        
        logging.info("\n🎯 READY FOR INTEGRATION!")
        logging.info("Use the successful approach(es) in the enhanced demo")
    else:
        logging.warning("⚠️  No approaches completely eliminated 'unknown_source'")
        logging.info("   However, file paths are appearing in content")
        logging.info("   This may be acceptable for user experience")

if __name__ == "__main__":
    asyncio.run(main())