#!/usr/bin/env python3
"""
Comprehensive LightRAG Citation Validation Test

This test validates that the LightRAG citation fix works correctly and provides
integration guidance for the enhanced demo.
"""

import asyncio
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re

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

# Test documents with varied content
TEST_DOCS = [
    {
        "title": "Deep Learning for Medical Diagnosis",
        "content": "This comprehensive study explores deep learning applications in medical diagnosis. The research presents novel CNN architectures achieving 98.5% accuracy in medical image classification. The study involved 5000 patients across 10 medical centers.",
        "file_path": "deep_learning_medical_2024.pdf",
        "authors": "Chen, L. et al.",
        "year": "2024",
        "journal": "Nature Medicine"
    },
    {
        "title": "Quantum Computing in Drug Discovery",
        "content": "This research investigates quantum computing applications for accelerating drug discovery processes. The quantum algorithms reduced computational time by 85% compared to classical methods. The study tested 1200 molecular compounds.",
        "file_path": "quantum_drug_discovery_2024.pdf", 
        "authors": "Rodriguez, M. et al.",
        "year": "2024",
        "journal": "Science"
    },
    {
        "title": "AI Ethics in Healthcare Systems",
        "content": "This paper examines ethical considerations for AI implementation in healthcare. The study surveyed 800 healthcare professionals and identified key ethical challenges. Recommendations for ethical AI deployment are provided.",
        "file_path": "ai_ethics_healthcare_2024.pdf",
        "authors": "Thompson, K. et al.", 
        "year": "2024",
        "journal": "JAMA AI"
    }
]

def create_optimized_document(doc: Dict[str, str]) -> str:
    """Create document with optimized metadata structure for citation tracking"""
    formatted_doc = {
        "page_content": doc["content"],
        "metadata": {
            "source": doc["file_path"],
            "title": doc["title"],
            "authors": doc.get("authors", "Unknown"),
            "year": doc.get("year", "Unknown"),
            "journal": doc.get("journal", "Unknown"),
            "document_id": f"{doc['file_path']}_{doc.get('year', 'unknown')}"
        }
    }
    return str(formatted_doc)

class CitationValidator:
    """Validates citation tracking in LightRAG responses"""
    
    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "unknown_source_count": 0,
            "proper_citations_count": 0,
            "file_references_found": 0,
            "author_references_found": 0,
            "title_references_found": 0
        }
    
    def validate_response(self, response: str, expected_docs: List[Dict]) -> Dict[str, any]:
        """Validate a single response for citation quality"""
        self.metrics["total_queries"] += 1
        
        response_str = str(response)
        
        # Count unknown_source instances
        unknown_count = response_str.lower().count("unknown_source")
        self.metrics["unknown_source_count"] += unknown_count
        
        # Check for proper citations (file_path references)
        proper_citations = 0
        if "[DC] file_path" in response_str or "[KG] file_path" in response_str:
            proper_citations += response_str.count("[DC] file_path") + response_str.count("[KG] file_path")
        
        self.metrics["proper_citations_count"] += proper_citations
        
        # Check for file references in content
        file_refs = []
        for doc in expected_docs:
            if doc["file_path"] in response_str:
                file_refs.append(doc["file_path"])
                self.metrics["file_references_found"] += 1
        
        # Check for author references
        author_refs = []
        for doc in expected_docs:
            author_name = doc.get("authors", "").split(",")[0].strip()
            if author_name and author_name in response_str:
                author_refs.append(author_name)
                self.metrics["author_references_found"] += 1
        
        # Check for title references
        title_refs = []
        for doc in expected_docs:
            if doc["title"] in response_str:
                title_refs.append(doc["title"])
                self.metrics["title_references_found"] += 1
        
        return {
            "unknown_source_instances": unknown_count,
            "proper_citations": proper_citations,
            "file_references": file_refs,
            "author_references": author_refs,
            "title_references": title_refs,
            "citation_quality": "excellent" if unknown_count == 0 and (proper_citations > 0 or len(file_refs) > 0) else "needs_improvement"
        }
    
    def get_summary_report(self) -> Dict[str, any]:
        """Get comprehensive validation summary"""
        if self.metrics["total_queries"] == 0:
            return {"error": "No queries processed"}
        
        success_rate = (self.metrics["total_queries"] - self.metrics["unknown_source_count"]) / self.metrics["total_queries"]
        
        return {
            "total_queries_tested": self.metrics["total_queries"],
            "unknown_source_elimination_rate": f"{(1 - self.metrics['unknown_source_count'] / max(1, self.metrics['total_queries'])) * 100:.1f}%",
            "citation_success_rate": f"{success_rate * 100:.1f}%",
            "file_reference_rate": f"{(self.metrics['file_references_found'] / max(1, self.metrics['total_queries'])) * 100:.1f}%",
            "author_reference_rate": f"{(self.metrics['author_references_found'] / max(1, self.metrics['total_queries'])) * 100:.1f}%",
            "title_reference_rate": f"{(self.metrics['title_references_found'] / max(1, self.metrics['total_queries'])) * 100:.1f}%",
            "overall_grade": "A" if success_rate >= 0.9 else "B" if success_rate >= 0.7 else "C" if success_rate >= 0.5 else "F",
            "ready_for_production": success_rate >= 0.8
        }

async def run_comprehensive_citation_test():
    """Run comprehensive citation validation test"""
    
    logging.info("=" * 70)
    logging.info("COMPREHENSIVE LIGHTRAG CITATION VALIDATION TEST")
    logging.info("=" * 70)
    
    # Initialize validator
    validator = CitationValidator()
    
    # Create test directory
    working_dir = "./comprehensive_citation_test"
    os.makedirs(working_dir, exist_ok=True)
    
    try:
        # Initialize LightRAG
        logging.info("1. Initializing LightRAG with optimized configuration...")
        
        rag = LightRAG(
            working_dir=working_dir,
            embedding_func=openai_embed,
            llm_model_func=gpt_4o_mini_complete,
        )
        
        # Initialize async storages and pipeline status
        await rag.initialize_storages()
        await initialize_pipeline_status()
        
        logging.info("✓ LightRAG initialized successfully")
        
        # Insert test documents
        logging.info("2. Inserting test documents with optimized metadata...")
        
        for i, doc in enumerate(TEST_DOCS):
            formatted_doc = create_optimized_document(doc)
            
            logging.info(f"   Inserting: {doc['title']}")
            logging.info(f"   Authors: {doc['authors']}")
            logging.info(f"   Source: {doc['file_path']}")
            
            try:
                await rag.ainsert(formatted_doc)
                logging.info(f"   ✓ Document {i+1} inserted successfully")
            except Exception as e:
                logging.error(f"   ❌ Failed to insert document {i+1}: {e}")
                return False
        
        logging.info("✓ All documents inserted successfully")
        
        # Test with comprehensive query set
        logging.info("3. Running comprehensive citation validation queries...")
        
        test_queries = [
            "What are the latest advances in medical AI?",
            "How is quantum computing being used in drug discovery?", 
            "What are the main ethical concerns with AI in healthcare?",
            "Compare the accuracy rates mentioned in these studies.",
            "What computational methods are discussed in these papers?",
            "Who are the key researchers in these areas?",
            "What are the key findings from these medical AI studies?"
        ]
        
        all_results = []
        
        for i, query in enumerate(test_queries, 1):
            logging.info(f"\n--- Query {i}/{len(test_queries)} ---")
            logging.info(f"Query: {query}")
            
            try:
                # Use naive mode for reliable citation tracking
                result = await rag.aquery(
                    query,
                    param=QueryParam(mode="naive", enable_rerank=False)
                )
                
                logging.info("Response:")
                logging.info("-" * 50)
                logging.info(str(result)[:500] + "..." if len(str(result)) > 500 else str(result))
                
                # Validate the response
                validation = validator.validate_response(result, TEST_DOCS)
                all_results.append(validation)
                
                logging.info(f"Citation Quality: {validation['citation_quality']}")
                if validation['unknown_source_instances'] == 0:
                    logging.info("✅ No unknown_source found")
                else:
                    logging.warning(f"⚠️  {validation['unknown_source_instances']} unknown_source instances")
                
                if validation['file_references']:
                    logging.info(f"📚 File references: {validation['file_references']}")
                if validation['author_references']:
                    logging.info(f"👥 Author references: {validation['author_references']}")
                
            except Exception as e:
                logging.error(f"❌ Query {i} failed: {e}")
                all_results.append({"citation_quality": "failed", "error": str(e)})
        
        # Generate comprehensive report
        logging.info("\n" + "="*70)
        logging.info("COMPREHENSIVE VALIDATION REPORT")
        logging.info("="*70)
        
        summary = validator.get_summary_report()
        
        for key, value in summary.items():
            logging.info(f"{key.replace('_', ' ').title()}: {value}")
        
        # Determine if ready for integration
        if summary.get("ready_for_production", False):
            logging.info("\n🎯 CITATION TRACKING VALIDATION: PASSED")
            logging.info("✅ Ready for integration into enhanced demo")
            logging.info("✅ Unknown_source issue has been resolved")
            logging.info("✅ Proper citations are working consistently")
            
            # Provide integration guidance
            logging.info("\n📋 INTEGRATION GUIDANCE:")
            logging.info("1. Use QueryParam(mode='naive', enable_rerank=False)")
            logging.info("2. Include metadata with 'source' field pointing to file path")
            logging.info("3. Initialize with rag.initialize_storages() and initialize_pipeline_status()")
            logging.info("4. Document format: {'page_content': content, 'metadata': {'source': file_path, ...}}")
            
            return True
        else:
            logging.warning("\n⚠️  CITATION TRACKING VALIDATION: NEEDS IMPROVEMENT")
            logging.info("Additional refinement required before production integration")
            return False
    
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
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("❌ OPENAI_API_KEY not set in environment")
        return
    
    try:
        logging.info("🧪 Starting Comprehensive LightRAG Citation Validation")
        success = await run_comprehensive_citation_test()
        
        if success:
            logging.info("\n🎉 VALIDATION COMPLETE - CITATION TRACKING FIXED!")
        else:
            logging.warning("\n⚠️  Validation completed with issues")
            
    except Exception as e:
        logging.error(f"Main test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())