#!/usr/bin/env python3
"""
Test PDF Manager - Organized Storage and Download System
"""

import asyncio
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add the research agent to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from research_agent.utils.pdf_manager import PDFManager

async def test_pdf_manager():
    """Test comprehensive PDF manager functionality"""
    logger.info("🔬 Testing PDF Manager - Organized Storage System")
    
    # Create temporary directory for testing
    temp_dir = Path(tempfile.mkdtemp(prefix="pdf_manager_test_"))
    logger.info(f"Test directory: {temp_dir}")
    
    try:
        # Initialize PDF Manager
        config = {
            'max_retries': 2,
            'timeout': 30,
            'max_concurrent_downloads': 3
        }
        
        pdf_manager = PDFManager(base_path=str(temp_dir / "papers"), config=config)
        
        # Test storage structure
        logger.info("\n📁 Testing Storage Structure")
        expected_dirs = [
            temp_dir / "papers" / "arxiv",
            temp_dir / "papers" / "pubmed", 
            temp_dir / "papers" / "sci_hub",
            temp_dir / "papers" / "metadata"
        ]
        
        for expected_dir in expected_dirs:
            if expected_dir.exists():
                logger.info(f"✅ Directory exists: {expected_dir}")
            else:
                logger.error(f"❌ Directory missing: {expected_dir}")
                return False
        
        # Test with real ArXiv papers
        test_papers = [
            {
                'id': 'arxiv_2301.07041',
                'title': 'Constitutional AI: Harmlessness from AI Feedback',
                'authors': ['Bai, Yuntao', 'Kadavath, Saurav', 'Kundu, Sandipan'],
                'source': 'arxiv',
                'publication_year': 2023,
                'pdf_url': 'https://arxiv.org/pdf/2301.07041.pdf',
                'metadata': {'arxiv_id': '2301.07041'}
            },
            {
                'id': 'arxiv_1706.03762',
                'title': 'Attention Is All You Need',
                'authors': ['Vaswani, Ashish', 'Shazeer, Noam'],
                'source': 'arxiv',
                'publication_year': 2017,
                'pdf_url': 'https://arxiv.org/pdf/1706.03762.pdf',
                'metadata': {'arxiv_id': '1706.03762'}
            }
        ]
        
        # Test individual download
        logger.info("\n📥 Testing Individual PDF Download")
        paper = test_papers[0]
        success, file_path, info = await pdf_manager.download_pdf(
            paper['pdf_url'], 
            paper
        )
        
        if success:
            logger.info(f"✅ Download successful: {file_path}")
            logger.info(f"   File size: {info.get('file_size', 0)} bytes")
            logger.info(f"   Status: {info.get('status')}")
            
            # Verify file exists and is valid
            if Path(file_path).exists():
                logger.info("✅ File exists on disk")
                
                # Check if it's a valid PDF
                with open(file_path, 'rb') as f:
                    header = f.read(5)
                    if header.startswith(b'%PDF-'):
                        logger.info("✅ Valid PDF file confirmed")
                    else:
                        logger.error("❌ Not a valid PDF file")
                        return False
            else:
                logger.error("❌ File not found on disk")
                return False
        else:
            logger.error(f"❌ Download failed: {info}")
            return False
        
        # Test duplicate detection
        logger.info("\n🔍 Testing Duplicate Detection")
        success2, file_path2, info2 = await pdf_manager.download_pdf(
            paper['pdf_url'], 
            paper
        )
        
        if success2 and info2.get('status') == 'already_exists':
            logger.info("✅ Duplicate detection working")
        else:
            logger.warning("⚠️ Duplicate detection may not be working")
        
        # Test batch download
        logger.info("\n📦 Testing Batch Download")
        batch_results = await pdf_manager.download_papers_batch(test_papers)
        
        logger.info(f"Batch results:")
        logger.info(f"  Successful: {len(batch_results['successful'])}")
        logger.info(f"  Already exists: {len(batch_results['already_exists'])}")
        logger.info(f"  Failed: {len(batch_results['failed'])}")
        logger.info(f"  Total size: {batch_results['total_size_bytes']} bytes")
        
        # Test storage statistics
        logger.info("\n📊 Testing Storage Statistics")
        stats = pdf_manager.get_storage_statistics()
        
        logger.info(f"Storage statistics:")
        logger.info(f"  Total PDFs: {stats['total_pdfs']}")
        logger.info(f"  Total size: {stats['total_size_mb']} MB")
        logger.info(f"  By source: {stats['by_source']}")
        logger.info(f"  By year: {stats['by_year']}")
        logger.info(f"  Download stats: {stats['download_stats']}")
        
        # Test search functionality
        logger.info("\n🔍 Testing PDF Search")
        search_results = pdf_manager.search_pdfs(query="constitutional", limit=10)
        
        logger.info(f"Search results for 'constitutional': {len(search_results)} found")
        for result in search_results:
            logger.info(f"  - {result['title'][:50]}... ({result['source']})")
        
        # Test organized storage paths
        logger.info("\n📂 Testing Organized Storage Structure")
        
        # Check if files are in correct year directories
        arxiv_2023_dir = temp_dir / "papers" / "arxiv" / "2023"
        arxiv_2017_dir = temp_dir / "papers" / "arxiv" / "2017"
        
        if arxiv_2023_dir.exists():
            files_2023 = list(arxiv_2023_dir.glob("*.pdf"))
            logger.info(f"✅ 2023 ArXiv directory has {len(files_2023)} PDFs")
        
        if arxiv_2017_dir.exists():
            files_2017 = list(arxiv_2017_dir.glob("*.pdf"))
            logger.info(f"✅ 2017 ArXiv directory has {len(files_2017)} PDFs")
        
        # Test metadata database
        logger.info("\n💾 Testing Metadata Database")
        db_path = temp_dir / "papers" / "pdf_metadata.db"
        if db_path.exists():
            logger.info(f"✅ Metadata database exists: {db_path}")
            logger.info(f"   Database size: {db_path.stat().st_size} bytes")
        else:
            logger.error("❌ Metadata database not found")
            return False
        
        # Test error handling with invalid URL
        logger.info("\n⚠️ Testing Error Handling")
        invalid_paper = {
            'id': 'invalid_test',
            'title': 'Invalid Paper Test',
            'source': 'other',
            'publication_year': 2024,
            'pdf_url': 'https://invalid-domain-xyz.com/fake.pdf'
        }
        
        success_invalid, _, info_invalid = await pdf_manager.download_pdf(
            invalid_paper['pdf_url'],
            invalid_paper
        )
        
        if not success_invalid and 'error' in info_invalid:
            logger.info("✅ Error handling working correctly")
        else:
            logger.warning("⚠️ Error handling may need improvement")
        
        # Generate final report
        final_stats = pdf_manager.get_storage_statistics()
        
        logger.info("\n" + "="*70)
        logger.info("📋 PDF MANAGER TEST REPORT")
        logger.info("="*70)
        logger.info(f"Storage Path: {final_stats['storage_path']}")
        logger.info(f"Total PDFs Managed: {final_stats['total_pdfs']}")
        logger.info(f"Total Storage Used: {final_stats['total_size_mb']} MB")
        logger.info(f"Sources: {list(final_stats['by_source'].keys())}")
        logger.info(f"Years: {list(final_stats['by_year'].keys())}")
        
        download_stats = final_stats['download_stats']
        logger.info(f"\nDownload Statistics:")
        logger.info(f"  Attempted: {download_stats['total_attempted']}")
        logger.info(f"  Successful: {download_stats['successful']}")
        logger.info(f"  Failed: {download_stats['failed']}")
        logger.info(f"  Already Existed: {download_stats['already_exists']}")
        logger.info(f"  Total Downloaded: {download_stats['total_size_bytes']} bytes")
        
        logger.info("="*70)
        logger.info("✅ PDF Manager Test PASSED")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ PDF Manager test failed: {e}")
        return False
    
    finally:
        # Cleanup (optional - comment out to inspect results)
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up test directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup: {e}")

async def main():
    """Main test execution"""
    success = await test_pdf_manager()
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    print(f"\n🎯 PDF Manager Test {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)