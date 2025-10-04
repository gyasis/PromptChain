#!/usr/bin/env python3
"""
Document Pipeline Fix and Test Script

This script demonstrates the complete document ingestion pipeline 
and fixes the PaperQA2 empty index issue by properly loading documents.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from research_agent.integrations.three_tier_rag import ThreeTierRAG, RAGTier
from research_agent.core.document_pipeline import DocumentPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'document_pipeline_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

logger = logging.getLogger(__name__)


class DocumentPipelineTest:
    """Test and demonstrate document pipeline functionality"""
    
    def __init__(self):
        self.test_results = {
            'pipeline_initialization': False,
            'sample_documents_created': False,
            'rag_system_initialized': False,
            'documents_processed': False,
            'tier1_query_test': False,
            'tier2_query_test': False,
            'tier3_query_test': False,
            'errors': []
        }
        
    async def run_comprehensive_test(self):
        """Run comprehensive document pipeline test"""
        logger.info("🚀 Starting comprehensive document pipeline test...")
        
        try:
            # Step 1: Initialize document pipeline
            await self.test_pipeline_initialization()
            
            # Step 2: Create sample documents
            await self.test_sample_document_creation()
            
            # Step 3: Initialize RAG system
            await self.test_rag_system_initialization()
            
            # Step 4: Process documents through RAG system
            await self.test_document_processing()
            
            # Step 5: Test queries on each tier
            await self.test_tier_queries()
            
            # Step 6: Generate report
            self.generate_test_report()
            
        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            self.test_results['errors'].append(str(e))
            self.generate_test_report()
            raise
    
    async def test_pipeline_initialization(self):
        """Test document pipeline initialization"""
        logger.info("📁 Testing document pipeline initialization...")
        
        try:
            # Create document pipeline with custom config
            config = {
                'inputs_directory': './inputs',
                'papers_directory': './papers',
                'processed_directory': './processed',
                'lightrag_working_dir': './lightrag_data',
                'paperqa2_working_dir': './paperqa2_data',
                'graphrag_working_dir': './graphrag_data'
            }
            
            self.document_pipeline = DocumentPipeline(config)
            
            # Verify directories were created
            required_dirs = [
                self.document_pipeline.inputs_dir,
                self.document_pipeline.papers_dir,
                self.document_pipeline.processed_dir,
                self.document_pipeline.inputs_dir / 'pdf',
                self.document_pipeline.inputs_dir / 'txt'
            ]
            
            for directory in required_dirs:
                if not directory.exists():
                    raise Exception(f"Required directory not created: {directory}")
            
            self.test_results['pipeline_initialization'] = True
            logger.info("✅ Document pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Pipeline initialization failed: {e}")
            self.test_results['errors'].append(f"Pipeline initialization: {e}")
            raise
    
    async def test_sample_document_creation(self):
        """Test creation of sample documents"""
        logger.info("📄 Creating sample research documents...")
        
        try:
            # Create sample documents
            sample_docs = self.document_pipeline.create_sample_documents()
            
            if len(sample_docs) == 0:
                raise Exception("No sample documents were created")
            
            # Verify documents exist and have content
            for doc_path in sample_docs:
                if not doc_path.exists():
                    raise Exception(f"Sample document not found: {doc_path}")
                
                content = doc_path.read_text()
                if len(content) < 100:
                    raise Exception(f"Sample document too short: {doc_path}")
            
            self.test_results['sample_documents_created'] = True
            logger.info(f"✅ Created {len(sample_docs)} sample documents")
            
            # Log document details
            for doc_path in sample_docs:
                size = doc_path.stat().st_size
                logger.info(f"  📄 {doc_path.name}: {size} bytes")
            
        except Exception as e:
            logger.error(f"❌ Sample document creation failed: {e}")
            self.test_results['errors'].append(f"Sample documents: {e}")
            raise
    
    async def test_rag_system_initialization(self):
        """Test RAG system initialization"""
        logger.info("🧠 Initializing RAG system...")
        
        try:
            # Initialize three-tier RAG system
            rag_config = {
                'lightrag_llm_model': 'gpt-4o-mini',
                'lightrag_working_dir': './lightrag_data',
                'paperqa2_llm_model': 'gpt-4o-mini',
                'paperqa2_summary_model': 'gpt-4o-mini',
                'paperqa2_working_dir': './paperqa2_data',
                'paperqa2_temperature': 0.1,
                'graphrag_llm_model': 'gpt-4o-mini',
                'graphrag_working_dir': './graphrag_data'
            }
            
            self.rag_system = ThreeTierRAG(rag_config)
            
            # Check available tiers
            available_tiers = self.rag_system.get_available_tiers()
            
            if len(available_tiers) == 0:
                raise Exception("No RAG tiers available")
            
            self.test_results['rag_system_initialized'] = True
            logger.info(f"✅ RAG system initialized with {len(available_tiers)} tiers")
            
            # Log tier details
            tier_status = self.rag_system.get_tier_status()
            for tier, status in tier_status.items():
                availability = "✅" if status['available'] else "❌"
                logger.info(f"  {availability} {tier.value}")
            
        except Exception as e:
            logger.error(f"❌ RAG system initialization failed: {e}")
            self.test_results['errors'].append(f"RAG initialization: {e}")
            raise
    
    async def test_document_processing(self):
        """Test document processing through all RAG tiers"""
        logger.info("⚙️ Processing documents through RAG system...")
        
        try:
            # Process all documents
            start_time = time.time()
            
            processing_results = await self.document_pipeline.process_all_documents(
                self.rag_system, 
                force_reprocess=True
            )
            
            processing_time = time.time() - start_time
            
            # Verify processing results
            if processing_results['processed'] == 0:
                raise Exception("No documents were successfully processed")
            
            self.test_results['documents_processed'] = True
            
            logger.info(f"✅ Document processing completed in {processing_time:.2f}s")
            logger.info(f"  📊 Total documents: {processing_results['total_documents']}")
            logger.info(f"  ✅ Processed: {processing_results['processed']}")
            logger.info(f"  ❌ Failed: {processing_results['failed']}")
            logger.info(f"  ⚡ Already processed: {processing_results['already_processed']}")
            
            # Log tier processing summary
            logger.info("📈 Tier processing summary:")
            for tier, count in processing_results['tiers_summary'].items():
                logger.info(f"  {tier}: {count} documents")
            
            # Store processing results for later analysis
            self.processing_results = processing_results
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}")
            self.test_results['errors'].append(f"Document processing: {e}")
            raise
    
    async def test_tier_queries(self):
        """Test queries on each RAG tier"""
        test_queries = [
            "What are the main applications of quantum computing?",
            "How do neural network optimization techniques compare?",
            "What are the key principles of AI ethics?"
        ]
        
        for query in test_queries:
            logger.info(f"🔍 Testing query: '{query}'")
            
            try:
                # Test each tier individually
                for tier in [RAGTier.TIER1_LIGHTRAG, RAGTier.TIER2_PAPERQA2, RAGTier.TIER3_GRAPHRAG]:
                    if tier in self.rag_system.get_available_tiers():
                        try:
                            results = await self.rag_system.process_query(query, [tier])
                            result = results[0]
                            
                            if result.success:
                                logger.info(f"  ✅ {tier.value}: {len(result.content)} chars, confidence {result.confidence:.2f}")
                                
                                # Mark successful tier tests
                                if tier == RAGTier.TIER1_LIGHTRAG:
                                    self.test_results['tier1_query_test'] = True
                                elif tier == RAGTier.TIER2_PAPERQA2:
                                    self.test_results['tier2_query_test'] = True
                                elif tier == RAGTier.TIER3_GRAPHRAG:
                                    self.test_results['tier3_query_test'] = True
                            else:
                                logger.warning(f"  ❌ {tier.value}: {result.error}")
                                
                        except Exception as e:
                            logger.error(f"  ❌ {tier.value} query failed: {e}")
                            self.test_results['errors'].append(f"{tier.value} query: {e}")
                
                # Small delay between queries
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Query test failed: {e}")
                self.test_results['errors'].append(f"Query test: {e}")
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("📋 Generating test report...")
        
        # Calculate success metrics
        total_tests = len([k for k in self.test_results.keys() if k != 'errors'])
        successful_tests = sum(1 for k, v in self.test_results.items() if k != 'errors' and v)
        
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Create detailed report
        report = {
            'test_execution_time': datetime.now().isoformat(),
            'overall_success_rate': f"{success_rate:.1f}%",
            'tests_passed': successful_tests,
            'tests_failed': total_tests - successful_tests,
            'total_tests': total_tests,
            'test_results': self.test_results,
            'document_processing_summary': getattr(self, 'processing_results', {}),
            'rag_system_status': self.rag_system.get_tier_status() if hasattr(self, 'rag_system') else {},
            'recommendations': self._generate_recommendations()
        }
        
        # Save report to file
        report_file = f"document_pipeline_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Log summary
        logger.info("=" * 60)
        logger.info("📊 DOCUMENT PIPELINE TEST REPORT")
        logger.info("=" * 60)
        logger.info(f"Overall Success Rate: {success_rate:.1f}%")
        logger.info(f"Tests Passed: {successful_tests}/{total_tests}")
        
        if success_rate >= 80:
            logger.info("🎉 DOCUMENT PIPELINE IS WORKING CORRECTLY!")
        elif success_rate >= 60:
            logger.warning("⚠️ Document pipeline has some issues but basic functionality works")
        else:
            logger.error("❌ Document pipeline has significant issues requiring attention")
        
        # Log specific results
        logger.info("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            if test_name != 'errors':
                status = "✅ PASS" if result else "❌ FAIL"
                logger.info(f"  {status}: {test_name}")
        
        if self.test_results['errors']:
            logger.info(f"\nErrors encountered ({len(self.test_results['errors'])}):")
            for i, error in enumerate(self.test_results['errors'], 1):
                logger.info(f"  {i}. {error}")
        
        logger.info(f"\n📄 Full report saved to: {report_file}")
        logger.info("=" * 60)
        
        return report
    
    def _generate_recommendations(self):
        """Generate recommendations based on test results"""
        recommendations = []
        
        if not self.test_results['pipeline_initialization']:
            recommendations.append("Fix document pipeline initialization - check directory permissions")
        
        if not self.test_results['rag_system_initialized']:
            recommendations.append("Verify RAG system dependencies and API keys")
        
        if not self.test_results['documents_processed']:
            recommendations.append("Check document processing pipeline - ensure RAG tiers can ingest documents")
        
        if not self.test_results['tier2_query_test']:
            recommendations.append("PaperQA2 index may still be empty - verify document indexing")
        
        if not any([self.test_results['tier1_query_test'], 
                   self.test_results['tier2_query_test'], 
                   self.test_results['tier3_query_test']]):
            recommendations.append("No RAG tiers responding to queries - check document ingestion")
        
        if len(self.test_results['errors']) > 3:
            recommendations.append("Multiple errors detected - review logs for systematic issues")
        
        if not recommendations:
            recommendations.append("System appears to be working correctly!")
        
        return recommendations


async def main():
    """Main test execution function"""
    print("🚀 Document Pipeline Fix and Test")
    print("=" * 50)
    print("This script will:")
    print("1. Create document input directories")
    print("2. Generate sample research documents")
    print("3. Initialize the RAG system")
    print("4. Process documents through all tiers")
    print("5. Test queries to verify functionality")
    print("6. Generate a comprehensive report")
    print("=" * 50)
    
    # Check prerequisites
    try:
        import paperqa
        import lightrag
        print("✅ Required dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages and try again")
        return
    
    # Run the test
    test_runner = DocumentPipelineTest()
    await test_runner.run_comprehensive_test()


if __name__ == "__main__":
    asyncio.run(main())