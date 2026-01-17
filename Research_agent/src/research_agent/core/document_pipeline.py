#!/usr/bin/env python3
"""
Document Pipeline for Research Agent RAG System

Handles document ingestion, indexing, and management across all RAG tiers.
Ensures documents are properly loaded into LightRAG, PaperQA2, and GraphRAG.
"""

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class DocumentIngestionResult:
    """Result of document ingestion process"""
    success: bool
    document_path: str
    tiers_processed: List[str]
    errors: List[str]
    metadata: Dict[str, Any]
    processing_time: float


class DocumentPipeline:
    """
    Manages document ingestion and indexing across all RAG tiers
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the document pipeline"""
        self.config = config or {}
        
        # Base paths for document storage
        self.papers_dir = Path(self.config.get('papers_directory', './papers'))
        self.inputs_dir = Path(self.config.get('inputs_directory', './inputs'))
        self.processed_dir = Path(self.config.get('processed_directory', './processed'))
        
        # RAG system directories
        self.lightrag_dir = Path(self.config.get('lightrag_working_dir', './lightrag_data'))
        self.paperqa2_dir = Path(self.config.get('paperqa2_working_dir', './paperqa2_data'))
        self.graphrag_dir = Path(self.config.get('graphrag_working_dir', './graphrag_data'))
        
        # Create directory structure
        self._setup_directories()
        
        # Track processed documents
        self.processed_docs_file = self.processed_dir / 'processed_documents.json'
        self.processed_docs = self._load_processed_docs()
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'already_processed': 0
        }
        
    def _setup_directories(self):
        """Create necessary directory structure"""
        directories = [
            self.papers_dir,
            self.inputs_dir,
            self.processed_dir,
            self.lightrag_dir,
            self.paperqa2_dir,
            self.graphrag_dir,
            self.graphrag_dir / 'input',
            
            # Input subdirectories by source
            self.inputs_dir / 'pdf',
            self.inputs_dir / 'txt',
            self.inputs_dir / 'arxiv',
            self.inputs_dir / 'pubmed',
            self.inputs_dir / 'manual',
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Create README files
        self._create_readme_files()
        
        logger.info(f"Document pipeline directories initialized at: {self.papers_dir}")
        
    def _create_readme_files(self):
        """Create informative README files in input directories"""
        readme_content = {
            self.inputs_dir: """# Document Input Directory

Place research documents here for automatic processing:

## Supported Formats:
- PDF files (.pdf) - Research papers, reports
- Text files (.txt) - Preprocessed content
- Any text content for RAG processing

## Subdirectories:
- pdf/ - Place PDF documents here
- txt/ - Place text documents here  
- arxiv/ - ArXiv papers
- pubmed/ - PubMed papers
- manual/ - Manually curated documents

## Processing:
Documents placed here will be automatically:
1. Parsed and indexed by LightRAG
2. Processed by PaperQA2 for Q&A extraction
3. Converted to knowledge graphs by GraphRAG
4. Made available for research queries

The system monitors this directory and processes new files automatically.
""",
            
            self.papers_dir: """# Papers Storage Directory

This directory contains organized research papers downloaded by the system:

## Structure:
- arxiv/YEAR/ - ArXiv papers by publication year
- pubmed/YEAR/ - PubMed papers by publication year  
- sci_hub/YEAR/ - Papers from Sci-Hub by year
- other/YEAR/ - Other sources by year

## Metadata:
- pdf_metadata.db - SQLite database tracking all papers
- cache/ - Download cache and temporary files
- metadata/ - Paper metadata and bibliographic information

Papers here are automatically processed by the RAG system.
""",
            
            self.processed_dir: """# Processed Documents Directory

Contains tracking information for processed documents:

- processed_documents.json - List of successfully processed documents
- processing_logs/ - Detailed processing logs
- failed/ - Documents that failed processing with error details

This helps avoid reprocessing the same documents multiple times.
"""
        }
        
        for directory, content in readme_content.items():
            readme_file = directory / 'README.md'
            if not readme_file.exists():
                readme_file.write_text(content)
    
    def _load_processed_docs(self) -> Dict[str, Dict[str, Any]]:
        """Load list of previously processed documents"""
        if self.processed_docs_file.exists():
            try:
                with open(self.processed_docs_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load processed docs: {e}")
        return {}
    
    def _save_processed_docs(self):
        """Save list of processed documents"""
        try:
            with open(self.processed_docs_file, 'w') as f:
                json.dump(self.processed_docs, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save processed docs: {e}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file for duplicate detection"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def discover_documents(self) -> List[Path]:
        """Discover all documents available for processing"""
        document_patterns = ['*.pdf', '*.txt', '*.md']
        discovered = []
        
        # Search in input directories
        search_dirs = [
            self.inputs_dir,
            self.papers_dir,
            Path('./test_papers'),  # Include test papers
        ]
        
        for search_dir in search_dirs:
            if search_dir.exists():
                for pattern in document_patterns:
                    discovered.extend(search_dir.rglob(pattern))
        
        # Filter out system files and directories
        filtered = []
        for doc in discovered:
            if not any(part.startswith('.') for part in doc.parts):
                if doc.is_file() and doc.stat().st_size > 100:  # At least 100 bytes
                    filtered.append(doc)
        
        logger.info(f"Discovered {len(filtered)} documents for processing")
        return filtered
    
    async def process_document(
        self, 
        doc_path: Path, 
        rag_system: Any,
        force_reprocess: bool = False
    ) -> DocumentIngestionResult:
        """Process a single document through all RAG tiers"""
        start_time = datetime.now()
        
        try:
            # Calculate file hash for duplicate detection
            file_hash = self._calculate_file_hash(doc_path)
            
            # Check if already processed
            if not force_reprocess and file_hash in self.processed_docs:
                self.stats['already_processed'] += 1
                return DocumentIngestionResult(
                    success=True,
                    document_path=str(doc_path),
                    tiers_processed=self.processed_docs[file_hash].get('tiers', []),
                    errors=[],
                    metadata={'status': 'already_processed', 'file_hash': file_hash},
                    processing_time=0.0
                )
            
            logger.info(f"Processing document: {doc_path}")
            
            tiers_processed = []
            errors = []
            
            # Prepare document content/path for each tier
            doc_content = self._prepare_document_content(doc_path)
            
            # Process through available RAG tiers
            available_tiers = rag_system.get_available_tiers()
            
            for tier in available_tiers:
                try:
                    success = await rag_system.insert_documents([doc_content], tier)
                    if success:
                        tiers_processed.append(tier.value)
                        logger.info(f"✅ Document processed by {tier.value}")
                    else:
                        errors.append(f"Failed to process in {tier.value}")
                        logger.warning(f"❌ Document failed processing in {tier.value}")
                except Exception as e:
                    error_msg = f"Error in {tier.value}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            # Record processing result
            processing_time = (datetime.now() - start_time).total_seconds()
            success = len(tiers_processed) > 0
            
            if success:
                self.processed_docs[file_hash] = {
                    'path': str(doc_path),
                    'tiers': tiers_processed,
                    'processed_at': datetime.now().isoformat(),
                    'file_size': doc_path.stat().st_size,
                    'processing_time': processing_time
                }
                self._save_processed_docs()
                self.stats['successful'] += 1
            else:
                self.stats['failed'] += 1
            
            self.stats['total_processed'] += 1
            
            return DocumentIngestionResult(
                success=success,
                document_path=str(doc_path),
                tiers_processed=tiers_processed,
                errors=errors,
                metadata={
                    'file_hash': file_hash,
                    'file_size': doc_path.stat().st_size,
                    'available_tiers': [t.value for t in available_tiers]
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            self.stats['failed'] += 1
            self.stats['total_processed'] += 1
            
            logger.error(f"Failed to process document {doc_path}: {e}")
            return DocumentIngestionResult(
                success=False,
                document_path=str(doc_path),
                tiers_processed=[],
                errors=[str(e)],
                metadata={},
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _prepare_document_content(self, doc_path: Path) -> Union[str, Path]:
        """Prepare document content for RAG processing"""
        if doc_path.suffix.lower() == '.pdf':
            # For PDFs, return the path - RAG systems will handle PDF parsing
            return doc_path
        elif doc_path.suffix.lower() in ['.txt', '.md']:
            # For text files, we can return content or path
            return doc_path
        else:
            # Try to read as text
            try:
                return doc_path.read_text(encoding='utf-8')
            except:
                # Fallback to path
                return doc_path
    
    async def process_all_documents(self, rag_system: Any, force_reprocess: bool = False) -> Dict[str, Any]:
        """Process all discovered documents through the RAG system"""
        logger.info("Starting bulk document processing...")
        
        documents = self.discover_documents()
        if not documents:
            logger.warning("No documents found for processing")
            return {
                'total_documents': 0,
                'processed': 0,
                'failed': 0,
                'already_processed': 0,
                'results': []
            }
        
        results = []
        semaphore = asyncio.Semaphore(3)  # Limit concurrent processing
        
        async def process_with_limit(doc_path):
            async with semaphore:
                return await self.process_document(doc_path, rag_system, force_reprocess)
        
        # Process all documents
        tasks = [process_with_limit(doc) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Compile results
        successful = [r for r in results if isinstance(r, DocumentIngestionResult) and r.success]
        failed = [r for r in results if isinstance(r, DocumentIngestionResult) and not r.success]
        exceptions = [r for r in results if isinstance(r, Exception)]
        
        summary = {
            'total_documents': len(documents),
            'processed': len(successful),
            'failed': len(failed) + len(exceptions),
            'already_processed': self.stats['already_processed'],
            'results': [r for r in results if isinstance(r, DocumentIngestionResult)],
            'processing_stats': self.stats,
            'tiers_summary': self._compile_tiers_summary(successful)
        }
        
        logger.info(f"Document processing complete: {summary['processed']} successful, "
                   f"{summary['failed']} failed, {summary['already_processed']} already processed")
        
        return summary
    
    def _compile_tiers_summary(self, successful_results: List[DocumentIngestionResult]) -> Dict[str, int]:
        """Compile summary of tier processing results"""
        tier_counts = {}
        for result in successful_results:
            for tier in result.tiers_processed:
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
        return tier_counts
    
    def create_sample_documents(self) -> List[Path]:
        """Create sample research documents for testing"""
        sample_docs = []
        
        # Sample research paper abstracts and content
        samples = [
            {
                'filename': 'quantum_computing_review.txt',
                'content': '''Quantum Computing: A Comprehensive Review of Current State and Future Prospects

Abstract:
Quantum computing represents a paradigm shift in computational capability, leveraging quantum mechanical phenomena such as superposition and entanglement to process information in fundamentally new ways. This paper provides a comprehensive review of the current state of quantum computing technology, examining both theoretical foundations and practical implementations.

Introduction:
Classical computers process information using bits that exist in definite states of 0 or 1. Quantum computers, by contrast, utilize quantum bits (qubits) that can exist in superposition states, enabling parallel computation across multiple solution paths simultaneously.

Key Findings:
1. Quantum supremacy has been demonstrated in specific computational tasks
2. Current quantum systems face significant challenges from decoherence and noise
3. Quantum error correction protocols show promise for fault-tolerant computation
4. Applications in cryptography, optimization, and simulation show immediate potential

Quantum Algorithms:
Several quantum algorithms have been developed that show exponential speedup over classical counterparts:
- Shor's algorithm for integer factorization
- Grover's algorithm for unstructured search
- Variational quantum eigensolver (VQE) for quantum chemistry

Conclusion:
While practical quantum computing faces significant technical challenges, continued advances in hardware, error correction, and algorithm development suggest transformative potential across multiple domains.

References:
[1] Nielsen, M. A., & Chuang, I. L. (2010). Quantum computation and quantum information.
[2] Preskill, J. (2018). Quantum computing in the NISQ era and beyond.
[3] Arute, F., et al. (2019). Quantum supremacy using a programmable superconducting processor.
'''
            },
            {
                'filename': 'machine_learning_optimization.txt',
                'content': '''Machine Learning Optimization Techniques: A Survey of Modern Approaches

Abstract:
This survey examines contemporary optimization techniques used in machine learning, focusing on gradient-based methods, evolutionary algorithms, and hybrid approaches. We analyze their effectiveness across different problem domains and computational constraints.

1. Introduction
Optimization lies at the heart of machine learning, determining how models learn from data and adapt to new information. This paper surveys modern optimization techniques and their applications.

2. Gradient-Based Methods
2.1 Stochastic Gradient Descent (SGD)
SGD remains the foundation of most neural network training, despite its limitations in handling non-convex optimization landscapes.

2.2 Adaptive Learning Rate Methods
- Adam: Combines momentum with adaptive learning rates
- RMSprop: Addresses diminishing gradients in recurrent networks
- AdaGrad: Adapts learning rates based on historical gradients

3. Second-Order Methods
Newton's method and quasi-Newton approaches like L-BFGS offer faster convergence but require significant computational resources.

4. Evolutionary Algorithms
4.1 Genetic Algorithms
Useful for discrete optimization problems and neural architecture search.

4.2 Differential Evolution
Effective for continuous optimization in high-dimensional spaces.

5. Hybrid Approaches
Combining gradient-based and evolutionary methods can leverage the strengths of both paradigms.

6. Applications
- Neural network training
- Hyperparameter optimization
- Feature selection
- Model architecture search

7. Conclusion
The choice of optimization method depends heavily on problem characteristics, computational constraints, and convergence requirements. Future research directions include quantum-inspired optimization and meta-learning approaches.
'''
            },
            {
                'filename': 'neural_network_architectures.txt',
                'content': '''Deep Neural Network Architectures: Evolution and Design Principles

Abstract:
Neural network architectures have evolved significantly since the inception of the perceptron. This paper traces the development of deep learning architectures and examines design principles that lead to effective models.

1. Historical Development
1.1 Early Networks
- Perceptron (1957): Single-layer linear classifier
- Multi-layer perceptron: Introduction of hidden layers
- Backpropagation (1986): Enabled training of deep networks

1.2 Deep Learning Renaissance
The resurgence of neural networks was driven by:
- Increased computational power (GPUs)
- Large datasets (ImageNet, etc.)
- Improved algorithms (ReLU, dropout, batch normalization)

2. Fundamental Architectures
2.1 Feedforward Networks
Dense layers with full connectivity between adjacent layers.

2.2 Convolutional Neural Networks (CNNs)
- Local connectivity and weight sharing
- Translation invariance
- Hierarchical feature learning

2.3 Recurrent Neural Networks (RNNs)
- Memory mechanisms for sequential data
- LSTM and GRU variants address vanishing gradients

2.4 Transformer Architecture
- Self-attention mechanisms
- Parallel processing of sequences
- Foundation for modern language models

3. Design Principles
3.1 Depth vs Width
Deep networks can represent more complex functions but are harder to train.

3.2 Skip Connections (ResNet)
Enable training of very deep networks by addressing gradient vanishing.

3.3 Attention Mechanisms
Allow models to focus on relevant input regions dynamically.

4. Modern Trends
4.1 Neural Architecture Search (NAS)
Automated design of neural network architectures.

4.2 Efficient Architectures
MobileNets, EfficientNets optimize for mobile and edge deployment.

4.3 Hybrid Architectures
Combining different architectural elements for specific tasks.

5. Future Directions
- Neuromorphic computing
- Quantum neural networks
- Biological inspiration

This evolution demonstrates the rapid advancement in neural network design and the continued potential for innovation.
'''
            },
            {
                'filename': 'ai_ethics_frameworks.txt',
                'content': '''Ethical Frameworks for Artificial Intelligence: Principles and Implementation

Abstract:
As AI systems become more prevalent and powerful, establishing robust ethical frameworks becomes crucial. This paper examines current approaches to AI ethics and proposes implementation strategies.

1. Introduction
The increasing deployment of AI systems in critical domains necessitates careful consideration of ethical implications. This work surveys existing frameworks and implementation challenges.

2. Core Ethical Principles
2.1 Fairness and Non-discrimination
AI systems should not perpetuate or amplify existing biases.

2.2 Transparency and Explainability
Users should understand how AI systems make decisions.

2.3 Privacy and Data Protection
Personal information must be handled responsibly.

2.4 Human Agency and Oversight
Humans should retain meaningful control over AI systems.

2.5 Robustness and Safety
AI systems should be reliable and secure.

3. Existing Frameworks
3.1 IEEE Standards
- IEEE 2857: Ethical Design for AI systems
- IEEE 2859: Framework for AI ethics

3.2 Partnership on AI
Industry collaboration on best practices.

3.3 European Ethics Guidelines
High-Level Expert Group recommendations.

4. Implementation Challenges
4.1 Technical Challenges
- Bias detection and mitigation
- Explainable AI methods
- Adversarial robustness

4.2 Organizational Challenges
- Ethics review processes
- Training and awareness
- Governance structures

4.3 Regulatory Challenges
- Cross-border coordination
- Technology-neutral regulation
- Innovation vs. precaution balance

5. Case Studies
5.1 Healthcare AI
Balancing diagnostic accuracy with patient privacy.

5.2 Criminal Justice AI
Ensuring fairness in risk assessment tools.

5.3 Autonomous Vehicles
Ethical decision-making in unavoidable accident scenarios.

6. Recommendations
- Multi-stakeholder approach to governance
- Continuous monitoring and evaluation
- Public engagement and education
- International cooperation

7. Conclusion
Effective AI ethics requires ongoing collaboration between technologists, ethicists, policymakers, and society at large.
'''
            }
        ]
        
        # Create sample documents
        for sample in samples:
            doc_path = self.inputs_dir / 'txt' / sample['filename']
            doc_path.write_text(sample['content'])
            sample_docs.append(doc_path)
            logger.info(f"Created sample document: {doc_path}")
        
        return sample_docs
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        return {
            'processing_stats': self.stats,
            'processed_documents_count': len(self.processed_docs),
            'directories': {
                'inputs': str(self.inputs_dir),
                'papers': str(self.papers_dir),
                'processed': str(self.processed_dir)
            },
            'available_documents': len(self.discover_documents()),
            'last_updated': datetime.now().isoformat()
        }


# Convenience functions
async def initialize_document_pipeline(config: Optional[Dict[str, Any]] = None) -> DocumentPipeline:
    """Initialize document pipeline with optional configuration"""
    return DocumentPipeline(config)


def create_default_config() -> Dict[str, Any]:
    """Get default configuration for document pipeline"""
    return {
        'papers_directory': './papers',
        'inputs_directory': './inputs',
        'processed_directory': './processed',
        'lightrag_working_dir': './lightrag_data',
        'paperqa2_working_dir': './paperqa2_data',
        'graphrag_working_dir': './graphrag_data',
        'auto_process_interval': 300,  # Check for new documents every 5 minutes
        'supported_formats': ['.pdf', '.txt', '.md'],
        'max_concurrent_processing': 3
    }