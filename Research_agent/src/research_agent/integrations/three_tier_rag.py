#!/usr/bin/env python3
"""
Three-Tier RAG System Implementation

Implements a sophisticated 3-tier RAG processing pipeline:
- Tier 1: LightRAG for initial content processing
- Tier 2: PaperQA2 for academic paper analysis  
- Tier 3: GraphRAG for knowledge graph construction
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
import os
import time

logger = logging.getLogger(__name__)


class RAGTier(Enum):
    """RAG processing tiers"""
    TIER1_LIGHTRAG = "tier1_lightrag"
    TIER2_PAPERQA2 = "tier2_paperqa2" 
    TIER3_GRAPHRAG = "tier3_graphrag"


@dataclass
class RAGResult:
    """Result from RAG processing"""
    tier: RAGTier
    query: str
    content: str
    sources: List[str]
    confidence: float
    metadata: Dict[str, Any]
    processing_time: float
    success: bool
    error: Optional[str] = None


class ThreeTierRAG:
    """Three-Tier RAG System Implementation"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the three-tier RAG system"""
        self.config = config or {}
        self.tier_processors = {}
        self._initialize_processors()
    
    def _initialize_processors(self):
        """Initialize RAG tier processors"""
        # Tier 1: LightRAG
        try:
            self.tier_processors[RAGTier.TIER1_LIGHTRAG] = self._create_lightrag_processor()
            logger.info("✅ Tier 1 LightRAG processor initialized")
        except ImportError as e:
            logger.warning(f"❌ LightRAG not available - Tier 1 disabled: {e}")
        except Exception as e:
            logger.error(f"❌ LightRAG initialization failed - Tier 1 disabled: {e}")
        
        # Tier 2: PaperQA2
        try:
            import paperqa
            self.tier_processors[RAGTier.TIER2_PAPERQA2] = self._create_paperqa_processor()
            logger.info("✅ Tier 2 PaperQA2 processor initialized")
        except ImportError as e:
            logger.warning(f"❌ PaperQA2 not available - Tier 2 disabled: {e}")
        except Exception as e:
            logger.error(f"❌ PaperQA2 initialization failed - Tier 2 disabled: {e}")
        
        # Tier 3: GraphRAG
        try:
            import graphrag
            import yaml  # Required for config management
            self.tier_processors[RAGTier.TIER3_GRAPHRAG] = self._create_graphrag_processor()
            logger.info("✅ Tier 3 GraphRAG processor initialized")
        except ImportError as e:
            logger.warning(f"❌ GraphRAG not available - Tier 3 disabled: {e}")
        except Exception as e:
            logger.error(f"❌ GraphRAG initialization failed - Tier 3 disabled: {e}")
    
    async def _create_production_rerank_func(self):
        """Create production-ready rerank function with error handling and fallbacks"""
        try:
            from lightrag.rerank import jina_rerank
            
            async def production_rerank_func(query: str, documents: list, top_n: int = None, **kwargs):
                """Production rerank function with comprehensive error handling"""
                if not documents:
                    logger.debug("No documents to rerank")
                    return documents
                
                # Get API key from environment
                api_key = os.getenv('JINA_API_KEY')
                if not api_key:
                    logger.debug("JINA_API_KEY not found, skipping reranking (fallback to no reranking)")
                    return documents
                
                # Set defaults
                if top_n is None:
                    top_n = min(len(documents), self.config.get('lightrag_rerank_top_n', 20))
                
                start_time = time.time()
                
                try:
                    # Use state-of-the-art BAAI/bge-reranker-v2-m3 model
                    reranked_docs = await jina_rerank(
                        query=query,
                        documents=documents,
                        model="BAAI/bge-reranker-v2-m3",
                        top_n=top_n,
                        api_key=api_key,
                        **kwargs
                    )
                    
                    processing_time = time.time() - start_time
                    logger.info(f"✅ Reranked {len(documents)} → {len(reranked_docs)} documents in {processing_time:.2f}s")
                    return reranked_docs
                    
                except Exception as rerank_error:
                    processing_time = time.time() - start_time
                    logger.warning(f"⚠️ Reranking failed after {processing_time:.2f}s: {rerank_error}")
                    logger.debug("Falling back to original document order")
                    # Graceful fallback - return original documents
                    return documents
            
            return production_rerank_func
            
        except ImportError as e:
            logger.warning(f"Reranking module not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to create rerank function: {e}")
            return None
    
    def _create_lightrag_processor(self):
        """Create LightRAG processor for Tier 1"""
        try:
            from lightrag import LightRAG, QueryParam
            from lightrag.kg.shared_storage import initialize_pipeline_status
            import os
            
            # Get working directory from config
            working_dir = self.config.get('lightrag_working_dir', './lightrag_data')
            os.makedirs(working_dir, exist_ok=True)
            
            # Determine provider and get appropriate functions
            llm_model_name = self.config.get('lightrag_llm_model', 'gpt-4o-mini')
            provider = self._detect_llm_provider(llm_model_name)
            
            if provider == 'openai':
                # Verify OpenAI API key is available
                if not os.getenv('OPENAI_API_KEY'):
                    raise ValueError("OPENAI_API_KEY environment variable not set for OpenAI provider")
                
                from lightrag.llm.openai import openai_complete_if_cache, openai_embed
                
                # Create proper LLM function with correct signature
                async def llm_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs):
                    return await openai_complete_if_cache(
                        llm_model_name,
                        prompt,
                        system_prompt=system_prompt,
                        history_messages=history_messages,
                        api_key=os.getenv('OPENAI_API_KEY'),
                        **kwargs
                    )
                
                embedding_func = openai_embed
                
            elif provider == 'anthropic':
                # Verify Anthropic API key is available  
                if not os.getenv('ANTHROPIC_API_KEY'):
                    raise ValueError("ANTHROPIC_API_KEY environment variable not set for Anthropic provider")
                
                from lightrag.llm.anthropic import anthropic_complete, anthropic_embed
                
                # Create proper LLM function with correct signature
                async def llm_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs):
                    return await anthropic_complete(
                        llm_model_name,
                        prompt,
                        system_prompt=system_prompt,
                        history_messages=history_messages,
                        api_key=os.getenv('ANTHROPIC_API_KEY'),
                        **kwargs
                    )
                
                embedding_func = anthropic_embed
                
            elif provider == 'ollama':
                # Ollama doesn't require API keys
                from lightrag.llm.ollama import ollama_model_complete, ollama_embed
                
                # Create proper LLM function with correct signature
                async def llm_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs):
                    return await ollama_model_complete(
                        llm_model_name,
                        prompt,
                        system_prompt=system_prompt,
                        history_messages=history_messages,
                        **kwargs
                    )
                
                embedding_func = ollama_embed
                
            else:
                # Default to OpenAI for unknown providers
                logger.warning(f"Unknown provider for model {llm_model_name}, defaulting to OpenAI")
                if not os.getenv('OPENAI_API_KEY'):
                    raise ValueError("OPENAI_API_KEY environment variable not set")
                
                from lightrag.llm.openai import openai_complete_if_cache, openai_embed
                
                # Create proper LLM function with correct signature
                async def llm_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs):
                    return await openai_complete_if_cache(
                        llm_model_name,
                        prompt,
                        system_prompt=system_prompt,
                        history_messages=history_messages,
                        api_key=os.getenv('OPENAI_API_KEY'),
                        **kwargs
                    )
                
                embedding_func = openai_embed
            
            # Create rerank function for production use
            rerank_func = None
            rerank_enabled = self.config.get('lightrag_enable_rerank', True)
            
            if rerank_enabled:
                try:
                    # Create async rerank function immediately to check availability
                    import asyncio
                    
                    # Try to create rerank function now to see if it's available
                    async def check_rerank_availability():
                        return await self._create_production_rerank_func()
                    
                    # Create event loop if needed
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # Can't run async function in running loop, defer to async init
                            rerank_func = "ASYNC_INIT_REQUIRED"
                        else:
                            rerank_func = loop.run_until_complete(check_rerank_availability())
                    except RuntimeError:
                        # No event loop, create one
                        rerank_func = asyncio.run(check_rerank_availability())
                    
                    if rerank_func and rerank_func != "ASYNC_INIT_REQUIRED":
                        logger.info("✅ Production reranking enabled with BAAI/bge-reranker-v2-m3")
                    elif rerank_func == "ASYNC_INIT_REQUIRED":
                        logger.info("ℹ️ Reranking initialization deferred to async setup")
                    else:
                        logger.info("ℹ️ Reranking disabled (no JINA_API_KEY or module unavailable)")
                        rerank_enabled = False  # Disable if no function available
                        
                except Exception as e:
                    logger.warning(f"Failed to initialize reranking: {e}")
                    rerank_func = None
                    rerank_enabled = False
            else:
                logger.info("ℹ️ Reranking disabled by configuration")
            
            # Initialize LightRAG with proper LLM, embedding, and rerank functions
            rag_kwargs = {
                'working_dir': working_dir,
                'llm_model_func': llm_func,
                'embedding_func': embedding_func,
                'llm_model_name': llm_model_name,
                'embedding_batch_num': self.config.get('lightrag_embedding_batch_num', 10),
                'llm_model_max_async': self.config.get('lightrag_llm_max_async', 4),
                'embedding_func_max_async': self.config.get('lightrag_embedding_max_async', 8),
                'chunk_token_size': self.config.get('lightrag_chunk_token_size', 1200),
                'chunk_overlap_token_size': self.config.get('lightrag_chunk_overlap', 100),
                'top_k': self.config.get('lightrag_top_k', 40),
                'chunk_top_k': self.config.get('lightrag_chunk_top_k', 10)
            }
            
            # Add rerank function if available (will be set during async init if needed)
            if rerank_func and rerank_func != "ASYNC_INIT_REQUIRED":
                rag_kwargs['rerank_model_func'] = rerank_func
            elif not rerank_enabled:
                # Disable reranking completely if not available
                pass  # Don't set rerank_model_func, LightRAG will not expect reranking
            
            rag = LightRAG(**rag_kwargs)
            
            # Store initialization function for later use
            # Note: We'll initialize storages asynchronously when needed
            
        except Exception as e:
            logger.error(f"Failed to initialize LightRAG: {e}")
            # Return a mock processor if initialization fails
            return {
                "type": "lightrag_mock", 
                "status": "failed",
                "processor": None,
                "working_dir": working_dir,
                "model": self.config.get('lightrag_llm_model', 'gpt-4o-mini'),
                "error": str(e)
            }
        
        return {
            "type": "lightrag", 
            "status": "initialized",
            "processor": rag,
            "working_dir": working_dir,
            "model": self.config.get('lightrag_llm_model', 'gpt-4o-mini'),
            "initialize_pipeline_status": initialize_pipeline_status,
            "requires_async_init": True,
            "rerank_enabled": rerank_enabled,
            "rerank_async_init_required": rerank_func == "ASYNC_INIT_REQUIRED"
        }
    
    def _detect_llm_provider(self, model_name: str) -> str:
        """Detect LLM provider based on model name"""
        model_name_lower = model_name.lower()
        
        if any(keyword in model_name_lower for keyword in ['gpt', 'openai', 'o1']):
            return 'openai'
        elif any(keyword in model_name_lower for keyword in ['claude', 'anthropic']):
            return 'anthropic'
        elif any(keyword in model_name_lower for keyword in ['llama', 'mistral', 'ollama']):
            return 'ollama'
        else:
            return 'unknown'
    
    def _create_paperqa_processor(self):
        """Create PaperQA2 processor for Tier 2"""  
        from paperqa import Docs, Settings
        
        # Get working directory for PaperQA2
        working_dir = self.config.get('paperqa2_working_dir', './paperqa2_data')
        Path(working_dir).mkdir(parents=True, exist_ok=True)
        
        # Create PaperQA Settings with proper configuration
        settings = Settings(
            llm=self.config.get('paperqa2_llm_model', 'gpt-4o-mini'),
            summary_llm=self.config.get('paperqa2_summary_model', 'gpt-4o-mini'),
            temperature=self.config.get('paperqa2_temperature', 0.1),
            paper_directory=working_dir
        )
        
        # Initialize PaperQA2 Docs instance
        docs = Docs()
        
        return {
            "type": "paperqa2", 
            "status": "initialized",
            "processor": docs,
            "settings": settings,
            "working_dir": working_dir,
            "llm_model": self.config.get('paperqa2_llm_model', 'gpt-4o-mini'),
            "summary_model": self.config.get('paperqa2_summary_model', 'gpt-4o-mini'),
            "temperature": self.config.get('paperqa2_temperature', 0.1)
        }
    
    def _create_graphrag_processor(self):
        """Create GraphRAG processor for Tier 3"""
        from pathlib import Path
        import os
        import yaml
        
        # Get working directory from config
        working_dir = Path(self.config.get('graphrag_working_dir', './graphrag_data'))
        working_dir.mkdir(parents=True, exist_ok=True)
        
        # Create input directory for documents
        input_dir = working_dir / "input"
        input_dir.mkdir(exist_ok=True)
        
        # Create GraphRAG configuration
        api_key = os.getenv('OPENAI_API_KEY', '')
        config_data = {
            "llm": {
                "api_key": api_key,
                "type": "openai_chat",
                "model": self.config.get('graphrag_llm_model', 'gpt-4o-mini'),
                "model_supports_json": True,
                "max_tokens": 4000,
                "temperature": 0.0,
                "top_p": 1.0,
            },
            "embeddings": {
                "api_key": api_key,
                "type": "openai_embedding",
                "model": self.config.get('graphrag_embed_model', 'text-embedding-3-small'),
                "batch_size": 16,
            },
            "chunks": {
                "size": 300,
                "overlap": 100,
                "group_by_columns": ["id"],
            },
            "input": {
                "type": "file",
                "file_type": "text",
                "base_dir": str(input_dir),
                "file_encoding": "utf-8",
                "file_pattern": ".*\\.txt$",
            },
            "cache": {
                "type": "file",
                "base_dir": str(working_dir / "cache"),
            },
            "storage": {
                "type": "file",
                "base_dir": str(working_dir / "output"),
            },
            "reporting": {
                "type": "file",
                "base_dir": str(working_dir / "reports"),
            },
            "entity_extraction": {
                "max_gleanings": 1,
                "strategy": {"type": "graph_intelligence"},
            },
            "summarize_descriptions": {
                "max_length": 500,
                "strategy": {"type": "graph_intelligence"},
            },
            "community_reports": {
                "max_length": 1500,
                "strategy": {"type": "graph_intelligence"},
            },
            "claim_extraction": {
                "enabled": True,
                "max_gleanings": 1,
                "strategy": {"type": "graph_intelligence"},
            },
        }
        
        # Save GraphRAG configuration
        config_path = working_dir / "settings.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        # Note: Environment variables are handled by the parent process
        
        return {
            "type": "graphrag", 
            "status": "initialized",
            "working_dir": str(working_dir),
            "input_dir": str(input_dir),
            "config_path": str(config_path),
            "llm_model": self.config.get('graphrag_llm_model', 'gpt-4o-mini'),
            "embed_model": self.config.get('graphrag_embed_model', 'text-embedding-3-small'),
            "indexed": False,  # Track if index has been built
        }
    
    async def process_query(self, query: str, tiers: Optional[List[RAGTier]] = None) -> List[RAGResult]:
        """Process query through specified RAG tiers"""
        import time
        
        if tiers is None:
            tiers = list(RAGTier)
        
        results = []
        
        for tier in tiers:
            if tier not in self.tier_processors:
                result = RAGResult(
                    tier=tier,
                    query=query,
                    content="",
                    sources=[],
                    confidence=0.0,
                    metadata={"error": "Processor not available"},
                    processing_time=0.0,
                    success=False,
                    error=f"Tier {tier.value} processor not available"
                )
                results.append(result)
                continue
            
            start_time = time.time()
            
            try:
                if tier == RAGTier.TIER1_LIGHTRAG:
                    # Real LightRAG processing
                    result = await self._process_lightrag_tier(query, start_time)
                elif tier == RAGTier.TIER2_PAPERQA2:
                    # Real PaperQA2 processing
                    result = await self._process_paperqa_tier(query, start_time)
                elif tier == RAGTier.TIER3_GRAPHRAG:
                    # Real GraphRAG processing with knowledge graph construction
                    result = await self._process_graphrag_tier(query, start_time)
                else:
                    # Fallback for unknown tiers
                    await asyncio.sleep(0.1)
                    result = RAGResult(
                        tier=tier,
                        query=query,
                        content=f"Unknown tier: {tier.value}",
                        sources=[],
                        confidence=0.0,
                        metadata={"tier": tier.value, "processor": "unknown"},
                        processing_time=time.time() - start_time,
                        success=False,
                        error=f"Unknown tier: {tier.value}"
                    )
                
                results.append(result)
                
            except Exception as e:
                result = RAGResult(
                    tier=tier,
                    query=query,
                    content="",
                    sources=[],
                    confidence=0.0,
                    metadata={"error": str(e)},
                    processing_time=time.time() - start_time,
                    success=False,
                    error=str(e)
                )
                results.append(result)
        
        return results
    
    async def insert_documents(self, documents: List[Union[str, Path]], tier: RAGTier = RAGTier.TIER1_LIGHTRAG) -> bool:
        """Insert documents into specified RAG tier for processing"""
        try:
            if tier == RAGTier.TIER1_LIGHTRAG and tier in self.tier_processors:
                processor_info = self.tier_processors[tier]
                lightrag_processor = processor_info["processor"]
                
                # Initialize LightRAG storages if needed
                if processor_info.get("requires_async_init", False):
                    try:
                        await lightrag_processor.initialize_storages()
                        initialize_pipeline_status_func = processor_info["initialize_pipeline_status"]
                        await initialize_pipeline_status_func()
                        processor_info["requires_async_init"] = False
                        logger.info("✅ LightRAG storages initialized for document insertion")
                    except Exception as init_error:
                        logger.error(f"Failed to initialize LightRAG storages: {init_error}")
                        raise init_error
                
                # Initialize rerank function if needed
                if processor_info.get("rerank_async_init_required", False):
                    try:
                        rerank_func = await self._create_production_rerank_func()
                        if rerank_func:
                            # Update the LightRAG instance with rerank function
                            lightrag_processor.rerank_model_func = rerank_func
                            logger.info("✅ Production reranking initialized for LightRAG")
                        processor_info["rerank_async_init_required"] = False
                    except Exception as rerank_error:
                        logger.warning(f"Failed to initialize reranking: {rerank_error}")
                        processor_info["rerank_async_init_required"] = False
                
                # Insert documents into LightRAG using async methods
                for doc in documents:
                    await lightrag_processor.ainsert(str(doc))
                
                logger.info(f"✅ Inserted {len(documents)} documents into LightRAG")
                return True
                
            elif tier == RAGTier.TIER2_PAPERQA2 and tier in self.tier_processors:
                processor_info = self.tier_processors[tier]
                paperqa_docs = processor_info["processor"]
                settings = processor_info["settings"]
                working_dir = Path(processor_info["working_dir"])
                
                # Process documents
                added_count = 0
                
                for i, doc in enumerate(documents):
                    if isinstance(doc, (str, Path)):
                        doc_path = Path(doc)
                        
                        if doc_path.exists() and doc_path.suffix.lower() in ['.pdf', '.txt']:
                            # Add existing files directly
                            await asyncio.to_thread(paperqa_docs.add, str(doc_path))
                            added_count += 1
                        elif isinstance(doc, str) and not doc_path.exists():
                            # This is text content - save to temporary file
                            temp_file = working_dir / f"temp_text_doc_{i}.txt"
                            temp_file.write_text(doc, encoding='utf-8')
                            await asyncio.to_thread(paperqa_docs.add, str(temp_file))
                            added_count += 1
                        else:
                            logger.warning(f"Unsupported document for PaperQA2: {doc}")
                            continue
                    else:
                        logger.warning(f"Invalid document type for PaperQA2: {type(doc)}")
                        continue
                
                logger.info(f"✅ Inserted {added_count} documents into PaperQA2")
                return True
                
            elif tier == RAGTier.TIER3_GRAPHRAG and tier in self.tier_processors:
                processor_info = self.tier_processors[tier]
                working_dir = Path(processor_info["working_dir"])
                input_dir = Path(processor_info["input_dir"])
                
                # Process documents for GraphRAG
                added_count = 0
                
                for i, doc in enumerate(documents):
                    if isinstance(doc, (str, Path)):
                        doc_path = Path(doc)
                        
                        if doc_path.exists() and doc_path.suffix.lower() in ['.txt']:
                            # Copy text files directly to input directory
                            target_file = input_dir / doc_path.name
                            target_file.write_text(doc_path.read_text(encoding='utf-8'), encoding='utf-8')
                            added_count += 1
                        elif doc_path.exists() and doc_path.suffix.lower() == '.pdf':
                            # Convert PDF to text for GraphRAG processing
                            try:
                                import pypdf
                                with open(doc_path, 'rb') as pdf_file:
                                    pdf_reader = pypdf.PdfReader(pdf_file)
                                    text_content = ""
                                    for page in pdf_reader.pages:
                                        text_content += page.extract_text() + "\n\n"
                                
                                # Save as text file in input directory
                                txt_filename = f"{doc_path.stem}.txt"
                                target_file = input_dir / txt_filename
                                target_file.write_text(text_content, encoding='utf-8')
                                added_count += 1
                            except Exception as e:
                                logger.warning(f"Failed to extract text from PDF {doc_path}: {e}")
                                continue
                        elif isinstance(doc, str) and not doc_path.exists():
                            # This is text content - save directly to input directory
                            text_file = input_dir / f"text_doc_{i}.txt"
                            text_file.write_text(doc, encoding='utf-8')
                            added_count += 1
                        else:
                            logger.warning(f"Unsupported document format for GraphRAG: {doc}")
                            continue
                    else:
                        logger.warning(f"Invalid document type for GraphRAG: {type(doc)}")
                        continue
                
                # Build GraphRAG index if documents were added
                if added_count > 0:
                    try:
                        await self._build_graphrag_index(working_dir)
                        # Mark as indexed
                        processor_info["indexed"] = True
                        logger.info(f"✅ Inserted {added_count} documents into GraphRAG and built index")
                    except Exception as e:
                        logger.error(f"Failed to build GraphRAG index: {e}")
                        logger.info(f"✅ Inserted {added_count} documents into GraphRAG (index build failed)")
                
                return added_count > 0
                
            else:
                logger.warning(f"Document insertion not implemented for {tier.value}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to insert documents into {tier.value}: {e}")
            return False
    
    async def _reload_paperqa_documents(self, processor_info: Dict[str, Any]):
        """Reload documents into PaperQA2 from working directory"""
        try:
            paperqa_docs = processor_info["processor"]
            working_dir = Path(processor_info["working_dir"])
            
            # Look for text files in the working directory
            text_files = list(working_dir.glob('*.txt'))
            
            if text_files:
                logger.info(f"Reloading {len(text_files)} documents into PaperQA2...")
                
                for text_file in text_files:
                    try:
                        await asyncio.to_thread(paperqa_docs.add, str(text_file))
                        logger.info(f"✅ Reloaded: {text_file.name}")
                    except Exception as e:
                        logger.warning(f"Failed to reload {text_file.name}: {e}")
                
                logger.info(f"PaperQA2 now has {len(paperqa_docs.docs)} documents")
            else:
                logger.warning(f"No documents found in {working_dir} to reload")
                
        except Exception as e:
            logger.error(f"Failed to reload PaperQA2 documents: {e}")
    
    async def _build_graphrag_index(self, working_dir: Path):
        """Build GraphRAG index from documents in the input directory"""
        from graphrag.cli.index import index_cli
        import subprocess
        import sys
        
        try:
            # Use subprocess to run GraphRAG index command
            cmd = [
                sys.executable, "-m", "graphrag", "index",
                "--root", str(working_dir),
                "--config", str(working_dir / "settings.yaml"),
                "--verbose"
            ]
            
            # Run indexing in background thread to avoid blocking
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                cwd=str(working_dir)
            )
            
            if result.returncode == 0:
                logger.info("GraphRAG index built successfully")
                return True
            else:
                logger.error(f"GraphRAG index build failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error building GraphRAG index: {e}")
            return False
    
    async def _process_lightrag_tier(self, query: str, start_time: float) -> RAGResult:
        """Process query using LightRAG (Tier 1)"""
        import time
        
        try:
            processor_info = self.tier_processors[RAGTier.TIER1_LIGHTRAG]
            lightrag_processor = processor_info["processor"]
            
            # Check if processor is available
            if lightrag_processor is None or processor_info["status"] == "failed":
                raise Exception(f"LightRAG processor not available: {processor_info.get('error', 'Unknown error')}")
            
            # Initialize LightRAG storages if needed
            if processor_info.get("requires_async_init", False):
                try:
                    # Initialize storages and pipeline status
                    await lightrag_processor.initialize_storages()
                    initialize_pipeline_status_func = processor_info["initialize_pipeline_status"]
                    await initialize_pipeline_status_func()
                    # Mark as initialized
                    processor_info["requires_async_init"] = False
                    logger.info("✅ LightRAG storages initialized successfully")
                except Exception as init_error:
                    logger.error(f"Failed to initialize LightRAG storages: {init_error}")
                    raise init_error
            
            # Initialize rerank function if needed
            if processor_info.get("rerank_async_init_required", False):
                try:
                    rerank_func = await self._create_production_rerank_func()
                    if rerank_func:
                        # Update the LightRAG instance with rerank function
                        lightrag_processor.rerank_model_func = rerank_func
                        logger.info("✅ Production reranking initialized for query processing")
                    processor_info["rerank_async_init_required"] = False
                except Exception as rerank_error:
                    logger.warning(f"Failed to initialize reranking for query: {rerank_error}")
                    processor_info["rerank_async_init_required"] = False
            
            # Use LightRAG for query processing with proper QueryParam API
            try:
                # Import QueryParam if not already available
                from lightrag import QueryParam
                
                # Define the async query function
                async def run_lightrag_query_async(processor, query_text, mode="hybrid"):
                    # Check if reranking is available and enabled
                    processor_info = self.tier_processors[RAGTier.TIER1_LIGHTRAG]
                    rerank_available = (
                        hasattr(processor, 'rerank_model_func') and 
                        processor.rerank_model_func is not None and
                        processor_info.get('rerank_enabled', False)
                    )
                    query_param = QueryParam(mode=mode, enable_rerank=rerank_available)
                    return await processor.aquery(query_text, param=query_param)
                
                # Use the correct QueryParam API for query processing
                response = await run_lightrag_query_async(lightrag_processor, query, "hybrid")
            except Exception as api_error:
                # If query fails, try to add some content first and retry with naive mode
                logger.warning(f"Query failed, trying to add content first: {api_error}")
                try:
                    # Define async insert function
                    async def run_lightrag_insert_async(processor, content):
                        return await processor.ainsert(content)
                    
                    await run_lightrag_insert_async(lightrag_processor, f"Research context: {query}")
                    response = await run_lightrag_query_async(lightrag_processor, query, "naive")
                except Exception as retry_error:
                    logger.error(f"Retry also failed: {retry_error}")
                    raise retry_error
            
            # Extract sources and metadata from response
            sources = []
            confidence = 0.8  # Default confidence, could be calculated from response quality
            
            if hasattr(response, 'sources'):
                sources = response.sources
            elif isinstance(response, dict) and 'sources' in response:
                sources = response['sources']
            
            return RAGResult(
                tier=RAGTier.TIER1_LIGHTRAG,
                query=query,
                content=str(response),
                sources=sources,
                confidence=confidence,
                metadata={
                    "tier": "tier1_lightrag",
                    "processor": "actual_lightrag",
                    "working_dir": processor_info["working_dir"],
                    "model": processor_info["model"]
                },
                processing_time=time.time() - start_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"LightRAG processing failed: {e}")
            return RAGResult(
                tier=RAGTier.TIER1_LIGHTRAG,
                query=query,
                content="",
                sources=[],
                confidence=0.0,
                metadata={"tier": "tier1_lightrag", "processor": "actual_lightrag", "error": str(e)},
                processing_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    async def _process_paperqa_tier(self, query: str, start_time: float) -> RAGResult:
        """Process query using PaperQA2 (Tier 2)"""
        import time
        from paperqa import ask
        
        try:
            processor_info = self.tier_processors[RAGTier.TIER2_PAPERQA2]
            paperqa_docs = processor_info["processor"]
            settings = processor_info["settings"]
            
            # Check if documents are available in the Docs instance
            if len(paperqa_docs.docs) == 0:
                # Try to reload documents from the working directory
                await self._reload_paperqa_documents(processor_info)
            
            # Use PaperQA ask function with our configured docs and settings
            # Create a temporary settings copy that uses our docs
            temp_settings = settings
            
            # If docs is empty, try with regular ask
            if len(paperqa_docs.docs) == 0:
                response = await asyncio.to_thread(
                    ask,
                    query,
                    temp_settings
                )
            else:
                # Use docs.query method if available, otherwise use ask with docs
                if hasattr(paperqa_docs, 'query'):
                    response = await asyncio.to_thread(
                        paperqa_docs.query,
                        query
                    )
                else:
                    # Use ask function but with our populated docs
                    response = await asyncio.to_thread(
                        ask,
                        query,
                        temp_settings,
                        docs=paperqa_docs
                    )
            
            # Extract information from the response
            content = str(response.answer) if hasattr(response, 'answer') else str(response)
            
            # Extract sources and citations
            sources = []
            confidence = 0.5  # Default confidence
            citations = []
            
            if hasattr(response, 'contexts') and response.contexts:
                sources = [ctx.text.name for ctx in response.contexts if hasattr(ctx.text, 'name')]
                # Calculate confidence based on context scores
                scores = [getattr(ctx, 'score', 0.5) for ctx in response.contexts]
                confidence = sum(scores) / len(scores) if scores else 0.5
                
                # Extract citations
                citations = [ctx.text.citation for ctx in response.contexts 
                           if hasattr(ctx.text, 'citation')]
            
            # Get status from response
            status_info = {}
            if hasattr(response, 'status'):
                status_info = {
                    'total_paper_count': getattr(response.status, 'total_paper_count', 0),
                    'relevant_paper_count': getattr(response.status, 'relevant_paper_count', 0),
                    'evidence_count': getattr(response.status, 'evidence_count', 0)
                }
            
            return RAGResult(
                tier=RAGTier.TIER2_PAPERQA2,
                query=query,
                content=content,
                sources=sources,
                confidence=min(confidence, 1.0),  # Ensure confidence is within bounds
                metadata={
                    "tier": "tier2_paperqa2",
                    "processor": "actual_paperqa2",
                    "working_dir": processor_info["working_dir"],
                    "llm_model": processor_info["llm_model"],
                    "summary_model": processor_info["summary_model"],
                    "temperature": processor_info["temperature"],
                    "citations": citations,
                    "num_contexts": len(response.contexts) if hasattr(response, 'contexts') else 0,
                    "status": status_info
                },
                processing_time=time.time() - start_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"PaperQA2 processing failed: {e}")
            return RAGResult(
                tier=RAGTier.TIER2_PAPERQA2,
                query=query,
                content="",
                sources=[],
                confidence=0.0,
                metadata={
                    "tier": "tier2_paperqa2", 
                    "processor": "actual_paperqa2", 
                    "error": str(e)
                },
                processing_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    async def _process_graphrag_tier(self, query: str, start_time: float) -> RAGResult:
        """Process query using GraphRAG (Tier 3)"""
        import time
        import os
        
        try:
            processor_info = self.tier_processors[RAGTier.TIER3_GRAPHRAG]
            working_dir = Path(processor_info["working_dir"])
            config_path = Path(processor_info["config_path"])
            
            # Check if index exists
            output_dir = working_dir / "output"
            if not processor_info.get("indexed", False) or not output_dir.exists():
                return RAGResult(
                    tier=RAGTier.TIER3_GRAPHRAG,
                    query=query,
                    content="GraphRAG index not built yet. Please add documents first.",
                    sources=[],
                    confidence=0.0,
                    metadata={
                        "tier": "tier3_graphrag", 
                        "processor": "actual_graphrag",
                        "error": "Index not built",
                        "working_dir": str(working_dir)
                    },
                    processing_time=time.time() - start_time,
                    success=False,
                    error="GraphRAG index not built yet"
                )
            
            # Determine search method based on query characteristics
            search_method = self._determine_search_method(query)
            
            # Run GraphRAG query using subprocess to avoid import issues
            import subprocess
            import sys
            
            cmd = [
                sys.executable, "-m", "graphrag", "query",
                "--method", search_method,
                "--query", query,
                "--root", str(working_dir),
                "--config", str(config_path),
                "--data", str(output_dir),
                "--community-level", str(self.config.get('graphrag_community_level', 2)),
                "--response-type", self.config.get('graphrag_response_type', 'Multiple Paragraphs')
            ]
            
            # Run query in background thread to avoid blocking
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                cwd=str(working_dir)
            )
            
            if result.returncode == 0:
                response = result.stdout.strip()
            else:
                logger.error(f"GraphRAG query failed: {result.stderr}")
                response = f"GraphRAG query failed: {result.stderr}"
            
            # Extract content and metadata from response
            content = str(response) if response else "No response from GraphRAG"
            
            # Calculate confidence based on response quality and length
            confidence = self._calculate_graphrag_confidence(content, query)
            
            # Extract sources from response (GraphRAG includes source information in text)
            sources = self._extract_graphrag_sources(content)
            
            return RAGResult(
                tier=RAGTier.TIER3_GRAPHRAG,
                query=query,
                content=content,
                sources=sources,
                confidence=confidence,
                metadata={
                    "tier": "tier3_graphrag",
                    "processor": "actual_graphrag",
                    "search_method": search_method,
                    "working_dir": str(working_dir),
                    "community_level": self.config.get('graphrag_community_level', 2),
                    "llm_model": processor_info["llm_model"],
                    "embed_model": processor_info["embed_model"],
                    "response_length": len(content)
                },
                processing_time=time.time() - start_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"GraphRAG processing failed: {e}")
            return RAGResult(
                tier=RAGTier.TIER3_GRAPHRAG,
                query=query,
                content="",
                sources=[],
                confidence=0.0,
                metadata={
                    "tier": "tier3_graphrag", 
                    "processor": "actual_graphrag", 
                    "error": str(e)
                },
                processing_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    def _determine_search_method(self, query: str) -> str:
        """Determine whether to use global or local search based on query characteristics"""
        # Keywords that suggest global conceptual search
        global_keywords = [
            'overview', 'summary', 'what is', 'explain', 'describe', 'how does', 
            'why', 'what are the', 'compare', 'contrast', 'relationship', 
            'impact', 'effect', 'influence', 'trend', 'pattern', 'approach',
            'methodology', 'framework', 'theory', 'concept', 'principle'
        ]
        
        # Keywords that suggest local entity-specific search
        local_keywords = [
            'who', 'when', 'where', 'which', 'specific', 'particular',
            'example', 'case study', 'instance', 'name', 'list', 'enumerate'
        ]
        
        query_lower = query.lower()
        
        # Count matches for each type
        global_matches = sum(1 for keyword in global_keywords if keyword in query_lower)
        local_matches = sum(1 for keyword in local_keywords if keyword in query_lower)
        
        # Default to global search for broad conceptual queries
        # Use local search for specific entity queries
        if local_matches > global_matches:
            return "local"
        else:
            return "global"
    
    def _calculate_graphrag_confidence(self, content: str, query: str) -> float:
        """Calculate confidence score based on GraphRAG response quality"""
        if not content or content == "No response from GraphRAG":
            return 0.0
        
        # Base confidence for having a response
        confidence = 0.5
        
        # Increase confidence based on response length (more detailed = higher confidence)
        content_length = len(content)
        if content_length > 1000:
            confidence += 0.2
        elif content_length > 500:
            confidence += 0.1
        elif content_length > 200:
            confidence += 0.05
        
        # Check for citations or references (GraphRAG includes these)
        if '[' in content and ']' in content:
            confidence += 0.15
        
        # Check for structured response (paragraphs, lists)
        if content.count('\n\n') > 1:
            confidence += 0.1
        
        # Check if query terms appear in response
        query_terms = query.lower().split()
        content_lower = content.lower()
        matching_terms = sum(1 for term in query_terms if term in content_lower and len(term) > 3)
        if matching_terms > 0:
            confidence += min(0.1 * (matching_terms / len(query_terms)), 0.15)
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def _extract_graphrag_sources(self, content: str) -> List[str]:
        """Extract sources from GraphRAG response content"""
        import re
        
        sources = []
        
        # Look for citation patterns like [1], [Entity1], etc.
        citation_pattern = r'\[([^\]]+)\]'
        citations = re.findall(citation_pattern, content)
        
        # Add unique citations as sources
        for citation in citations:
            if citation.isdigit():
                sources.append(f"GraphRAG_Citation_{citation}")
            else:
                sources.append(f"GraphRAG_Entity_{citation}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_sources = []
        for source in sources:
            if source not in seen:
                seen.add(source)
                unique_sources.append(source)
        
        # If no citations found, indicate graph-based reasoning
        if not unique_sources:
            unique_sources = ["GraphRAG_KnowledgeGraph"]
        
        return unique_sources[:10]  # Limit to top 10 sources
    
    def get_available_tiers(self) -> List[RAGTier]:
        """Get list of available RAG tiers"""
        return list(self.tier_processors.keys())
    
    def get_tier_status(self) -> Dict[RAGTier, Dict[str, Any]]:
        """Get status of all tiers"""
        return {
            tier: {
                "available": tier in self.tier_processors,
                "processor": self.tier_processors.get(tier, {})
            }
            for tier in RAGTier
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all tiers"""
        health_status = {
            "overall_status": "healthy",
            "available_tiers": len(self.tier_processors),
            "total_tiers": len(RAGTier),
            "tiers": {}
        }
        
        for tier in RAGTier:
            if tier in self.tier_processors:
                health_status["tiers"][tier.value] = {
                    "status": "available",
                    "processor": self.tier_processors[tier]
                }
            else:
                health_status["tiers"][tier.value] = {
                    "status": "unavailable",
                    "reason": "Processor not initialized"
                }
        
        if len(self.tier_processors) == 0:
            health_status["overall_status"] = "critical"
        elif len(self.tier_processors) < len(RAGTier):
            health_status["overall_status"] = "degraded"
        
        return health_status


# Convenience functions
async def create_three_tier_rag(config: Optional[Dict[str, Any]] = None) -> ThreeTierRAG:
    """Create and initialize ThreeTierRAG system"""
    system = ThreeTierRAG(config)
    return system


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for three-tier RAG"""
    return {
        "tier1_lightrag": {
            "enabled": True,
            "model": "text-embedding-ada-002",
            "chunk_size": 512,
            "rerank_top_n": 20
        },
        "tier2_paperqa2": {
            "enabled": True,
            "paperqa2_llm_model": "gpt-4o-mini",
            "paperqa2_summary_model": "gpt-4o-mini", 
            "paperqa2_temperature": 0.1,
            "paperqa2_working_dir": "./paperqa2_data"
        },
        "tier3_graphrag": {
            "enabled": True,
            "graph_type": "knowledge",
            "community_level": 2
        },
        # Reranking configuration
        "lightrag_enable_rerank": True,
        "lightrag_rerank_top_n": 20,
        "rerank_model": "BAAI/bge-reranker-v2-m3",
        "rerank_provider": "jina"
    }