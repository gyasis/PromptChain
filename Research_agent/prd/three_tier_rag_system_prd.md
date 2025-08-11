# Three-Tier RAG System: Product Requirements Document (PRD)

## Executive Summary

This document outlines the requirements for implementing a three-tier Retrieval-Augmented Generation (RAG) system for research paper analysis. The system will support two implementation approaches:

1. **Existing Library Approach**: Using LightRAG, PaperQA2, and GraphRAG with cloud APIs
2. **Ollama-Based Approach**: Using local models with equivalent functionality

## Technical Architecture

### System Overview
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Tier 1        │    │   Tier 2        │    │   Tier 3        │
│   Entity        │───▶│   Research      │───▶│   Knowledge     │
│   Extraction    │    │   Analysis      │    │   Graph         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Existing Library Approach

### Tier 1: LightRAG Implementation
**Purpose**: Entity and relationship extraction with knowledge graph construction

**Components**:
- **LightRAG Framework**: Open-source from HKU
- **Entity Extraction**: LLM-based entity identification
- **Relationship Mapping**: Dual-level retrieval system

**Implementation**:
```python
# PSEUDO CODE - LightRAG Tier 1
class LightRAGTier1:
    def __init__(self):
        self.llm_model = "gpt-4"  # Cloud API
        self.embedding_model = "sentence-transformers"
        self.vector_store = ChromaDB()
    
    def extract_entities(self, documents):
        # Extract entities using LLM
        entities = []
        for doc in documents:
            prompt = f"Extract entities from: {doc.text}"
            response = call_llm_api(prompt)
            entities.extend(parse_entities(response))
        return entities
```

### Tier 2: PaperQA2 Implementation
**Purpose**: Research analysis and question answering

**Components**:
- **PaperQA2 Framework**: FutureHouse Inc. implementation
- **Literature Retrieval**: Citation traversal system
- **Contradiction Detection**: Automated conflict identification

**Key Features**:
- 85.2% precision rate
- 66% accuracy rate
- 70% contradiction detection accuracy

### Tier 3: GraphRAG Implementation
**Purpose**: Knowledge graph reasoning and multi-hop analysis

**Components**:
- **GraphRAG Framework**: Microsoft Research implementation
- **Knowledge Graph**: LLM-generated graph construction
- **Multi-hop Reasoning**: Graph traversal algorithms

## Ollama-Based Approach (Using Existing Libraries)

### Tier 1: LightRAG with Ollama Models
**Purpose**: Entity extraction, relationship mapping, and initial clustering

**Components**:
- **LightRAG Framework**: Open-source from HKU with Ollama integration
- **Ollama Model**: mistral-nemo:latest (fast inference)
- **Embedding Model**: bge-m3:latest (high-quality embeddings)
- **Vector Store**: Built-in ChromaDB integration

**Implementation**:
```python
# PSEUDO CODE - LightRAG with Ollama Tier 1
from lightrag import LightRAG
from lightrag.embedding import EmbeddingFunc

class LightRAGOllamaTier1:
    def __init__(self):
        # Configure LightRAG with Ollama
        self.rag = LightRAG(
            working_dir="./lightrag_working_dir",
            llm_model_func=ollama_model_complete,  # Ollama integration
            llm_model_name='mistral-nemo:latest',
            embedding_func=EmbeddingFunc(
                embedding_dim=1024,
                func=lambda texts: ollama_embed(
                    texts,
                    embed_model="bge-m3:latest"
                )
            ),
        )
    
    def extract_entities(self, documents):
        # LightRAG handles entity extraction automatically
        for doc in documents:
            await self.rag.ainsert(doc.text)
        
        # Query for entity information
        entities = await self.rag.aquery(
            "Extract all entities and their relationships",
            param=QueryParam(mode="hybrid")
        )
        
        return entities.entities, entities.relationships
```

**Environment Configuration**:
```bash
# .env file for LightRAG with Ollama
LLM_BINDING=ollama
LLM_MODEL=mistral-nemo:latest
LLM_BINDING_HOST=http://localhost:11434
OLLAMA_LLM_NUM_CTX=16384

EMBEDDING_BINDING=ollama
EMBEDDING_BINDING_HOST=http://localhost:11434
EMBEDDING_MODEL=bge-m3:latest
EMBEDDING_DIM=1024
```

### Tier 2: PaperQA2 with Ollama Models
**Purpose**: Research analysis, question answering, and summarization

**Components**:
- **PaperQA2 Framework**: FutureHouse Inc. with Ollama integration
- **Ollama Model**: llama3.2 (balanced performance)
- **Embedding Model**: mxbai-embed-large (research-optimized)
- **Specialized LLMs**: Separate models for main, summary, and agent tasks

**Implementation**:
```python
# PSEUDO CODE - PaperQA2 with Ollama Tier 2
from paperqa import Settings, ask
from paperqa.settings import AgentSettings

class PaperQA2OllamaTier2:
    def __init__(self):
        # Configure PaperQA2 with Ollama
        self.local_llm_config = {
            "model_list": [{
                "model_name": "ollama/llama3.2",
                "litellm_params": {
                    "model": "ollama/llama3.2",
                    "api_base": "http://localhost:11434",
                }
            }]
        }
        
        self.settings = Settings(
            llm="ollama/llama3.2",
            llm_config=self.local_llm_config,
            summary_llm="ollama/llama3.2",
            summary_llm_config=self.local_llm_config,
            agent=AgentSettings(agent_llm="ollama/llama3.2"),
            embedding="ollama/mxbai-embed-large"
        )
    
    def analyze_research(self, question, papers):
        # PaperQA2 handles research analysis automatically
        answer_response = ask(
            question,
            settings=self.settings
        )
        
        return {
            "analysis": answer_response.session.answer,
            "sources": answer_response.session.contexts,
            "citations": answer_response.session.citations
        }
```

### Tier 3: GraphRAG with Ollama Models
**Purpose**: Knowledge graph generation and complex relationship analysis

**Components**:
- **GraphRAG Framework**: Microsoft Research with custom model support
- **Ollama Models**: Asymmetric usage (extraction vs. query models)
- **Graph Database**: Built-in Neo4J or NetworkX support
- **Map-Reduce Architecture**: Separate models for different operations

**Implementation**:
```python
# PSEUDO CODE - GraphRAG with Ollama Tier 3
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.manager import ModelManager

class GraphRAGOllamaTier3:
    def __init__(self):
        # Configure GraphRAG with Ollama
        self.extraction_config = LanguageModelConfig(
            api_base="http://localhost:11434",
            type=ModelType.OllamaChat,
            model="llama3.2",  # For entity extraction
            max_retries=20,
        )
        
        self.query_config = LanguageModelConfig(
            api_base="http://localhost:11434", 
            type=ModelType.OllamaChat,
            model="mistral",  # For complex reasoning
            max_retries=20,
        )
        
        # Initialize models
        self.extraction_model = ModelManager().get_or_create_chat_model(
            name="extraction",
            model_type=ModelType.OllamaChat,
            config=self.extraction_config,
        )
        
        self.query_model = ModelManager().get_or_create_chat_model(
            name="query",
            model_type=ModelType.OllamaChat,
            config=self.query_config,
        )
    
    def multi_hop_reasoning(self, question, graph_data):
        # GraphRAG handles multi-hop reasoning automatically
        search_engine = GlobalSearch(
            model=self.query_model,
            context_builder=context_builder,
            token_encoder=token_encoder,
            max_data_tokens=12_000,
            json_mode=True,
            concurrent_coroutines=32
        )
        
        result = search_engine.search(question)
        return result.answer
```

**YAML Configuration**:
```yaml
# config.yaml for GraphRAG with Ollama
models:
  extraction_chat_model:
    type: ollama_chat
    model: llama3.2
    api_base: http://localhost:11434
  query_chat_model:
    type: ollama_chat
    model: mistral
    api_base: http://localhost:11434
  default_embedding_model:
    type: ollama_embedding
    model: bge-m3
    api_base: http://localhost:11434

extract_graph:
  model_id: extraction_chat_model
  prompt: "prompts/extract_graph.txt"
  entity_types: [organization,person,geo,event]
  max_gleanings: 1

global_search:
  chat_model_id: query_chat_model
  map_prompt: "prompts/global_search_map_system_prompt.txt"
  reduce_prompt: "prompts/global_search_reduce_system_prompt.txt"
```

## Complete System Integration

```python
# PSEUDO CODE - Complete Three-Tier System (Using Existing Libraries)
class ThreeTierRAGSystem:
    def __init__(self, approach="ollama_libraries"):
        self.approach = approach
        
        if approach == "existing":
            self.tier1 = LightRAGTier1()
            self.tier2 = PaperQA2Tier2()
            self.tier3 = GraphRAGTier3()
        else:  # ollama_libraries
            self.tier1 = LightRAGOllamaTier1()
            self.tier2 = PaperQA2OllamaTier2()
            self.tier3 = GraphRAGOllamaTier3()
        
        self.vector_store = ChromaDB()
        self.knowledge_graph = None
    
    def analyze_papers(self, question, papers):
        """Complete three-tier analysis using existing libraries with Ollama"""
        
        print("🔄 Starting three-tier analysis with existing libraries...")
        
        # Tier 1: LightRAG with Ollama - Entity extraction
        print("📊 Tier 1: LightRAG extracting entities and relationships...")
        entities, relationships = await self.tier1.extract_entities(papers)
        self.knowledge_graph = await self.tier1.build_knowledge_graph(entities, relationships)
        
        # Tier 2: PaperQA2 with Ollama - Research analysis
        print("🔬 Tier 2: PaperQA2 analyzing research content...")
        analysis_result = self.tier2.analyze_research(question, papers)
        
        # Tier 3: GraphRAG with Ollama - Graph reasoning
        print("🕸️ Tier 3: GraphRAG performing multi-hop reasoning...")
        reasoning_result = await self.tier3.multi_hop_reasoning(question, self.knowledge_graph)
        
        # Combine results from all three libraries
        final_result = self.combine_library_results(analysis_result, reasoning_result, entities, relationships)
        
        return final_result
    
    def combine_library_results(self, analysis_result, reasoning_result, entities, relationships):
        """Combine results from LightRAG, PaperQA2, and GraphRAG"""
        return {
            "question": analysis_result.get("question", ""),
            "analysis": analysis_result.get("analysis", ""),
            "reasoning": reasoning_result,
            "entities": entities,
            "relationships": relationships,
            "sources": analysis_result.get("sources", []),
            "citations": analysis_result.get("citations", []),
            "summary": self.generate_summary(analysis_result, reasoning_result)
        }

# Usage example
def main():
    # Initialize system with existing libraries + Ollama
    rag_system = ThreeTierRAGSystem(approach="ollama_libraries")
    
    # Load papers
    papers = rag_system.load_papers([
        "paper1.pdf",
        "paper2.pdf",
        "paper3.pdf"
    ])
    
    # Analyze using all three libraries
    question = "What are the main research gaps in this literature?"
    result = await rag_system.analyze_papers(question, papers)
    
    # Print results
    print("📋 ANALYSIS RESULTS (Using Existing Libraries + Ollama):")
    print(f"Question: {result['question']}")
    print(f"LightRAG Analysis: {result['analysis']}")
    print(f"GraphRAG Reasoning: {result['reasoning']}")
    print(f"Entities Found: {len(result['entities'])}")
    print(f"Relationships Found: {len(result['relationships'])}")
    print(f"Sources: {len(result['sources'])} papers")
    print(f"Citations: {len(result['citations'])} references")

if __name__ == "__main__":
    main()
```

## Comparison Analysis

### Cost Comparison

| Metric | Existing Library Approach | Ollama-Based Approach |
|--------|---------------------------|----------------------|
| **Setup Cost** | $0 | $0 |
| **Per Query Cost** | $2-3 | $0 |
| **20 Papers Analysis** | $40-60 | $0 |
| **Annual Cost (100 queries)** | $2,400-3,600 | $0 |

### Performance Comparison

| Metric | Existing Library Approach | Ollama-Based Approach |
|--------|---------------------------|----------------------|
| **Speed** | Fast (cloud APIs) | Fast (local processing with optimized libraries) |
| **Accuracy** | High (85%+ precision) | High (85%+ precision using existing libraries) |
| **Privacy** | Low (data sent to cloud) | High (local processing) |
| **Reliability** | High (cloud infrastructure) | High (production-ready libraries) |
| **Development Time** | Medium (integration) | Low (existing libraries) |
| **Maintenance** | Medium (API dependencies) | Low (self-contained) |

### Use Case Recommendations

#### Choose Existing Library Approach When:
- Budget is not a constraint
- Maximum accuracy is required
- Processing large datasets (>100 papers)
- Need guaranteed uptime
- Prefer cloud-based solutions

#### Choose Ollama-Based Approach When:
- Budget is limited
- Privacy is critical
- Processing moderate datasets (20-50 papers)
- Need customization
- Want to leverage production-ready libraries
- Prefer local processing
- Need rapid development and deployment

## Technical Requirements

### Hardware Requirements
- **CPU**: 8+ cores recommended
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)
- **Storage**: 50GB+ free space for models and data

### Software Requirements
- **OS**: Linux, macOS, or Windows
- **Python**: 3.9+
- **Ollama**: Latest version

### Dependencies
```bash
# Core dependencies for existing libraries with Ollama
pip install lightrag
pip install paper-qa
pip install graphrag
pip install ollama

# Additional dependencies
pip install sentence-transformers
pip install chromadb
pip install networkx
pip install pypdf
pip install pymupdf

# Optional: For enhanced performance
pip install torch
pip install transformers
```

## Success Metrics

### Primary Metrics
- **Accuracy**: >80% for research analysis
- **Cost**: <$5 for 20-paper analysis
- **Speed**: <30 minutes for complete analysis
- **User Satisfaction**: >4.0/5.0 rating

### Technical Metrics
- **Entity Extraction Precision**: >85%
- **Relationship Mapping Accuracy**: >80%
- **Contradiction Detection Rate**: >70%
- **Multi-hop Reasoning Success**: >75%

## Implementation Timeline

### Phase 1: Foundation (Week 1-2)
- Set up development environment
- Install and configure Ollama
- Implement basic document processing

### Phase 2: Core Implementation (Week 3-4)
- Implement Tier 1 (Entity Extraction)
- Implement Tier 2 (Research Analysis)
- Implement Tier 3 (Graph Reasoning)

### Phase 3: Testing & Optimization (Week 5-6)
- Unit testing
- Integration testing
- Performance optimization

### Phase 4: Documentation & Deployment (Week 7-8)
- API documentation
- User guide creation
- Production deployment

## Conclusion

This PRD outlines a comprehensive three-tier RAG system that can be implemented using either existing library approaches or Ollama-based local models. The Ollama-based approach offers significant cost savings while maintaining most of the functionality of the cloud-based solutions.

The system is designed to be:
- **Cost-effective**: $0-5 vs $40-60 for 20 papers
- **Privacy-preserving**: Local processing
- **Customizable**: Open-source components
- **Scalable**: Modular architecture
- **Reliable**: Error handling and fallbacks

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Author**: Research Agent Team  
**Status**: Ready for Implementation 