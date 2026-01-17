# PaperQA2 Comprehensive Guide: Academic-Focused Tier 2 RAG System

## Overview

PaperQA2 is a specialized Retrieval Augmented Generation (RAG) system designed for high-accuracy information retrieval and question answering from scientific documents. As the **Tier 2** component in our 3-tier RAG architecture, PaperQA2 serves as the academic-focused layer optimized for processing scientific literature, PDFs, and research papers with superhuman performance in academic tasks.

### What Makes PaperQA2 Special

PaperQA2 represents a significant advancement over general-purpose RAG systems by focusing specifically on academic and scientific content. Version 5+ (PaperQA2) achieves:

- **85.2% precision rate** in scientific information retrieval
- **66% accuracy rate** in comprehensive academic evaluations
- **Superhuman performance** compared to PhD and postdoc-level biology researchers
- **Citation-grounded responses** with verifiable academic sources

## Key Features

### 1. Academic-Optimized RAG Pipeline

**Three-Phase Processing:**
1. **Paper Search**: LLM generates keyword-based queries to retrieve candidate papers
2. **Evidence Gathering**: Query embedding comparison with document chunks, ranking, classification, and summarization
3. **Answer Generation**: Contextualized prompts with best summaries for final response

### 2. Advanced Citation System

```python
from paperqa.clients import DocMetadataClient, ALL_CLIENTS

client = DocMetadataClient(clients=ALL_CLIENTS)
details = await client.query(title="Augmenting language models with chemistry tools")

print(details.formatted_citation)
# Andres M. Bran, Sam Cox, Oliver Schilter, Carlo Baldassari, Andrew D. White, and Philippe Schwaller.
#  Augmenting large language models with chemistry tools. Nature Machine Intelligence,
# 6:525-535, May 2024. URL: https://doi.org/10.1038/s42256-024-00832-8,
# doi:10.1038/s42256-024-00832-8.
# This article has 243 citations and is from a domain leading peer-reviewed journal.

print(details.citation_count)  # 243
print(details.license)         # cc-by
print(details.pdf_url)         # https://www.nature.com/articles/s42256-024-00832-8.pdf
```

**Citation Features:**
- **Automatic metadata retrieval** from Crossref, Semantic Scholar, Unpaywall
- **Citation count tracking** with retraction checks
- **Journal quality assessment** for source credibility
- **In-text citations** with page numbers and source attribution
- **Detailed bibliographic formatting** following academic standards

### 3. Document Processing Excellence

**Supported Formats:**
- PDF files (primary focus)
- TXT, MD, HTML files
- Academic papers from arXiv, PubMed
- Clinical trial data from clinicaltrials.gov

**Processing Pipeline:**
```python
from paperqa import Docs, Settings

docs = Docs()
# Add documents with automatic metadata extraction
await docs.aadd("scientific_paper.pdf")
await docs.aadd("research_data.txt")

# Configure academic-optimized settings
settings = Settings()
settings.answer.evidence_k = 15  # More evidence for accuracy
settings.answer.answer_max_sources = 5
settings.parsing.chunk_size = 5000

session = await docs.aquery("What are the implications for drug discovery?", settings=settings)
print(session.formatted_answer)
```

### 4. Academic Integrity Features

**Source Attribution:**
- Every claim linked to specific document sections
- Page-level citation accuracy
- Contradiction detection between papers
- Journal impact factor consideration

**Quality Assurance:**
- Metadata validation across multiple sources
- Retraction checking via academic databases
- Journal quality scoring
- Author verification

## Architecture Deep Dive

### Core Classes

#### 1. `Docs` Class - Document Management Hub

```python
from paperqa import Docs

docs = Docs()

# Asynchronous document addition (preferred)
await docs.aadd("paper.pdf")
await docs.aadd_url("https://arxiv.org/pdf/2409.13740")

# Synchronous alternatives
docs.add("paper.pdf")
docs.add_file("local_document.txt")

# Advanced addition with metadata
await docs.aadd(
    "code_file.js", 
    citation="File " + os.path.basename("code_file.js"), 
    docname="javascript_analysis"
)
```

**Key Methods:**
- `aadd()` / `add()`: Add documents with automatic processing
- `aadd_url()` / `add_url()`: Add from web URLs
- `aquery()` / `query()`: Query the document collection
- `aget_evidence()` / `get_evidence()`: Retrieve supporting evidence

#### 2. `Settings` Class - Configuration Management

```python
from paperqa import Settings
from paperqa.settings import AgentSettings, AnswerSettings, ParsingSettings

settings = Settings(
    # LLM Configuration
    llm="gpt-4o-2024-11-20",
    summary_llm="gpt-4o-2024-11-20",
    embedding="text-embedding-3-small",
    temperature=0.0,
    
    # Answer Generation
    answer=AnswerSettings(
        evidence_k=10,                    # Evidence pieces to retrieve
        answer_max_sources=5,             # Max sources in final answer
        evidence_detailed_citations=True, # Include detailed citations
        answer_length="about 200 words, but can be longer"
    ),
    
    # Document Parsing
    parsing=ParsingSettings(
        chunk_size=5000,                  # Characters per chunk
        overlap=250,                      # Overlap between chunks
        use_doc_details=True              # Extract metadata
    ),
    
    # Agent Configuration
    agent=AgentSettings(
        agent_llm="gpt-4o-2024-11-20",
        search_count=8,                   # Search iterations
        timeout=500.0                     # Agent timeout seconds
    )
)
```

### 3. Integration Patterns

#### Academic Database Integration

```python
# Zotero Integration
from paperqa.contrib import ZoteroDB

zotero = ZoteroDB(library_type="user")  # or "group"
for item in zotero.iterate(limit=20):
    if item.num_pages <= 30:  # Skip long papers
        await docs.aadd(item.pdf, docname=item.key)

# OpenReview Integration
from paperqa.contrib.openreview_paper_helper import OpenReviewPaperHelper

helper = OpenReviewPaperHelper(settings, venue_id="ICLR.cc/2025/Conference")
question = "What is the progress on brain activity research?"
submissions = helper.fetch_relevant_papers(question)
docs = await helper.aadd_docs(submissions)
```

#### Multi-Model Configuration

```python
# Local Models (Ollama)
local_llm_config = {
    "model_list": [{
        "model_name": "ollama/llama3.2",
        "litellm_params": {
            "model": "ollama/llama3.2",
            "api_base": "http://localhost:11434",
        }
    }]
}

settings = Settings(
    llm="ollama/llama3.2",
    llm_config=local_llm_config,
    embedding="ollama/mxbai-embed-large"
)

# Google Gemini
settings = Settings(
    llm="gemini/gemini-2.0-flash",
    agent=AgentSettings(agent_llm="gemini/gemini-2.0-flash"),
    embedding="gemini/text-embedding-004"
)

# Anthropic Claude
settings = Settings(
    llm="claude-3-5-sonnet-20240620",
    summary_llm="claude-3-5-sonnet-20240620"
)
```

## Academic Focus: Why PaperQA2 Excels for Scientific Literature

### 1. Citation-Aware Processing

Unlike general RAG systems, PaperQA2 understands academic document structure:

```python
# Automatic citation extraction and formatting
session = await docs.aquery("What are the latest advances in drug delivery?")
print(session.references)
# Detailed academic references with page numbers, DOIs, citation counts

# Access individual citations
for context in session.contexts:
    print(f"Source: {context.text.name}")
    print(f"Citation: {context.text.doc.citation}")
    print(f"Content: {context.context}")
    print(f"Relevance Score: {context.score}")
```

### 2. Metadata-Enriched Embeddings

PaperQA2 embeds documents with academic metadata awareness:

- **Journal impact factors** influence relevance scoring
- **Citation counts** boost authoritative sources
- **Author credentials** considered in ranking
- **Publication dates** for temporal relevance

### 3. Contradiction Detection

```python
# Built-in contradiction detection
settings = Settings.from_name("contracrow")  # Contradiction detection mode
session = await docs.aquery("Claim: Vitamin D supplements prevent COVID-19")
# Returns analysis of contradictory evidence across papers
```

### 4. Academic Workflow Integration

```python
# Clinical trials integration
from paperqa.agents.tools import DEFAULT_TOOL_NAMES

settings = Settings(
    agent={"tool_names": DEFAULT_TOOL_NAMES + ["clinical_trials_search"]}
)

# Query combining papers and clinical data
response = await agent_query(
    "What drugs effectively treat Ulcerative Colitis?",
    settings=settings
)
```

## Performance Characteristics

### Speed vs Accuracy Trade-offs

**High Quality Mode:**
```python
settings = Settings.from_name("high_quality")
# - evidence_k = 15 (more evidence)
# - Uses ToolSelector agent
# - Higher cost, superior accuracy
```

**Fast Mode:**
```python
settings = Settings.from_name("fast")
# - Reduced evidence gathering
# - Simpler processing pipeline
# - Lower cost, faster responses
```

**Custom Optimization:**
```python
settings = Settings(
    answer=AnswerSettings(
        evidence_k=10,           # Balance speed/accuracy
        max_concurrent_requests=4, # Parallel processing
        evidence_skip_summary=False # Keep summarization
    ),
    parsing=ParsingSettings(
        chunk_size=3000,         # Smaller chunks for speed
        defer_embedding=True     # Batch embedding
    )
)
```

### Resource Requirements

- **Memory**: ~2-4GB for typical academic corpus (1000 papers)
- **Storage**: Vector index scales with document count
- **API Costs**: Optimized for academic budgets with rate limiting
- **Processing Time**: 30-60 seconds per query (high quality mode)

## API Reference

### Core Methods

```python
# Document Management
docs = Docs()
await docs.aadd(path, citation=None, docname=None, settings=None)
await docs.aadd_url(url, citation=None, docname=None, settings=None) 
await docs.aadd_file(file_path, citation=None, docname=None, settings=None)

# Querying
session = await docs.aquery(query, settings=None, callbacks=None)
evidence = await docs.aget_evidence(query, settings=None)

# Persistence
import pickle
with open("academic_docs.pkl", "wb") as f:
    pickle.dump(docs, f)
```

### CLI Interface

```bash
# Basic usage
pqa ask 'What is the mechanism of action for mRNA vaccines?'

# With specific settings
pqa --settings high_quality ask 'Compare CRISPR editing efficiency across studies'

# Index management
pqa -i immunology index           # Create named index
pqa -i immunology ask 'Query'     # Query specific index
pqa -i immunology search 'terms'  # Full-text search

# Configuration
pqa --temperature 0.1 --parsing.chunk_size 3000 ask 'Question'
```

### Advanced Configuration

```python
# Custom embedding models
from paperqa import HybridEmbeddingModel, SparseEmbeddingModel, LiteLLMEmbeddingModel

model = HybridEmbeddingModel(
    models=[LiteLLMEmbeddingModel(), SparseEmbeddingModel(ndim=1024)]
)
await docs.aadd("paper.pdf", embedding_model=model)

# Rate limiting
settings = Settings(
    llm_config={"rate_limit": {"gpt-4o-2024-11-20": "30000 per 1 minute"}},
    summary_llm_config={"rate_limit": {"gpt-4o-2024-11-20": "30000 per 1 minute"}}
)

# Custom prompts for domain-specific needs
from paperqa.prompts import qa_prompt
custom_qa_prompt = (
    "Answer the biological question '{question}' based on peer-reviewed research. "
    "Cite context using (Author2024) format. "
    "Context: {context}"
)
settings.prompts.qa = custom_qa_prompt
```

## Integration with 3-Tier RAG System

### Position in Architecture

**Tier 1 (LightRAG)**: Fast general knowledge retrieval
↓
**Tier 2 (PaperQA2)**: Academic literature processing ← **YOU ARE HERE**
↓
**Tier 3 (GraphRAG)**: Complex knowledge graph reasoning

### Handoff Patterns

```python
# From Tier 1 to PaperQA2
if query_requires_academic_sources(query):
    # Route to PaperQA2 for scholarly analysis
    academic_response = await paperqa_agent.aquery(query, settings=academic_settings)
    return academic_response

# From PaperQA2 to Tier 3
if response_requires_complex_reasoning(response):
    # Forward to GraphRAG for multi-hop analysis
    return await graphrag_agent.complex_query(response.knowledge_base)
```

### Data Flow

```python
# Academic query pipeline
async def academic_pipeline(query: str, docs_collection: Docs):
    # 1. PaperQA2 evidence gathering
    evidence = await docs_collection.aget_evidence(query)
    
    # 2. Citation-aware answer generation
    session = await docs_collection.aquery(query)
    
    # 3. Metadata enrichment
    enriched_response = {
        "answer": session.answer,
        "citations": [ctx.text.doc.citation for ctx in session.contexts],
        "evidence_quality": calculate_evidence_quality(session.contexts),
        "contradiction_check": detect_contradictions(session.contexts)
    }
    
    return enriched_response
```

## Latest Updates (2024-2025)

### Version 5+ Enhancements

1. **LiteLLM Integration**: Universal model compatibility
2. **Enhanced Metadata Pipeline**: Multi-source validation
3. **Improved Agent Framework**: ToolSelector agent with iterative refinement
4. **Clinical Trials Integration**: Direct clinicaltrials.gov querying
5. **Hybrid Embedding Support**: Sparse + dense vector combinations
6. **Rate Limiting**: Built-in cost control for academic budgets

### Recent Research Breakthroughs

- **Superhuman Performance**: Outperforms PhD-level researchers in information retrieval
- **Contradiction Detection**: Averages 2.34 contradictions per biology paper
- **Scale Analysis**: Enables literature analysis at unprecedented scale
- **Cost Efficiency**: Fraction of human researcher time and cost

### Future Roadmap

- **Multi-modal Document Processing**: Images, tables, figures
- **Real-time Literature Monitoring**: Automated paper tracking
- **Collaborative Annotation**: Shared academic knowledge bases
- **Domain-Specific Optimization**: Specialized models for biology, chemistry, physics

## Limitations

### Current Constraints

1. **PDF Processing**: Complex layouts may cause text extraction issues
2. **Language Support**: Primarily optimized for English academic literature
3. **Mathematical Content**: Limited support for complex equations and formulas
4. **Real-time Data**: No access to very recent publications (indexing delay)
5. **Full-Text Access**: Depends on open access or institutional subscriptions

### Mitigation Strategies

```python
# Handle PDF processing issues
settings = Settings(
    parsing=ParsingSettings(
        chunk_size=3000,          # Smaller chunks for complex layouts
        overlap=500,              # More overlap for continuity
        use_doc_details=True      # Extract metadata when possible
    )
)

# Multiple source validation
client = DocMetadataClient(clients=ALL_CLIENTS)  # Cross-reference multiple databases
```

## Integration Examples

### Basic Academic Research Assistant

```python
import asyncio
from paperqa import Docs, Settings

async def research_assistant():
    docs = Docs()
    
    # Load academic corpus
    papers = ["immunology_review.pdf", "vaccine_mechanisms.pdf", "clinical_trial_data.txt"]
    for paper in papers:
        await docs.aadd(paper)
    
    # Configure for academic accuracy
    settings = Settings.from_name("high_quality")
    
    # Research query
    session = await docs.aquery(
        "What are the mechanisms by which mRNA vaccines induce long-term immunity?",
        settings=settings
    )
    
    print("Academic Analysis:")
    print(session.formatted_answer)
    print("\nSources:")
    for ref in session.references:
        print(f"- {ref}")

asyncio.run(research_assistant())
```

### Literature Review Pipeline

```python
async def literature_review_pipeline(research_topic: str, paper_directory: str):
    from paperqa.agents import build_index, agent_query
    
    # 1. Build comprehensive index
    settings = Settings(
        agent=AgentSettings(
            index=IndexSettings(
                paper_directory=paper_directory,
                manifest_file="academic_manifest.csv",
                concurrency=10
            )
        )
    )
    
    await build_index(settings=settings)
    
    # 2. Generate research questions
    research_questions = [
        f"What are the current approaches to {research_topic}?",
        f"What are the limitations of existing {research_topic} methods?",
        f"What contradictions exist in {research_topic} literature?",
        f"What are the future directions for {research_topic} research?"
    ]
    
    # 3. Process each question
    review_sections = {}
    for question in research_questions:
        response = await agent_query(question, settings=settings)
        review_sections[question] = {
            "answer": response.session.answer,
            "citations": response.session.references,
            "evidence_quality": len(response.session.contexts)
        }
    
    return review_sections
```

This comprehensive guide positions PaperQA2 as the essential academic-focused component in our 3-tier RAG system, optimized for scholarly research, citation accuracy, and scientific literature analysis. Its specialized features make it indispensable for any application requiring high-accuracy academic information retrieval.