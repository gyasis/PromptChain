# RAG Libraries Data Ingestion Analysis

## Executive Summary

This analysis examines the data ingestion capabilities of LightRAG, PaperQA2, and GraphRAG to inform the design of a pipeline from Sci-Hub MCP → PDF processing → 3-tier RAG system. Each library has distinct strengths and requirements for handling academic papers and scientific documents.

## 1. LightRAG (Hong Kong University Data Science Lab)

### Supported Data Formats
- **PDF Files**: Primary format, processed via textract
- **Microsoft Office**: DOC, DOCX, PPT, PPTX, XLS, XLSX
- **Text Files**: TXT, MD (Markdown)
- **Structured Data**: CSV, JSON, XML, HTML
- **Code Files**: Python, Java, and other programming languages

### PDF Processing Capabilities
- **Built-in Support**: Yes, via textract integration (as of October 2024)
- **Multimodal Processing**: Handles text, images, tables, and formulas through RAG-Anything integration
- **Processing Method**: Automatic text extraction with structured content preservation
- **Large File Handling**: Recommends dividing large files into smaller segments for incremental processing

### Document Chunking/Preprocessing
- **Default Chunk Size**: 1200 tokens
- **Default Overlap**: 100 tokens
- **Configurable Parameters**: `chunk_token_size` and `chunk_overlap_token_size`
- **Chunking Strategies**: 
  - Fixed-size chunking (default)
  - Recursive chunking
  - Semantic chunking
  - Document-based chunking
  - Hybrid approaches

### Metadata Requirements
- **Minimal Requirements**: No specific metadata fields required
- **Automatic Processing**: Performs entity extraction and relationship mapping automatically
- **Deduplication**: Merges identical entities and relationships to reduce overhead
- **Graph Construction**: Builds knowledge graphs automatically during ingestion

### Document Ingestion Pattern
```python
from lightrag import LightRAG

# Initialize LightRAG
rag = LightRAG(
    working_dir="./lightrag_working_dir",
    chunk_token_size=1200,
    chunk_overlap_token_size=100
)

# Ingest documents
await rag.ainsert(document_text)  # For text
# Or for files: supports PDF, DOC, PPT, CSV via extract
```

### Key Strengths for Academic Papers
- **Knowledge Graph Focus**: Excellent for entity and relationship extraction
- **Multimodal Support**: Can handle complex academic documents with figures and tables
- **Fast Processing**: Optimized for speed and performance
- **Local Processing**: Can run entirely locally with Ollama models

## 2. PaperQA2 (FutureHouse)

### Supported Data Formats
- **PDF Files**: Primary format for academic papers
- **Text Files**: TXT, MD
- **HTML**: Web-based content
- **Focus**: Specifically optimized for scientific literature

### PDF Processing Capabilities
- **Built-in Support**: Yes, with sophisticated academic paper processing
- **Document Parser**: Uses Grobid (state-of-the-art document parsing algorithm)
- **Academic Optimization**: Specifically designed for scientific paper structure
- **Automatic Processing**: Handles complex academic document layouts

### Document Chunking/Preprocessing
- **Intelligent Chunking**: Context-aware chunking for scientific content
- **Evidence Contexts**: Creates "chunked evidence contexts" for better retrieval
- **LLM-based Processing**: Uses LLM-based re-ranking and contextual summarization (RCS)
- **Metadata-aware Embeddings**: Incorporates document metadata into embedding process

### Metadata Requirements
- **Automatic Retrieval**: Fetches metadata automatically from:
  - Crossref API (requires API key for large datasets)
  - Semantic Scholar (requires API key for large datasets) 
  - Unpaywall
- **Metadata Fields**: 
  - DOI
  - Title
  - Authors
  - Citation counts
  - Journal quality data
  - Retraction status
- **Rate Limiting**: API keys recommended for 100+ papers to avoid rate limits

### Document Ingestion Pattern
```python
from paperqa import Settings, ask
import os

# Set API keys for metadata services
os.environ["CROSSREF_API_KEY"] = "your_key"
os.environ["SEMANTIC_SCHOLAR_API_KEY"] = "your_key"

# Configure settings
settings = Settings(
    llm="gpt-4o-2024-11-20",
    embedding="text-embedding-3-small"
)

# Ingest directory of PDFs
built_index = await get_directory_index(
    path="./papers",
    settings=settings
)

# Query the indexed papers
answer = ask(
    "Your research question",
    settings=settings
)
```

### Key Strengths for Academic Papers
- **Scientific Focus**: Purpose-built for academic literature analysis
- **High Accuracy**: 85.2% precision, 66% accuracy for scientific Q&A
- **Citation Integration**: Automatic citation and bibliographic management
- **Contradiction Detection**: Can identify conflicting information across papers
- **Research Workflow**: Optimized for literature review and research synthesis

## 3. GraphRAG (Microsoft Research)

### Supported Data Formats
- **Plain Text**: TXT files (entire content imported)
- **CSV**: Structured data with configurable columns
- **JSON**: Single objects or arrays with flexible schema
- **Note**: No direct PDF support - requires preprocessing to supported formats

### PDF Processing Capabilities
- **No Built-in Support**: Requires external PDF-to-text conversion
- **Preprocessing Required**: Must convert PDFs to TXT, CSV, or JSON format
- **Format Recommendation**: Convert PDFs while preserving structural elements (headers, lists, tables)

### Document Chunking/Preprocessing
- **Text Units**: Documents split into smaller "text units" (chunks)
- **Configurable Parameters**:
  - `chunk_size`: Recommended 500-800 tokens
  - `overlap`: Configurable overlap between chunks
- **Pre-chunked Support**: Can accept pre-chunked documents via CSV format
- **Metadata Integration**: Can prepend metadata to each chunk

### Metadata Requirements
- **DataFrame Schema**: Requires specific column structure:
  - `id`: Unique document identifier
  - `text`: Document content
  - `title`: Document title
  - `creation_date`: Optional timestamp
  - `metadata`: Optional structured metadata
- **Custom Metadata**: Supports additional custom columns
- **Metadata Propagation**: Can prepend metadata to text chunks for context

### Document Ingestion Pattern
```python
# Input DataFrame structure
documents_df = pd.DataFrame({
    'id': ['doc1', 'doc2'],
    'text': ['Document content...', 'More content...'],
    'title': ['Paper 1', 'Paper 2'],
    'creation_date': ['2024-01-01', '2024-01-02'],
    'metadata': [{'author': 'Smith'}, {'author': 'Jones'}]
})

# Configure chunking
settings = {
    'chunk_size': 600,
    'chunk_overlap': 100,
    'prepend_metadata': True
}
```

### Key Strengths for Academic Papers
- **Graph Reasoning**: Excellent for multi-hop reasoning and complex relationships
- **Community Detection**: Identifies related concepts and themes across documents
- **Hierarchical Structure**: Creates hierarchical knowledge representations
- **Global and Local Search**: Supports both focused and broad query types

## Pipeline Design Recommendations

### For Sci-Hub MCP → 3-Tier RAG Pipeline

#### Tier 1: LightRAG for Entity Extraction
**Recommended Approach:**
- Use LightRAG's built-in PDF processing via textract
- Configure for academic content: 1200 token chunks, 100 token overlap
- Leverage automatic entity and relationship extraction
- Build knowledge graph for foundational understanding

**PDF Processing Pipeline:**
```python
# Stage 1: PDF Ingestion
rag_tier1 = LightRAG(
    working_dir="./tier1_working",
    chunk_token_size=1200,
    chunk_overlap_token_size=100
)

# Direct PDF processing
await rag_tier1.ainsert(pdf_text_content)
```

#### Tier 2: PaperQA2 for Research Analysis
**Recommended Approach:**
- Use PaperQA2's automatic metadata retrieval
- Leverage Grobid for sophisticated academic parsing
- Focus on research question answering and synthesis
- Maintain citation tracking and source attribution

**PDF Processing Pipeline:**
```python
# Stage 2: Academic Analysis
settings = Settings(
    llm="gpt-4o-2024-11-20",
    embedding="text-embedding-3-small"
)

# Automatic PDF processing with metadata
index = await get_directory_index(
    path="./papers_from_scihub",
    settings=settings
)
```

#### Tier 3: GraphRAG for Knowledge Graph Reasoning
**Recommended Approach:**
- Convert processed content to structured format (JSON/CSV)
- Include rich metadata from PaperQA2's processing
- Configure for academic content chunking (600-800 tokens)
- Enable hierarchical reasoning and community detection

**PDF Processing Pipeline:**
```python
# Stage 3: Graph Construction
# Convert processed papers to GraphRAG format
documents_df = pd.DataFrame({
    'id': paper_ids,
    'text': processed_texts,
    'title': paper_titles,
    'metadata': paper_metadata  # From PaperQA2
})

# Configure for academic reasoning
graph_settings = {
    'chunk_size': 700,
    'chunk_overlap': 100,
    'prepend_metadata': True
}
```

### Integration Strategy

1. **PDF Acquisition**: Sci-Hub MCP retrieves academic papers
2. **Tier 1 Processing**: LightRAG ingests PDFs directly, extracts entities and relationships
3. **Tier 2 Enhancement**: PaperQA2 processes same PDFs with metadata enrichment and academic analysis
4. **Tier 3 Reasoning**: GraphRAG consumes structured output from Tiers 1&2 for complex reasoning
5. **Unified Interface**: Combine outputs for comprehensive academic research capabilities

### Preprocessing Requirements Summary

| Library | PDF Support | Chunking | Metadata | Preprocessing |
|---------|-------------|----------|----------|---------------|
| LightRAG | ✅ Built-in via textract | 1200 tokens, 100 overlap | Automatic extraction | Minimal - direct ingestion |
| PaperQA2 | ✅ Grobid parser | Intelligent academic chunks | Auto from Crossref/S2 | Academic structure aware |
| GraphRAG | ❌ Requires conversion | 500-800 tokens recommended | Structured DataFrame | Convert to TXT/CSV/JSON |

### Cost and Performance Considerations

- **LightRAG**: Most efficient for entity extraction, local processing capable
- **PaperQA2**: Best accuracy for academic Q&A, requires API keys for metadata at scale
- **GraphRAG**: Computationally intensive but excellent for complex reasoning

This analysis provides the foundation for implementing an effective Sci-Hub → PDF → 3-tier RAG pipeline optimized for academic research workflows.