# HybridRAG System Patterns & Architecture

## Core Architecture Overview

### Hybrid RAG Design Pattern
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │────│ Ingestion Layer │────│ Knowledge Graph │
│   (DeepLake)    │    │                 │    │   (LightRAG)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         │              ┌─────────────────┐                │
         │              │ Processing Core │                │
         │              │ - Batch Logic   │                │
         │              │ - Rate Limiting │                │
         │              │ - Progress Track│                │
         │              └─────────────────┘                │
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                      ┌─────────────────┐
                      │ Query Interface │
                      │ - Multi-Mode    │
                      │ - Interactive   │
                      │ - Colored CLI   │
                      └─────────────────┘
```

### Component Interaction Patterns

#### 1. Data Flow Architecture
**DeepLake → Ingestion → LightRAG → Query Interface**
- **One-way Flow**: Data moves from DeepLake through ingestion to LightRAG
- **Batch Processing**: Handles large datasets efficiently with progress tracking
- **Knowledge Graph Construction**: LightRAG automatically builds entity-relationship graphs
- **Multi-Mode Querying**: Different reasoning approaches based on query type

#### 2. Dual Interface Pattern
**Async/Sync Flexibility**
```python
# Async Pattern (for batch operations)
async def process_documents_async(self, batch_size: int = 50):
    await asyncio.gather(*[self.process_batch(batch) for batch in batches])

# Sync Pattern (for interactive queries)  
def query_interface(self, query: str, mode: str = "hybrid"):
    return self.rag.query(query, param=QueryParam(mode=mode))
```

#### 3. Progress Tracking Pattern
**Transparent Operations with User Feedback**
- TQDM progress bars for long-running operations
- Colored console output for status indication
- Batch-wise processing with intermediate checkpoints
- Rate limiting with visible delays and reasoning

### Technical Architecture Decisions

#### Storage Strategy
**Separated Concerns with Clear Boundaries**
- **DeepLake**: Source of truth for raw medical table data with embeddings
- **LightRAG**: Knowledge graph construction and multi-hop reasoning
- **File System**: Configuration, logs, and intermediate processing data

#### Query Mode Architecture
**Strategy Pattern Implementation**
```python
query_modes = {
    "local": "Focus on specific entity relationships",
    "global": "High-level overview and summaries", 
    "hybrid": "Combination of local and global (default)",
    "naive": "Simple retrieval without graph structure",
    "mix": "Mixed approach with vector and graph retrieval"
}
```

#### Error Handling Pattern
**Graceful Degradation with Informative Feedback**
- API rate limit handling with exponential backoff
- Partial success reporting for batch operations
- Context preservation during failures
- Clear error messages with suggested remediation

### Component Design Patterns

#### 1. DeepLakeToLightRAG Class
**Orchestration Pattern**
```python
class DeepLakeToLightRAG:
    """Handles conversion from DeepLake to LightRAG documents"""
    
    # Configuration Management
    def __init__(self, deeplake_path, lightrag_working_dir, api_key)
    
    # Data Processing Pipeline
    def load_deeplake_dataset(self) -> Dataset
    def format_document_for_lightrag(self, record) -> str
    def process_documents_in_batches(self, batch_size: int)
    
    # Async/Sync Bridge
    def run_async_ingestion(self)
    async def process_documents_async(self, batch_size: int)
```

#### 2. LightRAGQueryInterface Class  
**Interactive Interface Pattern**
```python
class LightRAGQueryInterface:
    """Interactive query interface for LightRAG database"""
    
    # Initialization and Setup
    def __init__(self, working_dir: str)
    def setup_lightrag(self) -> LightRAG
    
    # Interactive Command Loop
    def run_demo(self)
    def process_command(self, user_input: str)
    def handle_mode_change(self, mode: str)
    
    # Query Processing
    def query_database(self, query: str, mode: str) -> str
    def format_response(self, response: str) -> str
```

#### 3. CustomDeepLake Enhancement Pattern
**Extension Without Modification**
- Extends base DeepLake functionality
- Adds recency-based search capabilities
- Maintains backward compatibility
- Comprehensive documentation and examples

### Data Processing Patterns

#### Batch Processing Strategy
**Memory-Efficient Large Dataset Handling**
```python
def process_documents_in_batches(self, batch_size: int = 50):
    """Process documents in manageable chunks"""
    total_records = len(self.dataset)
    
    for i in tqdm(range(0, total_records, batch_size)):
        batch = self.dataset[i:i+batch_size]
        formatted_docs = [self.format_document_for_lightrag(record) 
                         for record in batch]
        await self.rag.ainsert(formatted_docs)
        
        # Rate limiting and progress feedback
        if i + batch_size < total_records:
            await asyncio.sleep(0.5)
```

#### Document Formatting Pattern
**Structured Data to Natural Language**
```python
def format_document_for_lightrag(self, record) -> str:
    """Convert JSONL medical table data to readable document"""
    parsed_data = json.loads(record['text'].data()['value'])
    
    # Extract key medical metadata
    table_id = parsed_data.get('TABLEID', 'Unknown')
    schema_name = parsed_data.get('SCHEMANAME', 'Unknown')
    description = parsed_data.get('TABLE DESCRIPTION', 'No description')
    
    # Format for knowledge graph construction
    return f"Medical Table {table_id} in schema {schema_name}: {description}"
```

### Query Processing Architecture

#### Multi-Mode Query Strategy
**Context-Aware Query Routing**
- **Local Mode**: Entity-focused queries with specific relationship exploration
- **Global Mode**: High-level summaries and broad pattern identification  
- **Hybrid Mode**: Balanced approach combining local detail with global context
- **Naive Mode**: Simple vector retrieval without graph reasoning
- **Mix Mode**: Advanced combining multiple retrieval strategies

#### Interactive Command Processing
**Command Pattern with State Management**
```python
def process_command(self, user_input: str) -> bool:
    """Process interactive commands with state changes"""
    if user_input.startswith('/mode '):
        self.current_mode = user_input.split()[1]
    elif user_input.startswith('/context'):
        self.context_only = not self.context_only  
    elif user_input.startswith('/topk '):
        self.top_k = int(user_input.split()[1])
    else:
        # Regular query processing
        return self.query_database(user_input, self.current_mode)
```

### Performance Patterns

#### Rate Limiting Strategy
**API-Friendly Batch Processing**
- Configurable delays between API calls
- Exponential backoff for rate limit errors
- Progress visibility during long operations
- Batch size tuning based on API limits

#### Memory Management Pattern
**Efficient Large Dataset Handling**
- Stream processing for large DeepLake datasets
- Batch-wise memory allocation and cleanup
- Lazy loading of LightRAG components
- Garbage collection between batches

### Integration Patterns

#### Environment Isolation Pattern
**UV-Based Dependency Management**
```bash
# Isolated environment creation
uv sync --no-install-project

# Execution within isolated environment  
uv run python deeplake_to_lightrag.py
uv run python lightrag_query_demo.py
```

#### Configuration Management Pattern
**Environment-Based Configuration**
```python
# .env file pattern
OPENAI_API_KEY=your_key_here
ACTIVELOOP_TOKEN=optional_token
ACTIVELOOP_USERNAME=optional_username

# Runtime configuration
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

### Testing and Quality Patterns

#### Validation Strategy
**Multi-Layer Verification**
- Environment validation before processing
- API key validation during initialization  
- Dataset existence checking before ingestion
- Query result validation and formatting
- Interactive command validation and feedback

#### Logging and Monitoring Pattern
**Comprehensive Operation Tracking**
- Structured logging with timestamps and levels
- Progress tracking with visual feedback
- Error logging with context preservation
- Performance metrics collection

This architecture enables scalable, maintainable, and user-friendly medical database schema exploration through intelligent knowledge graph construction and multi-mode querying capabilities.