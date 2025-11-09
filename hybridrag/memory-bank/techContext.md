# HybridRAG Technical Context

## Technology Stack

### Core Dependencies
**Primary Framework Components**
- **LightRAG-HKU v0.1.0+**: Knowledge graph construction and multi-hop reasoning
- **DeepLake v4.0.0+**: Vector database with 1536-dimensional embeddings  
- **OpenAI API v1.0.0+**: Embeddings (text-embedding-ada-002) and language models
- **Python 3.9+**: Runtime environment with async/await support

### Development Environment
**UV Package Manager Integration**
```toml
[project]
name = "hybridrag"
version = "0.1.0"
requires-python = ">=3.9"

dependencies = [
    "lightrag-hku>=0.1.0",
    "deeplake>=4.0.0", 
    "openai>=1.0.0",
    "python-dotenv>=1.0.0",
    "colorama>=0.4.6",
    "numpy>=1.24.0",
    "asyncio-throttle>=1.0.0",
    "tqdm>=4.65.0",
]
```

### Development Tools
**Code Quality and Testing**
- **Black v23.0.0+**: Code formatting with 88-character line length
- **isort v5.12.0+**: Import sorting with Black profile compatibility
- **flake8 v6.0.0+**: Linting and style checking
- **mypy v1.0.0+**: Static type checking with strict configuration
- **pytest v7.0.0+**: Testing framework with async support

## Architecture Constraints

### API Dependencies
**OpenAI API Integration Requirements**
- **Embeddings**: text-embedding-ada-002 model for 1536-dimensional vectors
- **Language Models**: GPT-3.5/GPT-4 for knowledge graph construction and querying
- **Rate Limits**: Must handle API throttling and implement backoff strategies
- **Cost Consideration**: Embedding generation and LLM calls can be expensive at scale

### Storage Architecture
**File System Layout**
```
hybridrag/
├── deeplake_to_lightrag.py           # Ingestion pipeline
├── lightrag_query_demo.py            # Interactive interface  
├── athena_lightrag_db/               # LightRAG knowledge graph storage
│   ├── kv_store_*.json              # Key-value stores for entities/relations
│   └── vdb_*.json                   # Vector database files
├── new_deeplake_features/            # Enhanced DeepLake functionality
│   └── customdeeplake_v4.py         # Recency-based search
└── .venv/                           # Isolated UV environment
```

### Performance Characteristics
**Scalability Constraints**
- **Ingestion Speed**: ~50 documents per batch with 0.5s delays (rate limit management)
- **Query Response**: Target <5 seconds for complex multi-hop queries
- **Memory Usage**: LightRAG loads full knowledge graph into memory
- **Dataset Size**: Tested with 15,149 medical table records successfully

## Data Architecture

### Source Data Format
**DeepLake athena_descriptions_v4 Structure**
```python
{
    'id': 'unique_identifier',
    'text': 'JSONL_structured_metadata',  # Medical table description
    'embedding': [1536_dimensional_vector],  # OpenAI embeddings
    'metadata': {
        'table_name': 'string',
        'table_id': 'string', 
        'source_file': 'string'
    }
}
```

### Processed Document Format
**LightRAG Input Structure**
```python
# Formatted for knowledge graph construction
document = f"""
Medical Table {table_id} in schema {schema_name}: {description}
Row Count: {row_count}
Columns: {column_info}
Categories: {table_categories}
"""
```

### Knowledge Graph Schema
**LightRAG Generated Entities and Relations**
- **Entities**: Medical tables, schemas, procedures, departments, data categories
- **Relations**: Table dependencies, schema memberships, procedural connections
- **Attributes**: Row counts, column information, medical categories, descriptions

## Integration Patterns

### Async/Sync Bridge Pattern
**Dual Execution Model**
```python
# Async for batch processing (ingestion)
async def process_documents_async(self, batch_size: int = 50):
    batches = self.create_batches(batch_size)
    tasks = [self.process_batch(batch) for batch in batches]
    await asyncio.gather(*tasks)

# Sync for interactive queries  
def query_database(self, query: str, mode: str = "hybrid") -> str:
    return self.rag.query(query, param=QueryParam(mode=mode))
```

### Error Handling Strategy
**Graceful Degradation Pattern**
- **API Failures**: Exponential backoff with retry logic
- **Rate Limits**: Automatic delay adjustment and progress feedback
- **Partial Processing**: Continue batch processing despite individual failures
- **Validation Errors**: Clear error messages with remediation suggestions

## Configuration Management

### Environment Variables
**Required Configuration**
```bash
OPENAI_API_KEY=sk-...                    # Required for embeddings and LLM
ACTIVELOOP_TOKEN=...                     # Optional for DeepLake cloud features
ACTIVELOOP_USERNAME=...                  # Optional for DeepLake authentication
```

### Runtime Configuration
**Configurable Parameters**
```python
# Ingestion Configuration
batch_size: int = 50                     # Documents per batch
rate_limit_delay: float = 0.5            # Seconds between batches
working_dir: str = "./athena_lightrag_db" # LightRAG storage location

# Query Configuration  
default_mode: str = "hybrid"             # Default query mode
top_k: int = 10                          # Number of results to retrieve
context_only: bool = False               # Show context without generation
```

## Development Workflow

### Environment Setup
**UV-Based Isolated Development**
```bash
# Environment creation and activation
uv sync --no-install-project
source .venv/bin/activate

# Development execution
uv run python deeplake_to_lightrag.py    # Batch ingestion
uv run python lightrag_query_demo.py     # Interactive queries

# Environment validation
uv run python -c "import lightrag, deeplake, openai; print('✅ Dependencies OK')"
```

### Code Quality Pipeline
**Automated Formatting and Checking**
```bash
# Code formatting
black . --line-length 88
isort . --profile black

# Quality checking
flake8 . --max-line-length 88
mypy . --strict

# Testing
pytest --asyncio-mode=auto
```

## Performance Considerations

### Optimization Strategies
**Current Implementation Choices**
- **Batch Processing**: 50-document batches balance memory usage and API efficiency
- **Rate Limiting**: 0.5s delays prevent API throttling while maintaining progress
- **Memory Management**: Stream processing avoids loading entire dataset into memory
- **Async Operations**: Concurrent batch processing where API limits allow

### Monitoring Points
**Performance Bottlenecks**
- **API Latency**: OpenAI embedding and completion call times
- **Knowledge Graph Construction**: LightRAG entity/relation extraction time
- **Query Processing**: Multi-hop reasoning complexity vs. response time
- **Memory Usage**: Large knowledge graphs can consume significant RAM

## Known Limitations

### Technical Constraints  
- **API Dependency**: Requires stable internet connection and OpenAI API availability
- **Memory Requirements**: Full knowledge graph loaded into memory during queries
- **Sequential Processing**: Ingestion pipeline processes batches sequentially
- **Single Database**: Currently focused on single DeepLake dataset

### Scaling Considerations
- **Dataset Size**: Performance degrades with >50K documents without optimization
- **Query Complexity**: Very complex multi-hop queries may exceed response time targets
- **Concurrent Users**: Single-user interactive interface, no multi-user support
- **API Costs**: Large-scale usage requires API cost management strategies

## Future Technical Enhancements

### Infrastructure Improvements
1. **Parallel Processing**: Multi-threaded ingestion with API rate limit pooling
2. **Caching Layer**: Redis or similar for query result caching
3. **Database Optimization**: Incremental knowledge graph updates
4. **Monitoring**: Comprehensive performance and usage analytics

### Integration Capabilities
1. **Multi-Database Support**: Handle multiple DeepLake datasets simultaneously
2. **Real-time Updates**: Live schema change detection and integration
3. **API Gateway**: RESTful API for programmatic access
4. **Web Interface**: Browser-based UI for broader accessibility

### Advanced Features
1. **Query Optimization**: Intelligent query routing based on complexity
2. **Federated Search**: Combine multiple knowledge graphs
3. **Machine Learning**: Query pattern analysis and optimization
4. **Natural Language to SQL**: Direct database query generation with schema context

This technical foundation supports the current proof-of-concept while providing clear paths for production-scale deployment and advanced feature development.