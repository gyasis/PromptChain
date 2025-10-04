# HybridRAG System

A comprehensive knowledge graph-based retrieval system that combines folder watching, document ingestion, and intelligent search capabilities using LightRAG and PromptChain.

## 🚀 Features

### 📁 Intelligent Folder Monitoring
- **Recursive watching** of multiple directories
- **Real-time detection** of new and modified files
- **Smart filtering** by file extensions and size limits
- **Deduplication** using SHA256 hashing
- **SQLite tracking** of processed files

### ⚡ Advanced Document Processing
- **Multi-format support**: TXT, MD, PDF, HTML, JSON, YAML, CSV, code files
- **Intelligent chunking** with token-aware splitting
- **PDF OCR capabilities** (optional)
- **Metadata preservation** throughout pipeline
- **Batch processing** with configurable concurrency

### 🧠 Sophisticated Search Interface
- **Multi-mode queries**: Local, Global, Hybrid, Agentic
- **Simple search**: Direct LightRAG queries
- **Agentic search**: Multi-hop reasoning using PromptChain
- **Multi-query synthesis**: Combine multiple searches
- **Interactive CLI** with command history

### 🔄 Production-Ready Architecture
- **Async/await** throughout for performance
- **Graceful error handling** and recovery
- **Comprehensive logging** with rotation
- **Health monitoring** and statistics
- **Signal handling** for clean shutdown

## 📋 Prerequisites

- Python 3.8+ 
- OpenAI API key
- Optional: Anthropic API key (for enhanced agentic features)

## 🛠️ Installation

1. **Clone or create the project directory**:
```bash
mkdir hybridrag && cd hybridrag
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install PromptChain** (for agentic features):
```bash
pip install git+https://github.com/gyasis/PromptChain.git
```

4. **Setup environment**:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

## 🎯 Usage

### Ingestion Mode
Continuously watch folders and ingest new documents:

```bash
python main.py ingestion
```

This will:
- Monitor configured folders for new/modified files
- Queue files for processing
- Extract text and create chunks
- Ingest into LightRAG knowledge graph
- Track processing status in SQLite

### Search Mode  
Interactive search interface:

```bash
python main.py search
```

Available commands:
- `search <query>` - Simple LightRAG search
- `agentic <query>` - Multi-hop reasoning search
- `multi <query1> | <query2>` - Multiple query synthesis
- `stats` - Show system statistics
- `history` - Show search history
- `health` - System health check
- `quit` - Exit

### One-Shot Query
Execute a single query and return JSON result:

```bash
# Simple search
python main.py query --query "What is machine learning?"

# Agentic search
python main.py query --query "Explain the relationship between AI and ML" --agentic
```

### System Status
Get comprehensive system status:

```bash
python main.py status
```

## ⚙️ Configuration

The system uses a hierarchical configuration in `config/config.py`:

### LightRAG Configuration
```python
@dataclass
class LightRAGConfig:
    working_dir: str = "./lightrag_db"
    model_name: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1200
    chunk_overlap: int = 100
```

### Ingestion Configuration
```python
@dataclass
class IngestionConfig:
    watch_folders: List[str] = ["./data"]
    file_extensions: List[str] = [".txt", ".md", ".pdf", ".json", ...]
    recursive: bool = True
    batch_size: int = 10
    max_file_size_mb: float = 50.0
```

### Search Configuration
```python
@dataclass
class SearchConfig:
    default_mode: str = "hybrid"
    default_top_k: int = 10
    enable_reranking: bool = True
    enable_context_accumulation: bool = True
```

## 📊 Query Modes

### Local Mode
Focus on specific entities and relationships:
```python
result = await search_interface.simple_search(
    "machine learning algorithms", 
    mode="local"
)
```

### Global Mode  
High-level overviews and summaries:
```python
result = await search_interface.simple_search(
    "overview of AI technologies",
    mode="global" 
)
```

### Hybrid Mode (Recommended)
Combines local and global approaches:
```python
result = await search_interface.simple_search(
    "deep learning applications",
    mode="hybrid"
)
```

### Agentic Mode
Multi-hop reasoning with tool access:
```python
result = await search_interface.agentic_search(
    "Compare machine learning and deep learning approaches",
    max_steps=5
)
```

## 🔧 Advanced Features

### Custom Document Processing
Extend the `DocumentProcessor` class to handle additional file types:

```python
class CustomDocumentProcessor(DocumentProcessor):
    def read_file(self, file_path: str, extension: str) -> str:
        if extension == '.custom':
            return self._read_custom_format(file_path)
        return super().read_file(file_path, extension)
```

### Custom Search Tools
Add domain-specific tools for agentic search:

```python
def custom_search_tool(query: str) -> str:
    # Your custom search logic
    return results

# Register with search interface
search_interface.register_tool(custom_search_tool)
```

### Monitoring and Metrics
The system provides comprehensive monitoring:

```python
# Get detailed statistics
stats = await system.get_system_status()

# Monitor ingestion progress
watcher_stats = folder_watcher.get_queue_stats()
ingestion_stats = ingestion_pipeline.get_stats()

# Track search performance
search_stats = search_interface.get_stats()
```

## 📁 Project Structure

```
hybridrag/
├── src/
│   ├── folder_watcher.py      # File monitoring system
│   ├── ingestion_pipeline.py  # Document processing
│   ├── lightrag_core.py       # LightRAG interface
│   └── search_interface.py    # Search functionality
├── config/
│   └── config.py              # Configuration classes
├── data/                      # Default watch folder
├── lightrag_db/              # LightRAG knowledge graph
├── ingestion_queue/          # Processing queue
├── logs/                     # System logs
├── main.py                   # Main orchestrator
├── requirements.txt          # Dependencies
├── .env.example              # Environment template
└── README.md                 # This file
```

## 🚨 Error Handling

The system implements comprehensive error handling:

- **File Processing Errors**: Files with errors are moved to `ingestion_queue/errors/`
- **API Rate Limits**: Built-in retry logic with exponential backoff
- **Network Issues**: Graceful degradation and fallback mechanisms
- **Resource Exhaustion**: Memory and disk usage monitoring
- **Graceful Shutdown**: Signal handling for clean termination

## 📈 Performance Tuning

### Ingestion Performance
- Adjust `batch_size` for throughput vs. memory usage
- Configure `max_concurrent_ingestions` based on system resources
- Set appropriate `poll_interval` for responsiveness vs. CPU usage

### Search Performance
- Use `local` mode for specific entity queries
- Use `global` mode for broad overviews
- Use `hybrid` mode for balanced results
- Enable `reranking` for improved relevance

### Memory Management
- Configure `chunk_size` and `chunk_overlap` for optimal indexing
- Set `max_file_size_mb` to prevent memory issues
- Monitor LightRAG database size and consider periodic cleanup

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PromptChain is installed for agentic features
2. **API Rate Limits**: Reduce batch sizes and increase delays
3. **Memory Issues**: Lower chunk sizes and concurrent processing
4. **File Processing**: Check file permissions and encoding
5. **LightRAG Errors**: Verify OpenAI API key and model access

### Debug Mode
Enable verbose logging:
```bash
export HYBRIDRAG_LOG_LEVEL=DEBUG
python main.py search
```

### Health Checks
Regular system health monitoring:
```bash
python main.py status | jq '.health'
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LightRAG**: Knowledge graph construction and querying
- **PromptChain**: Multi-hop reasoning and agent orchestration
- **OpenAI**: Language models and embeddings
- **Community**: Various document processing libraries

---

**Happy Searching! 🔍✨**