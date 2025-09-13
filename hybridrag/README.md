# LightRAG + DeepLake Hybrid RAG System

A proof-of-concept system that ingests medical table descriptions from DeepLake's Athena database into LightRAG for knowledge graph-based querying.

## Overview

This system demonstrates:
- **Data Ingestion**: Extracting JSONL-structured data from DeepLake (15,149 medical table descriptions)
- **Knowledge Graph Creation**: LightRAG automatically builds a knowledge graph from the ingested documents
- **Multi-Mode Querying**: Support for local, global, hybrid, naive, and mix query modes
- **Interactive Demo**: Command-line interface for exploring the medical database schema

## Architecture

```
DeepLake (athena_descriptions_v4)
    ↓
    [JSONL structured text with embeddings]
    ↓
deeplake_to_lightrag.py
    ↓
    [Formatted documents]
    ↓
LightRAG Database (./athena_lightrag_db)
    ↓
lightrag_query_demo.py
    ↓
[Interactive Query Interface]
```

## Setup

### Prerequisites
- [UV](https://docs.astral.sh/uv/) package manager
- Python 3.9+
- OpenAI API key

### 1. Quick Setup

Run the automated environment setup:

```bash
# Make the activation script executable and run it
chmod +x activate_env.sh
./activate_env.sh
```

### 2. Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Install dependencies with UV (creates isolated environment)
uv sync --no-install-project

# Or install with pip (not recommended - no isolation)
pip install -r requirements.txt
```

### 3. Environment Variables

Copy the example environment file and add your API key:

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

Example `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
ACTIVELOOP_TOKEN=your_activeloop_token_here  # Optional
ACTIVELOOP_USERNAME=your_username_here        # Optional
```

### 4. Running Scripts

**With UV (Recommended - Isolated Environment):**
```bash
# Run data ingestion
uv run python deeplake_to_lightrag.py

# Run query demo
uv run python lightrag_query_demo.py
```

**With Activated Environment:**
```bash
# Activate the UV environment
source .venv/bin/activate

# Run scripts normally
python deeplake_to_lightrag.py
python lightrag_query_demo.py
```

### 5. Environment Validation

Verify your setup:

```bash
# Check environment status
./activate_env.sh

# Or manually validate
uv run python -c "import lightrag, deeplake, openai; print('✅ All dependencies working!')"
```

### 6. Data Ingestion

This will read all 15,149 records from DeepLake and ingest them into LightRAG:

```bash
uv run python deeplake_to_lightrag.py
```

**Note**: First-time ingestion may take 30-60 minutes depending on API rate limits.

### 7. Query the Database

Launch the interactive query interface:

```bash
uv run python lightrag_query_demo.py
```

## Query Modes

- **local**: Focus on specific entity relationships (best for detailed questions)
- **global**: High-level overview and summaries (best for broad questions)
- **hybrid**: Combination of local and global (default, balanced approach)
- **naive**: Simple retrieval without graph structure
- **mix**: Mixed approach with vector and graph retrieval

## Example Queries

```
> What tables are related to patient appointments?
> Describe the anesthesia case management tables
> Show me all collector category tables
> What is the structure of allowable schedule categories?
> List tables with high row counts
```

## Interactive Commands

- `/mode <mode>` - Change query mode (local/global/hybrid/naive/mix)
- `/context` - Toggle context-only mode (shows retrieved context without generating response)
- `/topk <n>` - Set number of results to retrieve
- `/clear` - Clear screen
- `/help` - Show help
- `exit` or `quit` - Exit the demo

## Data Structure

Each DeepLake record contains:
- **id**: Unique identifier
- **text**: JSONL with table metadata (TABLEID, SCHEMANAME, TABLE DESCRIPTION, etc.)
- **embedding**: 1536-dimensional vector embeddings
- **metadata**: Table name, ID, and source file

## Files

- `deeplake_to_lightrag.py` - Main ingestion script
- `lightrag_query_demo.py` - Interactive query interface
- `requirements.txt` - Python dependencies
- `athena_lightrag_db/` - LightRAG database (created after ingestion)

## Performance Notes

- Initial ingestion processes in batches to avoid rate limits
- Query response time: 2-5 seconds depending on mode
- LightRAG builds and maintains a knowledge graph for efficient querying
- The system can handle complex multi-hop questions about table relationships

## Future Enhancements

1. **Hybrid Search**: Combine LightRAG graph search with DeepLake vector search
2. **Query Routing**: Intelligent routing between LightRAG and DeepLake based on query type
3. **Performance Optimization**: Parallel processing and caching
4. **Web Interface**: Build a web UI for easier interaction
5. **Advanced Analytics**: Add statistical analysis of table relationships

## Troubleshooting

- **OpenAI API Key Error**: Ensure your API key is set in the `.env` file
- **Rate Limit Errors**: The ingestion script includes delays; if errors persist, reduce batch_size
- **Database Not Found**: Run the ingestion script before the query demo
- **Memory Issues**: For large datasets, consider processing in smaller chunks

## License

MIT