# Sci-Hub MCP Server Integration Setup Guide

## Overview

This document provides comprehensive setup and configuration instructions for integrating the Sci-Hub MCP Server with the Research Agent system. The integration enables automated academic paper discovery and download through the Model Context Protocol (MCP).

## Installation Status

✅ **COMPLETED** - The Sci-Hub MCP Server is already installed as a uv dependency and configured for use.

## Configuration Files

### MCP Configuration (`/config/mcp_config.json`)

The main configuration file contains:

```json
{
  "mcpServers": {
    "scihub": {
      "command": "/home/gyasis/.local/bin/uv",
      "args": ["run", "--directory", "/home/gyasis/Documents/code/PromptChain/Research_agent", "python", "-m", "sci_hub_server"],
      "cwd": "/home/gyasis/Documents/code/PromptChain/Research_agent",
      "env": {
        "SCIHUB_BASE_URL": "https://sci-hub.se",
        "SCIHUB_TIMEOUT": "30",
        "SCIHUB_MAX_RETRIES": "3",
        "SCIHUB_USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
      },
      "enabled": true,
      "transport": "stdio",
      "description": "Sci-Hub MCP Server for academic paper search and download"
    }
  }
}
```

### Environment Variables

The following environment variables are configured for the Sci-Hub server:

| Variable | Value | Purpose |
|----------|--------|---------|
| `SCIHUB_BASE_URL` | `https://sci-hub.se` | Primary Sci-Hub mirror URL |
| `SCIHUB_TIMEOUT` | `30` | Request timeout in seconds |
| `SCIHUB_MAX_RETRIES` | `3` | Maximum retry attempts |
| `SCIHUB_USER_AGENT` | `Mozilla/5.0...` | Browser user agent string |

## Available MCP Tools

The Sci-Hub MCP Server provides the following tools:

### 1. `search_scihub_by_doi`
Search for academic papers using DOI (Digital Object Identifier).

**Parameters:**
- `doi` (string, required): The DOI of the paper

**Usage Example:**
```python
result = await mcp_client.call_tool('search_scihub_by_doi', {
    'doi': '10.1038/nature12373'
})
```

### 2. `search_scihub_by_title`
Search for papers by title with optional author filtering.

**Parameters:**
- `title` (string, required): Title of the paper to search for
- `author` (string, optional): Author name to refine search

**Usage Example:**
```python
result = await mcp_client.call_tool('search_scihub_by_title', {
    'title': 'CRISPR-Cas9 genome editing',
    'author': 'Jennifer Doudna'
})
```

### 3. `search_scihub_by_keyword`
Search for papers using keywords.

**Parameters:**
- `keywords` (string, required): Keywords to search for
- `limit` (integer, optional): Maximum number of results (default: 10)

**Usage Example:**
```python
result = await mcp_client.call_tool('search_scihub_by_keyword', {
    'keywords': 'machine learning healthcare',
    'limit': 20
})
```

### 4. `download_scihub_pdf`
Download PDF from Sci-Hub.

**Parameters:**
- `identifier` (string, required): DOI, title, or other identifier
- `output_path` (string, optional): Local path to save the PDF

**Usage Example:**
```python
result = await mcp_client.call_tool('download_scihub_pdf', {
    'identifier': '10.1038/nature12373',
    'output_path': './papers/nature_paper.pdf'
})
```

### 5. `get_paper_metadata`
Retrieve metadata for a paper from Sci-Hub.

**Parameters:**
- `identifier` (string, required): DOI, title, or other identifier

**Usage Example:**
```python
result = await mcp_client.call_tool('get_paper_metadata', {
    'identifier': '10.1038/nature12373'
})
```

## Integration Points

### 1. LiteratureSearchAgent Integration

The `LiteratureSearchAgent` at `/src/research_agent/agents/literature_searcher.py` is configured to use Sci-Hub MCP tools:

- **Parallel Search**: Executes searches across ArXiv, PubMed, and Sci-Hub simultaneously
- **Cleanup Fallback**: Uses Sci-Hub title search for additional coverage
- **Deduplication**: Removes duplicates across all sources
- **Quality Filtering**: Prioritizes papers with full text access

**Integration Configuration:**
```json
{
  "integration": {
    "literature_search": {
      "enabled": true,
      "priority": "high",
      "max_papers_per_query": 50,
      "timeout_per_search": 60
    }
  }
}
```

### 2. PDFManager Integration

The `PDFManager` at `/src/research_agent/integrations/pdf_manager.py` handles PDF storage:

- **Organized Storage**: Papers stored in `./papers/sci_hub/YYYY/` structure
- **Metadata Tracking**: SQLite database tracks all downloads
- **Duplicate Prevention**: Hash-based duplicate detection
- **Concurrent Downloads**: Configurable concurrent download limits

**Integration Configuration:**
```json
{
  "integration": {
    "pdf_manager": {
      "enabled": true,
      "auto_download": true,
      "storage_path": "./papers/sci_hub",
      "max_concurrent_downloads": 3
    }
  }
}
```

### 3. MultiQueryCoordinator Integration

Coordinates searches across multiple sources with intelligent merging:

**Integration Configuration:**
```json
{
  "integration": {
    "multi_query_coordinator": {
      "enabled": true,
      "parallel_searches": true,
      "merge_duplicates": true,
      "quality_filtering": true
    }
  }
}
```

## Server Startup Methods

### Development Mode
```bash
cd /home/gyasis/Documents/code/PromptChain/Research_agent
uv run python -m sci_hub_server
```

### Production Mode
```bash
cd /home/gyasis/Documents/code/PromptChain/Research_agent
uv run python -m sci_hub_server --production
```

### MCP Client Connection
The server uses stdio transport for communication with MCP clients. Clients connect using the configuration specified in `mcp_config.json`.

## Testing and Validation

### Test Script
Run the comprehensive test suite:

```bash
cd /home/gyasis/Documents/code/PromptChain/Research_agent
python tests/test_scihub_mcp_integration.py
```

### Test Coverage
The test script validates:
- ✅ Server installation and import
- ✅ Server startup capability  
- ✅ Client connection configuration
- ✅ Tool discovery and schemas
- ✅ Paper search functionality
- ✅ Metadata retrieval
- ✅ PDF download capability
- ✅ LiteratureSearchAgent integration
- ✅ PDFManager integration

### Test Results
Latest test run: **100% success rate** (9/9 tests passed)

## File Structure

```
Research_agent/
├── config/
│   └── mcp_config.json                 # MCP server configuration
├── src/
│   └── research_agent/
│       ├── agents/
│       │   └── literature_searcher.py  # Literature search agent
│       └── integrations/
│           └── pdf_manager.py          # PDF download manager
├── papers/                             # PDF storage directory
│   └── sci_hub/                       # Sci-Hub papers
│       └── YYYY/                      # Year-based organization
├── logs/                              # MCP client logs
├── test_scihub_mcp_integration.py     # Integration test script
└── SCIHUB_MCP_SETUP.md                # This documentation
```

## Usage Examples

### Basic Paper Search
```python
from research_agent.agents.literature_searcher import LiteratureSearchAgent

config = {
    'model': 'openai/gpt-4o-mini',
    'pubmed': {'email': 'researcher@example.com', 'tool': 'ResearchAgent'}
}

agent = LiteratureSearchAgent(config)

# Generate search strategy
strategy = {
    "database_allocation": {
        "sci_hub": {"max_papers": 30, "search_terms": ["machine learning", "healthcare"]},
        "arxiv": {"max_papers": 20, "search_terms": ["ML", "medical AI"]},
        "pubmed": {"max_papers": 20, "search_terms": ["artificial intelligence", "diagnosis"]}
    }
}

# Execute search
papers = await agent.search_papers(json.dumps(strategy), max_papers=50)
```

### PDF Download Management
```python
from research_agent.integrations.pdf_manager import PDFManager

pdf_manager = PDFManager(base_path="./papers", config={
    'max_retries': 3,
    'timeout': 60,
    'max_concurrent_downloads': 5
})

# Download paper
paper_metadata = {
    'id': 'scihub_nature_001',
    'title': 'CRISPR-Cas9 genome editing',
    'authors': ['Jinek, Martin', 'Chylinski, Krzysztof'],
    'source': 'sci_hub',
    'publication_year': 2012,
    'doi': '10.1126/science.1225829'
}

success, file_path, info = await pdf_manager.download_pdf(
    'https://sci-hub.se/10.1126/science.1225829',
    paper_metadata
)
```

## Troubleshooting

### Common Issues

1. **Server Import Error**
   ```bash
   # Solution: Ensure package is installed
   uv add sci-hub-mcp-server@git+https://github.com/gyasis/Sci-Hub-MCP-Server.git
   ```

2. **Connection Timeout**
   ```json
   // Solution: Increase timeout in config
   "env": {
     "SCIHUB_TIMEOUT": "60"
   }
   ```

3. **PDF Download Fails**
   - Check Sci-Hub mirror availability
   - Verify network connectivity
   - Check paper identifier format

### Monitoring and Logs

- MCP client logs: `./logs/mcp_client.log`
- Test reports: Generated with timestamps (e.g., `scihub_mcp_test_report_20250813_115153.json`)
- PDF download logs: Integrated with PDFManager logging

## Security Considerations

1. **User Agent**: Configured to use standard browser user agent
2. **Rate Limiting**: Built-in retry mechanisms with exponential backoff
3. **Error Handling**: Comprehensive error handling to prevent crashes
4. **Legal Compliance**: Users responsible for adhering to applicable laws and regulations

## Performance Configuration

### Optimal Settings
- **Max Concurrent Downloads**: 3-5 (balance speed vs. server load)
- **Request Timeout**: 30-60 seconds
- **Max Retries**: 3 attempts
- **Papers per Query**: 30-50 for Sci-Hub searches

### Monitoring
- Track download success rates
- Monitor response times
- Log failed requests for analysis

## Future Enhancements

1. **Mirror Failover**: Automatic switching between Sci-Hub mirrors
2. **Caching**: Local caching of search results
3. **Batch Downloads**: Enhanced batch processing capabilities
4. **Analytics**: Download and usage statistics dashboard

## Support

For issues or questions:
1. Check test results: Run `python tests/test_scihub_mcp_integration.py`
2. Review logs in `./logs/` directory
3. Validate configuration with schema
4. Check Sci-Hub server availability

---

**Status**: ✅ **PRODUCTION READY**  
**Last Updated**: 2025-08-13  
**Configuration Version**: 1.0.0