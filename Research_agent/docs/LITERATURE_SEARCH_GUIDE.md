# Literature Search Agent Guide

## Overview

The Literature Search Agent is a comprehensive system that searches academic literature across multiple databases including ArXiv, PubMed, and Sci-Hub. It uses intelligent search strategies and provides deduplication and quality filtering.

## Architecture

### Core Components

1. **SearchStrategistAgent** (`search_strategist.py`)
   - Creates optimized search strategies
   - Determines database allocation
   - Generates search terms and filters

2. **LiteratureSearchAgent** (`literature_searcher.py`)
   - Executes searches across multiple databases
   - Handles deduplication and quality filtering
   - Manages search result processing

### Supported Databases

- **ArXiv**: Preprints and research papers
- **PubMed**: Medical and biological literature
- **Sci-Hub**: Full-text paper access (via MCP)

## How It Works

### 1. Search Strategy Creation

The system first creates a search strategy that includes:

```json
{
  "search_strategy": {
    "primary_keywords": ["machine learning", "deep learning"],
    "secondary_keywords": ["neural networks", "AI"],
    "filters": {
      "publication_years": [2020, 2024],
      "paper_types": ["research", "review"]
    }
  },
  "database_allocation": {
    "arxiv": {
      "max_papers": 30,
      "search_terms": ["machine learning methods"]
    },
    "pubmed": {
      "max_papers": 20,
      "search_terms": ["machine learning applications"]
    },
    "sci_hub": {
      "max_papers": 40,
      "search_terms": ["machine learning research"]
    }
  }
}
```

### 2. Parallel Search Execution

The agent executes searches across all databases simultaneously:

```python
# Search tasks are created for each database
search_tasks = []
if 'arxiv' in db_allocation:
    search_tasks.append(self._search_arxiv(terms, limit))
if 'pubmed' in db_allocation:
    search_tasks.append(self._search_pubmed(terms, limit))
if 'sci_hub' in db_allocation:
    search_tasks.append(self._search_scihub_mcp(terms, limit))

# Execute all searches concurrently
results = await asyncio.gather(*search_tasks)
```

### 3. Result Processing

After collecting results, the system:

1. **Deduplicates** papers using DOI, title similarity, and ArXiv ID matching
2. **Filters** by quality (abstract length, publication year, etc.)
3. **Ranks** by database priority and quality scores
4. **Limits** total results to specified maximum

## Key Features

### Intelligent Search Optimization

- **Database-specific queries**: Optimizes search terms for each database's syntax
- **Boolean logic**: Supports complex queries with AND/OR operators
- **Field-specific search**: Title, abstract, author, and journal searches

### Quality Filtering

- **Abstract validation**: Ensures papers have meaningful abstracts
- **Publication year filtering**: Focuses on recent research
- **Full-text availability**: Prioritizes papers with accessible content
- **Database priority scoring**: Weights results by source reliability

### Deduplication Strategy

- **DOI matching**: Highest confidence duplicate detection
- **Title similarity**: Fuzzy matching for similar titles
- **ArXiv ID matching**: Cross-database preprint identification
- **Author + year matching**: Fallback for papers without DOIs

### Fallback Mechanisms

- **Sci-Hub cleanup**: Title-based search for missed papers
- **Query expansion**: Generates alternative search terms
- **Source filtering**: Can limit searches to specific databases

## Usage Examples

### Basic Usage

```python
import asyncio
from research_agent.agents.literature_searcher import LiteratureSearchAgent

# Configuration
config = {
    'model': 'openai/gpt-4o-mini',
    'pubmed': {
        'email': 'your-email@example.com',
        'tool': 'ResearchAgent'
    }
}

# Initialize agent
searcher = LiteratureSearchAgent(config)

# Create search strategy
strategy = {
    "database_allocation": {
        "arxiv": {"max_papers": 10, "search_terms": ["machine learning"]},
        "pubmed": {"max_papers": 5, "search_terms": ["machine learning"]}
    }
}

# Execute search
papers = await searcher.search_papers(
    strategy=json.dumps(strategy),
    max_papers=15
)
```

### Advanced Usage with Source Filtering

```python
# Search only ArXiv
arxiv_papers = await searcher.search_papers(
    strategy=strategy,
    max_papers=10,
    source_filter='arxiv'
)

# Search only PubMed
pubmed_papers = await searcher.search_papers(
    strategy=strategy,
    max_papers=10,
    source_filter='pubmed'
)
```

### Custom Search Strategy

```python
# Create detailed search strategy
detailed_strategy = {
    "search_strategy": {
        "primary_keywords": ["deep learning", "neural networks"],
        "secondary_keywords": ["computer vision", "natural language processing"],
        "boolean_queries": [
            {
                "database": "arxiv",
                "query": 'ti:"deep learning" AND (computer vision OR NLP)',
                "priority": 1.0
            }
        ],
        "filters": {
            "publication_years": [2022, 2024],
            "paper_types": ["research"],
            "languages": ["en"]
        }
    },
    "database_allocation": {
        "arxiv": {
            "priority": 1.0,
            "max_papers": 20,
            "search_terms": ["deep learning", "neural networks"],
            "rationale": "Latest research in AI"
        }
    }
}
```

## Configuration Options

### PubMed Configuration

```yaml
pubmed:
  email: "your-email@example.com"  # Required for PubMed API
  tool: "ResearchAgent"
  timeout: 30
  rate_limit: 10  # requests per second
```

### ArXiv Configuration

```yaml
arxiv:
  timeout: 30
  rate_limit: 30  # requests per minute
  categories:
    - "cs.AI"
    - "cs.LG"
    - "stat.ML"
```

### Sci-Hub Configuration

```yaml
sci_hub:
  timeout: 120
  rate_limit: 10  # requests per minute
  retry_attempts: 3
```

## Paper Data Structure

Each paper returned has the following structure:

```python
{
    'id': 'arxiv_1234.5678',
    'title': 'Paper Title',
    'authors': ['Author, First', 'Author, Second'],
    'abstract': 'Paper abstract...',
    'doi': '10.1000/example.123',
    'publication_year': 2023,
    'source': 'arxiv',  # arxiv, pubmed, sci_hub
    'url': 'https://arxiv.org/abs/1234.5678',
    'pdf_url': 'https://arxiv.org/pdf/1234.5678.pdf',
    'full_text_available': True,
    'metadata': {
        'journal': 'Journal Name',
        'arxiv_id': '1234.5678',
        'categories': ['cs.AI', 'cs.LG'],
        'database_priority': 0.8,
        'retrieved_at': '2024-01-15T10:30:00'
    }
}
```

## Error Handling

The system includes robust error handling:

- **API failures**: Individual database failures don't stop the entire search
- **Rate limiting**: Respects API rate limits with automatic retries
- **Timeout handling**: Configurable timeouts for each database
- **Fallback mechanisms**: Alternative search strategies when primary fails

## Performance Considerations

- **Parallel execution**: Searches run concurrently across databases
- **Caching**: Results are cached to avoid duplicate API calls
- **Rate limiting**: Built-in rate limiting to respect API constraints
- **Memory management**: Efficient processing of large result sets

## Testing

Use the provided test scripts:

```bash
# Simple test
python simple_literature_search_test.py

# Comprehensive test
python tests/test_literature_search_agent.py
```

## Troubleshooting

### Common Issues

1. **PubMed API errors**: Ensure you provide a valid email in configuration
2. **Rate limiting**: Reduce search frequency or increase rate limit settings
3. **No results**: Check search terms and try broader queries
4. **Import errors**: Ensure all dependencies are installed

### Debug Mode

Enable debug logging to see detailed search execution:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration with Research Agent

The Literature Search Agent is part of a larger research system that includes:

- **3-tier RAG processing**: LightRAG, PaperQA2, GraphRAG
- **Multi-query coordination**: Handles multiple research questions
- **Interactive chat**: Follow-up questions and analysis
- **Report generation**: Automated literature review creation

For full research workflow integration, see the main Research Agent documentation. 