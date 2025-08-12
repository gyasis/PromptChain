# Configuration Guide

This guide provides comprehensive documentation for configuring the Research Agent System through the YAML configuration file and environment variables.

## Table of Contents

- [Overview](#overview)
- [Configuration File Structure](#configuration-file-structure)
- [System Configuration](#system-configuration)
- [Literature Search Configuration](#literature-search-configuration)
- [3-Tier RAG System](#3-tier-rag-system)
- [PromptChain Integration](#promptchain-integration)
- [Research Session Management](#research-session-management)
- [MCP Server Configuration](#mcp-server-configuration)
- [Caching Configuration](#caching-configuration)
- [Report Generation](#report-generation)
- [Web Interface](#web-interface)
- [Performance & Security](#performance--security)
- [Environment Variables](#environment-variables)
- [Configuration Examples](#configuration-examples)

## Overview

The Research Agent System uses a hierarchical YAML configuration system that allows for flexible deployment across different environments. The main configuration file is located at `config/research_config.yaml`.

### Configuration Loading Priority

1. Command-line arguments (highest priority)
2. Environment variables
3. YAML configuration file
4. Default values (lowest priority)

### Basic Configuration Loading

```python
from research_agent.core.config import ResearchConfig

# Load default configuration
config = ResearchConfig()

# Load from YAML file
config = ResearchConfig.from_yaml("config/research_config.yaml")

# Load from dictionary
config_dict = {...}
config = ResearchConfig.from_dict(config_dict)
```

## Configuration File Structure

```yaml
# research_config.yaml
system:                          # System metadata
literature_search:               # Literature search settings
three_tier_rag:                 # RAG processing configuration
promptchain:                    # Multi-agent orchestration
research_session:               # Session management
mcp_servers:                    # MCP integrations
caching:                        # Caching configuration
report_generation:              # Report generation settings
web_interface:                  # Web interface settings
logging:                        # Logging configuration
performance:                    # Performance settings
security:                       # Security settings
development:                    # Development settings
env_overrides:                  # Environment variable mappings
```

## System Configuration

Basic system metadata and identification.

```yaml
system:
  name: "Research Agent System"
  version: "0.1.0"
  description: "Advanced research agent with 3-tier RAG"
```

**Fields:**
- `name` (str): System name for identification
- `version` (str): System version
- `description` (str): System description

## Literature Search Configuration

Configuration for literature search engines and databases.

### Overview

```yaml
literature_search:
  enabled_sources:
    - sci_hub
    - arxiv
    - pubmed
```

### Sci-Hub Configuration

Direct paper retrieval through Sci-Hub MCP server.

```yaml
literature_search:
  sci_hub:
    enabled: true
    timeout: 120              # Request timeout in seconds
    rate_limit: 10           # Requests per minute
    retry_attempts: 3        # Number of retry attempts
    cache_pdfs: true         # Cache downloaded PDFs
    pdf_storage: "./data/cache/pdfs"  # PDF storage directory
```

**Fields:**
- `enabled` (bool): Enable/disable Sci-Hub integration
- `timeout` (int): Request timeout in seconds
- `rate_limit` (int): Maximum requests per minute
- `retry_attempts` (int): Number of retry attempts for failed requests
- `cache_pdfs` (bool): Whether to cache downloaded PDFs locally
- `pdf_storage` (str): Directory for storing cached PDFs

### ArXiv Configuration

Configuration for ArXiv API integration.

```yaml
literature_search:
  arxiv:
    enabled: true
    timeout: 30
    rate_limit: 30           # Requests per minute
    categories:              # ArXiv categories to search
      - "cs.AI"             # Artificial Intelligence
      - "cs.CL"             # Computation and Language
      - "cs.LG"             # Machine Learning
      - "stat.ML"           # Machine Learning (Statistics)
      - "q-bio"             # Quantitative Biology
    max_results: 50         # Maximum results per query
```

**Fields:**
- `enabled` (bool): Enable/disable ArXiv integration
- `timeout` (int): Request timeout in seconds
- `rate_limit` (int): Maximum requests per minute
- `categories` (List[str]): ArXiv categories to search
- `max_results` (int): Maximum results per query

**Common ArXiv Categories:**
- `cs.AI`: Artificial Intelligence
- `cs.CL`: Computation and Language
- `cs.CV`: Computer Vision and Pattern Recognition
- `cs.LG`: Machine Learning
- `cs.NE`: Neural and Evolutionary Computing
- `stat.ML`: Machine Learning (Statistics)
- `q-bio`: Quantitative Biology

### PubMed Configuration

Configuration for PubMed/NCBI integration.

```yaml
literature_search:
  pubmed:
    enabled: true
    timeout: 30
    rate_limit: 10           # Requests per second (Note: different from others)
    email: "${OPENALEX_EMAIL}"  # Required for PubMed API
    tool_name: "research-agent"
    max_results: 50
```

**Fields:**
- `enabled` (bool): Enable/disable PubMed integration
- `timeout` (int): Request timeout in seconds
- `rate_limit` (int): Maximum requests per second
- `email` (str): Email address for PubMed API (required)
- `tool_name` (str): Tool name for PubMed API identification
- `max_results` (int): Maximum results per query

## 3-Tier RAG System

Configuration for the three-tier RAG processing pipeline.

### Execution Mode

```yaml
three_tier_rag:
  execution_mode: "sequential"  # sequential, parallel, adaptive
```

**Options:**
- `sequential`: Process tiers one after another
- `parallel`: Process all tiers simultaneously
- `adaptive`: Dynamically choose based on query complexity

### Tier 1: LightRAG Configuration

Entity extraction and knowledge graph construction.

```yaml
three_tier_rag:
  tier1_lightrag:
    enabled: true
    mode: "ollama"           # cloud, ollama, hybrid
    
    # Cloud API Configuration
    cloud:
      llm_model: "gpt-4o-mini"
      embedding_model: "text-embedding-3-large"
      temperature: 0.1
      
    # Ollama Configuration
    ollama:
      host: "http://localhost:11434"
      llm_model: "mistral-nemo:latest"
      embedding_model: "bge-m3:latest"
      embedding_dim: 1024
      context_window: 16384
      
    # Processing Settings
    processing:
      chunk_size: 1200
      chunk_overlap: 100
      max_parallel_insert: 2
      llm_model_max_async: 4
      entity_extract_max_gleaning: 1
      working_dir: "./data/lightrag_working"
      
    # PDF Processing
    pdf_processing:
      use_textract: true
      extract_images: false
      extract_tables: true
      ocr_enabled: false
```

**Mode Options:**
- `cloud`: Use cloud APIs (OpenAI, Anthropic, etc.)
- `ollama`: Use local Ollama models
- `hybrid`: Mix of cloud and local (not applicable for Tier 1)

**Cloud Configuration:**
- `llm_model`: Language model for entity extraction
- `embedding_model`: Embedding model for vector representations
- `temperature`: Model temperature for generation

**Ollama Configuration:**
- `host`: Ollama server URL
- `llm_model`: Local LLM model name
- `embedding_model`: Local embedding model name
- `embedding_dim`: Embedding dimension size
- `context_window`: Model context window size

**Processing Settings:**
- `chunk_size`: Text chunk size for processing
- `chunk_overlap`: Overlap between chunks
- `max_parallel_insert`: Maximum parallel insertions
- `llm_model_max_async`: Maximum async LLM calls
- `entity_extract_max_gleaning`: Maximum extraction iterations
- `working_dir`: LightRAG working directory

### Tier 2: PaperQA2 Configuration

Research-focused question answering with detailed citations.

```yaml
three_tier_rag:
  tier2_paperqa2:
    enabled: true
    mode: "ollama"           # cloud, ollama
    
    # Cloud API Configuration
    cloud:
      llm: "gpt-4o-2024-11-20"
      summary_llm: "gpt-4o-2024-11-20"
      embedding: "text-embedding-3-small"
      temperature: 0.0
      
    # Ollama Configuration
    ollama:
      llm: "ollama/llama3.2"
      summary_llm: "ollama/llama3.2"
      embedding: "ollama/mxbai-embed-large"
      api_base: "http://localhost:11434"
      
    # PaperQA2 Settings
    settings:
      batch_size: 1
      verbosity: 1
      answer:
        evidence_k: 10                    # Number of evidence pieces
        evidence_detailed_citations: true
        evidence_retrieval: true
        evidence_summary_length: "about 100 words"
        answer_max_sources: 5
        answer_length: "about 200 words, but can be longer"
        max_concurrent_requests: 4
      parsing:
        chunk_size: 5000
        overlap: 250
        page_size_limit: "1,280,000"
        use_doc_details: true
        
    # Paper Processing
    paper_processing:
      auto_citation: true
      auto_metadata: true
      pdf_parsing: "grobid"      # grobid, textract, pymupdf
```

**Mode Options:**
- `cloud`: Use cloud APIs
- `ollama`: Use local Ollama models

**Answer Settings:**
- `evidence_k`: Number of evidence pieces to retrieve
- `evidence_detailed_citations`: Include detailed citation information
- `evidence_summary_length`: Length of evidence summaries
- `answer_max_sources`: Maximum sources per answer
- `answer_length`: Target answer length

**PDF Parsing Options:**
- `grobid`: Scientific document parsing (recommended)
- `textract`: AWS Textract (cloud service)
- `pymupdf`: Basic PDF text extraction

### Tier 3: GraphRAG Configuration

Advanced reasoning over knowledge graphs.

```yaml
three_tier_rag:
  tier3_graphrag:
    enabled: true
    mode: "ollama"           # cloud, ollama
    
    # Cloud API Configuration
    cloud:
      extraction_model: "gpt-4o-mini"
      query_model: "gpt-4"
      embedding_model: "text-embedding-3-large"
      
    # Ollama Configuration
    ollama:
      extraction_model: "llama3.2"
      query_model: "mistral"
      embedding_model: "bge-m3"
      api_base: "http://localhost:11434"
      
    # GraphRAG Settings
    settings:
      max_retries: 20
      chunk_size: 300
      chunk_overlap: 100
      entity_types:            # Entity types to extract
        - "organization"
        - "person" 
        - "geo"
        - "event"
        - "concept"
        - "method"
      max_gleanings: 1
      
    # Graph Processing
    graph_processing:
      community_detection: true
      max_data_tokens: 12000
      concurrent_coroutines: 32
      json_mode: true
```

**Entity Types:**
- `organization`: Companies, institutions, organizations
- `person`: People, authors, researchers
- `geo`: Geographic locations
- `event`: Events, conferences, milestones
- `concept`: Abstract concepts, theories
- `method`: Methodologies, algorithms, techniques

## PromptChain Integration

Configuration for multi-agent orchestration using PromptChain.

### Agent Chain Configuration

```yaml
promptchain:
  agent_chain:
    execution_mode: "pipeline"    # pipeline, router, round_robin, broadcast
    max_agents: 10
    cache_enabled: true
    cache_name: "research_session"
    cache_path: "./data/cache"
    verbose: true
```

**Execution Modes:**
- `pipeline`: Sequential agent execution
- `router`: Dynamic agent selection
- `round_robin`: Cyclic agent execution
- `broadcast`: Parallel execution with synthesis

### Agent Configurations

Each agent can be configured with specific models, instructions, and tools.

#### Query Generator Agent

```yaml
promptchain:
  agents:
    query_generator:
      model: "openai/gpt-4o"
      type: "agentic"
      instructions:
        - "Analyze research topic: {topic}"
        - "Generate 8-12 specific research questions"
        - "Format as structured query list with priority scores"
      processor:
        type: "agentic_step"
        objective: "Generate comprehensive research questions"
        max_internal_steps: 5
      tools:
        - "topic_analysis"
        - "question_generation"
        - "priority_scoring"
```

**Agent Types:**
- `agentic`: Uses AgenticStepProcessor for internal reasoning
- `function_calling`: Direct function/tool calling
- `orchestrator`: Coordinates other agents
- `interactive`: Interactive chat agent

**Processor Types:**
- `agentic_step`: Multi-step reasoning with tools
- `custom`: Custom processor class
- None: Direct instruction processing

#### Search Strategist Agent

```yaml
promptchain:
  agents:
    search_strategist:
      model: "openai/gpt-4o-mini"
      type: "agentic"
      instructions:
        - "Current queries: {queries}"
        - "Previous findings: {findings}"
        - "Determine optimal search strategy and keywords"
      processor:
        type: "agentic_step"
        objective: "Optimize search strategy"
        max_internal_steps: 3
      tools:
        - "search_optimization"
        - "keyword_generation"
        - "database_selection"
```

#### Literature Searcher Agent

```yaml
promptchain:
  agents:
    literature_searcher:
      model: "openai/gpt-4o-mini"
      type: "function_calling"
      instructions:
        - "Execute search strategy: {strategy}"
        - "Find relevant papers using multiple sources"
        - "Collect and validate paper metadata"
      tools:
        - "mcp_scihub_search_by_keyword"
        - "mcp_scihub_search_by_title"
        - "arxiv_search"
        - "pubmed_search"
        - "paper_validation"
        - "metadata_extraction"
```

#### ReAct Analysis Agent

```yaml
promptchain:
  agents:
    react_analyzer:
      model: "anthropic/claude-3-5-sonnet-20240620"
      type: "agentic"
      instructions:
        - "Analyze multi-tier results: {results}"
        - "Identify research gaps and coverage issues"
        - "Determine if additional queries are needed"
      processor:
        type: "agentic_step"
        objective: "Perform ReAct-style gap analysis"
        max_internal_steps: 7
      tools:
        - "gap_detection"
        - "coverage_analysis"
        - "query_generation"
        - "completeness_assessment"
```

#### Synthesis Agent

```yaml
promptchain:
  agents:
    synthesis_agent:
      model: "openai/gpt-4o"
      type: "agentic"
      instructions:
        - "All research findings: {findings}"
        - "Original research topic: {topic}"
        - "Synthesize comprehensive literature review"
      processor:
        type: "agentic_step"
        objective: "Synthesize comprehensive literature review"
        max_internal_steps: 10
        store_steps: true        # Store intermediate steps
      tools:
        - "literature_synthesis"
        - "statistical_analysis"
        - "citation_analysis"
        - "document_generation"
```

### Router Configuration

For router-based agent selection.

```yaml
promptchain:
  router:
    model: "openai/gpt-4o-mini"
    decision_templates:
      research_workflow_dispatch: |
        User input: {user_input}
        Current research phase: {current_phase}
        Available phases: {available_phases}
        Choose phase and agent: {"phase": "phase_name", "agent": "agent_name"}
        
      chat_mode_dispatch: |
        User input: {user_input}
        Research session: {research_session}
        Available actions: {available_actions}
        Choose action: {"action": "action_name", "parameters": {...}}
```

## Research Session Management

Configuration for research session lifecycle and persistence.

```yaml
research_session:
  # Session Limits
  max_iterations: 5
  max_queries_per_iteration: 15
  min_papers_per_query: 5
  max_papers_total: 100
  
  # Quality Thresholds
  completeness_threshold: 0.85      # Minimum completion score
  gap_detection_threshold: 0.7      # Gap detection sensitivity
  citation_minimum: 3               # Minimum citations per paper
  
  # Session Persistence
  save_intermediate_results: true
  session_timeout: 7200            # 2 hours in seconds
  auto_save_interval: 300          # 5 minutes in seconds
```

**Quality Thresholds:**
- `completeness_threshold`: Minimum research completion score (0.0-1.0)
- `gap_detection_threshold`: Sensitivity for detecting research gaps
- `citation_minimum`: Minimum number of citations required per paper

## MCP Server Configuration

Configuration for Model Context Protocol server integrations.

```yaml
mcp_servers:
  # Context7 for library documentation
  context7:
    enabled: true
    server_type: "context7"
    
  # Gemini for pair programming  
  gemini:
    enabled: true
    server_type: "gemini_mcp_server"
    
  # Git for version tracking
  git:
    enabled: true
    server_type: "git"
    
  # Sci-Hub MCP
  scihub:
    enabled: true
    server_type: "stdio"
    command: "uv"
    args: ["run", "fastmcp", "dev", "sci_hub_server.py"]
    cwd: "./integrations/scihub_mcp"
```

**Server Types:**
- `stdio`: Standard I/O based servers
- `context7`: Context7 library documentation
- `gemini_mcp_server`: Gemini AI integration
- `git`: Git repository operations

## Caching Configuration

Multi-tier caching system for performance optimization.

```yaml
caching:
  enabled: true
  
  # Memory Cache
  memory:
    size_mb: 500              # Memory cache size in MB
    ttl_seconds: 3600         # Time-to-live in seconds
    
  # Disk Cache
  disk:
    enabled: true
    directory: "./data/cache"
    size_mb: 2000            # Disk cache size in MB
    cleanup_interval: 86400   # Cleanup interval in seconds
    
  # Redis Cache (optional)
  redis:
    enabled: false
    url: "${REDIS_URL}"
    ttl_seconds: 86400
    
  # Database Cache
  database:
    enabled: true
    path: "./data/research_cache.db"
    session_timeout: 3600
```

**Cache Types:**
- `memory`: In-memory caching for fast access
- `disk`: Persistent disk caching
- `redis`: External Redis caching (optional)
- `database`: SQLite database for session persistence

## Report Generation

Configuration for generating research reports and visualizations.

```yaml
report_generation:
  # Output Formats
  formats:
    - "markdown"
    - "html" 
    - "pdf"
    - "json"
    
  # Report Sections
  sections:
    executive_summary: true
    methodology: true
    findings: true
    statistics: true
    visualizations: true
    references: true
    appendix: false
    
  # Statistics and Analytics
  statistics:
    paper_count: true
    citation_analysis: true
    temporal_analysis: true
    keyword_frequency: true
    author_networks: true
    
  # Visualizations  
  visualizations:
    enabled: true
    library: "plotly"         # plotly, matplotlib
    formats:
      - "png"
      - "html"
      - "svg"
    graphs:
      - "citation_network"
      - "temporal_trends"
      - "keyword_clouds"
      - "author_collaboration"
      - "research_gaps"
```

**Visualization Libraries:**
- `plotly`: Interactive web-based visualizations
- `matplotlib`: Static publication-quality plots

**Graph Types:**
- `citation_network`: Citation relationship graphs
- `temporal_trends`: Time-based analysis charts
- `keyword_clouds`: Word cloud visualizations
- `author_collaboration`: Author collaboration networks
- `research_gaps`: Gap analysis visualizations

## Web Interface

Configuration for web-based interfaces.

```yaml
web_interface:
  # FastAPI Configuration
  fastapi:
    enabled: true
    host: "0.0.0.0"
    port: 8000
    debug: false
    reload: true
    cors_enabled: true
    cors_origins:
      - "http://localhost:3000"
      - "http://localhost:8001"
      
  # Chainlit Configuration  
  chainlit:
    enabled: true
    host: "0.0.0.0"
    port: 8001
    auth_secret: "${CHAINLIT_AUTH_SECRET}"
    session_timeout: 3600
    
  # Interface Features
  features:
    mode_selection: true
    real_time_updates: true
    export_reports: true
    session_management: true
    progress_tracking: true
```

## Performance & Security

### Performance Configuration

```yaml
performance:
  # Concurrency
  max_concurrent_requests: 10
  request_timeout: 120
  batch_processing: true
  batch_size: 5
  
  # Memory Management
  max_memory_mb: 4096
  gc_threshold: 0.8           # Garbage collection threshold
  
  # Rate Limiting
  enable_rate_limiting: true
  requests_per_minute: 60
  burst_size: 10
```

### Security Configuration

```yaml
security:
  # API Security
  api_key_enabled: false
  api_key: "${API_KEY}"
  
  # Data Privacy
  anonymize_data: false
  encrypt_cache: false
  secure_headers: true
  
  # Input Validation
  validate_inputs: true
  sanitize_queries: true
  max_query_length: 1000
```

### Logging Configuration

```yaml
logging:
  level: "INFO"               # DEBUG, INFO, WARNING, ERROR
  
  # Console Logging
  console:
    enabled: true
    format: "structured"      # structured, simple
    level: "INFO"
    
  # File Logging
  file:
    enabled: true
    directory: "./logs"
    filename: "research_agent.log"
    level: "DEBUG"
    max_size_mb: 10
    backup_count: 5
    
  # JSONL Logging
  jsonl:
    enabled: true
    directory: "./logs"
    filename: "research_events.jsonl"
    events:
      - "query_start"
      - "literature_search"
      - "tier_processing"
      - "report_generation"
      - "query_complete"
```

## Environment Variables

Environment variables can override configuration values. Define mappings in the configuration:

```yaml
env_overrides:
  OPENAI_API_KEY: "llm_config.api_key"
  ANTHROPIC_API_KEY: "llm_config.api_key" 
  OLLAMA_HOST: "three_tier_rag.tier1_lightrag.ollama.host"
  CACHE_DIR: "caching.disk.directory"
  LOG_LEVEL: "logging.level"
```

### Required Environment Variables

```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Email for PubMed (required)
OPENALEX_EMAIL=your_email@domain.com

# Optional
OLLAMA_HOST=http://localhost:11434
REDIS_URL=redis://localhost:6379
CHAINLIT_AUTH_SECRET=your_secret_key
API_KEY=your_api_key
```

## Configuration Examples

### Cloud-Only Configuration

Optimized for cloud APIs with maximum accuracy.

```yaml
three_tier_rag:
  tier1_lightrag:
    mode: "cloud"
    cloud:
      llm_model: "gpt-4o"
      embedding_model: "text-embedding-3-large"
  tier2_paperqa2:
    mode: "cloud"
    cloud:
      llm: "gpt-4o-2024-11-20"
      embedding: "text-embedding-3-small"
  tier3_graphrag:
    mode: "cloud"
    cloud:
      extraction_model: "gpt-4o-mini"
      query_model: "gpt-4"

promptchain:
  agents:
    query_generator:
      model: "openai/gpt-4o"
    synthesis_agent:
      model: "openai/gpt-4o"
```

### Local-Only Configuration

Privacy-focused with all local models.

```yaml
three_tier_rag:
  tier1_lightrag:
    mode: "ollama"
    ollama:
      host: "http://localhost:11434"
      llm_model: "mistral-nemo:latest"
      embedding_model: "bge-m3:latest"
  tier2_paperqa2:
    mode: "ollama"
    ollama:
      llm: "ollama/llama3.2"
      embedding: "ollama/mxbai-embed-large"
  tier3_graphrag:
    mode: "ollama"
    ollama:
      extraction_model: "llama3.2"
      query_model: "mistral"

promptchain:
  agents:
    query_generator:
      model: "ollama/mistral-nemo"
    synthesis_agent:
      model: "ollama/llama3.2"
```

### Hybrid Configuration

Balance between accuracy and cost.

```yaml
three_tier_rag:
  tier1_lightrag:
    mode: "ollama"              # Local for entity extraction
  tier2_paperqa2:
    mode: "cloud"               # Cloud for accuracy in Q&A
  tier3_graphrag:
    mode: "ollama"              # Local for graph processing

promptchain:
  agents:
    query_generator:
      model: "openai/gpt-4o-mini"    # Cloud for query generation
    search_strategist:
      model: "ollama/mistral"        # Local for strategy
    literature_searcher:
      model: "ollama/llama3.2"       # Local for search
    react_analyzer:
      model: "anthropic/claude-3-sonnet-20240229"  # Cloud for analysis
    synthesis_agent:
      model: "openai/gpt-4o"         # Cloud for final synthesis
```

### Development Configuration

Settings optimized for development and testing.

```yaml
development:
  debug_mode: true
  mock_apis: true               # Use mock APIs for testing
  test_data_enabled: true       # Enable test data
  hot_reload: true
  profiling_enabled: true

logging:
  level: "DEBUG"
  console:
    level: "DEBUG"
  
performance:
  max_concurrent_requests: 2    # Reduced for development
  batch_size: 2

research_session:
  max_papers_total: 20          # Reduced for faster testing
  max_iterations: 2
```

### Production Configuration

Settings optimized for production deployment.

```yaml
performance:
  max_concurrent_requests: 50
  batch_processing: true
  batch_size: 10
  max_memory_mb: 8192

caching:
  enabled: true
  memory:
    size_mb: 1000
  disk:
    size_mb: 5000
  redis:
    enabled: true

security:
  api_key_enabled: true
  encrypt_cache: true
  validate_inputs: true

logging:
  level: "INFO"
  file:
    enabled: true
    max_size_mb: 50
    backup_count: 10
```

## Configuration Validation

The system validates configuration on startup and provides helpful error messages.

### Common Configuration Errors

1. **Missing API Keys**: Required environment variables not set
2. **Invalid Model Names**: Specified models not available
3. **Resource Limits**: Memory or disk limits too low
4. **Network Configuration**: Invalid URLs or ports
5. **File Permissions**: Insufficient permissions for cache directories

### Validation Commands

```bash
# Validate configuration
uv run python -m research_agent.cli validate-config config/research_config.yaml

# Test configuration with dry run
uv run python -m research_agent.cli research "test topic" --dry-run --config config/test_config.yaml
```

This configuration guide provides comprehensive documentation for all configuration options in the Research Agent System. Use this as a reference when customizing the system for your specific needs and deployment environment.