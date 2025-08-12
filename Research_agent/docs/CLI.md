# CLI Reference Guide

This guide provides comprehensive documentation for the Research Agent System command-line interface, built with Typer and Rich for an enhanced terminal experience.

## Table of Contents

- [Installation & Setup](#installation--setup)
- [Global Options](#global-options)
- [Commands Overview](#commands-overview)
- [Command Reference](#command-reference)
- [Interactive Features](#interactive-features)
- [Output Formats](#output-formats)
- [Configuration](#configuration)
- [Examples & Workflows](#examples--workflows)
- [Troubleshooting](#troubleshooting)

## Installation & Setup

### Prerequisites

```bash
# Install Python 3.11+
python --version  # Should be 3.11+

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup project
cd Research_agent
uv sync
```

### Environment Variables

Create `.env` file with required API keys:

```bash
# Copy example environment file
cp .env.example .env

# Edit with your keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
OPENALEX_EMAIL=your_email@domain.com
```

### CLI Access

The CLI can be accessed in multiple ways:

```bash
# Method 1: Direct module execution (recommended)
uv run python -m research_agent.cli [command]

# Method 2: Direct script execution
uv run python src/research_agent/cli.py [command]

# Method 3: Install as package (optional)
uv pip install -e .
research-agent [command]
```

## Global Options

These options are available across all commands:

- `--help`: Show help information
- `--version`: Show version information
- `--verbose, -v`: Enable verbose output
- `--config, -c PATH`: Specify configuration file

## Commands Overview

| Command | Purpose | Primary Use Case |
|---------|---------|------------------|
| `research` | Conduct comprehensive research | Main research workflow |
| `chat` | Interactive chat with results | Follow-up questions and analysis |
| `analyze` | Direct paper analysis | Analyze existing paper collections |
| `list-sessions` | List saved sessions | Session management |
| `stats` | Show session statistics | Performance and results analysis |
| `export` | Export results to files | Generate reports and documentation |

## Command Reference

### `research` - Conduct Research

The primary command for conducting comprehensive research on a topic.

```bash
uv run python -m research_agent.cli research [TOPIC] [OPTIONS]
```

#### Arguments

- `TOPIC` (required): Research topic to investigate

#### Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--config` | `-c` | PATH | None | Configuration YAML file path |
| `--max-papers` | `-p` | INT | 100 | Maximum papers to retrieve |
| `--iterations` | `-i` | INT | 5 | Maximum research iterations |
| `--mode` | `-m` | CHOICE | cloud | Processing mode (cloud/ollama/hybrid) |
| `--output` | `-o` | PATH | auto | Output directory for results |
| `--interactive/--no-interactive` | - | BOOL | True | Enable interactive chat after research |
| `--verbose` | `-v` | FLAG | False | Enable verbose output |

#### Processing Modes

- **Cloud Mode**: Uses cloud APIs (OpenAI, Anthropic) for maximum accuracy
- **Ollama Mode**: Uses local Ollama models for privacy and cost control
- **Hybrid Mode**: Combines cloud and local models for balance

#### Examples

```bash
# Basic research with default settings
uv run python -m research_agent.cli research "machine learning interpretability"

# Research with custom parameters
uv run python -m research_agent.cli research "quantum computing" \
  --max-papers 50 \
  --iterations 3 \
  --mode hybrid \
  --verbose

# Research with custom configuration
uv run python -m research_agent.cli research "transformer attention mechanisms" \
  --config config/custom_config.yaml \
  --output ./my_research \
  --no-interactive

# Quick research for testing (reduced scope)
uv run python -m research_agent.cli research "neural networks" \
  --max-papers 20 \
  --iterations 2 \
  --mode ollama
```

#### Output

The research command generates:

1. **Console Output**: Real-time progress updates with Rich formatting
2. **Session Files**: Complete session data in JSON format
3. **Literature Review**: Comprehensive review in Markdown and JSON
4. **Statistics**: Research metrics and analysis
5. **Interactive Chat**: Optional follow-up interface

#### Expected Workflow

1. **Query Generation**: Creates 8-12 targeted research questions
2. **Search Strategy**: Optimizes search across databases
3. **Literature Search**: Retrieves papers from Sci-Hub, ArXiv, PubMed
4. **3-Tier Processing**: Processes papers through RAG pipeline
5. **Gap Analysis**: Identifies research gaps using ReAct reasoning
6. **Iteration**: Repeats if gaps found (up to max iterations)
7. **Synthesis**: Generates comprehensive literature review
8. **Interactive Chat**: Optional Q&A session

### `chat` - Interactive Chat

Start interactive chat session with saved research results.

```bash
uv run python -m research_agent.cli chat [SESSION_FILE] [OPTIONS]
```

#### Arguments

- `SESSION_FILE` (required): Path to saved research session JSON

#### Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--mode` | `-m` | CHOICE | qa | Chat mode |

#### Chat Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `qa` | Question & Answer | General questions about research |
| `deep_dive` | Deep Analysis | Detailed exploration of specific topics |
| `compare` | Comparative Analysis | Compare findings across papers |
| `summary` | Summary Generation | Generate focused summaries |
| `custom` | Custom Analysis | User-defined analysis tasks |

#### Chat Commands

During chat session, use these commands:

| Command | Description |
|---------|-------------|
| `/mode` | Change chat mode |
| `/suggest` | Get question suggestions |
| `/history` | Show chat history |
| `/stats` | Show session statistics |
| `/export` | Export chat session |
| `/help` | Show help information |
| `/quit` | Exit chat session |

#### Examples

```bash
# Start chat with default Q&A mode
uv run python -m research_agent.cli chat research_output/session.json

# Start chat in comparative analysis mode
uv run python -m research_agent.cli chat results/ml_interpretability/session.json --mode compare

# Chat workflow
uv run python -m research_agent.cli chat session.json
# Interactive session:
# You> What are the key limitations of current attention mechanisms?
# Assistant> Based on the research findings...
# You> /suggest
# Assistant> Here are some suggested questions:...
# You> /quit
```

#### Chat Interface Features

- **Rich Formatting**: Markdown rendering for responses
- **Syntax Highlighting**: Code and data formatting
- **Progress Indicators**: Real-time processing feedback
- **Command Completion**: Tab completion for commands
- **History Management**: Persistent chat history
- **Export Capabilities**: Save chat sessions

### `analyze` - Direct Paper Analysis

Analyze papers with specific queries (planned feature).

```bash
uv run python -m research_agent.cli analyze [PAPERS_DIR] [QUERY] [OPTIONS]
```

#### Arguments

- `PAPERS_DIR` (required): Directory containing papers
- `QUERY` (required): Analysis query

#### Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--format` | `-f` | CHOICE | markdown | Output format |

#### Output Formats

- `json`: Structured JSON output
- `markdown`: Formatted Markdown report
- `text`: Plain text output
- `html`: HTML report

#### Status

⚠️ **Note**: This command is planned but not yet implemented. Use the `research` command for paper analysis.

### `list-sessions` - List Sessions

Display all saved research sessions with metadata.

```bash
uv run python -m research_agent.cli list-sessions [OPTIONS]
```

#### Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--dir` | `-d` | PATH | ./research_output | Sessions directory |

#### Examples

```bash
# List sessions in default directory
uv run python -m research_agent.cli list-sessions

# List sessions in custom directory
uv run python -m research_agent.cli list-sessions --dir ./my_research
```

#### Output Format

```
📋 Research Sessions

┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓
┃ Session  ┃ Topic                        ┃ Date       ┃ Papers ┃ Status    ┃
┃ ID       ┃                              ┃            ┃        ┃           ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩
│ a1b2c3d4 │ machine learning interpret.. │ 2024-01-15 │     87 │ completed │
│ e5f6g7h8 │ quantum computing           │ 2024-01-14 │     45 │ completed │
│ i9j0k1l2 │ transformer attention       │ 2024-01-13 │     23 │ processing│
└──────────┴──────────────────────────────┴────────────┴────────┴───────────┘
```

### `stats` - Session Statistics

Show detailed statistics for a research session.

```bash
uv run python -m research_agent.cli stats [SESSION_FILE]
```

#### Arguments

- `SESSION_FILE` (required): Path to research session JSON

#### Examples

```bash
# Show statistics for specific session
uv run python -m research_agent.cli stats research_output/ml_interpretability/session.json

# View statistics for the latest session
uv run python -m research_agent.cli stats $(find research_output -name "session.json" | head -1)
```

#### Output Sections

1. **Overview Panel**: Basic session information
2. **Research Metrics**: Detailed statistics table
3. **Paper Distribution**: Papers by year and source
4. **Performance Metrics**: Processing times and success rates

#### Sample Output

```
📈 Session Statistics

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                              Overview                                 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Research Topic: Machine Learning Interpretability                     │
│ Session ID: a1b2c3d4                                                  │
│ Status: completed                                                      │
│ Created: 2024-01-15 14:30:22                                         │
└────────────────────────────────────────────────────────────────────────┘

                           Research Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric                       ┃                                   Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Total Papers                 │                                      87 │
│ Queries Processed           │                                      12 │
│ Iterations                   │                                       4 │
│ Success Rate                 │                                   92.1% │
│ Processing Time              │                               1,847.3s │
└──────────────────────────────┴─────────────────────────────────────────┘

📅 Papers by Year                    📚 Papers by Source
├── 2024: 23 papers                 ├── ArXiv: 45 papers
├── 2023: 31 papers                 ├── Sci-Hub: 28 papers
├── 2022: 19 papers                 └── PubMed: 14 papers
└── 2021: 14 papers
```

### `export` - Export Results

Export research results to various formats.

```bash
uv run python -m research_agent.cli export [SESSION_FILE] [OUTPUT_FILE] [OPTIONS]
```

#### Arguments

- `SESSION_FILE` (required): Path to research session JSON
- `OUTPUT_FILE` (required): Output file path

#### Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--format` | `-f` | CHOICE | markdown | Export format |
| `--citations/--no-citations` | - | BOOL | True | Include bibliography |

#### Export Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `markdown` | Formatted Markdown | Documentation, GitHub |
| `json` | Structured JSON | Data processing, APIs |
| `html` | HTML report | Web viewing, publishing |
| `text` | Plain text | Simple viewing, processing |

#### Examples

```bash
# Export to Markdown with citations
uv run python -m research_agent.cli export session.json report.md

# Export to HTML without citations
uv run python -m research_agent.cli export session.json report.html \
  --format html \
  --no-citations

# Export to JSON for processing
uv run python -m research_agent.cli export session.json data.json \
  --format json

# Batch export multiple formats
SESSION="research_output/ml_interpretability/session.json"
uv run python -m research_agent.cli export $SESSION report.md --format markdown
uv run python -m research_agent.cli export $SESSION report.html --format html
uv run python -m research_agent.cli export $SESSION report.json --format json
```

#### Export Content

Exported files include:

- **Executive Summary**: Key findings and statistics
- **Literature Review Sections**: Organized by topics
- **Key Findings**: Bullet-pointed insights
- **Research Gaps**: Identified limitations and opportunities
- **Bibliography**: Formatted citations (if enabled)
- **Methodology**: Research approach and tools used
- **Statistics**: Paper counts, sources, temporal analysis

## Interactive Features

### Rich Console Output

The CLI uses Rich library for enhanced terminal experience:

- **Progress Bars**: Real-time progress tracking
- **Colored Output**: Syntax highlighting and status colors
- **Panels**: Organized information display
- **Tables**: Formatted data presentation
- **Tree Views**: Hierarchical information
- **Live Updates**: Dynamic content updates

### Progress Tracking

During research execution:

```
🔬 Research Agent
Topic: Machine Learning Interpretability

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                              Configuration                             ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ • Max Papers: 100                                                     │
│ • Max Iterations: 5                                                   │
│ • Mode: cloud                                                         │
│ • Output: ./research_output/machine_learning_interpretability         │
└────────────────────────────────────────────────────────────────────────┘

► Initializing research system...
► Generating research queries...
► Found 12 research questions
► Optimizing search strategy...
► Searching literature databases...
► Processing papers through 3-tier RAG...
    ├── Tier 1: LightRAG entity extraction (45 papers)
    ├── Tier 2: PaperQA2 research analysis (45 papers)
    └── Tier 3: GraphRAG knowledge reasoning (45 papers)
► Analyzing research gaps...
► Synthesis in progress...
```

### Error Handling

The CLI provides detailed error messages and suggestions:

```bash
# Configuration error
❌ Error: Configuration file not found: config/missing.yaml
💡 Tip: Check the file path or use --config to specify a different file

# API key error  
❌ Error: OpenAI API key not found
💡 Tip: Set OPENAI_API_KEY in your .env file or environment variables

# Session not found
❌ Error: Session file not found: missing_session.json
💡 Tip: Use 'list-sessions' command to see available sessions
```

### Command Completion

Enable bash/zsh completion for enhanced usability:

```bash
# Install completion for current session
eval "$(_RESEARCH_AGENT_COMPLETE=bash_source research-agent)"

# Or add to your shell profile
echo 'eval "$(_RESEARCH_AGENT_COMPLETE=bash_source research-agent)"' >> ~/.bashrc
```

## Output Formats

### Console Output

- **Structured Display**: Rich panels and tables
- **Color Coding**: Status-based coloring (green=success, red=error, yellow=warning)
- **Progress Indicators**: Spinners and progress bars
- **Hierarchical Views**: Tree structures for nested information

### File Outputs

#### JSON Format
```json
{
    "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "topic": "Machine Learning Interpretability",
    "status": "completed",
    "created_at": "2024-01-15T14:30:22Z",
    "statistics": {
        "total_papers": 87,
        "queries_completed": 12,
        "iterations_completed": 4,
        "success_rate": 0.921,
        "total_time": 1847.3
    },
    "literature_review": {
        "executive_summary": {...},
        "sections": {...}
    }
}
```

#### Markdown Format
```markdown
# Literature Review: Machine Learning Interpretability

## Executive Summary
- **Total Papers Analyzed**: 87
- **Key Themes**: SHAP, LIME, attention mechanisms
- **Research Gaps**: Limited multi-modal interpretability
- **Completion Score**: 92%

## Key Findings
1. Attention-based interpretability shows 23% improvement...
2. SHAP values provide most consistent explanations...

## Bibliography
- Vaswani, A., et al. (2017). Attention Is All You Need...
```

## Configuration

### Configuration File

Use custom configuration files to modify behavior:

```bash
# Create custom config
cp config/research_config.yaml config/my_config.yaml

# Edit configuration
# ... modify settings ...

# Use custom config
uv run python -m research_agent.cli research "topic" --config config/my_config.yaml
```

### Environment Variables

Override specific settings:

```bash
# Set processing mode
export PROCESSING_MODE=ollama

# Set cache directory
export CACHE_DIR=./my_cache

# Use environment overrides
uv run python -m research_agent.cli research "topic"
```

## Examples & Workflows

### Complete Research Workflow

```bash
#!/bin/bash
# research_workflow.sh

# 1. Set up environment
export OPENAI_API_KEY="your_key"
export ANTHROPIC_API_KEY="your_key"
export OPENALEX_EMAIL="your@email.com"

# 2. Conduct research
echo "Starting research on transformers..."
uv run python -m research_agent.cli research "transformer attention mechanisms" \
  --max-papers 100 \
  --iterations 5 \
  --mode hybrid \
  --output ./transformer_research \
  --verbose \
  --no-interactive

# 3. Generate statistics
echo "Generating statistics..."
uv run python -m research_agent.cli stats ./transformer_research/session.json

# 4. Export to multiple formats
echo "Exporting results..."
SESSION="./transformer_research/session.json"
uv run python -m research_agent.cli export $SESSION report.md --format markdown
uv run python -m research_agent.cli export $SESSION report.html --format html
uv run python -m research_agent.cli export $SESSION data.json --format json

echo "Research complete! Files available in ./transformer_research/"
```

### Batch Processing Multiple Topics

```bash
#!/bin/bash
# batch_research.sh

TOPICS=(
  "machine learning interpretability"
  "transformer attention mechanisms"
  "neural network optimization"
  "adversarial machine learning"
)

for topic in "${TOPICS[@]}"; do
  echo "Researching: $topic"
  
  # Create safe directory name
  dir_name=$(echo "$topic" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')
  
  # Conduct research
  uv run python -m research_agent.cli research "$topic" \
    --max-papers 50 \
    --iterations 3 \
    --output "./batch_research/$dir_name" \
    --no-interactive
  
  # Export results
  uv run python -m research_agent.cli export \
    "./batch_research/$dir_name/session.json" \
    "./batch_research/$dir_name/report.md"
done

# List all sessions
echo "All research sessions:"
uv run python -m research_agent.cli list-sessions --dir ./batch_research
```

### Interactive Analysis Session

```bash
# Start research
uv run python -m research_agent.cli research "quantum machine learning" \
  --max-papers 75 \
  --mode hybrid

# Interactive chat automatically starts
# During chat:
# You> What are the main quantum algorithms used in machine learning?
# Assistant> Based on the literature review, the main quantum algorithms...

# You> /suggest
# Assistant> Here are some suggested questions:
# 1. How do quantum neural networks compare to classical ones?
# 2. What are the current limitations of quantum machine learning?
# 3. Which quantum computing platforms are most suitable for ML?

# You> How do quantum neural networks compare to classical ones?
# Assistant> The research shows several key differences...

# You> /export
# Assistant> Chat exported to chat_export_a1b2c3d4.json

# You> /quit
```

### Development and Testing

```bash
# Quick test with minimal resources
uv run python -m research_agent.cli research "test topic" \
  --max-papers 10 \
  --iterations 1 \
  --mode ollama \
  --no-interactive

# Validate configuration
uv run python -m research_agent.cli research "test" \
  --config config/test_config.yaml \
  --dry-run

# Debug mode
uv run python -m research_agent.cli research "debug test" \
  --verbose \
  --config config/debug_config.yaml
```

## Troubleshooting

### Common Issues

#### 1. API Key Errors
```bash
❌ Error: OpenAI API key not found
```
**Solution:**
```bash
# Check environment variables
echo $OPENAI_API_KEY

# Set in .env file
echo "OPENAI_API_KEY=your_key" >> .env

# Or export directly
export OPENAI_API_KEY="your_key"
```

#### 2. Configuration File Not Found
```bash
❌ Error: Configuration file not found: config/missing.yaml
```
**Solution:**
```bash
# Check file exists
ls -la config/

# Use default config
uv run python -m research_agent.cli research "topic"  # No --config flag

# Copy from template
cp config/research_config.yaml config/my_config.yaml
```

#### 3. Session File Issues
```bash
❌ Error: Session file not found or corrupted
```
**Solution:**
```bash
# List available sessions
uv run python -m research_agent.cli list-sessions

# Check session file format
file session.json
head session.json

# Regenerate if corrupted
uv run python -m research_agent.cli research "topic"  # New session
```

#### 4. Memory/Performance Issues
```bash
❌ Error: Out of memory during processing
```
**Solution:**
```bash
# Reduce paper count
uv run python -m research_agent.cli research "topic" --max-papers 25

# Use local models (less memory)
uv run python -m research_agent.cli research "topic" --mode ollama

# Adjust configuration
# Edit config/research_config.yaml:
# performance:
#   max_memory_mb: 2048
#   batch_size: 2
```

#### 5. Network/API Issues
```bash
❌ Error: Request timeout or API unavailable
```
**Solution:**
```bash
# Check network connectivity
curl -I https://api.openai.com/v1/models

# Increase timeout in config
# performance:
#   request_timeout: 300

# Use hybrid mode as fallback
uv run python -m research_agent.cli research "topic" --mode hybrid
```

### Debug Commands

```bash
# Verbose output
uv run python -m research_agent.cli research "topic" --verbose

# Check system status
uv run python -c "from research_agent.core.config import ResearchConfig; print('Config loaded successfully')"

# Validate environment
env | grep -E "(OPENAI|ANTHROPIC|OPENALEX)"

# Test Ollama connection (if using local models)
curl http://localhost:11434/api/tags
```

### Log Analysis

```bash
# Check log files
ls -la logs/
tail -f logs/research_agent.log

# Parse JSONL events
cat logs/research_events.jsonl | jq '.event' | sort | uniq -c

# Filter errors
grep ERROR logs/research_agent.log
```

This CLI reference guide provides comprehensive documentation for using the Research Agent System's command-line interface. The rich terminal experience and comprehensive options make it suitable for both interactive use and automated workflows.