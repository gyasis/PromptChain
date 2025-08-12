# API Documentation

This document provides comprehensive API documentation for the Research Agent System components.

## Table of Contents

- [Core Components](#core-components)
- [Research Agents](#research-agents)
- [Session Management](#session-management)
- [Configuration](#configuration)
- [Integrations](#integrations)
- [Utilities](#utilities)

## Core Components

### AdvancedResearchOrchestrator

The main orchestration class that coordinates all research activities.

```python
from research_agent.core.orchestrator import AdvancedResearchOrchestrator
from research_agent.core.config import ResearchConfig

# Initialize orchestrator
config = ResearchConfig.from_yaml("config/research_config.yaml")
orchestrator = AdvancedResearchOrchestrator(config)
```

#### Methods

##### `conduct_research_session(research_topic: str, callbacks: List[Callable] = None) -> ResearchSession`

Conducts a complete research session for the given topic.

**Parameters:**
- `research_topic` (str): The research topic to investigate
- `callbacks` (List[Callable], optional): Progress callback functions

**Returns:**
- `ResearchSession`: Complete research session with results

**Example:**
```python
session = await orchestrator.conduct_research_session(
    research_topic="machine learning interpretability",
    callbacks=[lambda msg: print(f"Progress: {msg}")]
)
```

##### `process_iterative_queries(session: ResearchSession) -> ResearchSession`

Processes queries iteratively with gap analysis and refinement.

**Parameters:**
- `session` (ResearchSession): Current research session

**Returns:**
- `ResearchSession`: Updated session with processed queries

##### `shutdown() -> None`

Gracefully shuts down the orchestrator and cleans up resources.

### ResearchSession

Manages research session state and data.

```python
from research_agent.core.session import ResearchSession, SessionStatus

# Create new session
session = ResearchSession(
    topic="quantum computing applications",
    session_id="unique-session-id"
)
```

#### Properties

- `topic` (str): Research topic
- `session_id` (str): Unique session identifier
- `status` (SessionStatus): Current session status
- `papers` (List[Paper]): Retrieved papers
- `queries` (List[Query]): Generated research queries
- `completed_queries` (List[Query]): Completed queries
- `literature_review` (Dict): Final literature review
- `current_iteration` (int): Current iteration number
- `statistics` (Dict): Session statistics

#### Methods

##### `add_paper(paper: Paper) -> None`

Adds a paper to the session.

**Parameters:**
- `paper` (Paper): Paper object to add

##### `add_query(query: Query) -> None`

Adds a research query to the session.

**Parameters:**
- `query` (Query): Query object to add

##### `calculate_completion_score() -> float`

Calculates research completion score based on various metrics.

**Returns:**
- `float`: Completion score between 0.0 and 1.0

##### `get_session_statistics() -> Dict[str, Any]`

Returns comprehensive session statistics.

**Returns:**
- `Dict[str, Any]`: Statistics dictionary

##### `to_dict() -> Dict[str, Any]`

Serializes session to dictionary format.

**Returns:**
- `Dict[str, Any]`: Serialized session data

## Research Agents

### QueryGenerationAgent

Transforms research topics into comprehensive sets of targeted questions.

```python
from research_agent.agents.query_generator import QueryGenerationAgent

# Initialize agent
config = {
    'model': 'openai/gpt-4o',
    'processor': {
        'objective': 'Generate comprehensive research questions',
        'max_internal_steps': 5
    }
}
agent = QueryGenerationAgent(config)
```

#### Methods

##### `generate_queries(topic: str, context: Dict = None) -> Dict[str, Any]`

Generates research queries for a given topic.

**Parameters:**
- `topic` (str): Research topic
- `context` (Dict, optional): Additional context

**Returns:**
- `Dict[str, Any]`: Generated queries with metadata

**Response Format:**
```json
{
    "primary_queries": [
        {
            "query": "What are the main approaches to machine learning interpretability?",
            "priority": 0.9,
            "category": "overview",
            "keywords": ["interpretability", "explainable AI", "transparency"]
        }
    ],
    "secondary_queries": [...],
    "metadata": {
        "total_queries": 12,
        "generation_time": 15.3,
        "model_used": "openai/gpt-4o"
    }
}
```

### SearchStrategistAgent

Optimizes search strategies across multiple databases and sources.

```python
from research_agent.agents.search_strategist import SearchStrategistAgent

agent = SearchStrategistAgent(config)
```

#### Methods

##### `optimize_search_strategy(queries: List[Dict], findings: Dict = None) -> Dict[str, Any]`

Optimizes search strategy based on queries and previous findings.

**Parameters:**
- `queries` (List[Dict]): Research queries
- `findings` (Dict, optional): Previous search findings

**Returns:**
- `Dict[str, Any]`: Optimized search strategy

**Response Format:**
```json
{
    "search_terms": {
        "sci_hub": ["interpretability", "explainable AI"],
        "arxiv": ["XAI", "model interpretation"],
        "pubmed": ["machine learning explanation"]
    },
    "database_priorities": {
        "sci_hub": 0.8,
        "arxiv": 0.9,
        "pubmed": 0.6
    },
    "filters": {
        "year_range": [2020, 2024],
        "languages": ["en"],
        "document_types": ["journal", "conference"]
    }
}
```

### LiteratureSearchAgent

Coordinates multi-source paper retrieval and validation.

```python
from research_agent.agents.literature_searcher import LiteratureSearchAgent

agent = LiteratureSearchAgent(config)
```

#### Methods

##### `search_literature(strategy: Dict[str, Any]) -> Dict[str, Any]`

Executes literature search based on provided strategy.

**Parameters:**
- `strategy` (Dict[str, Any]): Search strategy from SearchStrategistAgent

**Returns:**
- `Dict[str, Any]`: Search results with papers and metadata

**Response Format:**
```json
{
    "papers": [
        {
            "title": "Attention Is All You Need",
            "authors": ["Vaswani, Ashish", "Shazeer, Noam"],
            "year": 2017,
            "venue": "NIPS",
            "doi": "10.1000/example",
            "abstract": "...",
            "pdf_url": "https://...",
            "source": "arxiv",
            "metadata": {...}
        }
    ],
    "statistics": {
        "total_found": 156,
        "total_retrieved": 89,
        "sources": {
            "arxiv": 45,
            "sci_hub": 28,
            "pubmed": 16
        }
    }
}
```

### ReActAnalysisAgent

Performs iterative gap analysis and research refinement using ReAct reasoning.

```python
from research_agent.agents.react_analyzer import ReActAnalysisAgent

agent = ReActAnalysisAgent(config)
```

#### Methods

##### `analyze_research_gaps(results: Dict[str, Any], original_topic: str) -> Dict[str, Any]`

Analyzes research gaps and determines if additional queries are needed.

**Parameters:**
- `results` (Dict[str, Any]): Multi-tier RAG processing results
- `original_topic` (str): Original research topic

**Returns:**
- `Dict[str, Any]`: Gap analysis results and recommendations

**Response Format:**
```json
{
    "completeness_assessment": {
        "coverage_score": 0.78,
        "topic_coverage": {
            "interpretability_methods": 0.9,
            "evaluation_metrics": 0.6,
            "applications": 0.8
        }
    },
    "identified_gaps": [
        {
            "gap": "Limited coverage of multi-modal interpretability",
            "severity": "medium",
            "suggested_queries": [
                "How do interpretability methods apply to vision-language models?"
            ]
        }
    ],
    "recommendations": {
        "continue_research": true,
        "additional_iterations": 2,
        "focus_areas": ["evaluation metrics", "multi-modal methods"]
    }
}
```

### SynthesisAgent

Generates comprehensive literature reviews from research findings.

```python
from research_agent.agents.synthesis_agent import SynthesisAgent

agent = SynthesisAgent(config)
```

#### Methods

##### `synthesize_literature_review(findings: Dict[str, Any], topic: str) -> Dict[str, Any]`

Synthesizes comprehensive literature review from all research findings.

**Parameters:**
- `findings` (Dict[str, Any]): Aggregated research findings
- `topic` (str): Original research topic

**Returns:**
- `Dict[str, Any]`: Complete literature review

**Response Format:**
```json
{
    "literature_review": {
        "executive_summary": {
            "overview": "This review analyzes 87 papers on machine learning interpretability...",
            "key_findings": [
                "Attention mechanisms provide 23% better interpretability than gradients",
                "SHAP values show highest consistency across domains"
            ],
            "research_gaps": [
                "Limited work on multi-modal interpretability",
                "Lack of standardized evaluation metrics"
            ],
            "statistics": {
                "total_papers": 87,
                "date_range": "2020-2024",
                "completion_score": 0.92
            }
        },
        "sections": {
            "methodology_analysis": {...},
            "key_findings": {...},
            "comparative_analysis": {...},
            "future_directions": {...}
        }
    },
    "citations": {
        "bibliography": [...],
        "citation_network": {...}
    },
    "metadata": {
        "synthesis_time": 45.2,
        "model_used": "openai/gpt-4o"
    }
}
```

## Session Management

### SessionManager

Handles session persistence and retrieval.

```python
from research_agent.core.session import SessionManager

manager = SessionManager(cache_path="./data/cache")
```

#### Methods

##### `save_session(session: ResearchSession, path: str = None) -> str`

Saves research session to disk.

**Parameters:**
- `session` (ResearchSession): Session to save
- `path` (str, optional): Custom save path

**Returns:**
- `str`: Path where session was saved

##### `load_session(session_id: str) -> ResearchSession`

Loads research session from disk.

**Parameters:**
- `session_id` (str): Session ID to load

**Returns:**
- `ResearchSession`: Loaded session

##### `list_sessions() -> List[Dict[str, Any]]`

Lists all available sessions.

**Returns:**
- `List[Dict[str, Any]]`: Session metadata list

## Configuration

### ResearchConfig

Configuration management for the research system.

```python
from research_agent.core.config import ResearchConfig

# Load from YAML
config = ResearchConfig.from_yaml("config/research_config.yaml")

# Load from dictionary
config = ResearchConfig.from_dict(config_dict)

# Get default configuration
config = ResearchConfig()
```

#### Methods

##### `from_yaml(file_path: str) -> ResearchConfig`

Creates configuration from YAML file.

**Parameters:**
- `file_path` (str): Path to YAML configuration file

**Returns:**
- `ResearchConfig`: Configuration instance

##### `from_dict(config_dict: Dict[str, Any]) -> ResearchConfig`

Creates configuration from dictionary.

**Parameters:**
- `config_dict` (Dict[str, Any]): Configuration dictionary

**Returns:**
- `ResearchConfig`: Configuration instance

##### `set_ollama_mode() -> None`

Configures system for Ollama-only processing.

##### `set_hybrid_mode() -> None`

Configures system for hybrid cloud/local processing.

##### `to_dict() -> Dict[str, Any]`

Exports configuration to dictionary.

**Returns:**
- `Dict[str, Any]`: Configuration dictionary

## Integrations

### MultiQueryCoordinator

Coordinates processing of multiple queries across RAG tiers.

```python
from research_agent.integrations.multi_query_coordinator import MultiQueryCoordinator

coordinator = MultiQueryCoordinator(config)
```

#### Methods

##### `coordinate_multi_query_processing(papers: List[Paper], queries: List[Query]) -> Dict[str, Any]`

Coordinates processing of papers across all RAG tiers for multiple queries.

**Parameters:**
- `papers` (List[Paper]): Papers to process
- `queries` (List[Query]): Queries to process

**Returns:**
- `Dict[str, Any]`: Coordinated processing results

### MCPIntegration

Manages Model Context Protocol server integrations.

```python
from research_agent.integrations.mcp_integration import MCPIntegration

mcp = MCPIntegration(mcp_config)
```

#### Methods

##### `initialize_servers() -> None`

Initializes all configured MCP servers.

##### `get_available_tools() -> List[Dict[str, Any]]`

Returns available tools from all MCP servers.

**Returns:**
- `List[Dict[str, Any]]`: Tool definitions

##### `call_tool(tool_name: str, parameters: Dict[str, Any]) -> Any`

Calls an MCP tool.

**Parameters:**
- `tool_name` (str): Name of tool to call
- `parameters` (Dict[str, Any]): Tool parameters

**Returns:**
- `Any`: Tool result

## Utilities

### InteractiveChatInterface

Provides interactive chat capabilities for research sessions.

```python
from research_agent.utils.chat_interface import InteractiveChatInterface, ChatMode

chat = InteractiveChatInterface(
    session=research_session,
    config=config,
    orchestrator=orchestrator
)
```

#### Methods

##### `set_mode(mode: ChatMode) -> None`

Sets the chat interaction mode.

**Parameters:**
- `mode` (ChatMode): Chat mode (qa, deep_dive, compare, summary, custom)

##### `process_message(message: str) -> str`

Processes a user message and returns response.

**Parameters:**
- `message` (str): User message

**Returns:**
- `str`: Chat response

##### `get_suggested_questions() -> List[str]`

Returns suggested questions based on research context.

**Returns:**
- `List[str]`: Suggested questions

##### `get_chat_history(limit: int = 50) -> List[Dict[str, Any]]`

Returns chat history.

**Parameters:**
- `limit` (int): Maximum number of messages to return

**Returns:**
- `List[Dict[str, Any]]`: Chat history

##### `export_chat_session() -> Dict[str, Any]`

Exports current chat session.

**Returns:**
- `Dict[str, Any]`: Exported chat data

##### `generate_final_summary() -> str`

Generates summary of chat session.

**Returns:**
- `str`: Chat session summary

## Error Handling

### Custom Exceptions

```python
from research_agent.exceptions import (
    ResearchAgentError,
    ConfigurationError,
    SessionNotFoundError,
    ProcessingError,
    IntegrationError
)

try:
    session = orchestrator.conduct_research_session(topic)
except ProcessingError as e:
    print(f"Processing failed: {e}")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except ResearchAgentError as e:
    print(f"General error: {e}")
```

### Exception Types

- `ResearchAgentError`: Base exception for all research agent errors
- `ConfigurationError`: Configuration-related errors
- `SessionNotFoundError`: Session not found errors
- `ProcessingError`: RAG processing errors
- `IntegrationError`: MCP/external integration errors

## Type Definitions

### Core Types

```python
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass

@dataclass
class Paper:
    title: str
    authors: List[str]
    year: int
    venue: str
    doi: Optional[str] = None
    abstract: Optional[str] = None
    pdf_url: Optional[str] = None
    source: str = "unknown"
    metadata: Dict[str, Any] = None

@dataclass
class Query:
    query: str
    priority: float
    category: str
    keywords: List[str]
    status: str = "pending"
    results: Optional[Dict[str, Any]] = None

class SessionStatus(Enum):
    CREATED = "created"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ChatMode(Enum):
    QUESTION_ANSWERING = "qa"
    DEEP_DIVE = "deep_dive"
    COMPARATIVE_ANALYSIS = "compare"
    SUMMARY_GENERATION = "summary"
    CUSTOM_ANALYSIS = "custom"
```

## Usage Examples

### Complete Research Workflow

```python
import asyncio
from research_agent.core.orchestrator import AdvancedResearchOrchestrator
from research_agent.core.config import ResearchConfig

async def main():
    # Load configuration
    config = ResearchConfig.from_yaml("config/research_config.yaml")
    
    # Initialize orchestrator
    orchestrator = AdvancedResearchOrchestrator(config)
    
    # Conduct research
    session = await orchestrator.conduct_research_session(
        research_topic="transformer attention mechanisms in NLP",
        callbacks=[lambda msg: print(f"Progress: {msg}")]
    )
    
    # Print results
    print(f"Research completed! Found {len(session.papers)} papers")
    print(f"Completion score: {session.calculate_completion_score():.2%}")
    
    # Start interactive chat
    from research_agent.utils.chat_interface import InteractiveChatInterface
    
    chat = InteractiveChatInterface(
        session=session,
        config=config.__dict__,
        orchestrator=orchestrator
    )
    
    # Process a question
    response = await chat.process_message(
        "What are the key limitations of current attention mechanisms?"
    )
    print(f"Answer: {response}")
    
    # Cleanup
    await orchestrator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### Custom Agent Integration

```python
from research_agent.agents.base import BaseAgent
from promptchain import PromptChain

class CustomAnalysisAgent(BaseAgent):
    def __init__(self, config):
        self.config = config
        self.chain = PromptChain(
            models=[config.get('model', 'openai/gpt-4o')],
            instructions=[
                "Analyze papers for: {focus_area}",
                "Generate insights and patterns"
            ]
        )
    
    async def analyze_papers(self, papers, focus_area):
        paper_texts = [p.abstract for p in papers if p.abstract]
        
        result = await self.chain.process_prompt_async(
            "\n\n".join(paper_texts),
            variables={"focus_area": focus_area}
        )
        
        return {
            "analysis": result,
            "papers_analyzed": len(papers),
            "focus_area": focus_area
        }

# Use custom agent
config = {"model": "anthropic/claude-3-sonnet-20240229"}
agent = CustomAnalysisAgent(config)

# Integrate with research session
papers = session.papers
analysis = await agent.analyze_papers(papers, "methodological approaches")
```

This API documentation provides comprehensive coverage of the Research Agent System's components, methods, and usage patterns. Each section includes practical examples and detailed parameter descriptions to help developers integrate and extend the system effectively.