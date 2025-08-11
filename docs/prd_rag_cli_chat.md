# Product Requirements Document: RAG-Enabled CLI Chat for PromptChain

## 1. Executive Summary

### 1.1 Overview
This PRD outlines the development of a **RAG-enabled CLI chat interface** for the PromptChain library. The feature will provide users with a persistent, interactive command-line chat experience that leverages the existing PromptChain infrastructure and DeepLake RAG capabilities to answer questions using a knowledge base.

### 1.2 Key Objectives
- Create a CLI-based chat interface that maintains conversation history
- Integrate existing DeepLake RAG functionality for context-aware responses
- Leverage existing PromptChain infrastructure for model management and processing
- Provide session management with persistent storage
- Enable seamless switching between chat sessions
- Support multiple RAG knowledge bases

### 1.3 Success Metrics
- Users can start a CLI chat session and maintain context across interactions
- RAG queries return relevant context from the knowledge base
- Session history is preserved and can be retrieved
- Multiple sessions can be managed independently
- Performance meets interactive chat requirements (< 3 seconds response time)

## 2. Current State Analysis

### 2.1 Existing Infrastructure
The PromptChain library already provides:

**Core Components:**
- `PromptChain` class with async support and model management
- `AgentChain` for orchestrated agent interactions
- `StateAgent` for session and history management
- `ExecutionHistoryManager` for conversation persistence
- Model management with support for multiple providers (OpenAI, Ollama, etc.)

**RAG Capabilities:**
- DeepLake integration in `utils/mcp_deeplake_rag.py`
- `VectorSearchV4` class for vector similarity search
- Functions: `retrieve_context()`, `get_summary()`, `search_document_content()`
- Support for multiple datasets and fuzzy title matching

**Chat Examples:**
- `advanced_router_web_chat/` demonstrates web-based RAG chat
- `simple_agent_user_chat.md` shows basic chat patterns
- `AgentChain_Usage.md` provides agent orchestration examples

### 2.2 Gaps to Address
- No CLI-specific chat interface
- RAG integration not optimized for CLI workflows
- Session management not designed for persistent CLI sessions
- Missing command-line argument parsing and configuration
- No interactive input handling for CLI environment

## 3. Feature Requirements

### 3.1 Core Functionality

#### 3.1.1 CLI Chat Interface
- **Interactive Command Line**: Provide a Python script that can be run from command line
- **Persistent Sessions**: Maintain conversation history across CLI sessions
- **Session Management**: Support multiple named sessions with independent histories
- **Exit Commands**: Graceful exit with session preservation
- **Help System**: Built-in help and command documentation

#### 3.1.2 RAG Integration
- **Knowledge Base Connection**: Connect to existing DeepLake RAG system
- **Context Retrieval**: Automatically retrieve relevant context for user queries
- **Query Enhancement**: Use RAG results to enhance LLM responses
- **Multiple Datasets**: Support switching between different knowledge bases
- **Fallback Handling**: Graceful degradation when RAG is unavailable

#### 3.1.3 Model Management
- **Model Selection**: Allow users to specify models via command line
- **Provider Support**: Support OpenAI, Ollama, and other PromptChain providers
- **Model Switching**: Ability to change models during a session
- **Configuration Persistence**: Remember model preferences across sessions

### 3.2 User Experience Requirements

#### 3.2.1 Command Line Interface
```
# Basic usage
python -m promptchain.cli_chat

# With specific model
python -m promptchain.cli_chat --model openai/gpt-4

# With specific session
python -m promptchain.cli_chat --session my_project

# With specific RAG dataset
python -m promptchain.cli_chat --rag-dataset /path/to/dataset

# With configuration file
python -m promptchain.cli_chat --config chat_config.json
```

#### 3.2.2 Interactive Commands
```
# Chat commands
/help                    - Show available commands
/exit, /quit            - Exit chat session
/new                    - Start new session
/sessions               - List available sessions
/switch <session_name>  - Switch to different session
/summary                - Show current session summary
/rag <query>            - Direct RAG query
/model <model_name>     - Switch model
/config                 - Show current configuration
/clear                  - Clear current session history
```

#### 3.2.3 Configuration Options
- **Model Configuration**: Default model, temperature, max tokens
- **RAG Configuration**: Default dataset path, number of results
- **Session Configuration**: Default session name, history retention
- **Logging Configuration**: Log level, log file location
- **API Configuration**: API keys, endpoints, timeouts

### 3.3 Technical Requirements

#### 3.3.1 Performance
- **Response Time**: < 3 seconds for typical queries
- **Memory Usage**: Efficient memory management for long sessions
- **Concurrent Sessions**: Support multiple CLI sessions simultaneously
- **RAG Performance**: Optimized vector search with caching

#### 3.3.2 Reliability
- **Error Handling**: Graceful handling of API failures, network issues
- **Session Recovery**: Ability to recover from interrupted sessions
- **Data Persistence**: Reliable storage of conversation history
- **Backup/Restore**: Session backup and restoration capabilities

#### 3.3.3 Security
- **API Key Management**: Secure storage of API keys
- **Session Isolation**: Proper isolation between sessions
- **Input Validation**: Sanitization of user inputs
- **Access Control**: Optional session-level access controls

## 4. Implementation Architecture

### 4.1 File Structure
```
promptchain/
├── cli/
│   ├── __init__.py
│   ├── chat.py              # Main CLI chat interface
│   ├── commands.py          # Command processing
│   ├── config.py            # Configuration management
│   ├── session_manager.py   # Session handling
│   └── rag_integration.py   # RAG-specific functionality
├── utils/
│   ├── cli_helpers.py       # CLI-specific utilities
│   └── chat_utils.py        # Chat-specific utilities
└── examples/
    └── cli_chat_example.py  # Usage examples
```

### 4.2 Core Classes

#### 4.2.1 CLIChatInterface
```python
class CLIChatInterface:
    """Main CLI chat interface class."""
    
    def __init__(self, config: ChatConfig):
        self.config = config
        self.session_manager = SessionManager(config)
        self.rag_integration = RAGIntegration(config)
        self.command_processor = CommandProcessor()
        self.prompt_chain = None
        
    async def start(self):
        """Start the interactive chat session."""
        
    async def process_input(self, user_input: str):
        """Process user input and generate response."""
        
    async def execute_command(self, command: str):
        """Execute CLI commands."""
```

#### 4.2.2 SessionManager
```python
class SessionManager:
    """Manages chat sessions and history."""
    
    def __init__(self, config: ChatConfig):
        self.config = config
        self.current_session = None
        self.history_manager = ExecutionHistoryManager()
        
    async def create_session(self, name: str):
        """Create a new chat session."""
        
    async def switch_session(self, name: str):
        """Switch to an existing session."""
        
    async def list_sessions(self):
        """List all available sessions."""
        
    async def save_session(self):
        """Save current session state."""
```

#### 4.2.3 RAGIntegration
```python
class RAGIntegration:
    """Integrates RAG functionality with chat."""
    
    def __init__(self, config: ChatConfig):
        self.config = config
        self.vector_search = None
        self.current_dataset = None
        
    async def initialize_rag(self, dataset_path: str):
        """Initialize RAG with specific dataset."""
        
    async def retrieve_context(self, query: str):
        """Retrieve relevant context for query."""
        
    async def enhance_prompt(self, user_input: str, context: str):
        """Enhance user input with RAG context."""
```

### 4.3 Integration Points

#### 4.3.1 PromptChain Integration
- Use existing `PromptChain` class for model execution
- Leverage `AgentChain` for complex interactions
- Integrate with `StateAgent` for session management
- Use `ExecutionHistoryManager` for conversation persistence

#### 4.3.2 RAG Integration
- Use existing `VectorSearchV4` class from `utils/customdeeplake.py`
- Leverage functions from `utils/mcp_deeplake_rag.py`
- Support multiple dataset paths and configurations
- Implement caching for improved performance

#### 4.3.3 Model Management
- Use existing `ModelManagerFactory` and related classes
- Support all PromptChain model providers
- Implement model switching during sessions
- Maintain model configuration across sessions

## 5. Implementation Plan

### 5.1 Phase 1: Core CLI Interface (Week 1-2)
**Deliverables:**
- Basic CLI chat interface with interactive input
- Session management with persistence
- Command processing system
- Configuration management

**Tasks:**
1. Create `promptchain/cli/` directory structure
2. Implement `CLIChatInterface` class
3. Implement `SessionManager` class
4. Implement `CommandProcessor` class
5. Create configuration management system
6. Add basic command-line argument parsing

### 5.2 Phase 2: RAG Integration (Week 3-4)
**Deliverables:**
- RAG integration with existing DeepLake system
- Context enhancement for user queries
- Multiple dataset support
- RAG-specific commands

**Tasks:**
1. Implement `RAGIntegration` class
2. Integrate with existing `VectorSearchV4`
3. Add RAG context enhancement
4. Implement dataset switching
5. Add RAG-specific CLI commands
6. Optimize RAG performance with caching

### 5.3 Phase 3: Advanced Features (Week 5-6)
**Deliverables:**
- Model switching during sessions
- Advanced session management
- Performance optimizations
- Error handling and recovery

**Tasks:**
1. Implement model switching functionality
2. Add advanced session features (backup, restore)
3. Optimize performance and memory usage
4. Implement comprehensive error handling
5. Add logging and debugging features
6. Create usage examples and documentation

### 5.4 Phase 4: Testing and Documentation (Week 7-8)
**Deliverables:**
- Comprehensive testing suite
- User documentation
- Performance benchmarks
- Deployment guide

**Tasks:**
1. Create unit tests for all components
2. Create integration tests
3. Write user documentation
4. Create performance benchmarks
5. Write deployment and configuration guide
6. Create usage examples

## 6. Technical Specifications

### 6.1 Dependencies
```python
# Core dependencies (already available)
promptchain.utils.promptchaining
promptchain.utils.agent_chain
promptchain.utils.execution_history_manager
promptchain.utils.model_management
promptchain.utils.strategies.state_agent

# RAG dependencies (already available)
utils.mcp_deeplake_rag
utils.customdeeplake

# New CLI dependencies
argparse
asyncio
readline  # For better CLI experience
rich      # For enhanced terminal output
```

### 6.2 Configuration Schema
```json
{
  "model": {
    "default": "openai/gpt-4",
    "temperature": 0.7,
    "max_tokens": 2000
  },
  "rag": {
    "default_dataset": "/path/to/dataset",
    "max_results": 5,
    "similarity_threshold": 0.7
  },
  "session": {
    "default_name": "default",
    "history_retention_days": 30,
    "max_history_length": 1000
  },
  "logging": {
    "level": "INFO",
    "file": "cli_chat.log"
  }
}
```

### 6.3 Database Schema
```sql
-- Extend existing schema for CLI sessions
CREATE TABLE IF NOT EXISTS cli_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    config TEXT,  -- JSON configuration
    status TEXT DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS cli_session_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    config_key TEXT NOT NULL,
    config_value TEXT,
    FOREIGN KEY (session_id) REFERENCES cli_sessions(id)
);
```

## 7. Usage Examples

### 7.1 Basic Usage
```bash
# Start CLI chat with default settings
python -m promptchain.cli_chat

# Start with specific model
python -m promptchain.cli_chat --model openai/gpt-4

# Start with specific session
python -m promptchain.cli_chat --session my_project

# Start with RAG dataset
python -m promptchain.cli_chat --rag-dataset /path/to/knowledge_base
```

### 7.2 Interactive Session
```
Welcome to PromptChain CLI Chat!
Session: default
Model: openai/gpt-4
RAG Dataset: /path/to/knowledge_base

Type /help for available commands.

You: What is the main purpose of this library?

Assistant: Based on the knowledge base, PromptChain is a Python library designed for creating and executing chains of prompts with LLMs. It provides...

You: /rag What are the key features?

Assistant: Searching knowledge base...
Found 3 relevant documents:

1. Core Features:
   - Chain-based prompt processing
   - Model management and switching
   - RAG integration with DeepLake
   - Session management and history

2. Advanced Features:
   - Agentic step processing
   - MCP server integration
   - Dynamic chain building
   - State management

You: /sessions

Available Sessions:
- default (last accessed: 2024-01-15 10:30)
- my_project (last accessed: 2024-01-14 15:45)
- research (last accessed: 2024-01-13 09:20)

You: /switch my_project

Switched to session: my_project
Previous conversation loaded (15 messages)

You: /exit

Saving session...
Goodbye!
```

### 7.3 Configuration File
```json
{
  "model": {
    "default": "openai/gpt-4o-mini",
    "temperature": 0.5,
    "max_tokens": 1500
  },
  "rag": {
    "default_dataset": "/home/user/knowledge_base",
    "max_results": 3,
    "similarity_threshold": 0.8
  },
  "session": {
    "default_name": "work",
    "history_retention_days": 60
  },
  "logging": {
    "level": "INFO",
    "file": "/home/user/.promptchain/cli_chat.log"
  }
}
```

## 8. Testing Strategy

### 8.1 Unit Tests
- Test each component in isolation
- Mock external dependencies (API calls, database)
- Test error conditions and edge cases
- Test configuration management

### 8.2 Integration Tests
- Test end-to-end chat sessions
- Test RAG integration with real datasets
- Test session persistence and recovery
- Test model switching and configuration

### 8.3 Performance Tests
- Measure response times for typical queries
- Test memory usage with long sessions
- Test concurrent session handling
- Test RAG query performance

### 8.4 User Acceptance Tests
- Test CLI usability and command discovery
- Test session management workflows
- Test RAG query workflows
- Test configuration and customization

## 9. Documentation Requirements

### 9.1 User Documentation
- Installation and setup guide
- Basic usage tutorial
- Command reference
- Configuration guide
- Troubleshooting guide

### 9.2 Developer Documentation
- Architecture overview
- API documentation
- Extension guide
- Contributing guidelines

### 9.3 Example Documentation
- Basic chat examples
- RAG integration examples
- Session management examples
- Configuration examples

## 10. Success Criteria

### 10.1 Functional Requirements
- [ ] Users can start CLI chat sessions
- [ ] RAG integration works with existing knowledge bases
- [ ] Session history is preserved across sessions
- [ ] Multiple sessions can be managed independently
- [ ] Model switching works during sessions
- [ ] All CLI commands function correctly

### 10.2 Performance Requirements
- [ ] Response time < 3 seconds for typical queries
- [ ] Memory usage remains stable during long sessions
- [ ] RAG queries complete within 2 seconds
- [ ] Session switching completes within 1 second

### 10.3 Quality Requirements
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Code coverage > 80%
- [ ] Documentation is complete and accurate
- [ ] No critical security vulnerabilities

## 11. Risk Assessment

### 11.1 Technical Risks
- **RAG Performance**: Large datasets may impact response times
  - *Mitigation*: Implement caching and result limiting
- **Memory Usage**: Long sessions may consume excessive memory
  - *Mitigation*: Implement memory management and cleanup
- **API Dependencies**: External API failures may break functionality
  - *Mitigation*: Implement retry logic and graceful degradation

### 11.2 User Experience Risks
- **Complexity**: CLI interface may be too complex for some users
  - *Mitigation*: Provide clear help system and examples
- **Learning Curve**: Users may struggle with command syntax
  - *Mitigation*: Provide interactive help and command discovery

### 11.3 Integration Risks
- **Database Conflicts**: CLI sessions may conflict with existing sessions
  - *Mitigation*: Use separate database schema and proper isolation
- **Configuration Conflicts**: CLI config may conflict with existing config
  - *Mitigation*: Use separate configuration files and clear precedence rules

## 12. Future Enhancements

### 12.1 Phase 2 Features
- **Multi-modal Support**: Support for image and audio inputs
- **Plugin System**: Extensible command and functionality system
- **Advanced RAG**: Support for multiple RAG providers
- **Collaboration**: Multi-user session sharing

### 12.2 Phase 3 Features
- **Web Interface**: Optional web-based chat interface
- **API Server**: REST API for programmatic access
- **Advanced Analytics**: Session analytics and insights
- **Integration APIs**: Integration with external tools and services

## 13. Conclusion

This PRD outlines a comprehensive plan for implementing a RAG-enabled CLI chat interface for the PromptChain library. The implementation leverages existing infrastructure while adding new capabilities specifically designed for command-line workflows.

The feature will provide users with a powerful, persistent chat interface that combines the flexibility of CLI tools with the intelligence of RAG-enhanced LLM interactions. The modular design ensures easy maintenance and future enhancements.

**Implementation Priority**: High
**Estimated Effort**: 8 weeks
**Resource Requirements**: 1-2 developers
**Dependencies**: Existing PromptChain infrastructure, DeepLake RAG system

---

*This PRD is designed to be comprehensive enough for an LLM to understand and implement the feature. It provides clear technical specifications, implementation guidance, and success criteria while leveraging the existing PromptChain infrastructure.* 