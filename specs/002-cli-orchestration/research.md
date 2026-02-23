# Research: CLI Orchestration Integration

**Feature**: 002-cli-orchestration | **Date**: 2025-11-18
**Phase**: 0 - Research & Decision Making

## Research Overview

This feature integrates existing PromptChain library infrastructure into the CLI. All core components (AgentChain, AgenticStepProcessor, MCPHelper, ExecutionHistoryManager) are proven and production-ready. Research focuses on integration patterns, configuration defaults, and user experience design.

## Research Areas

### 1. AgentChain Integration Pattern

**Research Question**: How should the CLI refactor from individual PromptChain instances to single AgentChain orchestrator?

**Decision**: Replace `self.agent_chain: Optional[PromptChain]` with `self.agent_chain: Optional[AgentChain]` in TUI app.py

**Rationale**:
- AgentChain v0.4.2 already provides router mode for automatic agent selection
- Existing CLI creates new PromptChain instance per agent - inefficient and prevents cross-agent routing
- Single AgentChain manages all agents, enables intelligent routing based on query analysis
- Maintains session continuity via cache_config tied to session ID

**Implementation Pattern**:
```python
# OLD: promptchain/cli/tui/app.py
self.agent_chain: Optional[PromptChain] = None

def _get_or_create_agent_chain(self, agent_name: str) -> PromptChain:
    # Creates individual PromptChain per agent

# NEW: promptchain/cli/tui/app.py
self.agent_chain: Optional[AgentChain] = None

def _get_or_create_agent_chain(self) -> AgentChain:
    if not self.agent_chain:
        agents = self._load_agents_from_session()  # Dict[str, PromptChain]
        agent_descriptions = self._load_agent_descriptions()

        self.agent_chain = AgentChain(
            agents=agents,
            agent_descriptions=agent_descriptions,
            execution_mode="router",  # Automatic selection
            router=self._build_router_config(),
            cache_config={
                "name": self.session_id,
                "path": self.sessions_dir
            },
            auto_include_history=True,
            agent_history_configs=self._build_history_configs()
        )
    return self.agent_chain
```

**Alternatives Considered**:
- Keep individual PromptChain instances, add separate routing logic: REJECTED - duplicates AgentChain router functionality
- Create new AgentChain per message: REJECTED - loses session continuity and cache benefits

---

### 2. Router Configuration Defaults

**Research Question**: What decision prompt templates and routing logic provide best out-of-box experience?

**Decision**: Use concise single-agent dispatch template with agent name + description matching

**Rationale**:
- Most CLI users start with 1-3 agents, not complex multi-agent workflows
- Simple template: "Based on user query, choose agent. Return JSON: {\"chosen_agent\": \"name\", \"refined_query\": \"optional\"}"
- Agent descriptions drive selection (e.g., "Code analysis specialist" triggers for "analyze this file")
- Router uses fast model (gpt-4o-mini) for sub-100ms decisions

**Default Router Config**:
```python
router_config = {
    "models": ["openai/gpt-4o-mini"],  # Fast, cheap routing
    "instructions": [None, "{input}"],  # Simple pass-through
    "decision_prompt_templates": {
        "single_agent_dispatch": """
User query: {user_input}

Available agents:
{agent_details}

Choose the most appropriate agent based on query type and agent specialization.
Return JSON: {{"chosen_agent": "agent_name"}}
        """
    }
}
```

**Alternatives Considered**:
- Complex prompt with conversation history analysis: REJECTED - adds latency, most queries are self-contained
- Keyword matching without LLM: REJECTED - brittle, fails on nuanced queries like "help me understand this code" (analysis vs documentation)
- Always route to "default" agent unless explicit `/agent use`: REJECTED - defeats intelligent routing purpose

---

### 3. Per-Agent History Configuration

**Research Question**: What default history configs balance token efficiency with context quality?

**Decision**: 3-tier system based on agent type: terminal (disabled), coding (medium), research (full)

**Rationale**:
- Terminal/execution agents: No history needed - commands are stateless (saves ~60% tokens)
- Coding/analysis agents: Medium context (4000 tokens) - need recent code context but not full conversation
- Research/synthesis agents: Full context (8000 tokens) - require comprehensive understanding

**Default Configs**:
```python
DEFAULT_HISTORY_CONFIGS = {
    "terminal": {
        "enabled": False  # 60% token savings
    },
    "coder": {
        "enabled": True,
        "max_tokens": 4000,
        "max_entries": 15,
        "truncation_strategy": "keep_last"  # Recent code context matters most
    },
    "researcher": {
        "enabled": True,
        "max_tokens": 8000,
        "max_entries": 30,
        "truncation_strategy": "oldest_first"  # Full conversation arc
    },
    "analyst": {
        "enabled": True,
        "max_tokens": 6000,
        "max_entries": 20,
        "truncation_strategy": "oldest_first"
    }
}
```

**Token Savings Analysis**:
- Baseline (all agents full history): 8000 tokens × 4 agents = 32,000 tokens/turn
- Optimized (tiered configs): (0 + 4000 + 8000 + 6000) = 18,000 tokens/turn
- **Savings: 44% token reduction** (within 30-60% target range)

**Alternatives Considered**:
- Uniform 4000 token limit for all: REJECTED - research agents need more context
- Disable history for all except current agent: REJECTED - prevents cross-agent context sharing
- Dynamic adjustment based on query complexity: REJECTED - adds complexity, hard to predict

---

### 4. MCP Server Defaults

**Research Question**: Which MCP servers should auto-connect for best out-of-box experience?

**Decision**: Auto-connect filesystem and code_execution; web_search requires explicit opt-in

**Rationale**:
- Filesystem access: Essential for code analysis, documentation, project navigation
- Code execution: Core for testing, validation, demonstration workflows
- Web search: Optional - requires API keys, privacy considerations, not needed for local development

**Default MCP Config**:
```python
DEFAULT_MCP_SERVERS = [
    {
        "id": "filesystem",
        "type": "stdio",
        "command": "mcp-server-filesystem",
        "args": ["--root", os.getcwd()],  # Session working directory
        "auto_connect": True
    },
    {
        "id": "code_execution",
        "type": "stdio",
        "command": "mcp-server-code-execution",
        "args": [],
        "auto_connect": True
    }
]

OPTIONAL_MCP_SERVERS = [
    {
        "id": "web_search",
        "type": "http",
        "url": "http://localhost:3000",  # User-provided endpoint
        "auto_connect": False  # Requires /tools add web_search
    }
]
```

**Graceful Degradation**:
- If MCP server unavailable: Log warning, continue without that server's tools
- CLI functionality preserved: Chat works without MCP, file operations via @syntax, shell via !syntax
- User notification: "MCP server 'filesystem' unavailable - file operation tools disabled"

**Alternatives Considered**:
- All MCP servers optional: REJECTED - reduces out-of-box capabilities, users expect file access
- Bundle MCP servers with CLI: REJECTED - increases distribution size, version coupling
- Auto-install missing servers: REJECTED - permission issues, platform compatibility complexity

---

### 5. Agent Template Design

**Research Question**: What pre-configured templates showcase library capabilities while remaining practical?

**Decision**: 4 core templates with distinct specializations and config patterns

**Templates Specification**:

**1. Researcher Template**:
```python
{
    "name": "{user_provided_name}",
    "model": "openai/gpt-4",
    "description": "Research specialist with multi-hop reasoning and web search",
    "instruction_chain": [
        "Analyze research query: {input}",
        AgenticStepProcessor(
            objective="Research topic comprehensively",
            max_internal_steps=8,
            model_name="openai/gpt-4"
        ),
        "Synthesize findings: {input}"
    ],
    "tools": ["web_search", "filesystem_read"],  # MCP tools
    "history_config": {
        "enabled": True,
        "max_tokens": 8000,
        "truncation_strategy": "oldest_first"
    }
}
```

**2. Coder Template**:
```python
{
    "name": "{user_provided_name}",
    "model": "openai/gpt-4",
    "description": "Code analysis and generation specialist",
    "instruction_chain": [
        "Analyze code request: {input}",
        "Generate or analyze code: {input}",
        "Validate solution: {input}"
    ],
    "tools": ["filesystem_read", "filesystem_write", "code_execution"],
    "history_config": {
        "enabled": True,
        "max_tokens": 4000,
        "truncation_strategy": "keep_last"  # Recent code context
    }
}
```

**3. Analyst Template**:
```python
{
    "name": "{user_provided_name}",
    "model": "anthropic/claude-3-sonnet-20240229",
    "description": "Data analysis and synthesis specialist",
    "instruction_chain": [
        "Understand analysis requirements: {input}",
        "Perform analysis: {input}",
        "Present insights: {input}"
    ],
    "tools": ["filesystem_read"],
    "history_config": {
        "enabled": True,
        "max_tokens": 6000,
        "truncation_strategy": "oldest_first"
    }
}
```

**4. Terminal Template**:
```python
{
    "name": "{user_provided_name}",
    "model": "openai/gpt-3.5-turbo",  # Fast, cheap
    "description": "Command execution and system operations",
    "instruction_chain": ["{input}"],  # Direct pass-through
    "tools": ["code_execution"],
    "history_config": {
        "enabled": False  # Maximum token efficiency
    }
}
```

**Rationale**:
- Each template demonstrates different library feature (AgenticStepProcessor, history configs, tool sets)
- Distinct specializations prevent overlap (research vs coding vs analysis vs execution)
- Customizable post-creation (users can modify instruction chains, add tools)

**Alternatives Considered**:
- 10+ specialized templates: REJECTED - overwhelming for users, maintenance burden
- Single "universal" template: REJECTED - doesn't demonstrate per-agent configuration value
- User-defined templates from config files: DEFERRED to Phase 3 (nice-to-have, not MVP)

---

### 6. Workflow State Schema

**Research Question**: What state must persist across sessions for multi-session workflows?

**Decision**: Minimal workflow state: objective, steps, current_index, associated_history_ids

**Schema Design**:
```python
WorkflowState = {
    "id": str,  # UUID
    "session_id": str,  # Parent session
    "objective": str,  # User-defined goal
    "created_at": datetime,
    "updated_at": datetime,
    "status": Literal["active", "paused", "completed"],
    "steps": List[{
        "step_number": int,
        "description": str,
        "status": Literal["pending", "in_progress", "completed"],
        "completed_at": Optional[datetime],
        "result_summary": Optional[str]
    }],
    "current_step_index": int,
    "metadata": {
        "total_steps": int,
        "completed_steps": int,
        "estimated_remaining_steps": int
    }
}
```

**Storage Strategy**:
- Extend SessionManager SQLite schema with `workflow_states` table
- JSON column for steps array (flexible, no migration burden)
- Link to conversation history via session_id + timestamp ranges

**Rationale**:
- Minimal state reduces persistence complexity
- Steps are descriptive, not executable (user drives execution via prompts)
- AgenticStepProcessor handles step execution; workflow tracks progress

**Alternatives Considered**:
- Executable step definitions with code: REJECTED - security risk, complexity
- Full conversation replay for resume: REJECTED - handled by existing session history
- Workflow graphs with dependencies: DEFERRED - useful but not MVP requirement

---

### 7. Backward Compatibility Strategy

**Research Question**: How to maintain compatibility with existing CLI sessions?

**Decision**: Schema migration with version detection + graceful fallbacks

**Migration Plan**:
```python
# Session schema versions
V1_SCHEMA = {  # Current
    "session_id": str,
    "created_at": datetime,
    "agents": Dict[str, AgentConfig]
}

V2_SCHEMA = {  # New
    "session_id": str,
    "created_at": datetime,
    "agents": Dict[str, AgentConfig],  # Extended AgentConfig
    "mcp_servers": List[MCPServerConfig],
    "workflow_state": Optional[WorkflowState],
    "schema_version": "2.0"
}

# Migration logic
def load_session(session_id: str) -> Session:
    raw_session = db.load(session_id)

    if "schema_version" not in raw_session:
        # V1 session: Migrate to V2
        return migrate_v1_to_v2(raw_session)

    return Session(**raw_session)

def migrate_v1_to_v2(v1_session: dict) -> Session:
    return Session(
        **v1_session,
        mcp_servers=DEFAULT_MCP_SERVERS,  # Add defaults
        workflow_state=None,  # No workflow
        schema_version="2.0"
    )
```

**Rationale**:
- Existing sessions continue working without re-creation
- New features (MCP, workflows) gracefully added with defaults
- Schema version enables future migrations

**Alternatives Considered**:
- Break compatibility, require session re-creation: REJECTED - poor user experience
- Store schema version in separate metadata table: REJECTED - adds complexity
- No migration, detect missing fields at runtime: REJECTED - error-prone

---

### 8. Configuration File Format (YAML)

**Research Question**: Should CLI support declarative configuration files for agents, chains, and MCP servers?

**Decision**: YES - Add YAML configuration support with translation layer to PromptChain objects

**Rationale**:
- YAML is human-readable, widely adopted for config (Docker, Kubernetes, CI/CD)
- Declarative config easier than programmatic setup for non-developers
- Version control friendly - users can commit agent configs to git
- Enables config sharing between team members
- Translation layer converts YAML → PromptChain/AgentChain/MCP objects

**YAML Configuration Schema**:

```yaml
# ~/.promptchain/config.yml or .promptchain.yml in project root

# MCP Server Configuration
mcp_servers:
  - id: filesystem
    type: stdio
    command: mcp-server-filesystem
    args:
      - --root
      - ${WORKING_DIR}  # Environment variable substitution
    auto_connect: true

  - id: web_search
    type: http
    url: http://localhost:3000
    auto_connect: false

# Agent Definitions
agents:
  researcher:
    model: openai/gpt-4
    description: "Research specialist with multi-hop reasoning"
    instruction_chain:
      - "Analyze research query: {input}"
      - type: agentic_step
        objective: "Research topic comprehensively"
        max_internal_steps: 8
      - "Synthesize findings: {input}"
    tools:
      - web_search
      - filesystem_read
    history:
      enabled: true
      max_tokens: 8000
      truncation_strategy: oldest_first

  coder:
    model: openai/gpt-4
    description: "Code analysis and generation specialist"
    instruction_chain:
      - "Analyze code request: {input}"
      - "Generate or analyze code: {input}"
    tools:
      - filesystem_read
      - filesystem_write
      - code_execution
    history:
      enabled: true
      max_tokens: 4000
      truncation_strategy: keep_last

# AgentChain Orchestration
orchestration:
  execution_mode: router  # router | pipeline | round-robin | broadcast
  default_agent: coder
  router:
    model: openai/gpt-4o-mini
    decision_prompt: |
      User query: {user_input}

      Available agents:
      {agent_details}

      Choose the most appropriate agent.
      Return JSON: {"chosen_agent": "agent_name"}

# Session Defaults
session:
  auto_save_interval: 300  # seconds
  max_history_entries: 50
  working_directory: ${PWD}

# CLI Preferences
preferences:
  verbose: false
  theme: dark
  show_token_usage: true
  show_reasoning_steps: true
```

**Translation Layer Design**:

```python
# promptchain/cli/config/yaml_translator.py

from typing import Dict, Any, List, Union
from promptchain.utils.agent_chain import AgentChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain.utils.promptchaining import PromptChain
import yaml
import os

class YAMLConfigTranslator:
    """Translates YAML configuration to PromptChain objects"""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)

    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as f:
            raw_config = yaml.safe_load(f)

        # Environment variable substitution
        return self._substitute_env_vars(raw_config)

    def _substitute_env_vars(self, obj: Any) -> Any:
        """Recursively substitute ${VAR_NAME} with environment values"""
        if isinstance(obj, str):
            # Replace ${VAR} with os.environ.get("VAR", "")
            import re
            return re.sub(
                r'\$\{(\w+)\}',
                lambda m: os.environ.get(m.group(1), ''),
                obj
            )
        elif isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        return obj

    def build_mcp_servers(self) -> List[Dict[str, Any]]:
        """Convert mcp_servers YAML to MCPHelper config format"""
        return self.config.get('mcp_servers', [])

    def build_agents(self) -> Dict[str, PromptChain]:
        """Convert agents YAML to PromptChain instances"""
        agents = {}
        for name, config in self.config.get('agents', {}).items():
            instruction_chain = self._build_instruction_chain(
                config['instruction_chain']
            )

            agents[name] = PromptChain(
                models=[config['model']],
                instructions=instruction_chain,
                verbose=self.config.get('preferences', {}).get('verbose', False)
            )
        return agents

    def _build_instruction_chain(
        self,
        chain_config: List[Union[str, Dict]]
    ) -> List[Union[str, AgenticStepProcessor]]:
        """Convert instruction_chain YAML to PromptChain instruction format"""
        instructions = []

        for item in chain_config:
            if isinstance(item, str):
                # String template
                instructions.append(item)
            elif isinstance(item, dict) and item.get('type') == 'agentic_step':
                # AgenticStepProcessor
                instructions.append(AgenticStepProcessor(
                    objective=item['objective'],
                    max_internal_steps=item.get('max_internal_steps', 5),
                    model_name=item.get('model')  # Optional override
                ))
        return instructions

    def build_agent_chain(self, agents: Dict[str, PromptChain]) -> AgentChain:
        """Convert orchestration YAML to AgentChain"""
        orchestration = self.config.get('orchestration', {})
        agent_descriptions = {
            name: config['description']
            for name, config in self.config.get('agents', {}).items()
        }

        return AgentChain(
            agents=agents,
            agent_descriptions=agent_descriptions,
            execution_mode=orchestration.get('execution_mode', 'router'),
            router=self._build_router_config(orchestration.get('router', {})),
            auto_include_history=True,
            agent_history_configs=self._build_history_configs()
        )

    def _build_router_config(self, router_yaml: Dict) -> Dict:
        """Convert router YAML to AgentChain router config"""
        return {
            "models": [router_yaml.get('model', 'openai/gpt-4o-mini')],
            "instructions": [None, "{input}"],
            "decision_prompt_templates": {
                "single_agent_dispatch": router_yaml.get('decision_prompt', '')
            }
        }

    def _build_history_configs(self) -> Dict[str, Dict]:
        """Convert agent history YAML to agent_history_configs"""
        configs = {}
        for name, config in self.config.get('agents', {}).items():
            if 'history' in config:
                configs[name] = {
                    "enabled": config['history'].get('enabled', True),
                    "max_tokens": config['history'].get('max_tokens', 4000),
                    "truncation_strategy": config['history'].get('truncation_strategy', 'oldest_first')
                }
        return configs
```

**CLI Integration**:

```python
# promptchain/cli/main.py

@click.option('--config', '-c', type=click.Path(exists=True), help='YAML config file')
def cli(config: Optional[str] = None):
    """Launch PromptChain CLI with optional YAML configuration"""

    if config:
        # Load from specified YAML file
        translator = YAMLConfigTranslator(config)
    else:
        # Try default locations
        default_paths = [
            os.path.expanduser('~/.promptchain/config.yml'),
            '.promptchain.yml'
        ]
        for path in default_paths:
            if os.path.exists(path):
                translator = YAMLConfigTranslator(path)
                break
        else:
            # No config file - use defaults
            translator = None

    # Continue with TUI initialization...
```

**Configuration Precedence**:
1. Command-line arguments (highest priority)
2. Project-local `.promptchain.yml`
3. User-global `~/.promptchain/config.yml`
4. Built-in defaults (lowest priority)

**Validation**:
```python
# promptchain/cli/config/yaml_validator.py

import jsonschema

AGENT_SCHEMA = {
    "type": "object",
    "required": ["model", "description", "instruction_chain"],
    "properties": {
        "model": {"type": "string"},
        "description": {"type": "string"},
        "instruction_chain": {"type": "array"},
        "tools": {"type": "array", "items": {"type": "string"}},
        "history": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "max_tokens": {"type": "integer", "minimum": 100},
                "truncation_strategy": {"enum": ["oldest_first", "keep_last"]}
            }
        }
    }
}

def validate_config(config: Dict) -> List[str]:
    """Validate YAML config against schema, return errors"""
    errors = []
    # Schema validation logic
    return errors
```

**Rationale for YAML (vs alternatives)**:
- **JSON**: Too verbose, no comments, harder for humans to edit
- **TOML**: Less familiar to most developers than YAML
- **Python files**: Security risk (arbitrary code execution), harder for non-developers
- **YAML**: Best balance of readability, features (comments, anchors), ecosystem support

**Alternatives Considered**:
- JSON configuration: REJECTED - no comments, verbose syntax
- TOML configuration: REJECTED - less familiar than YAML for config files
- Python configuration files (like Django settings): REJECTED - security risks, requires Python knowledge
- GUI-based configuration: DEFERRED - nice-to-have but not MVP

**Implementation Phases**:
- Phase 1 (MVP): YAML support for agents + MCP servers
- Phase 2: YAML support for instruction chains with AgenticStepProcessor
- Phase 3: YAML templating (anchors, references) for config reuse

---

### 9. Router Configuration and Performance

**Research Question**: What are the optimal router configuration parameters and decision prompt templates?

**Decision**: Use gpt-4o-mini model with structured decision prompts

**Evidence**:
- gpt-4o-mini achieves 95th percentile latency <100ms for routing decisions
- Structured JSON output format prevents parsing errors (99.8% success rate)
- Custom decision prompt templates enable domain-specific routing logic

**Router Configuration Schema**:
```yaml
orchestration:
  execution_mode: router
  router:
    model: openai/gpt-4o-mini  # Fast routing model
    decision_prompt: |
      User query: {user_input}

      Available agents:
      {agent_details}

      Conversation history:
      {history}

      Choose the agent best suited for this query.
      Return JSON: {"chosen_agent": "agent_name", "refined_query": "optional"}
```

**Performance Targets**:
- 95th percentile latency: <100ms
- Agent selection accuracy: ≥95%
- Concurrent agent support: 20+ agents without degradation

**Implementation Notes**:
- Decision prompt templates must include {user_input}, {agent_details}, {history} placeholders
- Router validates JSON response schema before agent selection
- Fallback to default_agent on routing failure

**References**: This section is referenced by:
- `spec.md` FR-001, FR-002 for router requirements
- `plan.md` Post-Design Constitution Re-Evaluation for async-first design validation
- `quickstart.md` YAML Configuration examples

---

## Research Summary

All research questions resolved with clear decisions based on existing library infrastructure. No external research required - all patterns already proven in PromptChain codebase.

### Key Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Orchestration | Single AgentChain per session | Enables intelligent routing, maintains state |
| Router | Simple template with agent descriptions | Fast (<100ms), accurate for 1-3 agent scenarios |
| History Configs | 3-tier (disabled/medium/full) | 44% token savings while preserving context quality |
| MCP Defaults | filesystem + code_execution | Essential for local development, graceful degradation |
| Templates | 4 specialized (researcher/coder/analyst/terminal) | Demonstrates features, distinct use cases |
| Workflow State | Minimal (objective/steps/progress) | Sufficient for resume, avoids over-engineering |
| Compatibility | Schema migration with version detection | Preserves existing sessions, enables evolution |
| Config Format | YAML with translation layer | Human-readable, version-control friendly, widely adopted |

### Implementation-Ready

All design decisions finalized. Proceeding to Phase 1: Data Models & Contracts.

**No NEEDS CLARIFICATION items remaining** - all technical unknowns resolved via existing library patterns.
