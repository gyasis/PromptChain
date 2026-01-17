# Quickstart: CLI Orchestration Integration

**Feature**: 002-cli-orchestration | **Date**: 2025-11-18
**Phase**: 1 - Quickstart Guide

## Overview

This guide demonstrates the enhanced PromptChain CLI with AgentChain orchestration, multi-hop reasoning, external tools, and token-efficient history management. Follow these examples to understand the new capabilities.

## Prerequisites

- Python 3.8+
- PromptChain library installed (`pip install -e .`)
- API keys configured in `.env` file:
  ```bash
  OPENAI_API_KEY=your_key_here
  ANTHROPIC_API_KEY=your_key_here
  ```

## Zero-Config Experience

### Launch CLI with Defaults

```bash
# Start CLI with automatic default agent
promptchain

# or with specific session name
promptchain --session my-project
```

**What happens**:
- Default agent created automatically (`model: gpt-4`, basic capabilities)
- MCP servers (filesystem, code_execution) attempt auto-connection
- Ready for conversation immediately - no configuration required

**Example conversation**:
```
> Analyze the authentication patterns in src/auth.py
[System automatically routes to default agent]
[Agent reads file via @syntax and provides analysis]

> Now generate documentation for this module
[Agent switches task, maintains context from previous exchange]
```

---

## YAML Configuration (Recommended)

### Create Project Configuration

**File**: `.promptchain.yml` (in project root)

```yaml
# MCP Servers
mcp_servers:
  - id: filesystem
    type: stdio
    command: mcp-server-filesystem
    args:
      - --root
      - ${PWD}
    auto_connect: true

  - id: code_execution
    type: stdio
    command: mcp-server-code-execution
    auto_connect: true

# Agent Definitions
agents:
  coder:
    model: openai/gpt-4
    description: "Code analysis and generation specialist"
    instruction_chain:
      - "Analyze code request: {input}"
      - "Generate or refactor code: {input}"
    tools:
      - filesystem_read
      - filesystem_write
      - code_execution
    history:
      enabled: true
      max_tokens: 4000
      truncation_strategy: keep_last

  researcher:
    model: openai/gpt-4
    description: "Research specialist with web access"
    instruction_chain:
      - "Analyze research query: {input}"
      - type: agentic_step
        objective: "Research topic comprehensively using available tools"
        max_internal_steps: 8
      - "Synthesize findings: {input}"
    tools:
      - web_search
      - filesystem_read
    history:
      enabled: true
      max_tokens: 8000
      truncation_strategy: oldest_first

# Orchestration
orchestration:
  execution_mode: router
  default_agent: coder
  router:
    model: openai/gpt-4o-mini
    decision_prompt: |
      User query: {user_input}

      Available agents:
      {agent_details}

      Choose the agent best suited for this query.
      Return JSON: {"chosen_agent": "agent_name"}

# Preferences
preferences:
  verbose: false
  show_token_usage: true
  show_reasoning_steps: true
```

### Launch with Configuration

```bash
# Automatically loads .promptchain.yml from current directory
promptchain

# or specify config file explicitly
promptchain --config /path/to/config.yml
```

**What happens**:
- Agents created from YAML (coder + researcher)
- MCP servers connect automatically
- Router mode enabled for intelligent agent selection
- Token-efficient history configs applied

---

## Multi-Agent Workflows

### Automatic Agent Routing

```bash
promptchain --session code-analysis
```

**Conversation flow**:
```
> Find all Python files with authentication logic
[Router selects: coder (file operations)]
[Coder uses filesystem_read tool, returns list of files]

> Research OAuth2 best practices for Python
[Router selects: researcher (research query)]
[Researcher uses web_search + multi-hop reasoning]
[Returns: comprehensive OAuth2 analysis with sources]

> Based on that research, refactor our auth system to use OAuth2
[Router selects: coder (code generation)]
[Coder uses previous context + research findings]
[Generates: refactored code with OAuth2 implementation]
```

**Key points**:
- No manual `/agent use` commands needed
- Router analyzes query type and conversation context
- Each agent maintains appropriate history (coder: recent code, researcher: full context)
- Token savings: 30-60% vs all-agents-full-history baseline

---

## Complex Multi-Hop Reasoning

### Using AgenticStepProcessor

**Agent configuration** (in YAML):
```yaml
agents:
  analyst:
    model: anthropic/claude-3-sonnet
    description: "Complex analysis with multi-step reasoning"
    instruction_chain:
      - "Understand requirements: {input}"
      - type: agentic_step
        objective: "Analyze problem from multiple angles using tools"
        max_internal_steps: 10
      - "Synthesize insights: {input}"
    tools:
      - filesystem_read
      - code_execution
    history:
      max_tokens: 6000
```

**Example usage**:
```
> Analyze our test coverage, identify gaps, and recommend improvements

[Agent begins multi-hop reasoning]
Step 1: List all test files
  → Uses filesystem_read to find tests/
Step 2: Analyze test patterns
  → Reads test files, identifies coverage patterns
Step 3: Execute tests to see actual coverage
  → Uses code_execution to run pytest with coverage
Step 4: Compare test files against source files
  → Identifies untested modules
Step 5: Analyze complexity of untested code
  → Prioritizes high-risk areas
Step 6: Generate recommendations
  → Synthesizes findings into actionable plan

[Returns: Comprehensive analysis with prioritized recommendations]
```

**Key points**:
- Agent autonomously executes 6 reasoning steps
- Calls tools (filesystem, code_execution) without asking user
- Synthesizes multi-source information
- User receives final analysis, not step-by-step prompts

---

## Command-Based Agent Management

### Creating Agents Manually

```bash
# Launch CLI
promptchain

# Inside CLI session:
> /agent create debugger --model gpt-4 --description "Debug specialist for error analysis"
Created agent 'debugger' with model gpt-4

> /agent list
  1. coder (gpt-4) - Code analysis and generation specialist
  2. researcher (gpt-4) - Research specialist with web access
  3. debugger (gpt-4) - Debug specialist for error analysis

> /agent use debugger
Now using agent: debugger

> Why is this function throwing a TypeError?
[Debugger agent analyzes error with focused context]
```

### Agent Templates

```bash
# Quick agent creation from templates
> /agent create-from-template researcher my-researcher
Created agent 'my-researcher' from template 'researcher'
  - Model: gpt-4
  - Tools: web_search, filesystem_read
  - Multi-hop reasoning: enabled (8 steps)
  - History: 8000 tokens, oldest_first

> /agent create-from-template terminal shell-helper
Created agent 'shell-helper' from template 'terminal'
  - Model: gpt-3.5-turbo (fast, cheap)
  - Tools: code_execution
  - History: disabled (60% token savings)
```

**Available templates**:
- `researcher`: Multi-hop reasoning + web search
- `coder`: File ops + code execution + validation
- `analyst`: Data analysis + synthesis
- `terminal`: Command execution, no history

---

## MCP Tool Integration

### Managing MCP Servers

```bash
# Inside CLI session:
> /tools list
Connected MCP Servers:
  filesystem (stdio) - 15 tools
    - filesystem_read
    - filesystem_write
    - filesystem_list
    - ...

  code_execution (stdio) - 3 tools
    - execute_python
    - execute_bash
    - execute_javascript

> /tools add web_search
Connecting to web_search MCP server...
Connected! Discovered 8 tools:
  - web_search
  - web_scrape
  - ...

> /tools remove web_search
Disconnected web_search MCP server
```

### Using Tools in Conversation

```
> List all Python files in src/ directory
[Agent calls filesystem_list tool automatically]
Found 42 Python files in src/

> Run the test suite
[Agent calls execute_python("pytest tests/") tool]
Test Results:
  ✓ 145 passed
  ✗ 3 failed
[Displays test output]

> Search for "Python async best practices"
[Agent calls web_search tool if connected]
[Returns: Top 5 results with summaries]
```

**Key points**:
- Tools called automatically based on query needs
- MCP servers gracefully degrade if unavailable
- CLI functions without MCP (uses @syntax and !syntax fallbacks)

---

## Workflow State Management

### Multi-Session Workflows

```bash
# Start complex workflow
> /workflow create "Implement user authentication system"
Created workflow: auth-implementation
Objective: Implement user authentication system

# Work on steps (CLI tracks progress)
> Research authentication libraries for Python
[Agent performs research, CLI marks as step 1]

> Design database schema for users and sessions
[Agent designs schema, CLI marks as step 2]

# Exit session
> /exit
Workflow 'auth-implementation' saved (2 of 5 steps completed)

# Resume later
promptchain --session auth-project

> /workflow status
Workflow: auth-implementation
Objective: Implement user authentication system
Progress: 2 of 5 steps completed (40%)

Steps:
  ✓ 1. Research authentication libraries
  ✓ 2. Design database schema
  ○ 3. Implement login endpoints
  ○ 4. Add JWT token management
  ○ 5. Write integration tests

> /workflow resume
Resuming from step 3: Implement login endpoints
[Full context restored, ready to continue]
```

**Key points**:
- Workflows persist across sessions
- Progress tracked automatically
- Context restoration on resume

---

## Token Optimization

### Per-Agent History Configs

**Scenario**: 4-agent workflow with 20-message conversation

**Baseline** (all agents with full 8000-token history):
```
Agent 1: 8000 tokens
Agent 2: 8000 tokens
Agent 3: 8000 tokens
Agent 4: 8000 tokens
Total: 32,000 tokens/turn
```

**Optimized** (tiered history configs):
```
Terminal agent: 0 tokens (disabled)
Coder agent: 4000 tokens (recent code context)
Analyst agent: 6000 tokens (moderate context)
Researcher agent: 8000 tokens (full context)
Total: 18,000 tokens/turn
**Savings: 44% token reduction**
```

**Configuration** (in YAML):
```yaml
agents:
  terminal:
    history:
      enabled: false  # 60% savings

  coder:
    history:
      max_tokens: 4000
      truncation_strategy: keep_last  # Recent code matters

  analyst:
    history:
      max_tokens: 6000
      truncation_strategy: oldest_first

  researcher:
    history:
      max_tokens: 8000
      truncation_strategy: oldest_first  # Full arc
```

---

## Advanced Configuration

### Environment Variable Substitution

```yaml
# .promptchain.yml
mcp_servers:
  - id: filesystem
    args:
      - --root
      - ${PROJECT_ROOT}  # Substituted from environment

session:
  working_directory: ${PWD}

# In shell:
export PROJECT_ROOT=/home/user/myproject
promptchain  # Uses $PROJECT_ROOT in filesystem server config
```

### Configuration Precedence

**Priority order** (highest to lowest):
1. Command-line arguments: `promptchain --verbose`
2. Project config: `./.promptchain.yml`
3. User config: `~/.promptchain/config.yml`
4. Built-in defaults

**Example**:
```bash
# User global config
cat ~/.promptchain/config.yml
preferences:
  verbose: false

# Project config overrides
cat .promptchain.yml
preferences:
  verbose: true

# Result: verbose=true (project config wins)
promptchain
```

---

## Troubleshooting

### MCP Server Connection Failures

**Problem**: `MCP server 'filesystem' unavailable - file operation tools disabled`

**Solution**:
1. Check MCP server is installed: `which mcp-server-filesystem`
2. Test server manually: `mcp-server-filesystem --root .`
3. Check YAML config has correct command and args
4. Verify auto_connect: true (or use `/tools add filesystem`)

**Graceful degradation**:
- CLI continues functioning
- File operations via @syntax still work
- Log shows which tools are unavailable

### Agent Routing Issues

**Problem**: Router selects wrong agent consistently

**Solution**:
1. Improve agent descriptions (more specific specializations)
2. Customize router decision_prompt with examples
3. Check router model (gpt-4o-mini may need upgrade to gpt-4 for complex routing)
4. Use manual `/agent use <name>` as override

### Token Limit Exceeded

**Problem**: `Token limit exceeded - conversation history too long`

**Solution**:
1. Reduce `max_tokens` in agent history config
2. Enable `truncation_strategy: keep_last` for recent-context workflows
3. Use `include_types` filter to exclude tool_result entries
4. Disable history entirely for terminal-style agents

---

## Next Steps

1. **Explore templates**: Try all 4 agent templates to see different capabilities
2. **Customize YAML**: Adapt example config to your workflow needs
3. **Create specialized agents**: Design agents for your specific use cases
4. **Set up MCP servers**: Connect additional tools (database, API clients, etc.)
5. **Multi-session workflows**: Practice complex objectives spanning multiple days

## Reference Documentation

- Full spec: [spec.md](./spec.md)
- Data models: [data-model.md](./data-model.md)
- JSON schemas: [contracts/](./contracts/)
- Implementation plan: [plan.md](./plan.md)
- Research decisions: [research.md](./research.md)

---

**Version**: 2.0 (CLI Orchestration Integration)
**Last Updated**: 2025-11-18
