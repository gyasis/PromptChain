# Agent Templates Documentation

## Overview

Agent templates provide pre-configured agent definitions for common use cases in the PromptChain CLI. Templates streamline agent creation with optimized configurations for specific workflows including history management, tool selection, and processing patterns.

## Available Templates

### 1. Researcher Template

**Template ID**: `researcher`
**Category**: Research
**Complexity**: High
**Token Usage**: High

#### Description
Deep research specialist with multi-hop reasoning and web search capabilities. Uses AgenticStepProcessor for 8-step autonomous research workflows.

#### Configuration
- **Model**: `openai/gpt-4`
- **History**: Enabled (8000 tokens, 50 entries)
- **Truncation**: Oldest-first
- **Tools**:
  - `web_search` - Web search capability
  - `mcp_web_browser` (optional) - MCP web browsing
  - `mcp_filesystem_read` (optional) - MCP file reading

#### Instruction Chain
1. Analyze research query
2. AgenticStepProcessor (8-step autonomous research)
   - Comprehensive research using multiple sources
   - Multi-hop reasoning for complex queries
   - Automatic tool invocation
3. Synthesize findings into comprehensive report

#### Use Cases
- Academic research and literature review
- Market research and competitive analysis
- Technical documentation research
- Multi-source information gathering
- Fact-checking and verification

#### Example Usage
```bash
# Create researcher agent
/agent create-from-template researcher my-researcher

# With custom model
/agent create-from-template researcher my-researcher --model anthropic/claude-3-opus-20240229

# Use the agent
/agent use my-researcher
> Research the latest advancements in quantum computing
```

---

### 2. Coder Template

**Template ID**: `coder`
**Category**: Development
**Complexity**: High
**Token Usage**: Moderate

#### Description
Code generation and execution specialist with AgenticStepProcessor for iterative development workflows. Optimized for coding tasks with moderate history for context retention.

#### Configuration
- **Model**: `openai/gpt-4`
- **History**: Enabled (4000 tokens, 20 entries)
- **Truncation**: Oldest-first
- **Tools**:
  - `execute_code` - Code execution in sandboxed environment
  - `mcp_filesystem_read` (optional) - Read source files
  - `mcp_filesystem_write` (optional) - Write generated code

#### Instruction Chain
1. Parse coding request and requirements
2. AgenticStepProcessor (5-step iterative development)
   - Code generation with validation
   - Testing and debugging
   - Iterative refinement
3. Final code review and documentation

#### Use Cases
- Code generation and implementation
- Bug fixing and debugging
- Code refactoring and optimization
- Test creation and validation
- Documentation generation

#### Example Usage
```bash
# Create coder agent
/agent create-from-template coder python-dev

# Use the agent
/agent use python-dev
> Implement a binary search tree with insert, delete, and search methods
```

---

### 3. Analyst Template

**Template ID**: `analyst`
**Category**: Analysis
**Complexity**: High
**Token Usage**: High

#### Description
Data analysis and interpretation specialist with comprehensive context retention. Optimized for complex analytical workflows requiring detailed historical context.

#### Configuration
- **Model**: `openai/gpt-4`
- **History**: Enabled (8000 tokens, 50 entries)
- **Truncation**: Oldest-first
- **Tools**:
  - `data_analysis` - Data analysis and visualization
  - `mcp_filesystem_read` (optional) - Read data files

#### Instruction Chain
1. Understand analysis requirements
2. Data loading and validation
3. Statistical analysis and pattern detection
4. Comprehensive interpretation and insights generation

#### Use Cases
- Statistical analysis and hypothesis testing
- Data visualization and reporting
- Pattern recognition and trend analysis
- Business intelligence and metrics analysis
- Scientific data interpretation

#### Example Usage
```bash
# Create analyst agent
/agent create-from-template analyst data-analyst

# Use the agent
/agent use data-analyst
> Analyze the sales trends in Q4 2024 and identify key drivers
```

---

### 4. Terminal Template

**Template ID**: `terminal`
**Category**: Execution
**Complexity**: Low
**Token Usage**: Minimal (60% savings vs standard agent)

#### Description
Fast execution agent for command-line operations and quick tasks. History disabled for maximum token efficiency. Uses fast model (gpt-3.5-turbo) for sub-second responses.

#### Configuration
- **Model**: `openai/gpt-3.5-turbo`
- **History**: **Disabled** (0 tokens, 0 entries)
- **Token Savings**: ~60% compared to agents with history
- **Tools**:
  - `execute_shell` - Shell command execution
  - `mcp_filesystem_read` (optional) - Read files
  - `mcp_filesystem_write` (optional) - Write files

#### Instruction Chain
1. Direct pass-through execution (no additional processing)

#### Use Cases
- Shell command execution
- File system operations
- Quick one-off tasks
- Stateless operations
- Batch processing

#### Example Usage
```bash
# Create terminal agent
/agent create-from-template terminal bash-exec

# Use the agent
/agent use bash-exec
> List all Python files in the current directory
```

---

## Template Selection Guide

### By Use Case

| Use Case | Recommended Template | Rationale |
|----------|---------------------|-----------|
| Research & Information Gathering | **Researcher** | Multi-hop reasoning, web search, high context |
| Code Development | **Coder** | Code execution, iterative refinement, moderate context |
| Data Analysis | **Analyst** | Statistical tools, high context for complex analysis |
| System Administration | **Terminal** | Fast execution, no context overhead |
| Document Analysis | **Analyst** | Comprehensive interpretation, high context |
| Rapid Prototyping | **Coder** | Code generation with validation |
| One-off Commands | **Terminal** | Zero overhead, instant execution |

### By Token Efficiency

| Priority | Template | Token Usage | Savings |
|----------|----------|-------------|---------|
| Maximum Efficiency | **Terminal** | 0 tokens (history disabled) | 60% |
| Balanced | **Coder** | 4000 tokens | 40% |
| Context-Rich | **Researcher/Analyst** | 8000 tokens | 20% |

### By Response Time

| Priority | Template | Model | Response Time |
|----------|----------|-------|---------------|
| Fastest | **Terminal** | gpt-3.5-turbo | Sub-second |
| Moderate | **Coder** | gpt-4 | 2-5 seconds |
| Comprehensive | **Researcher/Analyst** | gpt-4 | 5-15 seconds |

---

## Template Commands

### Creating Agents from Templates

```bash
# Basic syntax
/agent create-from-template <template_name> <agent_name>

# With custom model
/agent create-from-template <template_name> <agent_name> --model <model_string>

# With custom description
/agent create-from-template <template_name> <agent_name> --description "<description>"

# Examples
/agent create-from-template researcher my-researcher
/agent create-from-template coder python-dev --model anthropic/claude-3-sonnet-20240229
/agent create-from-template terminal bash-1 --description "Fast bash executor"
```

### Listing Available Templates

```bash
# Show all templates with details
/agent list-templates

# Output includes:
# - Template name and display name
# - Description
# - Default model
# - Category and complexity
# - Token usage
# - Available tools
# - History configuration
```

### Customizing Template-Created Agents

```bash
# Update agent model
/agent update <agent_name> --model <new_model>

# Update agent description
/agent update <agent_name> --description "<new_description>"

# Add tools
/agent update <agent_name> --add-tools web_search,mcp_filesystem_read

# Remove tools
/agent update <agent_name> --remove-tools mcp_web_browser

# Combined updates
/agent update my-researcher \
  --model anthropic/claude-3-opus-20240229 \
  --description "Specialized ML research agent" \
  --add-tools mcp_web_browser
```

---

## Template Validation

Templates include automatic tool validation to ensure referenced tools are available before agent creation.

### Standard Tools Registry

The following tools are considered standard and should be available:

- `web_search` - Web search capability
- `execute_code` - Code execution
- `execute_shell` - Shell execution
- `data_analysis` - Data analysis
- `mcp_web_browser` - MCP web browsing (requires MCP server)
- `mcp_filesystem_read` - MCP file reading (requires MCP server)
- `mcp_filesystem_write` - MCP file writing (requires MCP server)

### MCP Tool Requirements

Tools prefixed with `mcp_` require Model Context Protocol (MCP) server connections:

```bash
# Connect MCP server before using MCP tools
/mcp connect filesystem

# Verify tools are available
/tools list

# Create agent with MCP tools
/agent create-from-template researcher my-researcher
```

---

## Advanced Usage

### Multi-Agent Workflows

Combine multiple templates for sophisticated workflows:

```bash
# Create specialized agent team
/agent create-from-template researcher market-researcher
/agent create-from-template analyst data-analyst
/agent create-from-template coder python-dev
/agent create-from-template terminal bash-exec

# Use agents sequentially
/agent use market-researcher
> Research AI startup funding trends in 2024

/agent use data-analyst
> Analyze the research data and identify top investment categories

/agent use python-dev
> Create visualization script for the analysis results

/agent use bash-exec
> Run the visualization script
```

### Template Metadata

All template-created agents include metadata tracking:

```json
{
  "template": "researcher",
  "template_metadata": {
    "category": "research",
    "complexity": "high",
    "token_usage": "high"
  }
}
```

Access metadata:
```bash
# List agents to see template origin
/agent list

# Output shows template metadata for each agent
```

---

## Best Practices

### 1. Choose the Right Template

- **Research tasks**: Use `researcher` for multi-source information gathering
- **Coding tasks**: Use `coder` for iterative development
- **Analysis tasks**: Use `analyst` for data interpretation
- **Quick operations**: Use `terminal` for one-off commands

### 2. Optimize Token Usage

- Use `terminal` template for stateless operations (60% token savings)
- Use `coder` for development work requiring moderate context
- Reserve `researcher`/`analyst` for tasks needing comprehensive context

### 3. Customize After Creation

Templates provide starting points - customize as needed:

```bash
# Start with template
/agent create-from-template researcher my-researcher

# Customize for specific needs
/agent update my-researcher \
  --model anthropic/claude-3-opus-20240229 \
  --description "ML paper specialist" \
  --add-tools mcp_web_browser
```

### 4. Validate Tool Availability

Before creating agents, ensure required MCP servers are connected:

```bash
# Check connected MCP servers
/mcp list

# Connect required servers
/mcp connect filesystem
/mcp connect web_browser

# Then create template agent
/agent create-from-template researcher my-researcher
```

---

## Template Architecture

### History Configuration Strategy

Templates use optimized history configurations based on use case:

| Template | History | Max Tokens | Rationale |
|----------|---------|------------|-----------|
| **Researcher** | Enabled | 8000 | Needs full context for multi-hop reasoning |
| **Coder** | Enabled | 4000 | Moderate context for iterative development |
| **Analyst** | Enabled | 8000 | Comprehensive context for analysis |
| **Terminal** | Disabled | 0 | Stateless operations, maximum efficiency |

### AgenticStepProcessor Integration

`researcher` and `coder` templates integrate AgenticStepProcessor for autonomous multi-step workflows:

- **Researcher**: 8 internal reasoning steps
- **Coder**: 5 iterative development steps

This enables:
- Complex goal decomposition
- Autonomous tool usage
- Self-correction and iteration
- Objective completion detection

---

## Troubleshooting

### Issue: "Template 'X' not found"

**Cause**: Invalid template name

**Solution**:
```bash
# List available templates
/agent list-templates

# Use exact template name
/agent create-from-template researcher my-researcher
```

### Issue: "Tool 'X' requires MCP server connection"

**Cause**: MCP tools not available

**Solution**:
```bash
# Connect required MCP server
/mcp connect filesystem

# Verify tools available
/tools list

# Create agent
/agent create-from-template researcher my-researcher
```

### Issue: High token usage

**Cause**: Using context-heavy templates for simple tasks

**Solution**:
```bash
# For quick operations, use terminal template
/agent create-from-template terminal quick-exec

# For operations needing some context, use coder
/agent create-from-template coder moderate-agent
```

---

## API Reference

### Python API

Templates can be used programmatically:

```python
from promptchain.cli.utils.agent_templates import (
    create_from_template,
    list_templates,
    validate_template_tools,
    AGENT_TEMPLATES
)

# Create agent from template
agent = create_from_template(
    template_name="researcher",
    agent_name="my-researcher",
    model_override="anthropic/claude-3-opus-20240229"
)

# List all templates
templates = list_templates()

# Validate template tools
validation = validate_template_tools("researcher")
if not validation["valid"]:
    print(f"Missing tools: {validation['missing_tools']}")
```

---

## Version History

### Phase 8 (Current)
- Initial release of 4 core templates
- Template validation system
- Command integration (`/agent create-from-template`, `/agent list-templates`)
- Comprehensive documentation

---

**Last Updated**: November 23, 2025
**Phase**: 8 (002-cli-orchestration)
**Status**: Complete
