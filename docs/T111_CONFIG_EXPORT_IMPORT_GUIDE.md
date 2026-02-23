# T111: Configuration Export/Import Guide

## Overview

The `/config export` and `/config import` commands enable sharing and backing up PromptChain CLI configurations, including agents, MCP servers, orchestration settings, and history configurations.

## Commands

### `/config export <filename>`

Export current CLI configuration to YAML or JSON format.

**Usage:**
```bash
/config export my-config.yml     # Export as YAML
/config export my-config.yaml    # Export as YAML (alternative extension)
/config export my-config.json    # Export as JSON
```

**What Gets Exported:**

1. **Session Metadata**
   - Session name
   - Working directory
   - Active agent
   - Default model
   - Auto-save settings

2. **Agent Configurations**
   - All agents (model, description, tools)
   - Per-agent history configurations
   - Template references (if available)

3. **MCP Server Configurations**
   - Server ID, type, command/URL
   - Connection arguments
   - Auto-connect settings

4. **Orchestration Settings**
   - Execution mode (router, pipeline, round-robin)
   - Auto-include history setting
   - Router configuration (model, timeout)

5. **History Configuration**
   - Global history manager settings
   - Per-agent history overrides

**Example Output (YAML):**
```yaml
version: "1.0"
session:
  name: my-session
  working_directory: /path/to/project
  active_agent: researcher
  default_model: openai/gpt-4

agents:
  researcher:
    model: openai/gpt-4
    description: Research specialist
    tools: [web_search, mcp_filesystem_read]
    history_config:
      enabled: true
      max_tokens: 8000
      max_entries: 50

mcp_servers:
  - id: filesystem
    type: stdio
    command: npx
    args: [-y, "@modelcontextprotocol/server-filesystem", /path]
    auto_connect: true

orchestration:
  execution_mode: router
  auto_include_history: true
  router_config:
    model: openai/gpt-4o-mini
    timeout_seconds: 10
```

**Security Features:**

- **Path Traversal Protection**: Attempts to use `../` or absolute paths are blocked
- **No Sensitive Data**: API keys and passwords are never exported
- **Safe Serialization**: Uses `yaml.safe_dump()` and `json.dump()` for security
- **Extension Validation**: Only `.yml`, `.yaml`, and `.json` extensions allowed

---

### `/config import <filename>`

Import CLI configuration from YAML or JSON file.

**Usage:**
```bash
/config import my-config.yml     # Import from YAML
/config import my-config.json    # Import from JSON
```

**Import Behavior:**

1. **Non-Destructive**: Existing agents and servers are NOT overwritten
2. **Additive**: New agents and servers are added to current session
3. **Updates**: Orchestration settings are updated if present
4. **Validation**: All imported data is validated before applying

**What Gets Imported:**

- ✅ **Agents**: Creates new agents (skips if name exists)
- ✅ **MCP Servers**: Adds servers (skips if ID exists)
- ✅ **Orchestration Settings**: Updates execution mode and router config
- ✅ **History Settings**: Applies per-agent history configurations

**Example:**
```bash
> /config import production-config.yml

Configuration imported from production-config.yml

Changes:
  • Imported agent: researcher (openai/gpt-4)
  • Imported agent: coder (openai/gpt-4)
  • Imported MCP server: filesystem (stdio)
  • Updated execution mode: router
  • Skipped existing agent: default
```

**Error Handling:**

- **File Not Found**: Clear error message if file doesn't exist
- **Invalid Extension**: Only `.yml`, `.yaml`, `.json` accepted
- **Invalid Format**: YAML/JSON parse errors are caught and reported
- **Missing Version**: Configuration must include `version` field
- **Schema Validation**: All imported data is validated before applying

---

## Use Cases

### 1. Share Agent Configuration with Team

```bash
# Developer A exports their optimized agent setup
/config export team-agents.yml

# Developer B imports the same setup
/config import team-agents.yml
```

### 2. Backup Before Experiment

```bash
# Save current configuration
/config export backup-2024-11-23.yml

# Make experimental changes...

# Restore if needed
/config import backup-2024-11-23.yml
```

### 3. Environment-Specific Configurations

```bash
# Development environment
/config import dev-config.yml

# Production environment
/config import prod-config.yml
```

### 4. Template Sharing

```bash
# Export optimized multi-agent workflow
/config export data-analysis-workflow.yml

# Share with colleagues who work on similar projects
```

---

## Export/Import Roundtrip

**Complete Workflow:**

```bash
# 1. Create session with custom configuration
/agent create researcher --model openai/gpt-4 --template researcher
/mcp add filesystem --command "npx" --args "-y,@modelcontextprotocol/server-filesystem,/path"

# 2. Export configuration
/config export my-workflow.yml

# 3. Create new session
/session new other-project

# 4. Import configuration
/config import my-workflow.yml

# Result: Same agents and MCP servers in new session
```

---

## File Format Specifications

### YAML Format (Recommended)

**Advantages:**
- Human-readable
- Comments supported
- Easier to edit manually
- Industry standard for configuration

**Example:**
```yaml
version: "1.0"
agents:
  researcher:
    model: openai/gpt-4
    description: Research specialist
    tools:
      - web_search
      - mcp_filesystem_read
```

### JSON Format

**Advantages:**
- Universal compatibility
- Programmatic processing
- Strict schema validation

**Example:**
```json
{
  "version": "1.0",
  "agents": {
    "researcher": {
      "model": "openai/gpt-4",
      "description": "Research specialist",
      "tools": ["web_search", "mcp_filesystem_read"]
    }
  }
}
```

---

## Advanced Features

### Nested Directory Support

Export creates parent directories automatically:

```bash
/config export configs/production/main.yml
# Creates configs/production/ directory if needed
```

### Selective Import

Import only what you need by editing the exported file:

```yaml
# Remove sections you don't want to import
version: "1.0"
agents:
  researcher:  # Only import this agent
    model: openai/gpt-4
# mcp_servers: []  # Commented out - won't import
```

### Version Compatibility

All exported configurations include a `version` field for future compatibility:

```yaml
version: "1.0"  # Current version
```

Future versions will maintain backward compatibility with version-specific parsers.

---

## Security Considerations

### What IS Exported

- ✅ Agent models and descriptions
- ✅ Tool names (e.g., "web_search")
- ✅ MCP server commands and arguments
- ✅ History configuration settings
- ✅ Orchestration settings

### What IS NOT Exported

- ❌ API keys
- ❌ Passwords
- ❌ Authentication tokens
- ❌ Conversation history
- ❌ User data

### Best Practices

1. **Review Before Sharing**: Check exported files for sensitive data
2. **Use Version Control**: Store configs in private repositories
3. **Validate Imports**: Review changes list after import
4. **Regular Backups**: Export before major configuration changes

---

## Troubleshooting

### Common Errors

**1. "Invalid file extension"**
```bash
# ❌ Wrong
/config export config.txt

# ✅ Correct
/config export config.yml
```

**2. "Path traversal detected"**
```bash
# ❌ Security violation
/config export ../../../etc/config.yml

# ✅ Correct
/config export my-config.yml
```

**3. "Failed to parse YAML"**
- Check YAML syntax (indentation, colons, hyphens)
- Validate with online YAML validator
- Use JSON format for stricter validation

**4. "Missing version field"**
- All configs must have `version: "1.0"` at top level
- Manually add if editing exported file

### Debug Tips

1. **Validate Export**: Open exported file to verify contents
2. **Test Import**: Try importing in a test session first
3. **Check Logs**: Review session history for detailed error messages
4. **Start Simple**: Export minimal config to test, then expand

---

## API Reference

### CommandResult Structure

Both commands return `CommandResult` objects:

```python
CommandResult(
    success=True,
    message="Configuration exported to /path/to/config.yml\nFormat: YAML\nAgents: 3\nMCP Servers: 2",
    data={
        "file_path": "/absolute/path/to/config.yml",
        "format": "yaml",
        "agent_count": 3,
        "mcp_server_count": 2
    },
    error=None
)
```

### Export Data Schema

```typescript
{
  version: string,                    // Config format version
  session: {
    name: string,
    working_directory: string,
    active_agent: string,
    default_model: string,
    auto_save_enabled: boolean
  },
  agents: {
    [agent_name]: {
      model: string,
      description: string,
      tools: string[],
      history_config?: {
        enabled: boolean,
        max_tokens: number,
        max_entries: number,
        truncation_strategy: string
      }
    }
  },
  mcp_servers: Array<{
    id: string,
    type: "stdio" | "http",
    command?: string,
    args?: string[],
    url?: string,
    auto_connect: boolean
  }>,
  orchestration: {
    execution_mode: "router" | "pipeline" | "round-robin" | "broadcast",
    auto_include_history: boolean,
    router_config?: {
      model: string,
      timeout_seconds: number
    }
  },
  history_configuration: {
    global?: {
      max_tokens: number,
      max_entries: number,
      truncation_strategy: string
    },
    per_agent?: {
      [agent_name]: {
        enabled: boolean,
        max_tokens: number,
        max_entries: number,
        truncation_strategy: string
      }
    }
  }
}
```

---

## Examples

### Complete Example: Multi-Agent Data Pipeline

**1. Setup Session:**
```bash
/session new data-pipeline
/agent create researcher --model openai/gpt-4 --template researcher
/agent create analyst --model anthropic/claude-3-sonnet-20240229 --template analyst
/agent create coder --model openai/gpt-4 --template coder
/mcp add snowflake --command mcp-server-snowflake
/mcp add tableau --command mcp-server-tableau
```

**2. Export Configuration:**
```bash
/config export data-pipeline.yml
```

**3. Share with Team:**
```bash
# Team member creates new session
/session new my-data-work

# Import shared configuration
/config import data-pipeline.yml

# Result: Same agents and MCP servers available
```

**4. Exported File (data-pipeline.yml):**
```yaml
version: '1.0'
session:
  name: data-pipeline
  working_directory: /home/user/projects/data
  active_agent: researcher
  default_model: openai/gpt-4.1-mini-2025-04-14

agents:
  researcher:
    model: openai/gpt-4
    description: Research specialist
    tools: [web_search, mcp_snowflake_read_query]
    history_config:
      enabled: true
      max_tokens: 8000

  analyst:
    model: anthropic/claude-3-sonnet-20240229
    description: Data analysis specialist
    tools: [mcp_snowflake_read_query, mcp_tableau_view_get_data]
    history_config:
      enabled: true
      max_tokens: 8000

  coder:
    model: openai/gpt-4
    description: Code generation specialist
    tools: [mcp_filesystem_read, mcp_filesystem_write]
    history_config:
      enabled: true
      max_tokens: 6000

mcp_servers:
  - id: snowflake
    type: stdio
    command: mcp-server-snowflake
    auto_connect: false

  - id: tableau
    type: stdio
    command: mcp-server-tableau
    auto_connect: false

orchestration:
  execution_mode: router
  auto_include_history: true
```

---

## See Also

- **T110**: `/config show` - Display current configuration
- **Phase 9**: CLI Configuration Management
- **Agent Templates**: Pre-configured agent profiles
- **MCP Integration**: External tool server management
