# Quickstart Guide: PromptChain CLI

**Last Updated**: 2025-11-16
**Target Audience**: Developers and power users who want to use PromptChain's interactive CLI
**Prerequisites**: Python 3.8+, API keys for LLM providers (OpenAI, Anthropic, etc.)

---

## Installation

### 1. Install PromptChain with CLI Support

```bash
# Install from PyPI (once published)
pip install promptchain[cli]

# Or install from source
git clone https://github.com/promptchain/promptchain.git
cd promptchain
pip install -e ".[cli]"
```

### 2. Configure API Keys

Create a `.env` file in your home directory or project root:

```bash
# ~/.env or ./.env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
# Optional: Other LLM providers supported by LiteLLM
```

Alternatively, export environment variables:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Verify Installation

```bash
promptchain --version
# Output: PromptChain CLI v0.5.0

promptchain --help
# Output: Usage instructions...
```

---

## Quick Start: First Session

### Launch Interactive Session

```bash
# Start a new session
promptchain

# Welcome message:
# PromptChain CLI v0.5.0
# Type /help for commands or start chatting
# Session: default (auto-created)
# >
```

### Basic Conversation

```
> Hello! Can you help me understand this project?
Assistant: Of course! I'd be happy to help you understand the project.
To give you the best overview, could you tell me what aspect you're most
interested in? I can help with architecture, setup, usage, or specific features.

> What are the main components?
Assistant: The main components are...
```

### File References

Use `@syntax` to include files in your conversation:

```
> Analyze @README.md and summarize the project

# PromptChain automatically reads the file and includes it in the context
Assistant: Based on README.md, this project is...

> Review @src/main.py and suggest improvements
Assistant: Looking at src/main.py, I notice several areas for improvement...
```

### Exit Session

```
> /exit
# Auto-save triggered...
# Session 'default' saved
# Goodbye!
```

---

## Working with Agents

### Create Specialized Agents

Agents let you use different models for different tasks:

```
> /agent create coding --model openai/gpt-4 --description "Python coding specialist"
Agent 'coding' created with model openai/gpt-4

> /agent create fast --model ollama/llama2 --description "Fast local model for quick questions"
Agent 'fast' created with model ollama/llama2

> /agent create research --model anthropic/claude-3-opus-20240229
Agent 'research' created with model anthropic/claude-3-opus-20240229
```

**Model Flexibility**: You can use ANY model supported by LiteLLM:
- OpenAI: `openai/gpt-4`, `openai/gpt-4-turbo`, `openai/gpt-3.5-turbo`
- Anthropic: `anthropic/claude-3-opus-20240229`, `anthropic/claude-3-sonnet-20240229`
- Local: `ollama/llama2`, `ollama/codellama`, `ollama/mistral`
- Other: Any LiteLLM-compatible provider

### Switch Between Agents

```
> /agent list
Available agents:
- default (openai/gpt-4) - Default agent [ACTIVE]
- coding (openai/gpt-4) - Python coding specialist (used 0 times)
- fast (ollama/llama2) - Fast local model (used 0 times)
- research (anthropic/claude-3-opus-20240229) (used 0 times)

> /agent use coding
Switched to agent 'coding'

> Help me refactor @src/utils.py to use async/await
Assistant (coding agent): Looking at utils.py, I'll refactor it to use async/await...

> /agent use fast
Switched to agent 'fast'

> Quick question: what's the difference between lists and tuples in Python?
Assistant (fast local model): Lists are mutable, tuples are immutable...
```

### Delete Agents

```
> /agent delete old-agent
Agent 'old-agent' deleted
```

---

## Session Management

### Save Sessions

Sessions auto-save every 5 messages or 2 minutes, but you can manually save:

```
> /session save
Session 'default' saved

# Or rename while saving
> /session save my-project
Session renamed to 'my-project' and saved
```

### Resume Previous Sessions

```bash
# List saved sessions
promptchain

> /session list
Saved sessions:
- my-project (last accessed: 2 hours ago, 3 agents)
- research (last accessed: 3 days ago, 1 agent)
- experiment (last accessed: 1 week ago, 2 agents)

> /exit

# Resume specific session
promptchain --session my-project

# Or use /session resume command (if session list shown)
> /session resume research
# (Note: Implementation may use CLI arg instead)
```

### Delete Sessions

```
> /session delete old-experiment
Session 'old-experiment' deleted
```

---

## Advanced Features

### Multi-line Input

Press `Shift+Enter` to add newlines without submitting:

```
> Here's my code:
[Shift+Enter]
def hello():
[Shift+Enter]
    print("Hello")
[Enter to submit]

### Shell Command Execution

Execute shell commands with `!` prefix:

```
> !git status
On branch main
Your branch is up to date with 'origin/main'.

> !ls -la src/
total 24
drwxr-xr-x 2 user user 4096 Nov 16 10:00 .
drwxr-xr-x 8 user user 4096 Nov 16 09:30 ..
-rw-r--r-- 1 user user 1234 Nov 16 09:45 main.py
```

### Directory References

Reference entire directories to discover relevant files:

```
> Explain the structure of @src/
# PromptChain discovers relevant .py files in src/ directory
Assistant: The src/ directory contains:
- main.py: Entry point
- utils.py: Utility functions
- config.py: Configuration management
```

### Session Status

Check current session information:

```
> /status
Session: my-project
Working Directory: /home/user/projects/my-project
Active Agent: coding (openai/gpt-4)
Total Agents: 3
Messages: 47
Auto-save: Enabled (every 2 minutes)
```

---

## Common Workflows

### Code Review Workflow

```bash
promptchain --session code-review


> /agent create reviewer --model anthropic/claude-3-opus-20240229
> /agent use reviewer
> Review @src/main.py for code quality, performance, and best practices
> Review @tests/test_main.py and suggest additional test cases
> /session save
> /exit
```

### Multi-Agent Research Workflow

```bash
promptchain --session research-project

> /agent create researcher --model anthropic/claude-3-opus-20240229 --description "Deep research"
> /agent create summarizer --model openai/gpt-4 --description "Concise summaries"

> /agent use researcher
> Research async/await patterns in Python @docs/async.md

> /agent use summarizer
> Summarize the key findings from the previous conversation
```

### Quick Questions with Local Model

```bash
promptchain

> /agent create quick --model ollama/llama2
> /agent use quick
> What's the difference between __str__ and __repr__?
> How do I reverse a string in Python?
```

---

## Configuration

### Default Configuration File

PromptChain CLI reads configuration from `~/.promptchain/config.json`:

```json
{
  "default_model": "openai/gpt-4",
  "auto_save_enabled": true,
  "auto_save_interval": 120,
  "max_file_size": 102400,
  "sessions_dir": "~/.promptchain/sessions",
  "mcp_servers": [
    {
      "id": "filesystem",
      "type": "stdio",
      "command": "mcp-server-filesystem",
      "args": ["--root", "."]
    }
  ]
}
```

### Command-Line Options

```bash
promptchain --help

Usage: promptchain [OPTIONS]

Options:
  --session TEXT       Resume existing session by name
  --model TEXT         Override default model for this session
  --verbose           Enable verbose logging
  --no-autosave       Disable auto-save for this session
  --version           Show version and exit
  --help              Show this message and exit
```

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Enter` | Submit message |
| `Shift+Enter` | New line (multi-line input) |
| `Up/Down Arrow` | Navigate command history |
| `Ctrl+C` | Cancel current operation |
| `Ctrl+D` | Exit session (same as /exit) |
| `Tab` | Autocomplete slash commands |

---

## Slash Commands Reference

| Command | Description |
|---------|-------------|
| `/agent create <name> --model <model>` | Create new agent |
| `/agent list` | List all agents |
| `/agent use <name>` | Switch to agent |
| `/agent delete <name>` | Delete agent |
| `/session save [name]` | Save session (optional rename) |
| `/session list` | List saved sessions |
| `/session delete <name>` | Delete session |
| `/status` | Show session status |
| `/help [topic]` | Show help documentation |
| `/exit` | Exit session |

---

## Troubleshooting

### API Key Errors

```
Error: OpenAI API key not found
```

**Solution**: Set environment variable or create .env file:
```bash
export OPENAI_API_KEY=sk-...
# or
echo "OPENAI_API_KEY=sk-..." >> ~/.env
```

### Model Not Available

```
Error: Model 'ollama/llama2' not available
```

**Solution**: Ensure Ollama is running:
```bash
ollama serve
ollama pull llama2
```

### Session Not Found

```
Error: Session 'my-project' not found
```

**Solution**: List available sessions:
```bash
promptchain
> /session list
```

### File Reference Failed

```
Warning: Failed to load @nonexistent.txt: File not found
```

**Solution**: Check file path (relative to working directory):
```bash
> /status  # Check working directory
> !ls src/  # Verify file exists
```

---

## Next Steps

- **Library Usage**: Learn how to use PromptChain programmatically in [CLAUDE.md](../../../CLAUDE.md)
- **Advanced Agents**: Explore AgentChain and multi-agent orchestration
- **MCP Integration**: Configure external MCP servers for tool access
- **Extensions**: Explore optional plugins (TOON, APReL, APRICOT) once available

---

## Getting Help

- **In-CLI Help**: Type `/help` for commands or `/help <topic>` for specific topics
- **Documentation**: Visit https://promptchain.readthedocs.io
- **Issues**: Report bugs at https://github.com/promptchain/promptchain/issues
- **Community**: Join discussions at https://github.com/promptchain/promptchain/discussions

---

**Version**: 1.0.0 | **Last Updated**: 2025-11-16

