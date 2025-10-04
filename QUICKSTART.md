# Quick Start Guide

Get up and running with the Agentic Chat Team in 5 minutes.

## Step 1: Install PromptChain

```bash
cd /home/gyasis/Documents/code/PromptChain
pip install -e .
pip install -r agentic_chat/requirements.txt
```

## Step 2: Setup API Keys

Create `.env` file in the PromptChain root directory:

```bash
# Copy the example
cp agentic_chat/.env.example .env

# Edit with your API keys
nano .env  # or use your favorite editor
```

Add your keys:
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
```

## Step 3: Verify Gemini MCP Server

Make sure the Gemini MCP server is accessible:

```bash
ls /home/gyasis/Documents/code/claude_code-gemini-mcp/server.py
# Should exist

ls /home/gyasis/Documents/code/claude_code-gemini-mcp/.venv/bin/python
# Should exist
```

## Step 4: Run the System

```bash
cd /home/gyasis/Documents/code/PromptChain
python agentic_chat/agentic_team_chat.py
```

You should see:
```
================================================================================
🤖 AGENTIC CHAT TEAM - 5-Agent Collaborative System
================================================================================

Initializing agents and systems...

✅ Logging to: ./agentic_team_logs
✅ History manager: 8000 max tokens
📝 Session log: ./agentic_team_logs/session_agentic_team_session_20250103_120000.jsonl

🔧 Creating specialized agents with agentic reasoning capabilities...

✅ System initialized successfully!

================================================================================
TEAM MEMBERS:
================================================================================

🔍 RESEARCH - Gemini-powered web research specialist
   Exclusive Gemini MCP access for web search and fact verification

📊 ANALYSIS - Data analysis and insight extraction expert
   Critical thinking and pattern recognition specialist

💻 TERMINAL - System operations and command execution expert
   Safe terminal access with TerminalTool integration

📝 DOCUMENTATION - Technical writing and explanation specialist
   Clear documentation and tutorial creation expert

🎯 SYNTHESIS - Strategic planning and insight integration expert
   Combines multiple perspectives into actionable recommendations

================================================================================
AVAILABLE COMMANDS:
================================================================================
  - Type your question or task to get started
  - Type 'history' to see conversation history
  - Type 'stats' to see session statistics
  - Type 'save' to save session summary
  - Type 'exit' or 'quit' to end session
================================================================================

💬 You:
```

## Step 5: Try It Out

### Example 1: Web Research
```
💬 You: What are the latest AI trends in 2025?

⏳ Processing...

🤖 [Research Agent provides comprehensive answer with sources]
```

### Example 2: Terminal Commands
```
💬 You: List all Python files in the current directory

⏳ Processing...

🤖 [Terminal Agent executes ls command and explains results]
```

### Example 3: Check Statistics
```
💬 You: stats

📊 SESSION STATISTICS:
================================================================================
Session Duration: 0:05:23
Total Queries: 2
History Truncations: 0

History Status:
  Current Size: 1234 / 8000 tokens (15.4% full)
  Total Entries: 4

Agent Usage:
  - research: 1 queries
  - terminal: 1 queries
  - analysis: 0 queries
  - documentation: 0 queries
  - synthesis: 0 queries

Logs Directory: ./agentic_team_logs
================================================================================
```

## Common Usage Patterns

### Clean Mode (Default)
Terminal shows only Q&A, logs everything to file:
```bash
python agentic_chat/agentic_team_chat.py
```

### Ultra-Quiet Mode
Minimal terminal output:
```bash
python agentic_chat/agentic_team_chat.py --quiet
```

### No Logging (Testing)
No files created:
```bash
python agentic_chat/agentic_team_chat.py --no-logging
```

### Named Session
Organized logging:
```bash
python agentic_chat/agentic_team_chat.py --session-name quantum_research
```

## Troubleshooting

### "Module not found" errors
```bash
# Make sure PromptChain is installed
cd /home/gyasis/Documents/code/PromptChain
pip install -e .
```

### "API key not found" errors
```bash
# Check .env file exists and has keys
cat .env | grep API_KEY
```

### Gemini MCP connection issues
```bash
# Verify Gemini MCP server path
python -c "import sys; sys.path.insert(0, '/home/gyasis/Documents/code/claude_code-gemini-mcp'); import server; print('OK')"
```

### Permission denied for logs
```bash
# Create logs directory manually
mkdir -p agentic_team_logs
chmod 755 agentic_team_logs
```

## Next Steps

- Read [README.md](./README.md) for full documentation
- Check [USAGE.md](./USAGE.md) for detailed command-line options
- Review [example_usage.py](./example_usage.py) for programmatic usage
- Explore session logs in `./agentic_team_logs/`

## Quick Reference

**Commands:**
- `history` - View conversation history
- `history-summary` - View history stats
- `stats` - Session statistics
- `save` - Save session summary
- `exit` / `quit` - End session

**Flags:**
- `--quiet` - Suppress verbose output
- `--no-logging` - Disable file logging
- `--max-history-tokens N` - Set history limit
- `--session-name NAME` - Custom session name
- `--help` - Show all options

**Log Files:**
- `session_<name>.jsonl` - Growing session log (JSONL)
- `final_session_<name>.txt` - Session summary (text)
- SQLite cache for persistent conversations

## Support

For issues or questions:
1. Check the documentation in this directory
2. Review session logs for errors
3. Ensure all dependencies are installed
4. Verify API keys are correct

Happy chatting! 🚀
