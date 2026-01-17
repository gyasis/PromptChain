# Agentic Chat Team - Usage Guide

## Command-Line Options

The Agentic Chat Team supports various command-line flags to control logging and output behavior.

### Basic Usage

```bash
# Run with default settings (clean terminal, logs to file)
python agentic_team_chat.py

# Suppress all console output (ultra-quiet mode)
python agentic_team_chat.py --quiet

# Disable file logging completely
python agentic_team_chat.py --no-logging

# Both quiet and no logging (minimal mode)
python agentic_team_chat.py --quiet --no-logging
```

### Advanced Options

```bash
# Set custom log level
python agentic_team_chat.py --log-level ERROR    # Only show errors
python agentic_team_chat.py --log-level DEBUG    # Show debug info

# Disable automatic history truncation
python agentic_team_chat.py --no-history-management

# Set custom max history tokens
python agentic_team_chat.py --max-history-tokens 16000

# Set custom session name
python agentic_team_chat.py --session-name my_research_session
```

### Combined Examples

```bash
# Production mode: minimal terminal output, full file logging
python agentic_team_chat.py --quiet

# Development mode: verbose terminal, large history
python agentic_team_chat.py --log-level DEBUG --max-history-tokens 16000

# Demo mode: clean terminal, no logging
python agentic_team_chat.py --no-logging
```

## Output Behavior

### Default Mode (Clean Terminal)

**Terminal shows only:**
- User input (what you type)
- Agent responses (final answers)
- Error messages (if any occur)
- Command prompts (history, stats, etc.)

**Session log file captures:**
- All user queries with timestamps
- All agent responses (full text)
- Internal reasoning steps
- Tool calls and results
- History management events
- Routing decisions
- Error details
- Session statistics

### Quiet Mode (`--quiet`)

**Terminal shows only:**
- Minimal startup message
- User input prompt
- Agent responses
- Errors

**All verbose output suppressed:**
- No initialization messages
- No agent creation messages
- No processing indicators

### No Logging Mode (`--no-logging`)

**Effect:**
- No file logging at all
- Session log file not created
- Final summary not saved
- All information lost after session ends

**Use when:**
- Testing without saving history
- Privacy-sensitive conversations
- Temporary quick queries

## Log Files

### Session Log (Growing Dynamic Log)

**Location:** `./agentic_team_logs/session_<name>.jsonl`

**Format:** JSONL (JSON Lines) - one JSON object per line

**Contents:**
```jsonl
{"event": "system_initialized", "timestamp": "2025-01-03T10:00:00", "agents": [...], ...}
{"event": "user_query", "timestamp": "2025-01-03T10:01:00", "query": "What is AI?", ...}
{"event": "agent_response", "timestamp": "2025-01-03T10:01:15", "response": "...", ...}
{"event": "history_truncated", "timestamp": "2025-01-03T10:15:00", "old_size": 8500, ...}
{"event": "session_ended", "timestamp": "2025-01-03T10:30:00", "total_queries": 25, ...}
```

**Use for:**
- Detailed session analysis
- Debugging agent behavior
- Performance monitoring
- Audit trails

### Final Summary

**Location:** `./agentic_team_logs/final_session_<name>.txt`

**Format:** Human-readable text

**Contents:**
- Session metadata (name, duration, query count)
- Full conversation history
- All interactions formatted for review

**Use for:**
- Session review
- Conversation export
- Documentation
- Knowledge base creation

## History Management

### Automatic Truncation

The system automatically manages conversation history to prevent token overflow:

**Threshold:** 90% of max tokens (default: 7200/8000)

**When triggered:**
```
⚠️ History approaching limit (7200/8000 tokens)
   Automatically truncating oldest entries...
   ✅ History truncated: 7200 → 4500 tokens
```

**Strategy:** Oldest-first (keeps recent context)

**Logged events:**
- Truncation timestamp
- Before/after token counts
- Total truncations in session

### Disable Auto-Truncation

```bash
python agentic_team_chat.py --no-history-management
```

**Warning:** Without truncation, very long conversations may exceed token limits and cause errors.

### Custom History Limits

```bash
# Double the default limit
python agentic_team_chat.py --max-history-tokens 16000

# Minimal history (for simple Q&A)
python agentic_team_chat.py --max-history-tokens 2000
```

## Interactive Commands

While in a session, use these commands:

```
history          - View conversation history + summary
history-summary  - View history statistics only
stats            - View session statistics
save             - Save current session summary
exit/quit        - End session
```

## Session Names

Custom session names help organize logs:

```bash
# Default: auto-generated timestamp
python agentic_team_chat.py
# Creates: session_agentic_team_session_20250103_100000.jsonl

# Custom name for organization
python agentic_team_chat.py --session-name quantum_research
# Creates: session_quantum_research.jsonl
```

**Benefits:**
- Easy log file identification
- Organized by topic/project
- Reusable naming scheme
- Better log management

## Performance Tips

### For Fast Interactions
```bash
python agentic_team_chat.py --quiet --max-history-tokens 4000
```

### For Long Conversations
```bash
python agentic_team_chat.py --max-history-tokens 16000
```

### For Research Sessions
```bash
python agentic_team_chat.py --session-name "project_research_$(date +%Y%m%d)"
```

### For Demos/Testing
```bash
python agentic_team_chat.py --no-logging
```

## Log Analysis

### Parse Session Logs with jq

```bash
# Count queries
jq 'select(.event == "user_query")' logs/session_*.jsonl | wc -l

# Find errors
jq 'select(.event == "error")' logs/session_*.jsonl

# Get all agent responses
jq 'select(.event == "agent_response") | .response' logs/session_*.jsonl

# Calculate average response length
jq 'select(.event == "agent_response") | .response_length' logs/session_*.jsonl | \
  jq -s 'add/length'
```

### Monitor Real-Time

```bash
# Watch session log grow
tail -f agentic_team_logs/session_*.jsonl

# Pretty print in real-time
tail -f agentic_team_logs/session_*.jsonl | jq
```

## Troubleshooting

### Logs Not Being Created

**Check:**
1. Directory permissions: `ls -la agentic_team_logs/`
2. Disk space: `df -h`
3. Not using `--no-logging` flag

### History Growing Too Fast

**Solutions:**
```bash
# Reduce max tokens
python agentic_team_chat.py --max-history-tokens 4000

# Or disable history tracking (not recommended)
python agentic_team_chat.py --no-history-management
```

### Terminal Too Verbose

```bash
# Use quiet mode
python agentic_team_chat.py --quiet
```

### Want More Detail in Terminal

The system is designed for clean terminal output by default. All details are in log files.

To see more during development, modify the `verbose` parameter in agent creation (requires code changes).

## Best Practices

1. **Use descriptive session names** for easy log identification
2. **Keep default history limits** unless you have specific needs
3. **Review session logs** after important conversations
4. **Set appropriate log levels** for your use case
5. **Clean up old logs** periodically to save space
6. **Use quiet mode** for production/clean demos
7. **Monitor history usage** in long sessions
8. **Export important sessions** to external storage

## Help

```bash
python agentic_team_chat.py --help
```

Shows all available options with descriptions.
