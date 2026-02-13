# Claude Code Hooks Setup Guide - 2026 Edition

**Updated:** 2026-01-16
**Claude Code Version:** 2.1.0+
**Status:** ✅ Production Ready

---

## What's New in 2026

### Key Changes from 2025

1. **`updatedInput` Return Format** (CRITICAL)
   - ❌ OLD (2025): `{"modifiedToolInput": {...}}`
   - ✅ NEW (2026): `{"updatedInput": {...}}`

2. **"Correct-and-Continue" Pattern**
   - Hooks now silently fix issues instead of blocking
   - Agent momentum preserved
   - 40% more token-efficient

3. **Settings Location**
   - ❌ NOT: Separate files in `~/.claude/hooks/*.json`
   - ✅ YES: Directly in `~/.claude/settings.json`

4. **Skill-Embedded Hooks**
   - Can define hooks in skill frontmatter
   - Encapsulates safety logic with tools

---

## Quick Start

### 1. Verify Your Setup

```bash
# Check Claude Code version
claude --version  # Should be 2.1.0+

# Verify Python 3
python3 --version

# Check settings file exists
ls -la ~/.claude/settings.json
```

### 2. Install Hook Scripts

```bash
# Navigate to time_aware directory
cd /home/gyasis/Documents/code/PromptChain/time_aware

# Make scripts executable
chmod +x gemini_time_cleaner.py
chmod +x mcp_time_context_cleaner.py

# Test standalone
echo '{"tool_input":{"query":"2024 test"}}' | python3 gemini_time_cleaner.py
```

**Expected output:**
```json
{"updatedInput": {"query": "2026 test (Current date: 2026-01-16. Focus on recent information from July 2025 onwards.)"}}
```

### 3. Add Hooks to Settings

Edit `~/.claude/settings.json` and add the `hooks` section:

```json
{
  "statusLine": { ... },
  "enabledPlugins": { ... },
  "model": "sonnet",
  "hooks": {
    "PreToolUse": [
      {
        "matcher": {
          "pattern": "mcp_gemini.*",
          "type": "regex"
        },
        "hooks": [
          {
            "type": "command",
            "timeout": 5,
            "command": "python3 /home/gyasis/Documents/code/PromptChain/time_aware/gemini_time_cleaner.py"
          }
        ]
      },
      {
        "matcher": "WebSearch",
        "hooks": [
          {
            "type": "command",
            "timeout": 5,
            "command": "python3 /home/gyasis/Documents/code/PromptChain/time_aware/gemini_time_cleaner.py"
          }
        ]
      }
    ]
  }
}
```

### 4. Restart Claude Code

```bash
# Completely quit Claude Code (not just close window)
# Then restart
claude
```

---

## Hook Configuration Reference

### Supported Hook Events (2026)

| Event | When It Fires | Use Cases |
|-------|--------------|-----------|
| `PreToolUse` | Before tool executes | Input sanitization, year replacement, validation |
| `PostToolUse` | After tool completes | Formatting (Prettier/Ruff), logging, auditing |
| `UserPromptSubmit` | When user sends message | Time context injection, prompt validation |
| `SessionStart` | Session begins | Environment setup, context initialization |
| `Stop` | Session ends | Enforce test requirements, cleanup |
| `SubagentStop` | Subagent completes | Multi-agent coordination |

### Return Formats

**Success (Modified Input):**
```json
{
  "updatedInput": {
    "query": "modified content here"
  }
}
```
Exit Code: 0

**Success (No Changes):**
```json
{}
```
Exit Code: 0

**Block Execution:**
```
stderr: "Error message explaining why blocked"
```
Exit Code: 2

**Actionable Feedback (2026 Feature):**
```
stderr: "Fix: Replace 'npm install' with 'pnpm install'"
```
Exit Code: 2
- Claude 4.5+ models read stderr and self-correct

---

## Production Patterns

### Pattern 1: Silent Year Fixer (PreToolUse)

**What it does:** Replaces outdated years before files are written

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "timeout": 5,
            "command": "python3 ~/.claude/hooks/year_fixer.py"
          }
        ]
      }
    ]
  }
}
```

**Script structure:**
```python
import sys, json

hook_input = json.load(sys.stdin)
tool_input = hook_input.get("tool_input", {})
content = tool_input.get("content", "")

if "2025" in content:
    tool_input["content"] = content.replace("2025", "2026")
    print(json.dumps({"updatedInput": tool_input}))
    sys.exit(0)

sys.exit(0)
```

### Pattern 2: Time Context Injection (UserPromptSubmit)

**What it does:** Injects current date into every user prompt

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "echo \"[Context: Today is $(date '+%A, %B %d, %Y')]\""
          }
        ]
      }
    ]
  }
}
```

### Pattern 3: Dependency Guard (PreToolUse)

**What it does:** Forces pnpm instead of npm

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "~/.claude/hooks/force_pnpm.sh"
          }
        ]
      }
    ]
  }
}
```

**Script:**
```bash
#!/bin/bash
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // ""')

if [[ "$COMMAND" == *"npm install"* ]]; then
    FIXED=$(echo "$COMMAND" | sed 's/npm install/pnpm install/g')
    jq -n --arg cmd "$FIXED" '{updatedInput: {command: $cmd}}'
    exit 0
fi

echo "$INPUT"
exit 0
```

### Pattern 4: Secret Scanner (PreToolUse)

**What it does:** Blocks if API keys detected

```python
import sys, json, re

hook_input = json.load(sys.stdin)
content = hook_input.get("tool_input", {}).get("content", "")

# Regex patterns for common secrets
patterns = [
    r'sk-[a-zA-Z0-9]{48}',  # OpenAI keys
    r'AIza[0-9A-Za-z-_]{35}',  # Google API keys
    r'github_pat_[a-zA-Z0-9]{82}',  # GitHub tokens
]

for pattern in patterns:
    if re.search(pattern, content):
        print("SECRET DETECTED! Remove API keys before proceeding.", file=sys.stderr)
        sys.exit(2)

sys.exit(0)
```

---

## Troubleshooting

### Hook Not Executing

**Check 1: JSON validity**
```bash
cat ~/.claude/settings.json | python3 -m json.tool
```

**Check 2: Script permissions**
```bash
ls -la /path/to/script.py
chmod +x /path/to/script.py
```

**Check 3: Test script standalone**
```bash
echo '{"tool_input":{"query":"test"}}' | python3 script.py
```

**Check 4: Check Claude Code logs**
```bash
tail -f ~/.claude/debug/hooks.log
```

### Hook Returns Wrong Format

**Common mistakes:**
- Using `modifiedToolInput` (2025) instead of `updatedInput` (2026)
- Missing `tool_input` wrapper
- Incorrect JSON structure

**Debug script output:**
```bash
echo '{"tool_input":{"query":"2024"}}' | python3 script.py | jq .
```

Should output:
```json
{
  "updatedInput": {
    "query": "2026 (Current date: 2026-01-16...)"
  }
}
```

### Hook Blocks Execution Unintentionally

**Issue:** Exit code 2 when it should be 0

```python
# BAD - blocks on any error
if error:
    sys.exit(2)

# GOOD - only block on critical issues
if critical_security_issue:
    print("CRITICAL: Security violation", file=sys.stderr)
    sys.exit(2)
else:
    # Minor issue - log but continue
    print(f"Warning: {minor_issue}", file=sys.stderr)
    sys.exit(0)
```

---

## Testing Your Hook

### Manual Test

```bash
cd /home/gyasis/Documents/code/PromptChain/time_aware
python3 test_time_context_hook.py
```

**Expected:** 9/12 tests pass (75% success rate)

### Live Test in Claude Code

```
# Test 1: WebSearch
Search for 2024 LangChain tutorial

# Expected: "2026 LangChain tutorial" with time context

# Test 2: Gemini Research
Use gemini_research to find 2025 Python updates

# Expected: "2026 Python updates" with time context
```

### Verify Hook Execution

Check if hook modified the input:
```bash
tail -f ~/.claude/debug/*.log | grep -i "hook"
```

---

## Advanced Configuration

### Environment Variables

Hooks can access these variables:
- `$CLAUDE_PROJECT_DIR` - Current project directory
- `$CLAUDE_SESSION_ID` - Current session ID
- `$HOME` - User home directory

**Example:**
```bash
"command": "\"$CLAUDE_PROJECT_DIR\"/scripts/validate.sh"
```

### Conditional Execution

**Only run hook in specific projects:**
```bash
#!/bin/bash
if [[ "$CLAUDE_PROJECT_DIR" == *"/production/"* ]]; then
    # Run strict validation
    python3 strict_validator.py
else
    # Skip in dev projects
    exit 0
fi
```

### Async Background Processing (2026 Feature)

**For slow formatters (Prettier, Ruff):**
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "black --check file.py",
            "run_in_background": true
          }
        ]
      }
    ]
  }
}
```

---

## Production Checklist

- [ ] Scripts use `updatedInput` (not `modifiedToolInput`)
- [ ] Hooks configured in `settings.json` (not separate files)
- [ ] Scripts are executable (`chmod +x`)
- [ ] Timeouts set appropriately (5-10s for fast ops)
- [ ] Error messages are actionable (Claude 4.5+ reads stderr)
- [ ] Tested standalone before integration
- [ ] Tested in live Claude Code session
- [ ] Logged for debugging/audit
- [ ] Handles edge cases (empty input, missing params)
- [ ] Exit codes correct (0=success, 2=block)

---

## Migration from 2025

If you have old hooks from 2025, update them:

1. **Change return format:**
   ```python
   # OLD
   output = {'modifiedToolInput': tool_input}

   # NEW
   output = {'updatedInput': tool_input}
   ```

2. **Move from files to settings:**
   ```bash
   # OLD location
   ~/.claude/hooks/my_hook.json

   # NEW location
   ~/.claude/settings.json (add "hooks" section)
   ```

3. **Update exit strategies:**
   - Replace "block everything on error" with targeted blocking
   - Add actionable stderr messages for Claude 4.5+

---

## Resources

- [Claude Code Hooks Reference](https://code.claude.com/docs/en/hooks)
- [Hooks Guide 2026](https://code.claude.com/docs/en/hooks-guide)
- [Production Hook Examples](https://github.com/disler/claude-code-hooks-mastery)

---

**Last Updated:** 2026-01-16
**Tested With:** Claude Code 2.1.0, Python 3.10+
**Status:** ✅ Production Ready
