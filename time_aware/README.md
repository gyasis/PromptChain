# Time-Aware Hooks for Claude Code

Automatically remove outdated year references (2024-2025) and inject current time context into Gemini MCP tools and Claude Code's WebSearch tool.

## 📋 Quick Start

1. **Read the PRD**: Start with `PRD_TIME_AWARE_HOOKS.md` to understand the full solution
2. **Follow Setup Guide**: Use `CLAUDE_CODE_HOOKS_SETUP_GUIDE.md` for installation
3. **Install Hook**: Copy `comprehensive_time_context_hook.json` to `~/.claude/hooks/`
4. **Approve in Claude Code**: Type `/hooks` and approve the hook

## 📁 Files

| File | Purpose |
|------|---------|
| `PRD_TIME_AWARE_HOOKS.md` | **START HERE** - Complete product requirements and implementation status |
| `CLAUDE_CODE_HOOKS_SETUP_GUIDE.md` | Step-by-step installation and setup instructions |
| `comprehensive_time_context_hook.json` | **RECOMMENDED** - Combined hook for Gemini + WebSearch |
| `gemini_time_context_hook.json` | Gemini-only hook (if you only need Gemini tools) |
| `mcp_time_context_hook.json` | Generic MCP hook (if you need all MCP tools) |
| `gemini_time_cleaner.py` | Standalone Python script (optional, for debugging) |
| `mcp_time_context_cleaner.py` | Standalone Python script (optional, for debugging) |

## ✅ Implementation Status

**Status**: ✅ **IMPLEMENTED** - All files created and ready for installation

- [x] Hook files created
- [x] Documentation written
- [x] Setup guide created
- [x] PRD completed

**Next Steps**: Installation and testing (see setup guide)

## 🎯 What It Does

### Before Hook:
```
ask_gemini(prompt="What are 2024 AI trends?")
WebSearch(query="2025 LangChain tutorial")
```

### After Hook:
```
ask_gemini(prompt="What are 2026 AI trends? (Current date: 2026-01-XX. Focus on recent info...)")
WebSearch(query="2026 LangChain tutorial (Current date: 2026-01-XX. Focus on recent info...)")
```

## 🔧 Tools Supported

**Gemini MCP Tools:**
- `ask_gemini`
- `start_deep_research`
- `gemini_research`
- `gemini_brainstorm`
- `gemini_code_review`
- `gemini_debug`

**Claude Code Tools:**
- `WebSearch`

## 📖 Documentation

1. **PRD_TIME_AWARE_HOOKS.md** - Complete product requirements, implementation details, testing plan
2. **CLAUDE_CODE_HOOKS_SETUP_GUIDE.md** - Installation, troubleshooting, testing instructions

## 🚀 Installation

```bash
# 1. Copy hook to Claude Code hooks directory
cp comprehensive_time_context_hook.json ~/.claude/hooks/

# 2. In Claude Code, type: /hooks
# 3. Find and approve: comprehensive_time_context_hook.json
```

See `CLAUDE_CODE_HOOKS_SETUP_GUIDE.md` for detailed instructions.

## 🧪 Testing

After installation, test with:
- "Search for 2024 LangChain tutorial"
- "Use gemini_research to find 2025 AI frameworks"
- "Ask Gemini about 2024 Python updates"

All should automatically update to current year with time context.

## 📝 Notes

- Hook requires manual approval in Claude Code (security feature)
- Python 3 must be available in PATH
- Works transparently - no changes to user workflow needed
- Only modifies tool inputs, not outputs

---

**For Implementation Review**: See `PRD_TIME_AWARE_HOOKS.md` section 3 for complete implementation status.

