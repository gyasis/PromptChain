# PromptChain Development References

## Claude Code - Hooks & Best Practices

### Official Documentation
- [Claude Code Hooks Guide](https://code.claude.com/docs/en/hooks-guide) - Official Anthropic documentation for hooks configuration and usage

### 2026 Best Practices
- [Claude Code Hooks: Complete Guide 2026](https://aiorg.dev/blog/claude-code-hooks) - Comprehensive guide with 20+ ready-to-use hook examples
- [Claude Code Best Practices 2026](https://aiorg.dev/blog/claude-code-best-practices) - 15 tips from running 6 production projects
- [Hooks Development Guide](https://claude-world.com/articles/hooks-development-guide/) - Complete workflow automation guide

### Additional Resources
- [Claude Code Hooks Mastery (GitHub)](https://github.com/disler/claude-code-hooks-mastery) - Repository with hook examples and patterns
- [Hooks in Claude Code - Complete Guide](https://www.eesel.ai/blog/hooks-in-claude-code) - Detailed automation workflow guide
- [Claude Code Best Practices: The 2026 Guide to 10x Productivity](https://www.morphllm.com/claude-code-best-practices) - Productivity optimization guide

---

## Project-Specific Hook Configuration

**Location:** `.claude/hooks/`
**Configuration:** `.claude/settings.json`
**Status:** `.claude/HOOKS_STATUS.md`

### Implemented Hooks
- `PreToolUse` - Security blocking
- `PostToolUse` - Auto-formatting + task completion
- `PostToolUseFailure` - Error logging
- `Stop` - Test enforcement
- `PreCompact` - State backup
- `UserPromptSubmit` - Context injection
- `SessionStart` - Context restoration
- `SessionEnd` - Session finalization

---

## PromptChain Project Documentation

### Core Documentation
- `README.md` - Project overview and quick start
- `CLAUDE.md` - Claude Code configuration and architecture
- `.claude/HOOKS_STATUS.md` - Hook configuration status and security features

### Development
- `examples/` - Usage examples and patterns
- `tests/` - Test suite
- `memory-bank/` - Institutional memory (dev-kid managed)

---

**Last Updated:** 2026-02-25
