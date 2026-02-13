# Product Requirements Document: Time-Aware Hooks for Claude Code

## Executive Summary

**Problem**: Claude Code agents and MCP tools (especially Gemini research tools) default to searching for information from their knowledge cutoff years (2024-2025) even when we're in 2026, leading to outdated search results and recommendations.

**Solution**: Create Claude Code hooks that automatically intercept tool calls, remove outdated year references, and inject current time context to ensure agents always search for current information.

**Status**: ✅ **IMPLEMENTED** - All hook files and documentation created, ready for installation and testing.

---

## 1. Problem Statement

### Current Issues

1. **Knowledge Cutoff Bias**: LLMs trained with 2024/2025 cutoff dates default to those years when generating search queries
2. **Static Year References**: Agents append "2024" or "2025" to queries even in 2026
3. **Temporal Disconnect**: No automatic time awareness in tool calls
4. **Outdated Results**: Search results return information from 1-2 years ago instead of current

### Impact

- **Gemini Research Tools**: `gemini_research`, `ask_gemini`, `start_deep_research` return outdated information
- **WebSearch Tool**: Claude Code's internal search defaults to old year references
- **User Experience**: Users receive outdated tutorials, guides, and information
- **Productivity**: Developers waste time on deprecated practices

---

## 2. Solution Overview

### Approach

Create **PreToolUse hooks** in Claude Code that:
1. Intercept tool calls before execution
2. Detect and remove outdated year references (2024, 2025)
3. Replace with current year
4. Inject relative time context (e.g., "Focus on recent information from last 6 months")
5. Work transparently without changing user workflow

### Target Tools

**Gemini MCP Tools:**
- `mcp_gemini-mcp_ask_gemini` (prompt parameter)
- `mcp_gemini-mcp_start_deep_research` (query parameter)
- `mcp_gemini-mcp_gemini_research` (topic parameter)
- `mcp_gemini-mcp_gemini_brainstorm` (topic parameter)
- `mcp_gemini-mcp_gemini_code_review` (code parameter)
- `mcp_gemini-mcp_gemini_debug` (error_message parameter)

**Claude Code Internal Tools:**
- `WebSearch` (query parameter)

---

## 3. Implementation Status

### ✅ Completed Components

1. **Hook Files Created:**
   - `comprehensive_time_context_hook.json` - Combined hook for Gemini + WebSearch
   - `gemini_time_context_hook.json` - Gemini-specific hook
   - `mcp_time_context_hook.json` - Generic MCP hook

2. **Supporting Files:**
   - `gemini_time_cleaner.py` - Standalone Python script (optional, for debugging)
   - `mcp_time_context_cleaner.py` - Standalone Python script (optional, for debugging)

3. **Documentation:**
   - `CLAUDE_CODE_HOOKS_SETUP_GUIDE.md` - Complete setup instructions
   - `PRD_TIME_AWARE_HOOKS.md` - This document

### 🔧 Implementation Details

**Hook Architecture:**
- **Type**: PreToolUse hooks (intercept before tool execution)
- **Matcher**: Regex patterns for Gemini tools, direct name for WebSearch
- **Processing**: Python inline command that:
  - Parses tool input JSON
  - Detects outdated years (2024, 2025)
  - Replaces with current year
  - Adds time context if missing
  - Returns modified tool input

**Key Features:**
- ✅ Automatic year detection and replacement
- ✅ Pattern matching for "2024 tutorial" → "2026 tutorial"
- ✅ Time context injection ("Current date: 2026-01-XX. Focus on recent info...")
- ✅ Works on multiple parameter types (query, prompt, topic, etc.)
- ✅ Non-destructive (only modifies when needed)

---

## 4. Installation Requirements

### Prerequisites

1. **Claude Code** installed and running
2. **Python 3** available in PATH (for hook execution)
3. **Hooks directory** exists:
   - macOS/Linux: `~/.claude/hooks/`
   - Windows: `%USERPROFILE%\.claude\hooks\`

### Installation Steps

1. **Create hooks directory** (if doesn't exist):
   ```bash
   mkdir -p ~/.claude/hooks  # macOS/Linux
   mkdir %USERPROFILE%\.claude\hooks  # Windows
   ```

2. **Copy hook file**:
   ```bash
   cp comprehensive_time_context_hook.json ~/.claude/hooks/
   ```

3. **Approve in Claude Code**:
   - Type `/hooks` in Claude Code
   - Find `comprehensive_time_context_hook.json`
   - Review and approve

4. **Verify**:
   - Test with: "Search for 2024 LangChain tutorial"
   - Should become: "Search for 2026 LangChain tutorial (Current date: 2026-01-XX...)"

---

## 5. Technical Specifications

### Hook JSON Structure

```json
{
  "description": "...",
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "ToolName" or {"pattern": "regex", "type": "regex"},
        "hooks": [
          {
            "type": "command",
            "timeout": 5,
            "command": "python3 -c \"...inline Python code...\""
          }
        ]
      }
    ]
  }
}
```

### Processing Logic

1. **Input**: Tool input JSON from Claude Code
2. **Extract**: Current year, current date, 6-month window
3. **Detect**: Outdated years (2024, 2025) in query parameters
4. **Replace**: Outdated years with current year
5. **Inject**: Time context if not present
6. **Output**: Modified tool input JSON

### Parameter Mapping

| Tool | Parameter Name | Example |
|------|---------------|---------|
| ask_gemini | `prompt` | "What are 2024 AI trends?" |
| start_deep_research | `query` | "2025 ML breakthroughs" |
| gemini_research | `topic` | "Python async 2024" |
| WebSearch | `query` | "LangChain 2024 tutorial" |

---

## 6. Testing & Validation

### Test Cases

**Test 1: Year Replacement**
- Input: `query="2024 LangChain tutorial"`
- Expected: `query="2026 LangChain tutorial (Current date: 2026-01-XX...)"`

**Test 2: Pattern Matching**
- Input: `query="2025 best practices"`
- Expected: `query="2026 best practices (Current date: 2026-01-XX...)"`

**Test 3: Time Context Addition**
- Input: `query="Python async patterns"`
- Expected: `query="Python async patterns (Current date: 2026-01-XX. Focus on recent info...)"`

**Test 4: No Duplication**
- Input: `query="AI trends (Current date: 2026-01-15)"`
- Expected: `query="AI trends (Current date: 2026-01-15)"` (no change)

### Validation Checklist

- [ ] Hook file created in correct directory
- [ ] Hook approved in Claude Code (`/hooks` menu)
- [ ] Python 3 available and working
- [ ] Year replacement works (2024 → 2026)
- [ ] Time context added when missing
- [ ] No duplicate time context added
- [ ] Works for Gemini tools (ask_gemini, research, etc.)
- [ ] Works for WebSearch tool
- [ ] No performance degradation (< 50ms overhead)

---

## 7. Edge Cases & Limitations

### Edge Cases Handled

1. **Current Year is 2024/2025**: Hook won't replace if current year matches
2. **Already Has Time Context**: Won't duplicate time context
3. **Non-String Parameters**: Safely handles non-string values
4. **Missing Parameters**: Only processes parameters that exist
5. **Empty Queries**: Handles empty/null values gracefully

### Known Limitations

1. **Tool Name Variations**: MCP tool names may vary by installation
   - **Solution**: Use regex pattern matching
   - **Workaround**: Check actual tool names with `/tools` command

2. **Parameter Name Variations**: Different tools use different parameter names
   - **Solution**: Check multiple common names (query, prompt, topic, etc.)

3. **Python Dependency**: Requires Python 3 in PATH
   - **Solution**: Verify with `python3 --version`
   - **Workaround**: Use absolute path to python3 if needed

4. **Hook Approval Required**: Security feature requires manual approval
   - **Solution**: Document approval process clearly
   - **Note**: This is a security feature, not a bug

---

## 8. Future Enhancements

### Potential Improvements

1. **Relative Timeframes**: Support `++Websearch 3m` syntax like the WebSearch hook
2. **Configurable Defaults**: Allow users to set default time window (3m, 6m, 1y)
3. **Tool-Specific Rules**: Different time contexts for different tools
4. **Year Range Detection**: Detect and update year ranges (e.g., "2024-2025" → "2025-2026")
5. **Date Format Normalization**: Standardize date formats in queries

### Not in Scope (v1.0)

- Modifying tool results (only modifies inputs)
- Historical date handling (only current/recent focus)
- Multi-language support (English only for now)
- Custom time context templates

---

## 9. Success Metrics

### Key Performance Indicators

1. **Year Replacement Rate**: % of queries with outdated years that get updated
   - **Target**: > 95%

2. **Time Context Injection**: % of queries that receive time context
   - **Target**: > 90% (excluding queries that already have context)

3. **Performance Impact**: Average hook execution time
   - **Target**: < 50ms per tool call

4. **User Satisfaction**: Reduction in outdated result complaints
   - **Target**: Measurable improvement in search relevance

---

## 10. Rollout Plan

### Phase 1: Setup (Current)
- ✅ Create hook files
- ✅ Write documentation
- ✅ Create setup guide

### Phase 2: Testing
- [ ] Install hook in development environment
- [ ] Test with various Gemini tools
- [ ] Test with WebSearch tool
- [ ] Validate year replacement
- [ ] Validate time context injection

### Phase 3: Deployment
- [ ] Install in production Claude Code
- [ ] Monitor hook execution logs
- [ ] Collect user feedback
- [ ] Measure success metrics

### Phase 4: Iteration
- [ ] Refine based on feedback
- [ ] Add additional tool support if needed
- [ ] Optimize performance if needed

---

## 11. Files & Structure

```
time_aware/
├── PRD_TIME_AWARE_HOOKS.md              # This document
├── CLAUDE_CODE_HOOKS_SETUP_GUIDE.md    # Setup instructions
├── comprehensive_time_context_hook.json # Main hook (Gemini + WebSearch)
├── gemini_time_context_hook.json       # Gemini-only hook
├── mcp_time_context_hook.json          # Generic MCP hook
├── gemini_time_cleaner.py              # Standalone Python script (optional)
└── mcp_time_context_cleaner.py        # Standalone Python script (optional)
```

---

## 12. Dependencies

### Required
- Claude Code (with hooks support)
- Python 3.x (for hook execution)
- JSON parsing capability

### Optional
- Python script files (for easier debugging/maintenance)

---

## 13. Maintenance

### Regular Updates Needed

1. **Year List**: Update `outdated_years` list annually
   - Currently: `['2024', '2025']`
   - In 2027: Add `'2026'` to list

2. **Tool Name Patterns**: Update regex if new Gemini tools added
   - Current pattern: `mcp_gemini-mcp_(ask_gemini|start_deep_research|...)`
   - Add new tools to pattern as needed

3. **Parameter Names**: Update if tool parameter names change
   - Current: `['prompt', 'query', 'topic', 'question', 'context']`
   - Add new parameter names as discovered

---

## 14. Support & Troubleshooting

### Common Issues

**Issue**: Hook not appearing in `/hooks` menu
- **Solution**: Verify file is in `~/.claude/hooks/` directory

**Issue**: Hook not working
- **Solution**: Check hook was approved, verify Python 3 available

**Issue**: Year not replacing
- **Solution**: Verify current year is not 2024/2025, check query contains year

**Issue**: Time context not adding
- **Solution**: Check if query already has time context, verify Python datetime works

### Getting Help

1. Check `CLAUDE_CODE_HOOKS_SETUP_GUIDE.md` for detailed troubleshooting
2. Verify hook JSON syntax is valid
3. Test Python script standalone if using external script
4. Check Claude Code logs for errors

---

## 15. Conclusion

**Status**: ✅ **IMPLEMENTATION COMPLETE**

All hook files, documentation, and setup guides have been created. The solution is ready for:
1. Installation in Claude Code
2. Testing and validation
3. Production deployment

**Next Steps**:
1. Install hook following `CLAUDE_CODE_HOOKS_SETUP_GUIDE.md`
2. Test with various Gemini and WebSearch queries
3. Monitor performance and user feedback
4. Iterate based on results

---

**Document Version**: 1.0  
**Last Updated**: 2026-01-XX  
**Status**: Ready for Implementation Review

