# TUI Crash Fix - 2026-01-11

## Issue: MissingStyle Error Causing TUI Crash

**Error**: `MissingStyle: Failed to get style 'entrypath'; unable to parse 'entrypath' as color`

**Symptoms**: CLI crashes when rendering tool results or other streaming content that contains Rich text markup with unknown style names.

**Root Cause**: Tool outputs and other dynamic content may contain Rich text formatting (including spans with custom styles like 'entrypath'). When this content was embedded directly into our Rich markup strings (e.g., `f"[green]✓ {content}[/green]"`), the Rich parser tried to resolve these unknown styles, causing a crash.

**Example**:
```python
# Tool result content contains: "...file at [entrypath]package.json[/entrypath]..."
# We were doing:
content = "[green]✓ read_file_range completed: ...file at [entrypath]package.json[/entrypath]...[/green]"
# Rich parser sees 'entrypath' and fails because it's not a valid style
```

## Fix Applied

Applied the same bracket escaping solution from `observe_panel.py:209` to all streaming event handlers in `app.py`.

**Changed locations in `promptchain/cli/tui/app.py`**:

### 1. Thinking Events (lines 1259-1269)
```python
# Before:
content=f"[dim italic]🧠 {content}[/dim italic]"

# After:
safe_content = content.replace('[', '\\[').replace(']', '\\]')
content=f"[dim italic]🧠 {safe_content}[/dim italic]"
```

### 2. Tool Call Events (lines 1270-1278)
```python
# Before:
content=f"[cyan]🔧 Calling: {content}[/cyan]"

# After:
safe_content = content.replace('[', '\\[').replace(']', '\\]')
content=f"[cyan]🔧 Calling: {safe_content}[/cyan]"
```

### 3. Tool Result Events (lines 1279-1290) ⭐ Primary Fix
```python
# Before:
preview = content[:300] + "..." if len(content) > 300 else content
content=f"[green]✓ {preview}[/green]"

# After:
preview = content[:300] + "..." if len(content) > 300 else content
safe_preview = preview.replace('[', '\\[').replace(']', '\\]')
content=f"[green]✓ {safe_preview}[/green]"
```

### 4. Error Events (lines 1291-1303)
```python
# Before:
content=f"[red]⚠️ {content}[/red]"

# After:
safe_content = content.replace('[', '\\[').replace(']', '\\]')
content=f"[red]⚠️ {safe_content}[/red]"
```

### 5. User Input Events (lines 1304-1313)
```python
# Before:
content=f"[yellow]📥 User input received: {content[:100]}...[/yellow]"

# After:
preview = content[:100] + "..." if len(content) > 100 else content
safe_preview = preview.replace('[', '\\[').replace(']', '\\]')
content=f"[yellow]📥 User input received: {safe_preview}[/yellow]"
```

### 6. Router Fallback (lines 3128-3134)
```python
# Before:
content=f"[italic]Router failed ({type(e).__name__}: {str(e)}), using fallback: {default_agent_name}[/italic]"

# After:
safe_error = str(e).replace('[', '\\[').replace(']', '\\]')
content=f"[italic]Router failed ({type(e).__name__}: {safe_error}), using fallback: {default_agent_name}[/italic]"
```

## How It Works

The fix converts literal square brackets in dynamic content to escaped brackets before Rich text parsing:
- `[entrypath]` becomes `\[entrypath\]`
- Rich parser treats `\[...\]` as literal characters, not markup tags
- Prevents crashes from unknown styles while preserving the content

## Impact

**Fixed**:
- ✅ TUI no longer crashes when tool results contain Rich markup
- ✅ All streaming events (thinking, tool calls, tool results, errors, user input) now safe
- ✅ Router error messages with brackets now display correctly

**Scope**:
- All streaming event handlers in `_handle_agentic_step_processor_stream()`
- Router fallback error messages
- Similar to previous fix in `observe_panel.py` for ObservePanel crash

## Testing

**To Verify**:
```bash
# Run CLI with any operation that uses tools
promptchain --verbose

# Try operations that previously crashed:
> Use the read_file_range tool to read package.json
> [Should display tool results without crashing]
```

**Expected Result**:
- ✅ Tool results display correctly with escaped brackets
- ✅ No MissingStyle exceptions
- ✅ TUI remains stable during all streaming events

## Related Fixes

This is part of a series of TUI stability fixes:
1. **2026-01-10**: Fixed MarkupError with bracket escaping in `observe_panel.py:209` (OBSERVABILITY_FIXES_2026-01-10.md)
2. **2026-01-10**: Improved step numbering display with visual hierarchy (OBSERVABILITY_FIXES_2026-01-10.md)
3. **2026-01-11**: Fixed MissingStyle error in streaming events (this fix)

## Prevention

**Pattern to Follow**:
Whenever embedding dynamic content in Rich markup strings, always escape brackets:

```python
# GOOD ✅
safe_content = content.replace('[', '\\[').replace(']', '\\]')
message_content = f"[green]{safe_content}[/green]"

# BAD ❌ (can crash if content has Rich markup)
message_content = f"[green]{content}[/green]"
```

---

**Date**: 2026-01-11
**Session**: TUI crash investigation
**Branch**: 005-mlflow-observability
**Files Modified**: `promptchain/cli/tui/app.py` (6 locations)
