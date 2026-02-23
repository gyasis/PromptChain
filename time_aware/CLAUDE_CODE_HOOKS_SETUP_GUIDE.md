# Claude Code Hooks Setup Guide

## Complete Guide to Setting Up Time Context Hooks in Claude Code

This guide shows you how to set up hooks in Claude Code to automatically remove outdated year references (2024-2025) and inject current time context into both **Gemini MCP tools** and **Claude Code's internal WebSearch** tool.

---

## 📁 Step 1: Find Your Hooks Directory

Claude Code stores hooks in one of two locations:

### Option A: User-Level Hooks (Recommended - applies to all projects)
- **macOS/Linux**: `~/.claude/hooks/`
- **Windows**: `%USERPROFILE%\.claude\hooks\`

### Option B: Project-Level Hooks (only for current project)
- **Any OS**: `.claude/hooks/` in your project root

**Recommendation**: Use user-level hooks so they work across all your projects.

---

## 📝 Step 2: Create the Hooks Directory

If the directory doesn't exist, create it:

### macOS/Linux:
```bash
mkdir -p ~/.claude/hooks
cd ~/.claude/hooks
```

### Windows:
```cmd
mkdir %USERPROFILE%\.claude\hooks
cd %USERPROFILE%\.claude\hooks
```

---

## 📄 Step 3: Create the Hook File

Copy the hook JSON file to the hooks directory:

### macOS/Linux:
```bash
# Copy the comprehensive hook file
cp comprehensive_time_context_hook.json ~/.claude/hooks/
```

### Windows:
```cmd
copy comprehensive_time_context_hook.json %USERPROFILE%\.claude\hooks\
```

---

## ✅ Step 4: Verify the File

Check that the file was created correctly:

### macOS/Linux:
```bash
ls -la ~/.claude/hooks/comprehensive_time_context_hook.json
cat ~/.claude/hooks/comprehensive_time_context_hook.json | head -5
```

### Windows:
```cmd
dir %USERPROFILE%\.claude\hooks\comprehensive_time_context_hook.json
type %USERPROFILE%\.claude\hooks\comprehensive_time_context_hook.json
```

You should see the JSON content starting with `{` and containing `"description"`.

---

## 🔐 Step 5: Review and Approve in Claude Code

**This is the most important step!** Claude Code requires manual approval for security.

1. **Open Claude Code** (or restart if already open)

2. **Open the hooks menu**:
   - Type `/hooks` in the Claude Code prompt
   - OR look for a "Hooks" menu item in the UI

3. **Find your hook**:
   - You should see `comprehensive_time_context_hook.json` listed
   - Status will show as "Not Reviewed" or "Pending"

4. **Review the hook**:
   - Click on the hook to view its contents
   - Verify it matches what you intended to install
   - Check that it targets the tools you want (Gemini MCP + WebSearch)

5. **Approve the hook**:
   - Click "Approve" or "Allow" button
   - The status should change to "Active" or "Approved"

**⚠️ Important**: If you don't see the hooks menu:
- Restart Claude Code completely
- Check Claude Code settings/preferences for a "Hooks" section
- Ensure you're using a recent version of Claude Code that supports hooks

---

## 🔄 Step 6: Restart Claude Code (if needed)

In most cases, approved hooks become active immediately. However, if you experience issues:

1. **Fully quit Claude Code** (don't just close the window)
2. **Restart Claude Code**
3. **Verify the hook is still approved** by checking `/hooks` again

---

## 🧪 Step 7: Test the Hook

Test both Gemini tools and WebSearch:

### Test Gemini Tools:
```
# In Claude Code, try:
"Use gemini_research to find 2024 AI frameworks"
"Ask Gemini about 2025 Python updates"
"Start deep research on 2024 best practices"
```

### Test WebSearch:
```
# In Claude Code, try:
"Search for 2024 LangChain tutorials"
"Find 2025 AI news"
"Look up 2024 best practices"
```

**What to look for:**
- The year "2024" or "2025" should be replaced with current year (2026)
- Time context should be automatically added
- Queries should include "(Current date: 2026-01-XX. Focus on recent information...)"

---

## 📋 What the Hook Does

### For Gemini MCP Tools:
- **ask_gemini**: Cleans `prompt` parameter
- **start_deep_research**: Cleans `query` parameter  
- **gemini_research**: Cleans `topic` parameter
- **gemini_brainstorm**: Cleans `topic` parameter
- **gemini_code_review**: Cleans `code` parameter (if it contains year references)
- **gemini_debug**: Cleans `error_message` parameter

### For WebSearch Tool:
- Cleans `query` parameter
- Removes outdated years
- Adds time context

### Transformations Applied:
1. **Year Replacement**: "2024" → "2026" (current year)
2. **Pattern Updates**: "2024 tutorial" → "2026 tutorial"
3. **Time Context Injection**: Adds current date and 6-month recency window

---

## 🛠️ Troubleshooting

### Hook Not Appearing:
- ✅ Check file is in correct directory: `~/.claude/hooks/` or `.claude/hooks/`
- ✅ Verify JSON syntax is valid (use a JSON validator)
- ✅ Check file extension is `.json` (not `.json.txt` on Windows)

### Hook Not Working:
- ✅ Verify hook was approved in `/hooks` menu
- ✅ Check Python 3 is available: `python3 --version`
- ✅ Restart Claude Code after approval
- ✅ Check Claude Code logs for errors

### Year Not Replacing:
- ✅ Verify current year is not 2024 or 2025 (hook only replaces outdated years)
- ✅ Check the query actually contains "2024" or "2025"
- ✅ Verify the tool parameter name matches (query, prompt, topic, etc.)

### Time Context Not Adding:
- ✅ Check if query already has time context (hook won't duplicate)
- ✅ Verify Python datetime module is working
- ✅ Check hook timeout (default 5 seconds)

---

## 📚 Additional Resources

- **Claude Code Hooks Documentation**: Check Claude Code's official docs
- **Hook Examples**: See `update-search-year` hook from the articles we discussed
- **Python Script Version**: Use `gemini_time_cleaner.py` for easier debugging

---

## 🎯 Quick Reference

**Hook File Location:**
- macOS/Linux: `~/.claude/hooks/comprehensive_time_context_hook.json`
- Windows: `%USERPROFILE%\.claude\hooks\comprehensive_time_context_hook.json`

**Activation Command:**
- Type `/hooks` in Claude Code
- Find and approve `comprehensive_time_context_hook.json`

**Tools Covered:**
- ✅ All Gemini MCP research tools
- ✅ Claude Code WebSearch tool

**What It Fixes:**
- ✅ Removes "2024" and "2025" year references
- ✅ Replaces with current year
- ✅ Adds time context automatically

---

That's it! Once approved, the hook will automatically clean up all your Gemini and WebSearch queries. 🚀

