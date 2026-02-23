# T041-T042 Visual Examples: Router Decision Display

This document provides visual examples of the router decision display and agent switching detection features implemented in T041-T042.

---

## Feature Overview

**T041**: Status bar displays selected agent with router icon (⚙)
**T042**: Chat view shows notifications when router switches agents

---

## Example 1: Single-Agent Mode (No Router)

**Setup**:
- Session: "my-project"
- Agents: 1 ("default" using gpt-4)
- Router Mode: Disabled

**Conversation**:
```
┌─────────────────────────────────────────────────────────────────┐
│ Session: my-project                                             │
└─────────────────────────────────────────────────────────────────┘

User: Hello, how can you help me?

[Assistant response from default agent...]

┌─────────────────────────────────────────────────────────────────┐
│ * Session: my-project | Agent: default | Model: gpt-4 | Messages: 2
└─────────────────────────────────────────────────────────────────┘
```

**Key Observations**:
- ✅ NO router icon (⚙) in status bar
- ✅ NO switch notifications
- ✅ Agent shown as "Agent: default" (not "⚙ Agent: default")

---

## Example 2: Multi-Agent Mode - First Message

**Setup**:
- Session: "code-review"
- Agents: 3 ("coder", "researcher", "writer")
- Router Mode: Enabled
- Active Agent: "coder" (default)

**Conversation (First Message)**:
```
┌─────────────────────────────────────────────────────────────────┐
│ Session: code-review                                            │
└─────────────────────────────────────────────────────────────────┘

User: Analyze this authentication code for security issues

[Router Decision: Selected "coder" agent]

[Assistant response from coder agent analyzing code security...]

┌─────────────────────────────────────────────────────────────────┐
│ * Session: code-review | ⚙ Agent: coder | Model: gpt-4 | Messages: 2
└─────────────────────────────────────────────────────────────────┘
```

**Key Observations**:
- ✅ Router icon (⚙) shown in cyan color
- ✅ Agent name "coder" displayed
- ✅ NO switch notification (first router decision)
- ✅ `last_displayed_agent` set to "coder" internally

---

## Example 3: Agent Switch Detected

**Setup**: Continue from Example 2

**Conversation (Second Message)**:
```
┌─────────────────────────────────────────────────────────────────┐
│ * Session: code-review | ⚙ Agent: coder | Model: gpt-4 | Messages: 2
└─────────────────────────────────────────────────────────────────┘

User: Research the latest OAuth 2.1 security best practices

→ Router switched to agent: researcher

[Assistant response from researcher agent with research findings...]

┌─────────────────────────────────────────────────────────────────┐
│ * Session: code-review | ⚙ Agent: researcher | Model: gpt-4 | Messages: 4
└─────────────────────────────────────────────────────────────────┘
```

**Key Observations**:
- ✅ Switch notification displayed: "→ Router switched to agent: researcher"
- ✅ Notification in dim, italic text
- ✅ Agent name "researcher" in bold
- ✅ Status bar updated to show "researcher"
- ✅ Router icon (⚙) maintained

**Rich Markup Rendering**:
```
[dim]→ Router switched to agent: [bold]researcher[/bold][/dim]
```

---

## Example 4: Same Agent Selected (No Switch)

**Setup**: Continue from Example 3

**Conversation (Third Message)**:
```
┌─────────────────────────────────────────────────────────────────┐
│ * Session: code-review | ⚙ Agent: researcher | Model: gpt-4 | Messages: 4
└─────────────────────────────────────────────────────────────────┘

User: Continue researching PKI certificate management standards

[Router Decision: Selected "researcher" again]

[Assistant response from researcher agent with additional research...]

┌─────────────────────────────────────────────────────────────────┐
│ * Session: code-review | ⚙ Agent: researcher | Model: gpt-4 | Messages: 6
└─────────────────────────────────────────────────────────────────┘
```

**Key Observations**:
- ✅ NO switch notification (same agent)
- ✅ Router icon (⚙) still displayed
- ✅ Status bar shows "researcher" consistently
- ✅ Clean chat view without unnecessary notifications

---

## Example 5: Multiple Agent Switches

**Setup**: Continue from Example 4

**Conversation (Multiple Messages)**:
```
┌─────────────────────────────────────────────────────────────────┐
│ * Session: code-review | ⚙ Agent: researcher | Model: gpt-4 | Messages: 6
└─────────────────────────────────────────────────────────────────┘

User: Write a comprehensive security audit report

→ Router switched to agent: writer

[Assistant response from writer agent with formatted report...]

User: Fix the SQL injection vulnerability in the login function

→ Router switched to agent: coder

[Assistant response from coder agent with code fix...]

User: Research OWASP Top 10 2023 changes

→ Router switched to agent: researcher

[Assistant response from researcher agent with OWASP findings...]

┌─────────────────────────────────────────────────────────────────┐
│ * Session: code-review | ⚙ Agent: researcher | Model: gpt-4 | Messages: 12
└─────────────────────────────────────────────────────────────────┘
```

**Key Observations**:
- ✅ Switch notifications for each agent change
- ✅ Status bar tracks current agent accurately
- ✅ Clear visual feedback for routing decisions
- ✅ No notification clutter (only shows on actual switches)

---

## Example 6: Router Icon Color Coding

**Status Bar Display Breakdown**:

**Router Mode Active**:
```
⚙ Agent: coder
└─┬─┘
  └─ Cyan color (#00ffff)
```

**Single-Agent Mode**:
```
Agent: default
└────┬────┘
     └─ Cyan color for agent name only
```

**CSS Color Mappings**:
```python
# Router icon
"[cyan]⚙[/cyan]"  # Unicode: U+2699 (Gear Symbol)

# Switch arrow
"[dim]→[/dim]"    # Unicode: U+2192 (Rightwards Arrow)

# Agent name in notification
"[bold]agent_name[/bold]"
```

---

## Example 7: System Message Styling

**CSS Applied to Switch Notifications**:

```css
.system-message {
    padding: 0 1;
    color: $text-muted;
    text-style: italic;
}
```

**Visual Rendering**:
```
Normal chat message: Bold, full color
→ Router switched to agent: researcher   ← Dim, italic, subtle
[Assistant response...]
```

**Purpose**:
- Non-intrusive notifications
- Clear distinction from user/assistant messages
- Professional, clean appearance

---

## Example 8: Edge Case - Empty Session

**Setup**:
- Session: "new-session"
- Agents: 0 (no agents configured)

**Conversation**:
```
┌─────────────────────────────────────────────────────────────────┐
│ Session: new-session                                            │
└─────────────────────────────────────────────────────────────────┘

User: Hello

[red]No agents configured. Please create an agent using /agent create[/red]

┌─────────────────────────────────────────────────────────────────┐
│ * Session: new-session | Messages: 1
└─────────────────────────────────────────────────────────────────┘
```

**Key Observations**:
- ✅ Graceful error handling
- ✅ No router icon (no agents to route to)
- ✅ Clear user guidance

---

## Example 9: Token Usage Display Integration

**Status Bar with All Features**:
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ * Session: my-project | ⚙ Agent: researcher | Model: gpt-4 | Messages: 10 | ● Tokens: 3456/8000 (43%)
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Feature Breakdown**:
- `*` - Session active indicator (green)
- `⚙` - Router mode active (cyan)
- `Agent: researcher` - Currently selected agent
- `Model: gpt-4` - Agent's model
- `Messages: 10` - Conversation length
- `● Tokens: 3456/8000 (43%)` - Token usage (green indicator)

**Color Coding**:
- Green indicators: Healthy state
- Yellow indicators: Warning state (60-85% tokens)
- Red indicators: Critical state (>85% tokens)

---

## Example 10: MCP Server Integration

**Status Bar with MCP Servers**:
```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ * Session: data-analysis | ⚙ Agent: analyst | Model: gpt-4 | Messages: 5 | MCP: ✓filesystem ✓database
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

**MCP Status Indicators**:
- `✓filesystem` - Filesystem MCP server connected (green)
- `✓database` - Database MCP server connected (green)
- `✗failed-server` - Server connection failed (red)
- `○disconnected` - Server disconnected (dim)

---

## Implementation Details Reference

### Status Bar Reactive Properties

```python
# T041: Router-selected agent
selected_agent: reactive[str] = reactive("")

# Other router-related properties
router_mode: reactive[bool] = reactive(False)
last_agent_switch: reactive[str] = reactive("")
```

### Agent Switch Detection Logic

```python
# T042: Detect agent switch
if self.last_displayed_agent and selected_agent_name != self.last_displayed_agent:
    # Show switch notification
    switch_message = f"[dim]→ Router switched to agent: [bold]{selected_agent_name}[/bold][/dim]"
    await self._display_system_message(switch_message)

# Update tracking
self.last_displayed_agent = selected_agent_name
```

### Status Bar Update Calls

```python
# Always update selected_agent for router icon (T041)
status_bar.update_session_info(
    selected_agent=selected_agent_name,  # T041: Show router selection
    active_agent=selected_agent_name,    # Update active agent
    model_name=selected_model,           # Update model
    last_agent_switch=selected_agent_name,  # Track switch
)
```

---

## User Experience Guidelines

### When to Show Switch Notifications

**✅ Show Notification**:
- Agent changes from previous message
- Router makes different selection than last time
- User needs awareness of routing decision

**❌ Don't Show Notification**:
- First message in session (no previous agent)
- Same agent selected as previous message
- Single-agent mode (no routing happening)

### Visual Hierarchy

**Priority Order**:
1. User message (highest priority, bold)
2. Assistant response (high priority, normal text)
3. Switch notification (low priority, dim/italic)
4. System messages (low priority, dim/italic)

**Rationale**:
- User and assistant messages are primary content
- System notifications provide context without distraction
- Dim styling prevents notification clutter

---

## Testing Checklist

**Visual Tests**:
- [ ] Router icon (⚙) displays in cyan color
- [ ] Agent name shown correctly in status bar
- [ ] Switch notification appears on agent change
- [ ] Switch notification is dim and italic
- [ ] Agent name in notification is bold
- [ ] Arrow symbol (→) renders correctly
- [ ] No notification on first message
- [ ] No notification when same agent selected

**Functional Tests**:
- [ ] `last_displayed_agent` tracks correctly
- [ ] Switch detection logic works for multiple switches
- [ ] Status bar updates reflect router decisions
- [ ] Single-agent mode shows no router icon
- [ ] Multi-agent mode always shows router icon
- [ ] Edge cases handled (no agents, empty session)

**Integration Tests**:
- [ ] Works with token usage display
- [ ] Works with MCP server status
- [ ] Works with session state indicators
- [ ] Works with activity log viewer
- [ ] No conflicts with existing chat flow

---

## Conclusion

The T041-T042 implementation provides clear, professional visual feedback for router decisions:

**Key Benefits**:
- Transparent routing behavior
- Non-intrusive notifications
- Clear agent attribution
- Professional UI/UX
- Consistent styling

**User Impact**:
- Users always know which agent is responding
- Router decisions are visible and understandable
- Clean, uncluttered chat interface
- Professional appearance suitable for production use
