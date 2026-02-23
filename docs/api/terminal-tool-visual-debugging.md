# TerminalTool Visual Debugging API Documentation

The PromptChain TerminalTool includes powerful visual debugging features that provide terminal-like formatting for better debugging of LLM interactions. These features use the Rich library to create visually appealing, terminal-style output that helps developers understand command execution flow and troubleshoot issues.

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [TerminalTool Visual Parameters](#terminaltool-visual-parameters)
4. [TerminalTool Visual Methods](#terminaltool-visual-methods)
5. [VisualTerminalFormatter Class](#visualterminalformatter-class)
6. [Usage Examples](#usage-examples)
7. [Integration Patterns](#integration-patterns)
8. [Best Practices](#best-practices)

## Overview

Visual debugging transforms the TerminalTool's output from plain text to rich, terminal-like displays. This feature is particularly valuable for:

- **Debugging LLM terminal interactions**: See exactly what commands LLMs are executing
- **Development workflows**: Visual feedback during complex multi-step processes  
- **Multi-agent systems**: Track command execution across different agents
- **Complex reasoning processes**: Understand how AgenticStepProcessor uses terminal commands
- **Educational purposes**: Demonstrate terminal interactions in a clear, visual format

## Requirements

Visual debugging requires the Rich library:

```bash
pip install rich
```

If Rich is not installed, visual debugging features will be disabled gracefully with warning messages.

## TerminalTool Visual Parameters

### Constructor Parameter

```python
class TerminalTool:
    def __init__(
        self,
        # ... other parameters ...
        visual_debug: bool = False
    ):
```

**`visual_debug: bool = False`**
- **Description**: Enable visual terminal debugging with Rich formatting
- **Type**: `bool`
- **Default**: `False`
- **Behavior**: 
  - When `True`: Enables visual formatting and creates a `VisualTerminalFormatter` instance
  - When `False`: Standard text output only
  - If `True` but Rich is not available: Prints warning and continues without visual features

## TerminalTool Visual Methods

### show_visual_terminal()

```python
def show_visual_terminal(
    self, 
    max_entries: int = 10, 
    show_timestamps: bool = False
) -> None
```

Display the visual terminal history with terminal-like formatting.

**Parameters:**
- `max_entries: int = 10` - Maximum number of recent commands to display
- `show_timestamps: bool = False` - Whether to show execution timestamps

**Example:**
```python
terminal = TerminalTool(visual_debug=True)
terminal("ls -la")
terminal("pwd")
terminal.show_visual_terminal(max_entries=5)  # Shows last 5 commands
```

### get_visual_history_summary()

```python
def get_visual_history_summary(self) -> Optional[Dict[str, Any]]
```

Get statistical summary of the visual command history.

**Returns:**
- `Dict[str, Any]` with keys:
  - `total_commands: int` - Total number of commands executed
  - `errors: int` - Number of failed commands
  - `success: int` - Number of successful commands
  - `error_rate: float` - Error rate (0.0 to 1.0)
  - `recent_commands: List[str]` - Last 5 command strings
- `None` if visual debugging is disabled

**Example:**
```python
terminal = TerminalTool(visual_debug=True)
terminal("echo 'test'")
terminal("invalid_command")  # This will fail
summary = terminal.get_visual_history_summary()
print(f"Success rate: {1 - summary['error_rate']:.2%}")
```

### clear_visual_history()

```python
def clear_visual_history(self) -> None
```

Clear the visual terminal history without affecting functionality.

**Example:**
```python
terminal.clear_visual_history()  # Start fresh visual history
```

### enable_visual_debug()

```python
def enable_visual_debug(self) -> None
```

Dynamically enable visual debugging if Rich is available.

**Behavior:**
- Creates a new `VisualTerminalFormatter` instance
- Enables visual formatting for subsequent commands
- Prints confirmation message if `verbose=True`
- Prints warning if Rich is not available

### disable_visual_debug()

```python
def disable_visual_debug(self) -> None
```

Dynamically disable visual debugging.

**Behavior:**
- Sets `visual_debug=False`
- Removes the visual formatter instance
- Subsequent commands will use standard text output
- Prints confirmation message if `verbose=True`

## VisualTerminalFormatter Class

The standalone `VisualTerminalFormatter` class provides terminal-like formatting capabilities that can be used independently of TerminalTool.

### Constructor

```python
class VisualTerminalFormatter:
    def __init__(
        self, 
        style: str = "dark", 
        max_history: int = 50
    ):
```

**Parameters:**
- `style: str = "dark"` - Visual style ("dark" or "light")
- `max_history: int = 50` - Maximum commands to keep in history

### Core Methods

#### add_command()

```python
def add_command(
    self, 
    command: str, 
    output: str, 
    working_dir: str = "/tmp",
    user: str = "user", 
    host: str = "system", 
    error: bool = False
) -> None
```

Add a command execution to the visual history.

**Parameters:**
- `command: str` - The executed command
- `output: str` - Command output
- `working_dir: str = "/tmp"` - Current working directory
- `user: str = "user"` - Username for display
- `host: str = "system"` - Hostname for display  
- `error: bool = False` - Whether command failed

#### format_as_terminal()

```python
def format_as_terminal(
    self, 
    max_entries: int = 10, 
    show_timestamps: bool = False
) -> Panel
```

Format commands as a Rich Panel with terminal-like appearance.

**Returns:** `rich.panel.Panel` object for display

#### print_terminal()

```python
def print_terminal(
    self, 
    max_entries: int = 10, 
    show_timestamps: bool = False
) -> None
```

Print the terminal display directly to console.

#### get_live_display()

```python
def get_live_display(self, max_entries: int = 10) -> Live
```

Get a Rich Live display for real-time updates.

**Usage:**
```python
formatter = VisualTerminalFormatter()
with formatter.get_live_display() as live:
    # Commands will update live display automatically
    formatter.add_command("ls", "file1.txt\nfile2.txt")
    live.update()
```

## Usage Examples

### Basic Visual Debugging

```python
from promptchain.tools.terminal import TerminalTool

# Enable visual debugging
terminal = TerminalTool(
    visual_debug=True,
    verbose=True,
    use_persistent_session=True
)

# Execute commands with visual output
terminal("mkdir -p /tmp/test-project")
terminal("cd /tmp/test-project")
terminal("echo 'Hello Visual Terminal!' > readme.txt")
terminal("ls -la")

# Show visual history
terminal.show_visual_terminal(max_entries=5)

# Get statistics
summary = terminal.get_visual_history_summary()
print(f"Executed {summary['total_commands']} commands")
```

### PromptChain Integration

```python
from promptchain import PromptChain
from promptchain.tools.terminal import TerminalTool

# Create visual terminal tool
def create_visual_terminal():
    terminal = TerminalTool(
        visual_debug=True,
        verbose=True,
        use_persistent_session=True,
        session_name="llm_development"
    )
    
    def execute_with_visual_feedback(command: str) -> str:
        result = terminal(command)
        # Show recent history after each command
        terminal.show_visual_terminal(max_entries=3)
        return result
    
    return execute_with_visual_feedback

# Use in PromptChain
chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=[
        """You are a development assistant with visual terminal access.
        Execute commands to help with the user's request.
        
        User request: {input}"""
    ],
    verbose=True
)

visual_terminal = create_visual_terminal()
chain.register_tool_function(visual_terminal)
chain.add_tools([{
    "type": "function",
    "function": {
        "name": "execute_with_visual_feedback",
        "description": "Execute terminal commands with visual debugging display",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Terminal command to execute"
                }
            },
            "required": ["command"]
        }
    }
}])

# LLM will use terminal with visual feedback
result = chain.process_prompt("Set up a Python project with virtual environment")
```

### AgenticStepProcessor Integration

```python
from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain.tools.terminal import TerminalTool

# Visual terminal for agentic reasoning
terminal = TerminalTool(
    visual_debug=True,
    verbose=True,
    session_name="agentic_development"
)

# Create agentic step with visual terminal access
agentic_step = AgenticStepProcessor(
    objective="Set up a complete web development environment",
    max_internal_steps=10,
    model_name="openai/gpt-4"
)

# Register visual terminal tool
def visual_terminal_tool(command: str) -> str:
    result = terminal(command)
    # Show progress after each command
    print("\n" + "="*50)
    terminal.show_visual_terminal(max_entries=3)
    summary = terminal.get_visual_history_summary()
    print(f"Progress: {summary['success']}/{summary['total_commands']} successful")
    print("="*50 + "\n")
    return result

# Use in chain with agentic reasoning
chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=[
        "Analyze requirements: {input}",
        agentic_step,  # Complex reasoning with visual terminal
        "Provide summary: {input}"
    ]
)

chain.register_tool_function(visual_terminal_tool)
chain.add_tools([{
    "type": "function", 
    "function": {
        "name": "visual_terminal_tool",
        "description": "Execute terminal commands with visual progress tracking",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"]
        }
    }
}])

result = chain.process_prompt("Set up Node.js development environment with testing")
```

### Multi-Agent System Integration

```python
from promptchain.utils.agent_chain import AgentChain
from promptchain.tools.terminal import TerminalTool

# Create specialized agents with visual terminals
def create_visual_agent(name: str, specialty: str):
    terminal = TerminalTool(
        visual_debug=True,
        session_name=f"{name}_session",
        verbose=True
    )
    
    def agent_terminal(command: str) -> str:
        result = terminal(command)
        print(f"\n🤖 {name} Agent Terminal:")
        terminal.show_visual_terminal(max_entries=2)
        return result
    
    agent = PromptChain(
        models=["openai/gpt-4"],
        instructions=[f"You are a {specialty} specialist. {input}"],
        verbose=True
    )
    
    agent.register_tool_function(agent_terminal)
    agent.add_tools([{
        "type": "function",
        "function": {
            "name": "agent_terminal",
            "description": f"Execute {specialty} commands with visual feedback",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            }
        }
    }])
    
    return agent

# Create multi-agent system
agent_chain = AgentChain(
    agents={
        "backend": create_visual_agent("Backend", "backend development"),
        "frontend": create_visual_agent("Frontend", "frontend development"),
        "devops": create_visual_agent("DevOps", "deployment and infrastructure")
    },
    agent_descriptions={
        "backend": "Handles server-side development and APIs",
        "frontend": "Manages client-side development and UI",
        "devops": "Manages deployment, CI/CD, and infrastructure"
    },
    execution_mode="router",
    verbose=True
)

# Each agent will use visual terminal debugging
await agent_chain.run_chat()
```

### Live Display for Real-Time Monitoring

```python
from promptchain.tools.terminal.visual_formatter import VisualTerminalFormatter
from promptchain.tools.terminal import TerminalTool
import time

# Create formatter for live monitoring
formatter = VisualTerminalFormatter()
terminal = TerminalTool()

# Monitor commands in real-time
commands = [
    "cd /tmp",
    "mkdir -p live-demo",
    "cd live-demo", 
    "git init",
    "echo '# Live Demo' > README.md",
    "git add .",
    "git commit -m 'Initial commit'"
]

# Live updating display
with formatter.get_live_display(max_entries=5) as live:
    for cmd in commands:
        # Execute command
        result = terminal(cmd)
        current_dir = terminal("pwd").strip()
        
        # Add to visual formatter
        formatter.add_command(cmd, result, current_dir)
        
        # Update live display
        live.update()
        time.sleep(1)  # Simulate processing time

print("Live demo complete!")
```

## Integration Patterns

### Pattern 1: Optional Visual Enhancement

Make visual debugging an optional feature that doesn't break existing code:

```python
def create_terminal_with_optional_visual(enable_visual: bool = False):
    """Create terminal with optional visual debugging"""
    terminal = TerminalTool(
        visual_debug=enable_visual,
        verbose=enable_visual,
        use_persistent_session=True
    )
    
    def execute_command(command: str) -> str:
        result = terminal(command)
        
        # Show visual feedback only if enabled
        if enable_visual and terminal.visual_formatter:
            terminal.show_visual_terminal(max_entries=3)
            
        return result
    
    return execute_command

# Usage
# For development: enable visual debugging
dev_terminal = create_terminal_with_optional_visual(enable_visual=True)

# For production: disable visual debugging  
prod_terminal = create_terminal_with_optional_visual(enable_visual=False)
```

### Pattern 2: Contextual Visual Feedback

Show different levels of visual detail based on context:

```python
def contextual_visual_terminal(debug_level: str = "basic"):
    """Terminal with contextual visual feedback"""
    terminal = TerminalTool(visual_debug=True, verbose=True)
    
    def execute_with_context(command: str) -> str:
        result = terminal(command)
        
        if debug_level == "detailed":
            terminal.show_visual_terminal(max_entries=10, show_timestamps=True)
            summary = terminal.get_visual_history_summary()
            print(f"📊 Stats: {summary['success']}/{summary['total_commands']} successful")
            
        elif debug_level == "basic":
            terminal.show_visual_terminal(max_entries=3)
            
        elif debug_level == "minimal":
            # Just show the last command
            if terminal.visual_formatter:
                terminal.visual_formatter.print_last_command()
        
        return result
    
    return execute_with_context
```

### Pattern 3: Error-Focused Visual Debugging

Enhance visual feedback specifically for error scenarios:

```python
def error_aware_visual_terminal():
    """Terminal with enhanced error visualization"""
    terminal = TerminalTool(visual_debug=True, verbose=True)
    
    def execute_with_error_focus(command: str) -> str:
        try:
            result = terminal(command)
            
            # Check if command failed
            if terminal.last_result and terminal.last_result.return_code != 0:
                print("🚨 Command failed! Visual history:")
                terminal.show_visual_terminal(max_entries=5)
                
                # Show error summary
                summary = terminal.get_visual_history_summary()
                if summary['errors'] > 0:
                    print(f"⚠️ Error rate: {summary['error_rate']:.1%}")
            else:
                # Success: show minimal visual feedback
                terminal.show_visual_terminal(max_entries=2)
                
            return result
            
        except Exception as e:
            print(f"💥 Exception during command execution: {e}")
            terminal.show_visual_terminal(max_entries=5)
            raise
    
    return execute_with_error_focus
```

## Best Practices

### When to Use Visual Debugging

**✅ Recommended Use Cases:**
- Development and debugging phases
- Educational demonstrations
- Complex multi-step workflows
- LLM behavior analysis
- Interactive debugging sessions
- Multi-agent system monitoring

**❌ Avoid for:**
- Production environments (unless specifically needed)
- Automated scripts without human oversight
- High-frequency command execution
- Resource-constrained environments

### Performance Considerations

```python
# Good: Enable visual debugging only when needed
terminal = TerminalTool(
    visual_debug=os.getenv("DEBUG_MODE", "false").lower() == "true",
    verbose=os.getenv("VERBOSE", "false").lower() == "true"
)

# Good: Clear history periodically for long-running processes
if len(terminal.command_history) > 100:
    terminal.clear_visual_history()

# Good: Use appropriate max_entries for display
terminal.show_visual_terminal(max_entries=5)  # Not 50
```

### Error Handling

```python
def robust_visual_terminal():
    """Terminal with robust visual debugging"""
    terminal = TerminalTool(visual_debug=True)
    
    def execute_safely(command: str) -> str:
        try:
            result = terminal(command)
            
            # Safe visual display
            if terminal.visual_formatter:
                try:
                    terminal.show_visual_terminal(max_entries=3)
                except Exception as ve:
                    print(f"Visual display warning: {ve}")
                    # Continue without visual display
            
            return result
            
        except Exception as e:
            # Show error context visually if possible
            if terminal.visual_formatter:
                try:
                    terminal.show_visual_terminal(max_entries=5)
                except:
                    pass  # Ignore visual errors
            raise
    
    return execute_safely
```

### Configuration Management

```python
# Environment-based configuration
VISUAL_DEBUG_CONFIG = {
    "development": {
        "visual_debug": True,
        "verbose": True,
        "max_entries": 10
    },
    "testing": {
        "visual_debug": True,
        "verbose": False,
        "max_entries": 5
    },
    "production": {
        "visual_debug": False,
        "verbose": False,
        "max_entries": 0
    }
}

def create_environment_terminal(env: str = "development"):
    """Create terminal configured for specific environment"""
    config = VISUAL_DEBUG_CONFIG.get(env, VISUAL_DEBUG_CONFIG["development"])
    
    return TerminalTool(
        visual_debug=config["visual_debug"],
        verbose=config["verbose"],
        use_persistent_session=True
    )
```

### Dynamic Control

```python
class DynamicVisualTerminal:
    """Terminal with runtime visual debugging control"""
    
    def __init__(self):
        self.terminal = TerminalTool(visual_debug=False)
        self._visual_enabled = False
    
    def enable_visual(self):
        """Enable visual debugging at runtime"""
        self.terminal.enable_visual_debug()
        self._visual_enabled = True
        print("✨ Visual debugging enabled")
    
    def disable_visual(self):
        """Disable visual debugging at runtime"""
        self.terminal.disable_visual_debug()
        self._visual_enabled = False
        print("🔕 Visual debugging disabled")
    
    def execute(self, command: str, show_visual: bool = None) -> str:
        """Execute command with optional visual override"""
        result = self.terminal(command)
        
        # Use parameter override or instance setting
        should_show = show_visual if show_visual is not None else self._visual_enabled
        
        if should_show and self.terminal.visual_formatter:
            self.terminal.show_visual_terminal(max_entries=3)
        
        return result

# Usage
terminal = DynamicVisualTerminal()
terminal.execute("ls")  # No visual

terminal.enable_visual()
terminal.execute("pwd")  # With visual

terminal.execute("date", show_visual=False)  # Override: no visual
```

This comprehensive documentation covers all aspects of the TerminalTool's visual debugging features, from basic usage to advanced integration patterns. The visual debugging capabilities significantly enhance the development experience when working with LLM-driven terminal interactions, providing clear insights into command execution flow and system behavior.