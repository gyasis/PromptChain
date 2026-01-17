# CLI Prompts Integration Design

**Date**: 2025-11-20
**Status**: 🟡 DESIGN PHASE
**Purpose**: Enable quick access, setup, and management of stored prompts for CLI agents

## Overview

This document outlines the design for integrating PromptChain's existing prompt infrastructure (`PrePrompt` and `prompt_loader`) with the CLI's agent management system to enable:

1. **Quick Agent Setup**: Create agents from pre-made prompt templates
2. **Prompt Management**: Save, load, list, and organize custom agent prompts
3. **Template Library**: Access built-in prompts for common agent types
4. **CLI Commands**: Intuitive slash commands for prompt operations

## Existing Infrastructure Analysis

### 1. PrePrompt System (`promptchain/utils/preprompt.py`)

**Purpose**: Load prompt templates from directories with strategy support

**Key Features**:
- **Prompt Loading by ID**: Filenames without extension (e.g., `analysis.md` → ID: `analysis`)
- **Strategy Application**: Prepend reusable instructions (e.g., `analysis:concise`)
- **Directory Prioritization**: Custom directories override standard library prompts
- **Supported Extensions**: `.txt`, `.md`, `.xml`, `.json`

**Architecture**:
```python
class PrePrompt:
    def __init__(self, additional_prompt_dirs: Optional[List[str]] = None):
        # Searches in order:
        # 1. additional_prompt_dirs (custom, user-defined)
        # 2. standard_prompt_dir (library: promptchain/prompts/)

    def load(self, promptID_with_strategy: str) -> str:
        # Examples:
        # - "analysis" → loads prompts/analysis.md
        # - "analysis:concise" → loads prompts/analysis.md + strategies/concise.json
```

**Standard Locations**:
- **Prompts**: `promptchain/prompts/` (relative to library installation)
- **Strategies**: `promptchain/prompts/strategies/` (JSON files with `"prompt"` key)

**Current Integration**:
- ✅ Used by `PromptChain` via `additional_prompt_dirs` parameter
- ✅ Used by `AgentChain` (passes to router configuration)
- ❌ **NOT** used by CLI TUI agent system yet

### 2. Prompt Loader (`promptchain/utils/prompt_loader.py`)

**Purpose**: Automatically discover and load prompts from directory structures

**Key Functions**:
```python
def load_prompts(prompts_dir: str = "src/prompts") -> Dict[str, Tuple[str, str]]:
    # Returns: {VAR_NAME: (category, content)}
    # Example: {"RESEARCH_ANALYSIS": ("research", "Analyze research papers...")}

def get_prompt_by_name(name: str, prompts_dir: Optional[str] = None) -> str:
    # Get prompt content by variable name

def list_available_prompts(prompts_dir: Optional[str] = None) -> dict:
    # Returns organized structure:
    # {
    #   "research": [{"name": "ANALYSIS", "description": "...", "path": "..."}],
    #   "coding": [{"name": "REVIEW", "description": "...", "path": "..."}]
    # }
```

**Features**:
- Walks directory structure for `.md` files
- Creates uppercase variable names from filenames
- Organizes by category (directory structure)
- Extracts descriptions from file content

**Current Integration**:
- ✅ Available as utility module
- ❌ **NOT** integrated with CLI or AgentChain

### 3. Integration Points

**PromptChain**:
```python
chain = PromptChain(
    models=["gpt-4"],
    instructions=[
        "analysis",              # Loaded via PrePrompt
        "analysis:concise",      # Loaded via PrePrompt with strategy
        "./custom_prompt.txt"    # Loaded as file path
    ],
    additional_prompt_dirs=["/path/to/custom/prompts"],
    verbose=True
)
```

**AgentChain** (passes through to PromptChain):
```python
agent_chain = AgentChain(
    agents={"researcher": researcher_agent},
    additional_prompt_dirs=["/path/to/prompts"]  # Line 104 in agent_chain.py
)
```

## Current Gaps

### What's Missing for CLI Integration:

1. **No Prompts Directory in CLI**: The CLI doesn't have a dedicated prompts directory
2. **No Agent-Prompt Association**: Agents created in CLI don't reference prompts
3. **No Prompt Management Commands**: Can't save/load/list prompts via CLI
4. **No Template Library**: No pre-made agent prompt templates
5. **No CLI Integration**: PrePrompt system not connected to TUI agent creation

## Proposed Design

### 1. Directory Structure

```
~/.promptchain/
├── sessions/              # Existing: Session storage
│   └── sessions.db
├── logs/                  # Existing: RunLogger output
├── prompts/              # NEW: User prompts directory
│   ├── agents/           # NEW: Agent-specific prompts
│   │   ├── researcher.md
│   │   ├── coder.md
│   │   ├── analyst.md
│   │   └── writer.md
│   ├── strategies/       # NEW: Strategy templates
│   │   ├── concise.json
│   │   ├── detailed.json
│   │   └── step-by-step.json
│   └── custom/           # NEW: User custom prompts
│       └── my_prompt.md
└── templates/            # NEW: Built-in template library
    └── agents/
        ├── research-agent.json
        ├── code-review-agent.json
        └── data-analyst-agent.json
```

### 2. Prompt File Format

**Agent Prompt** (`.md` file):
```markdown
# Agent: Researcher

You are a research specialist focused on gathering and analyzing information.

## Capabilities
- Web search and information gathering
- Document analysis and summarization
- Citation and source verification

## Instructions
{instructions}

## Context
{context}
```

**Strategy** (`.json` file):
```json
{
  "name": "Concise",
  "prompt": "Provide concise, bullet-point responses. Focus on key information without elaboration.",
  "description": "Strategy for brief, focused outputs"
}
```

**Agent Template** (`.json` file):
```json
{
  "name": "Research Agent",
  "model": "gpt-4",
  "prompt_id": "researcher",
  "strategy": "detailed",
  "description": "Specialized agent for research and analysis tasks",
  "tags": ["research", "analysis", "information-gathering"]
}
```

### 3. CLI Commands

#### Prompt Management Commands

**List Available Prompts**:
```bash
/prompt list                    # List all available prompts
/prompt list --category agents  # List agent prompts only
/prompt list --search research  # Search prompts by keyword
```

**Show Prompt Content**:
```bash
/prompt show researcher         # Display researcher.md content
/prompt show researcher:concise # Display with strategy applied
```

**Save Current Agent Prompt**:
```bash
/prompt save <name>                    # Save active agent's prompt
/prompt save researcher --category agents  # Save to specific category
```

**Create Agent from Prompt**:
```bash
/agent create-from-prompt researcher    # Create agent from researcher.md
/agent create researcher --prompt researcher:concise  # With strategy
```

**Edit Prompt**:
```bash
/prompt edit researcher         # Open in editor (if available)
/prompt create my_prompt        # Create new blank prompt
```

#### Template Management Commands

**List Templates**:
```bash
/template list                  # List all agent templates
/template list --tag research   # Filter by tag
```

**Create Agent from Template**:
```bash
/template use research-agent            # Create agent from template
/template use research-agent --name my-researcher  # Custom name
```

**Save Agent as Template**:
```bash
/template save researcher               # Save current agent config
```

### 4. Data Models

**Prompt Model** (`promptchain/cli/models/prompt.py`):
```python
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Prompt:
    """Represents a prompt template."""
    id: str                      # Filename without extension
    content: str                 # Full prompt text
    category: str                # Directory category (e.g., 'agents')
    description: Optional[str]   # Extracted from file
    path: str                    # Full file path
    strategies: List[str]        # Available strategies for this prompt

@dataclass
class Strategy:
    """Represents a prompt strategy."""
    id: str                      # Filename without extension
    name: str                    # Display name
    prompt: str                  # Strategy instruction text
    description: Optional[str]   # Strategy description

@dataclass
class AgentTemplate:
    """Represents a pre-configured agent template."""
    name: str
    model: str
    prompt_id: str
    strategy: Optional[str]
    description: str
    tags: List[str]
```

**Extended Agent Model**:
```python
# In promptchain/cli/models/agent.py
@dataclass
class Agent:
    name: str
    model_name: str
    description: str
    prompt_id: Optional[str] = None      # NEW: Reference to prompt
    strategy_id: Optional[str] = None    # NEW: Reference to strategy
    # ... existing fields ...
```

### 5. Prompt Manager Class

**Location**: `promptchain/cli/prompt_manager.py`

```python
from typing import List, Dict, Optional
from pathlib import Path
from promptchain.utils.preprompt import PrePrompt
from promptchain.utils.prompt_loader import list_available_prompts
from promptchain.cli.models.prompt import Prompt, Strategy, AgentTemplate

class PromptManager:
    """Manages prompts, strategies, and templates for CLI agents."""

    def __init__(self, prompts_dir: Path):
        self.prompts_dir = prompts_dir
        self.agents_dir = prompts_dir / "agents"
        self.strategies_dir = prompts_dir / "strategies"
        self.custom_dir = prompts_dir / "custom"
        self.templates_dir = prompts_dir.parent / "templates" / "agents"

        # Create directories if they don't exist
        self._ensure_directories()

        # Initialize PrePrompt with custom directories
        self.preprompt = PrePrompt(
            additional_prompt_dirs=[
                str(self.agents_dir),
                str(self.custom_dir)
            ]
        )

    def _ensure_directories(self):
        """Create prompts directory structure if it doesn't exist."""
        for directory in [self.agents_dir, self.strategies_dir,
                         self.custom_dir, self.templates_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def list_prompts(self, category: Optional[str] = None) -> List[Prompt]:
        """List all available prompts, optionally filtered by category."""
        all_prompts = list_available_prompts(str(self.prompts_dir))
        prompts = []

        for cat, prompt_list in all_prompts.items():
            if category and cat != category:
                continue
            for p in prompt_list:
                prompts.append(Prompt(
                    id=p['name'].lower(),
                    content=self._load_prompt_content(p['path']),
                    category=cat,
                    description=p['description'],
                    path=p['path'],
                    strategies=self.list_strategies()
                ))

        return prompts

    def load_prompt(self, prompt_id: str, strategy: Optional[str] = None) -> str:
        """Load prompt content using PrePrompt system."""
        instruction = f"{prompt_id}:{strategy}" if strategy else prompt_id
        return self.preprompt.load(instruction)

    def save_prompt(self, prompt_id: str, content: str, category: str = "custom") -> Path:
        """Save a new prompt to the specified category."""
        category_dir = self.prompts_dir / category
        category_dir.mkdir(exist_ok=True)

        prompt_path = category_dir / f"{prompt_id}.md"
        prompt_path.write_text(content, encoding='utf-8')

        # Rescan prompts
        self.preprompt._scan_all_prompt_dirs()

        return prompt_path

    def list_strategies(self) -> List[Strategy]:
        """List all available strategies."""
        strategies = []
        if not self.strategies_dir.exists():
            return strategies

        for strategy_file in self.strategies_dir.glob("*.json"):
            import json
            data = json.loads(strategy_file.read_text())
            strategies.append(Strategy(
                id=strategy_file.stem,
                name=data.get('name', strategy_file.stem),
                prompt=data.get('prompt', ''),
                description=data.get('description')
            ))

        return strategies

    def list_templates(self, tag: Optional[str] = None) -> List[AgentTemplate]:
        """List all agent templates, optionally filtered by tag."""
        templates = []
        if not self.templates_dir.exists():
            return templates

        for template_file in self.templates_dir.glob("*.json"):
            import json
            data = json.loads(template_file.read_text())
            template = AgentTemplate(**data)

            if tag and tag not in template.tags:
                continue

            templates.append(template)

        return templates

    def save_template(self, template: AgentTemplate) -> Path:
        """Save an agent template."""
        import json
        template_path = self.templates_dir / f"{template.name.lower().replace(' ', '-')}.json"
        template_path.write_text(
            json.dumps(template.__dict__, indent=2),
            encoding='utf-8'
        )
        return template_path

    def _load_prompt_content(self, path: str) -> str:
        """Load prompt file content."""
        return Path(path).read_text(encoding='utf-8')
```

### 6. TUI Integration

**Modified Files**:
1. `promptchain/cli/tui/app.py` - Add prompt commands
2. `promptchain/cli/command_handler.py` - Add prompt/template command handlers
3. `promptchain/cli/session_manager.py` - Store prompt references with agents

**Example Integration in `command_handler.py`**:
```python
class CommandHandler:
    def __init__(self, session_manager, prompt_manager):
        self.session_manager = session_manager
        self.prompt_manager = prompt_manager  # NEW
        # ... existing code ...

    async def handle_prompt_list(self, args: List[str]) -> str:
        """Handle /prompt list command."""
        # Parse args for --category, --search filters
        prompts = self.prompt_manager.list_prompts(category=category)

        # Format output
        output = "Available Prompts:\n"
        for prompt in prompts:
            output += f"  {prompt.id} ({prompt.category}): {prompt.description}\n"

        return output

    async def handle_agent_create_from_prompt(self, args: List[str]) -> str:
        """Handle /agent create-from-prompt command."""
        prompt_id = args[0]
        strategy = args[1] if len(args) > 1 else None

        # Load prompt
        prompt_content = self.prompt_manager.load_prompt(prompt_id, strategy)

        # Create agent with prompt
        agent_name = prompt_id
        agent = Agent(
            name=agent_name,
            model_name="gpt-4",
            description=f"Agent created from prompt: {prompt_id}",
            prompt_id=prompt_id,
            strategy_id=strategy
        )

        # Add to session
        self.session_manager.add_agent(agent)

        return f"Created agent '{agent_name}' from prompt '{prompt_id}'"
```

### 7. Built-in Templates

**Initial Template Library** (`~/.promptchain/templates/agents/`):

1. **research-agent.json**:
```json
{
  "name": "Research Agent",
  "model": "gpt-4",
  "prompt_id": "researcher",
  "strategy": "detailed",
  "description": "Comprehensive research and analysis agent",
  "tags": ["research", "analysis", "web-search"]
}
```

2. **code-review-agent.json**:
```json
{
  "name": "Code Reviewer",
  "model": "claude-3-opus-20240229",
  "prompt_id": "code_reviewer",
  "strategy": "step-by-step",
  "description": "Code review specialist with security focus",
  "tags": ["coding", "security", "review"]
}
```

3. **data-analyst-agent.json**:
```json
{
  "name": "Data Analyst",
  "model": "gpt-4",
  "prompt_id": "analyst",
  "strategy": "concise",
  "description": "Data analysis and visualization specialist",
  "tags": ["data", "analysis", "statistics"]
}
```

### 8. Built-in Prompts

**Initial Prompt Library** (`~/.promptchain/prompts/agents/`):

1. **researcher.md**:
```markdown
# Research Agent

You are a specialized research agent focused on gathering, analyzing, and synthesizing information.

## Core Capabilities
- Comprehensive web search and information gathering
- Academic and technical research
- Source verification and citation
- Data synthesis and summarization

## Guidelines
- Always cite sources when providing information
- Distinguish between facts and opinions
- Present multiple perspectives when relevant
- Highlight limitations or gaps in available information

## Output Format
Provide structured responses with:
1. Summary of findings
2. Key insights and analysis
3. Sources and citations
4. Additional context or recommendations

{context}
```

2. **code_reviewer.md**:
```markdown
# Code Review Agent

You are a code review specialist focused on identifying issues and suggesting improvements.

## Review Focus Areas
- Code quality and maintainability
- Security vulnerabilities (OWASP Top 10)
- Performance optimization opportunities
- Best practices and design patterns
- Test coverage and edge cases

## Review Process
1. Analyze code structure and organization
2. Identify potential bugs and security issues
3. Suggest specific improvements with examples
4. Prioritize findings by severity

## Output Format
- **Critical**: Security vulnerabilities, breaking bugs
- **High**: Major code quality issues, performance problems
- **Medium**: Refactoring opportunities, style inconsistencies
- **Low**: Minor improvements, suggestions

{context}
```

3. **analyst.md**:
```markdown
# Data Analysis Agent

You are a data analyst specialized in extracting insights from datasets.

## Analysis Capabilities
- Statistical analysis and hypothesis testing
- Data visualization recommendations
- Trend identification and pattern recognition
- Anomaly detection
- Predictive modeling suggestions

## Analysis Process
1. Data exploration and profiling
2. Statistical summary and distributions
3. Correlation and relationship analysis
4. Key findings and insights
5. Actionable recommendations

## Output Format
Provide structured analysis with:
- Data summary statistics
- Key findings (bullet points)
- Visualizations or charts (described)
- Recommendations for further analysis

{context}
```

### 9. Built-in Strategies

**Initial Strategy Library** (`~/.promptchain/prompts/strategies/`):

1. **concise.json**:
```json
{
  "name": "Concise",
  "prompt": "Provide concise, bullet-point responses. Focus on key information without elaboration. Maximum 5 bullet points per section.",
  "description": "Brief, focused outputs with essential information only"
}
```

2. **detailed.json**:
```json
{
  "name": "Detailed",
  "prompt": "Provide comprehensive, detailed responses. Include context, examples, and thorough explanations. Use multiple paragraphs and structured sections.",
  "description": "In-depth, comprehensive responses with full context"
}
```

3. **step-by-step.json**:
```json
{
  "name": "Step-by-Step",
  "prompt": "Break down your response into clear, numbered steps. Explain each step thoroughly and show the progression from start to finish.",
  "description": "Sequential, instructional format with numbered steps"
}
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)
- [ ] Create PromptManager class
- [ ] Create Prompt, Strategy, AgentTemplate models
- [ ] Implement directory structure creation
- [ ] Add PrePrompt integration
- [ ] Write unit tests for PromptManager

### Phase 2: Built-in Content (Week 1)
- [ ] Create initial agent prompts (researcher, coder, analyst)
- [ ] Create initial strategies (concise, detailed, step-by-step)
- [ ] Create initial templates (3-5 common agent types)
- [ ] Test prompt loading and strategy application

### Phase 3: CLI Commands (Week 2)
- [ ] Implement `/prompt list` command
- [ ] Implement `/prompt show` command
- [ ] Implement `/prompt save` command
- [ ] Implement `/agent create-from-prompt` command
- [ ] Implement `/template list` command
- [ ] Implement `/template use` command

### Phase 4: TUI Integration (Week 2)
- [ ] Integrate PromptManager with TUI app
- [ ] Add prompt references to Agent model
- [ ] Update agent creation flow to support prompts
- [ ] Update session persistence to store prompt IDs
- [ ] Add UI feedback for prompt operations

### Phase 5: Testing & Documentation (Week 3)
- [ ] Write integration tests for prompt commands
- [ ] Create user documentation with examples
- [ ] Test end-to-end workflows
- [ ] Create CLI usage guide for prompts

## Benefits

1. **Quick Agent Setup**: Create specialized agents from templates in seconds
2. **Reusability**: Save and reuse effective prompts across sessions
3. **Consistency**: Standardized prompt structure across agents
4. **Flexibility**: Mix and match prompts with strategies
5. **Shareability**: Export/import templates between users
6. **Maintainability**: Centralized prompt management

## Future Enhancements

1. **Prompt Versioning**: Track prompt versions and changes
2. **Prompt Marketplace**: Share and download community prompts
3. **Dynamic Prompts**: Variables and conditional sections
4. **Prompt Testing**: Test prompts against sample inputs
5. **Prompt Analytics**: Track which prompts perform best
6. **AI-Assisted Prompt Creation**: Generate prompts from descriptions

---

*Prompts Integration Design | 2025-11-20 | CLI Enhancement Initiative*
