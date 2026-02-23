# Prompts Folder Research Summary

**Date**: 2025-11-20
**Status**: ✅ RESEARCH COMPLETE
**Purpose**: Research existing prompt infrastructure and design CLI integration

## Research Findings

### 1. Existing Prompt Infrastructure

PromptChain has **two separate prompt management systems**:

#### **A. PrePrompt System** (`promptchain/utils/preprompt.py`) - PRIMARY
**Purpose**: Production-ready prompt template loader for PromptChain

**Key Features**:
- ✅ Load prompts by ID (filename without extension)
- ✅ Strategy application (e.g., `analysis:concise`)
- ✅ Custom directory support with prioritization
- ✅ Supports `.txt`, `.md`, `.xml`, `.json`
- ✅ Already integrated with PromptChain and AgentChain

**How It Works**:
```python
from promptchain.utils.preprompt import PrePrompt

# Initialize with custom directories
preprompt = PrePrompt(additional_prompt_dirs=["/path/to/custom/prompts"])

# Load prompt by ID
prompt = preprompt.load("analysis")              # From analysis.md
prompt = preprompt.load("analysis:concise")      # With concise strategy
```

**Standard Locations**:
- **Prompts**: `promptchain/prompts/` (library installation, currently doesn't exist)
- **Strategies**: `promptchain/prompts/strategies/` (JSON files with `"prompt"` key)

**Integration Points**:
- `PromptChain`: Uses via `additional_prompt_dirs` parameter (line 140 in promptchaining.py)
- `AgentChain`: Passes through to router (line 104 in agent_chain.py)

#### **B. Prompt Loader** (`promptchain/utils/prompt_loader.py`) - UTILITY
**Purpose**: Directory scanning utility for prompt discovery

**Key Functions**:
- `load_prompts(prompts_dir)` - Walk directory and load all .md files
- `get_prompt_by_name(name)` - Get specific prompt content
- `list_available_prompts()` - List all prompts organized by category

**How It Works**:
```python
from promptchain.utils.prompt_loader import list_available_prompts

# Scan directory structure
prompts = list_available_prompts(prompts_dir="/path/to/prompts")
# Returns: {
#   "research": [{"name": "ANALYSIS", "description": "...", "path": "..."}],
#   "coding": [{"name": "REVIEW", "description": "...", "path": "..."}]
# }
```

**Not Currently Integrated**: Available but unused in production code

### 2. Current CLI Status

**What Works**:
- ✅ AgentChain has `additional_prompt_dirs` parameter
- ✅ Can pass custom prompt directories to agents
- ✅ PrePrompt system is production-ready

**What's Missing**:
- ❌ No prompts directory in `~/.promptchain/`
- ❌ No CLI commands for prompt management
- ❌ No agent-prompt association in TUI
- ❌ No built-in prompt templates
- ❌ No easy way to save/load prompts for agents

### 3. Integration Opportunities

**PrePrompt is the right choice** because:
1. Already integrated with PromptChain and AgentChain
2. Production-tested and stable
3. Supports strategies (reusable instruction prefixes)
4. Has directory prioritization (custom overrides standard)
5. Clean API for loading prompts

**Prompt Loader is useful for**:
1. CLI discovery of available prompts
2. Listing prompts by category
3. Extracting descriptions from files

## Proposed Architecture

### Directory Structure
```
~/.promptchain/
├── prompts/              # NEW: User prompts
│   ├── agents/           # Agent-specific prompts
│   │   ├── researcher.md
│   │   ├── coder.md
│   │   └── analyst.md
│   ├── strategies/       # Strategy templates
│   │   ├── concise.json
│   │   ├── detailed.json
│   │   └── step-by-step.json
│   └── custom/           # User custom prompts
└── templates/            # NEW: Agent templates
    └── agents/
        ├── research-agent.json
        └── code-review-agent.json
```

### CLI Commands

**Prompt Management**:
```bash
/prompt list                    # List all prompts
/prompt show researcher         # Display prompt content
/prompt save my_prompt          # Save custom prompt
/agent create-from-prompt researcher    # Create agent from prompt
```

**Template Management**:
```bash
/template list                  # List agent templates
/template use research-agent    # Create agent from template
/template save researcher       # Save agent as template
```

### PromptManager Class

**New Component**: `promptchain/cli/prompt_manager.py`

**Responsibilities**:
- Initialize PrePrompt with custom directories
- Manage prompt/strategy/template CRUD operations
- Provide CLI interface to prompt system
- Integrate with session management

**Key Methods**:
```python
class PromptManager:
    def list_prompts(self, category: Optional[str] = None) -> List[Prompt]
    def load_prompt(self, prompt_id: str, strategy: Optional[str] = None) -> str
    def save_prompt(self, prompt_id: str, content: str, category: str) -> Path
    def list_strategies(self) -> List[Strategy]
    def list_templates(self, tag: Optional[str] = None) -> List[AgentTemplate]
    def save_template(self, template: AgentTemplate) -> Path
```

### Data Models

**New Models** (`promptchain/cli/models/prompt.py`):
- `Prompt` - Represents a prompt template
- `Strategy` - Represents a strategy (instruction prefix)
- `AgentTemplate` - Represents a pre-configured agent

**Extended Agent Model**:
```python
@dataclass
class Agent:
    # ... existing fields ...
    prompt_id: Optional[str] = None      # NEW: Reference to prompt
    strategy_id: Optional[str] = None    # NEW: Reference to strategy
```

## Implementation Roadmap

### Phase 1: Core Infrastructure
1. Create `PromptManager` class
2. Create data models (Prompt, Strategy, AgentTemplate)
3. Implement directory structure creation
4. Add PrePrompt integration

### Phase 2: Built-in Content
1. Create initial agent prompts (researcher, coder, analyst)
2. Create initial strategies (concise, detailed, step-by-step)
3. Create initial templates (3-5 agent types)

### Phase 3: CLI Commands
1. Implement `/prompt` commands
2. Implement `/template` commands
3. Integrate with `CommandHandler`

### Phase 4: TUI Integration
1. Integrate PromptManager with TUI app
2. Update agent creation to support prompts
3. Update session persistence

### Phase 5: Testing & Documentation
1. Write integration tests
2. Create user documentation
3. Test end-to-end workflows

## Next Steps

1. **Review Design Document**: See `docs/cli/prompts-integration-design.md` for full details
2. **Create Built-in Prompts**: Write initial researcher, coder, analyst prompts
3. **Implement PromptManager**: Build core prompt management class
4. **Add CLI Commands**: Implement `/prompt` and `/template` commands
5. **Test Integration**: Verify end-to-end workflows

## Key Technical Details

### PrePrompt Priority Order
1. `additional_prompt_dirs[0]` (first custom directory)
2. `additional_prompt_dirs[1]` (second custom directory)
3. ... (remaining custom directories)
4. `standard_prompt_dir` (library default)

**First match wins**: If `researcher.md` exists in both custom and standard, custom is used.

### Strategy Format (JSON)
```json
{
  "name": "Concise",
  "prompt": "Provide brief, bullet-point responses.",
  "description": "Strategy description"
}
```

### Agent Template Format (JSON)
```json
{
  "name": "Research Agent",
  "model": "gpt-4",
  "prompt_id": "researcher",
  "strategy": "detailed",
  "description": "Comprehensive research agent",
  "tags": ["research", "analysis"]
}
```

## Benefits

1. **Quick Setup**: Create specialized agents from templates in seconds
2. **Reusability**: Save and reuse effective prompts
3. **Consistency**: Standardized prompt structure
4. **Flexibility**: Mix prompts with strategies
5. **Integration**: Leverages existing PrePrompt infrastructure

## Files Referenced

- `promptchain/utils/preprompt.py` - Main prompt loading system
- `promptchain/utils/prompt_loader.py` - Directory scanning utility
- `promptchain/utils/agent_chain.py` - Line 104: `additional_prompt_dirs` parameter
- `promptchain/utils/promptchaining.py` - Line 140: `additional_prompt_dirs` parameter
- `docs/preprompt_usage.md` - PrePrompt documentation
- `docs/AgentChain_Usage.md` - AgentChain parameter reference

## Conclusion

✅ **Research Complete**

PromptChain has a robust prompt infrastructure via `PrePrompt` that's already integrated with the core library. The CLI needs:

1. **PromptManager** class to bridge PrePrompt with CLI
2. **CLI commands** for prompt/template management
3. **Built-in templates** for common agent types
4. **TUI integration** for agent creation from prompts

The design is straightforward because PrePrompt handles the heavy lifting. We just need to expose it via CLI commands and provide a curated template library.

---

*Prompts Research Summary | 2025-11-20 | CLI Enhancement Research*
