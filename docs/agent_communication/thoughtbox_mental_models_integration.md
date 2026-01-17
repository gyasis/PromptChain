# Thoughtbox Mental Models Integration for PromptChain
## Native Implementation of Structured Reasoning Frameworks

**Source**: [Thoughtbox MCP Server](https://github.com/Kastalien-Research/thoughtbox)  
**Purpose**: Extract mental models concepts and integrate natively into PromptChain so agents can select appropriate reasoning frameworks during task execution.

---

## Table of Contents
1. [Overview](#overview)
2. [Mental Models Catalog](#mental-models-catalog)
3. [Tag System](#tag-system)
4. [Operations](#operations)
5. [Native Integration Design](#native-integration-design)
6. [Implementation Plan](#implementation-plan)
7. [Usage Examples](#usage-examples)

---

## Overview

### Core Concept

Mental models are **process scaffolds** that tell agents **HOW to think** about problems, not **WHAT to think**. They provide structured reasoning frameworks that agents can select based on the task type.

### Key Features

- **15 Mental Models**: Structured reasoning frameworks for different problem types
- **Tag-Based Organization**: Models organized by category (debugging, planning, decision-making, etc.)
- **Dynamic Selection**: Agents can discover and select appropriate models during task execution
- **Process Scaffolds**: Each model provides step-by-step reasoning processes

### Integration Goal

Enable PromptChain agents to:
1. **Discover** relevant mental models based on task characteristics
2. **Select** appropriate reasoning framework automatically
3. **Apply** the mental model's process during task execution
4. **Switch** between models as needed during complex tasks

---

## Mental Models Catalog

### 1. **rubber-duck** - Rubber Duck Debugging
**Tags**: `debugging`, `communication`  
**Description**: Explain a problem step-by-step to find issues through articulation  
**Use Case**: When stuck on a bug or issue, explaining it out loud helps identify the problem

### 2. **five-whys** - Five Whys
**Tags**: `debugging`, `validation`  
**Description**: Iteratively ask why to drill down from symptoms to root cause  
**Use Case**: Root cause analysis, understanding why something failed

### 3. **pre-mortem** - Pre-mortem Analysis
**Tags**: `risk-analysis`, `planning`  
**Description**: Imagine failure has occurred, then work backward to identify causes  
**Use Case**: Risk assessment before starting a project, identifying potential failure points

### 4. **assumption-surfacing** - Assumption Surfacing
**Tags**: `validation`, `planning`  
**Description**: Explicitly identify and examine hidden assumptions  
**Use Case**: Before making decisions, surface and validate assumptions

### 5. **steelmanning** - Steelmanning
**Tags**: `decision-making`, `validation`  
**Description**: Present the strongest version of opposing views before deciding  
**Use Case**: Making balanced decisions, avoiding confirmation bias

### 6. **trade-off-matrix** - Trade-off Matrix
**Tags**: `decision-making`, `prioritization`  
**Description**: Map competing concerns explicitly to make balanced decisions  
**Use Case**: Choosing between options with multiple competing factors

### 7. **fermi-estimation** - Fermi Estimation
**Tags**: `estimation`  
**Description**: Make reasonable order-of-magnitude estimates with limited data  
**Use Case**: Quick estimates when precise data isn't available

### 8. **abstraction-laddering** - Abstraction Laddering
**Tags**: `architecture`, `communication`  
**Description**: Move up and down levels of abstraction to find the right framing  
**Use Case**: System design, explaining concepts at different detail levels

### 9. **decomposition** - Decomposition
**Tags**: `planning`, `architecture`  
**Description**: Break complex problems into smaller, tractable pieces  
**Use Case**: Breaking down large tasks, system architecture

### 10. **adversarial-thinking** - Adversarial Thinking
**Tags**: `risk-analysis`, `validation`  
**Description**: Adopt an attacker mindset to identify vulnerabilities  
**Use Case**: Security analysis, finding edge cases, stress testing

### 11. **opportunity-cost** - Opportunity Cost Analysis
**Tags**: `decision-making`, `prioritization`  
**Description**: Consider what you give up by choosing one option over others  
**Use Case**: Resource allocation, choosing between alternatives

### 12. **constraint-relaxation** - Constraint Relaxation
**Tags**: `planning`, `architecture`  
**Description**: Temporarily remove constraints to explore solution space, then reapply  
**Use Case**: Creative problem solving, exploring alternatives

### 13. **time-horizon-shifting** - Time Horizon Shifting
**Tags**: `planning`, `decision-making`  
**Description**: Evaluate decisions across multiple time scales (1 week, 1 year, 10 years)  
**Use Case**: Long-term planning, understanding short vs long-term implications

### 14. **impact-effort-grid** - Impact/Effort Grid
**Tags**: `prioritization`  
**Description**: Prioritize by plotting options on impact vs effort axes  
**Use Case**: Task prioritization, resource allocation

### 15. **inversion** - Inversion
**Tags**: `risk-analysis`, `planning`  
**Description**: Instead of seeking success, identify and avoid paths to failure  
**Use Case**: Risk mitigation, avoiding common pitfalls

---

## Tag System

### Tag Definitions

| Tag | Description |
|-----|-------------|
| **debugging** | Finding and fixing issues in code, logic, or systems |
| **planning** | Breaking down work, sequencing tasks, project organization |
| **decision-making** | Choosing between options under uncertainty |
| **risk-analysis** | Identifying what could go wrong and how to prevent it |
| **estimation** | Making reasonable guesses with limited information |
| **prioritization** | Deciding what to do first, resource allocation |
| **communication** | Explaining clearly to humans, documentation |
| **architecture** | System design, component relationships, structure |
| **validation** | Checking assumptions, testing hypotheses, verification |

### Model-to-Tag Mapping

```
debugging:        rubber-duck, five-whys
planning:         pre-mortem, assumption-surfacing, decomposition, constraint-relaxation, time-horizon-shifting, inversion
decision-making:  steelmanning, trade-off-matrix, opportunity-cost, time-horizon-shifting
risk-analysis:    pre-mortem, adversarial-thinking, inversion
estimation:       fermi-estimation
prioritization:   trade-off-matrix, opportunity-cost, impact-effort-grid
communication:    rubber-duck, abstraction-laddering
architecture:     abstraction-laddering, decomposition, constraint-relaxation
validation:       five-whys, assumption-surfacing, steelmanning, adversarial-thinking
```

---

## Operations

### 1. `get_model(model_name: str)`
**Purpose**: Retrieve the full prompt content for a specific mental model  
**Returns**: Complete mental model with process steps, examples, and pitfalls  
**Use Case**: Agent needs detailed reasoning framework for a specific problem

### 2. `list_models(tag: Optional[str] = None)`
**Purpose**: List all available mental models, optionally filtered by tag  
**Returns**: List of models with name, title, description, and tags  
**Use Case**: Agent discovers relevant models for current task

### 3. `list_tags()`
**Purpose**: List all available tags with their descriptions  
**Returns**: All tags and their descriptions  
**Use Case**: Agent understands available reasoning categories

### 4. `get_capability_graph()`
**Purpose**: Get structured data for knowledge graph initialization  
**Returns**: Entities and relations for all mental models  
**Use Case**: Building agent knowledge about available reasoning tools

---

## Native Integration Design

### Architecture Overview

```
Agent Task → Mental Model Selector → Selected Model → Apply Process → Task Execution
                ↓
         Tag-Based Discovery
                ↓
         Model Registry
```

### Core Components

#### 1. Mental Model Registry

```python
class MentalModelRegistry:
    """
    Registry of mental models with tag-based organization.
    """
    def __init__(self):
        self.models = {}  # model_name -> MentalModel
        self.tags = {}    # tag -> [model_names]
        self._initialize_models()
    
    def get_model(self, model_name: str) -> Optional[MentalModel]:
        """Get a specific mental model by name."""
        return self.models.get(model_name)
    
    def list_models(self, tag: Optional[str] = None) -> List[MentalModel]:
        """List models, optionally filtered by tag."""
        if tag:
            return [self.models[name] for name in self.tags.get(tag, [])]
        return list(self.models.values())
    
    def list_tags(self) -> List[Tag]:
        """List all available tags."""
        return list(self.tags.keys())
    
    def find_models_for_task(self, task_description: str, 
                            task_type: Optional[str] = None) -> List[MentalModel]:
        """
        Find relevant models based on task description and type.
        Uses keyword matching and tag inference.
        """
        # Implementation: Analyze task description and match to tags/models
        pass
```

#### 2. Mental Model Selector

```python
class MentalModelSelector:
    """
    Selects appropriate mental model based on task characteristics.
    """
    def __init__(self, registry: MentalModelRegistry, llm_model: str = "gpt-4o-mini"):
        self.registry = registry
        self.llm_model = llm_model
    
    async def select_model(self, task_description: str, 
                          context: Optional[Dict] = None) -> Optional[str]:
        """
        Use LLM to select the most appropriate mental model for the task.
        
        Returns:
            Name of selected mental model, or None if no model needed
        """
        # Get candidate models based on task keywords
        candidates = self.registry.find_models_for_task(task_description)
        
        if not candidates:
            return None
        
        # Use LLM to select best model
        selection_prompt = self._build_selection_prompt(
            task_description, candidates, context
        )
        
        # Call LLM to select model
        selected = await self._llm_select(selection_prompt)
        
        return selected
    
    def _build_selection_prompt(self, task: str, candidates: List[MentalModel], 
                                context: Dict) -> str:
        """Build prompt for LLM to select mental model."""
        candidates_str = "\n".join([
            f"- {m.name}: {m.description} (tags: {', '.join(m.tags)})"
            for m in candidates
        ])
        
        return f"""
You are selecting a mental model (reasoning framework) to help solve this task:

Task: {task}

Available mental models:
{candidates_str}

Select the most appropriate mental model for this task. Consider:
1. What type of problem is this? (debugging, planning, decision-making, etc.)
2. What reasoning approach would be most helpful?
3. What are the key challenges in this task?

Return only the model name (e.g., "five-whys" or "decomposition").
If no model is needed, return "none".
"""
```

#### 3. Mental Model Application

```python
class MentalModelApplicator:
    """
    Applies a mental model's reasoning process to a task.
    """
    def __init__(self, registry: MentalModelRegistry):
        self.registry = registry
    
    async def apply_model(self, model_name: str, task: str, 
                         agent_chain: 'AgentChain') -> str:
        """
        Apply a mental model's process to the task.
        
        This integrates with AgenticStepProcessor to guide reasoning.
        """
        model = self.registry.get_model(model_name)
        if not model:
            raise ValueError(f"Model not found: {model_name}")
        
        # Get the model's process prompt
        process_prompt = model.get_process_prompt(task)
        
        # Apply using AgenticStepProcessor or direct LLM call
        result = await agent_chain.process_prompt_async(process_prompt)
        
        return result
```

---

## Implementation Plan

### Phase 1: Core Registry (Foundation)

**File**: `promptchain/utils/mental_models.py`

```python
# promptchain/utils/mental_models.py

from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class MentalModel:
    """Represents a mental model with its process scaffold."""
    name: str
    title: str
    description: str
    tags: List[str]
    process_prompt: str  # The actual reasoning framework prompt
    examples: List[str]  # Example use cases
    pitfalls: List[str]  # Common mistakes to avoid

@dataclass
class Tag:
    """Represents a tag category."""
    name: str
    description: str

class MentalModelRegistry:
    """Registry of mental models for structured reasoning."""
    
    def __init__(self):
        self.models: Dict[str, MentalModel] = {}
        self.tags: Dict[str, List[str]] = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all 15 mental models."""
        # Define each model with its process prompt
        models = [
            MentalModel(
                name="rubber-duck",
                title="Rubber Duck Debugging",
                description="Explain a problem step-by-step to find issues through articulation",
                tags=["debugging", "communication"],
                process_prompt=self._rubber_duck_prompt(),
                examples=["Debugging code", "Troubleshooting issues"],
                pitfalls=["Skipping steps", "Not being thorough"]
            ),
            # ... (all 15 models)
        ]
        
        for model in models:
            self.models[model.name] = model
            for tag in model.tags:
                if tag not in self.tags:
                    self.tags[tag] = []
                self.tags[tag].append(model.name)
    
    def get_model(self, model_name: str) -> Optional[MentalModel]:
        """Get a specific mental model."""
        return self.models.get(model_name)
    
    def list_models(self, tag: Optional[str] = None) -> List[MentalModel]:
        """List models, optionally filtered by tag."""
        if tag:
            model_names = self.tags.get(tag, [])
            return [self.models[name] for name in model_names]
        return list(self.models.values())
    
    def list_tags(self) -> List[Tag]:
        """List all tags with descriptions."""
        tag_definitions = {
            "debugging": "Finding and fixing issues in code, logic, or systems",
            "planning": "Breaking down work, sequencing tasks, project organization",
            "decision-making": "Choosing between options under uncertainty",
            "risk-analysis": "Identifying what could go wrong and how to prevent it",
            "estimation": "Making reasonable guesses with limited information",
            "prioritization": "Deciding what to do first, resource allocation",
            "communication": "Explaining clearly to humans, documentation",
            "architecture": "System design, component relationships, structure",
            "validation": "Checking assumptions, testing hypotheses, verification",
        }
        
        return [
            Tag(name=name, description=desc)
            for name, desc in tag_definitions.items()
        ]
    
    def find_models_for_task(self, task_description: str) -> List[MentalModel]:
        """
        Find relevant models based on task keywords.
        """
        task_lower = task_description.lower()
        relevant_models = []
        
        # Keyword-based matching
        keyword_to_tags = {
            "bug": ["debugging"],
            "error": ["debugging"],
            "fix": ["debugging"],
            "plan": ["planning"],
            "design": ["planning", "architecture"],
            "decide": ["decision-making"],
            "choose": ["decision-making", "prioritization"],
            "risk": ["risk-analysis"],
            "estimate": ["estimation"],
            "prioritize": ["prioritization"],
            "explain": ["communication"],
            "architecture": ["architecture"],
            "validate": ["validation"],
            "assumption": ["validation"],
        }
        
        matched_tags = set()
        for keyword, tags in keyword_to_tags.items():
            if keyword in task_lower:
                matched_tags.update(tags)
        
        # Get models for matched tags
        for tag in matched_tags:
            if tag in self.tags:
                for model_name in self.tags[tag]:
                    model = self.models[model_name]
                    if model not in relevant_models:
                        relevant_models.append(model)
        
        return relevant_models
    
    # Model process prompts (simplified examples)
    def _rubber_duck_prompt(self) -> str:
        return """
You are using the Rubber Duck Debugging mental model.

Process:
1. Explain the problem out loud, step by step
2. Describe what you expected to happen
3. Describe what actually happened
4. Identify where the discrepancy occurs
5. Formulate hypothesis about the cause
6. Test the hypothesis

Apply this process to the task: {task}
"""
    
    def _five_whys_prompt(self) -> str:
        return """
You are using the Five Whys mental model.

Process:
1. State the problem clearly
2. Ask "Why did this happen?" - Answer and ask why again
3. Repeat 5 times (or until root cause is found)
4. Identify the root cause
5. Propose solution addressing root cause

Apply this process to the task: {task}
"""
    
    # ... (implement all 15 model prompts)
```

### Phase 2: Integration with AgentChain

**File**: `promptchain/utils/agent_chain.py` (modifications)

```python
# Add to AgentChain class

class AgentChain:
    def __init__(self, ..., enable_mental_models: bool = False, **kwargs):
        # ... existing initialization ...
        
        # Mental models integration
        self.enable_mental_models = enable_mental_models
        if enable_mental_models:
            from promptchain.utils.mental_models import MentalModelRegistry, MentalModelSelector
            self.mental_model_registry = MentalModelRegistry()
            self.mental_model_selector = MentalModelSelector(
                self.mental_model_registry,
                llm_model=kwargs.get("mental_model_selector_model", "gpt-4o-mini")
            )
            
            # Add mental model tools to agents
            self._register_mental_model_tools()
    
    def _register_mental_model_tools(self):
        """Register mental model tools with each agent."""
        for agent_name, agent in self.agents.items():
            tools = self._create_mental_model_tools(agent_name)
            agent.add_tools(tools)
    
    def _create_mental_model_tools(self, agent_name: str) -> List[Dict]:
        """Create mental model tools for an agent."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "select_mental_model",
                    "description": "Select the most appropriate mental model for the current task",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_description": {
                                "type": "string",
                                "description": "Description of the current task"
                            },
                            "task_type": {
                                "type": "string",
                                "enum": ["debugging", "planning", "decision-making", "risk-analysis", "estimation", "prioritization", "communication", "architecture", "validation"],
                                "description": "Optional: Type of task to help with model selection"
                            }
                        },
                        "required": ["task_description"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_mental_model",
                    "description": "Get the reasoning process for a specific mental model",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "enum": self.mental_model_registry.get_model_names(),
                                "description": "Name of the mental model to retrieve"
                            }
                        },
                        "required": ["model_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_mental_models",
                    "description": "List available mental models, optionally filtered by tag",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tag": {
                                "type": "string",
                                "enum": list(self.mental_model_registry.tags.keys()),
                                "description": "Optional: Filter models by tag"
                            }
                        }
                    }
                }
            }
        ]
    
    async def _handle_mental_model_tool(self, tool_name: str, arguments: Dict, 
                                       agent_name: str) -> str:
        """Handle mental model tool calls."""
        if tool_name == "select_mental_model":
            task = arguments.get("task_description")
            task_type = arguments.get("task_type")
            
            # Use selector to choose model
            selected = await self.mental_model_selector.select_model(
                task, {"task_type": task_type}
            )
            
            return f"Selected mental model: {selected}"
        
        elif tool_name == "get_mental_model":
            model_name = arguments.get("model_name")
            model = self.mental_model_registry.get_model(model_name)
            
            if model:
                return f"""
Mental Model: {model.title}
Description: {model.description}
Tags: {', '.join(model.tags)}

Process:
{model.process_prompt}

Examples: {', '.join(model.examples)}
Pitfalls to avoid: {', '.join(model.pitfalls)}
"""
            return f"Model not found: {model_name}"
        
        elif tool_name == "list_mental_models":
            tag = arguments.get("tag")
            models = self.mental_model_registry.list_models(tag=tag)
            
            result = f"Available mental models"
            if tag:
                result += f" (tag: {tag})"
            result += ":\n\n"
            
            for model in models:
                result += f"- {model.name}: {model.description} (tags: {', '.join(model.tags)})\n"
            
            return result
        
        return f"Unknown tool: {tool_name}"
```

### Phase 3: Automatic Model Selection

**Enhancement**: Auto-select mental model at task start

```python
class AgentChain:
    async def process_input(self, user_input: str, ...):
        # ... existing code ...
        
        # Auto-select mental model if enabled
        if self.enable_mental_models:
            selected_model = await self.mental_model_selector.select_model(
                user_input, context={"conversation_history": self._format_chat_history()}
            )
            
            if selected_model and selected_model != "none":
                # Apply model's reasoning process
                model = self.mental_model_registry.get_model(selected_model)
                if model:
                    # Enhance user input with mental model guidance
                    enhanced_input = f"""
Using mental model: {model.title}

{model.process_prompt.format(task=user_input)}

Now proceed with the task: {user_input}
"""
                    user_input = enhanced_input
        
        # Continue with normal processing
        # ...
```

---

## Usage Examples

### Example 1: Automatic Model Selection

```python
from promptchain import PromptChain
from promptchain.utils.agent_chain import AgentChain

# Create agent with mental models enabled
agent = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Solve: {input}"]
)

agent_chain = AgentChain(
    agents={"solver": agent},
    agent_descriptions={"solver": "Solves problems using structured reasoning"},
    enable_mental_models=True,  # Enable mental model integration
    execution_mode="router"
)

# Agent automatically selects appropriate model
result = await agent_chain.process_input_async(
    "I need to debug why my API is returning 500 errors"
)
# Agent will likely select "five-whys" or "rubber-duck" model
```

### Example 2: Manual Model Selection via Tools

```python
# Agent can use tools to discover and select models
# Agent's internal reasoning:
# "I need to plan a complex project. Let me find relevant mental models."

# Agent calls: list_mental_models(tag="planning")
# Returns: decomposition, pre-mortem, constraint-relaxation, etc.

# Agent calls: get_mental_model(model_name="decomposition")
# Gets the full reasoning process

# Agent applies the model to plan the project
```

### Example 3: Model Switching During Task

```python
# Agent starts with one model, switches as needed
# Task: "Design and implement a secure authentication system"

# Step 1: Use "decomposition" to break down the task
# Step 2: Use "adversarial-thinking" to identify security vulnerabilities
# Step 3: Use "pre-mortem" to identify potential failure points
# Step 4: Use "trade-off-matrix" to decide between implementation options
```

---

## Benefits of Native Integration

1. **No External Dependencies**: Works without MCP server setup
2. **Seamless Integration**: Mental models are part of agent's toolset
3. **Automatic Selection**: Agents can auto-select models based on task
4. **Flexible Application**: Models can be applied at any point in task execution
5. **Extensible**: Easy to add new mental models or customize existing ones

---

## Next Steps

1. **Implement Core Registry**: Create `mental_models.py` with all 15 models
2. **Create Process Prompts**: Write detailed reasoning frameworks for each model
3. **Integrate with AgentChain**: Add mental model tools and auto-selection
4. **Test with Examples**: Verify models are selected and applied correctly
5. **Document Usage**: Create examples showing mental models in action

---

## References

- **Thoughtbox Repository**: https://github.com/Kastalien-Research/thoughtbox
- **Mental Models Operations**: `src/mental-models/operations.ts`
- **Model Definitions**: `src/mental-models/contents/*.js`

---

**Status**: Design document - Ready for implementation

