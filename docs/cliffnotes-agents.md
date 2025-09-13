# AgenticStepProcessor vs Regular Agent Classes in PromptChain

Based on my analysis of the codebase, I'll explain how the AgenticStepProcessor works, how it differs from regular agent classes, and what makes it unique in the PromptChain framework.

## 🧠 How the AgenticStepProcessor Works

The **AgenticStepProcessor** is a specialized component that enables agentic behavior within a single PromptChain step, implementing an internal loop for complex reasoning and tool usage:

### **Core Architecture**
```python
class AgenticStepProcessor:
    def __init__(self, objective: str, max_internal_steps: int = 5, 
                 model_name: str = None, model_params: Dict[str, Any] = None):
        self.objective = objective
        self.max_internal_steps = max_internal_steps
        self.model_name = model_name
        self.model_params = model_params or {}
```

### **Internal Agentic Loop**

1. **Objective-Driven Processing**
   - Works toward a specific defined objective
   - Runs internal loop for up to `max_internal_steps` iterations
   - Makes multiple LLM calls and tool invocations within a single step

2. **Tool Integration**
   - Seamlessly uses tools registered on the parent PromptChain
   - Supports both local tools and MCP (Model Context Protocol) tools
   - Automatic tool discovery and execution

3. **Self-Contained Model Configuration**
   - Specifies its own `model_name` and `model_params`
   - Independent of the parent chain's model configuration
   - Optimized for agentic reasoning tasks

### **Key Agentic Features**

- **Internal Reasoning Loop**: Multiple LLM calls within one step
- **Tool Usage**: Automatic tool selection and execution
- **Objective-Driven**: Works toward specific goals
- **Model Independence**: Self-contained model configuration
- **Iterative Processing**: Refines output through multiple iterations

## 🔄 How It Differs from Regular Agent Classes

### **Regular Agent (Simple PromptChain)**
```python
class PromptChain:
    def __init__(self, models, instructions, ...):
        # Single instruction execution per step
        # Linear processing flow
        # Basic model management
```

**Characteristics:**
- **Single Step Execution**: One instruction, one LLM call
- **Linear Processing**: Sequential instruction execution
- **Simple State**: Basic input/output flow
- **Direct Execution**: Input → Single Process → Output
- **No Internal Loops**: Each step executes once

### **AgenticStepProcessor**
```python
class AgenticStepProcessor:
    def __init__(self, objective, max_internal_steps, model_name, model_params):
        # Internal agentic loop
        # Multiple LLM calls and tool executions
        # Objective-driven reasoning
```

**Characteristics:**
- **Internal Agentic Loop**: Multiple iterations within one step
- **Complex Reasoning**: Multi-step reasoning and tool usage
- **Objective-Driven**: Works toward specific goals
- **Tool Integration**: Automatic tool selection and execution
- **Iterative Refinement**: Improves output through multiple passes

### **Key Differences**

| Aspect | Regular Agent | AgenticStepProcessor |
|--------|---------------|---------------------|
| **Execution** | Single step, single call | Internal loop, multiple calls |
| **Processing** | Linear, sequential | Iterative, reasoning-based |
| **Tools** | Manual tool usage | Automatic tool integration |
| **Model Config** | Inherits from chain | Self-contained configuration |
| **Complexity** | Simple input/output | Complex reasoning and planning |
| **Use Case** | Basic tasks | Complex, multi-step reasoning |

## 🎯 AgenticStepProcessor Capabilities

### **Core Capabilities**

#### **Internal Agentic Loop**
```python
# AgenticStepProcessor runs an internal loop
async def run_async(self, initial_input, available_tools, llm_runner, tool_executor):
    # Runs up to max_internal_steps iterations
    # Each iteration can make LLM calls and tool executions
    # Works toward achieving the objective
```

#### **Tool Integration**
```python
# Automatically uses tools from parent PromptChain
calculator_schema = {
    "type": "function",
    "function": {
        "name": "simple_calculator",
        "description": "Evaluates mathematical expressions"
    }
}

agentic_step = AgenticStepProcessor(
    objective="You are a math assistant. Use tools when needed.",
    max_internal_steps=3,
    model_name="openai/gpt-4o",
    model_params={"tool_choice": "auto"}
)
```

#### **Objective-Driven Processing**
```python
# Works toward specific objectives
agentic_step = AgenticStepProcessor(
    objective="Generate comprehensive research questions for multi-query processing",
    max_internal_steps=5,
    model_name="openai/gpt-4o"
)
```

### **Advanced Features**

#### **Multi-Step Reasoning**
- **Reasoning**: Analyze current state and plan next actions
- **Acting**: Execute tools and gather information
- **Observing**: Evaluate results and refine approach
- **Iterating**: Continue until objective is achieved

#### **Tool Coordination**
- **Automatic Tool Discovery**: Finds available tools from parent chain
- **Intelligent Tool Selection**: Chooses appropriate tools based on context
- **Tool Result Integration**: Incorporates tool outputs into reasoning
- **Error Handling**: Manages tool failures gracefully

#### **Model Independence**
- **Self-Contained Configuration**: Specifies its own model and parameters
- **Optimized for Agentic Tasks**: Uses models suited for reasoning
- **Parameter Tuning**: Configurable temperature, tool choice, etc.
- **Model Switching**: Can use different models for different agentic steps

### **Real-World Example**

```python
# Regular PromptChain (simple)
simple_chain = PromptChain(
    models=["openai/gpt-4o"],
    instructions=[
        "Analyze this research topic: {input}",
        "Generate a summary: {input}"
    ]
)

# PromptChain with AgenticStepProcessor (advanced)
agentic_chain = PromptChain(
    models=[],  # No models needed for other steps
    instructions=[
        "Prepare research context: {input}",
        AgenticStepProcessor(
            objective="Conduct comprehensive research analysis using available tools",
            max_internal_steps=5,
            model_name="openai/gpt-4o",
            model_params={"tool_choice": "auto", "temperature": 0.2}
        ),
        "Finalize research report: {input}"
    ]
)
```

## 🚀 Key Capabilities Summary

### **AgenticStepProcessor Capabilities**
- **Internal Reasoning Loop**: Multiple LLM calls within one step
- **Automatic Tool Integration**: Seamless tool usage and coordination
- **Objective-Driven Processing**: Works toward specific goals
- **Self-Contained Configuration**: Independent model and parameter settings
- **Iterative Refinement**: Improves output through multiple iterations
- **Complex Reasoning**: Multi-step planning and execution

### **Regular Agent Limitations**
- **Single Step Execution**: One instruction, one call
- **Linear Processing**: No internal loops or reasoning
- **Manual Tool Usage**: Requires explicit tool integration
- **Chain-Dependent Configuration**: Inherits model settings from parent
- **Simple Input/Output**: Basic processing without iteration
- **Limited Complexity**: Suitable for straightforward tasks

The AgenticStepProcessor represents a significant evolution from basic PromptChain usage, providing sophisticated agentic capabilities within a single chain step, enabling complex reasoning, tool coordination, and iterative refinement that regular agents cannot achieve.


## 🎯 Integration with AgentChain Orchestrator

The **AgenticStepProcessor** can be integrated with the **AgentChain** orchestrator to create sophisticated multi-agent systems:

### **AgentChain Integration**

```python
# Create agents with AgenticStepProcessor
research_agent = PromptChain(
    models=[],
    instructions=[
        AgenticStepProcessor(
            objective="Conduct comprehensive research analysis",
            max_internal_steps=5,
            model_name="openai/gpt-4o"
        )
    ]
)

analysis_agent = PromptChain(
    models=[],
    instructions=[
        AgenticStepProcessor(
            objective="Analyze research findings and generate insights",
            max_internal_steps=3,
            model_name="openai/gpt-4o"
        )
    ]
)

# Orchestrate with AgentChain
agent_chain = AgentChain(
    agents={"research": research_agent, "analysis": analysis_agent},
    agent_descriptions={
        "research": "Conducts comprehensive research using tools",
        "analysis": "Analyzes findings and generates insights"
    },
    execution_mode="router",
    router=router_config
)
```

### **Combined Capabilities**

- **AgenticStepProcessor**: Provides internal reasoning and tool usage within each agent
- **AgentChain**: Orchestrates multiple agentic agents with intelligent routing
- **Tool Integration**: Both local tools and MCP tools work seamlessly
- **Complex Workflows**: Multi-agent systems with sophisticated reasoning

The combination of AgenticStepProcessor and AgentChain creates a powerful framework for building complex, reasoning-based AI systems that can handle sophisticated tasks requiring multiple steps, tool usage, and agent coordination. 