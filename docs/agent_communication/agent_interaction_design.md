# Agent-to-Agent Interaction Design for PromptChain
## Enabling Direct Communication and Mutual Assistance

**Problem**: Agents in PromptChain's AgentChain need to interact directly with each other, request help, and provide mutual assistance without always going through the orchestrator.

**Goal**: Design patterns and mechanisms for peer-to-peer agent communication while maintaining the existing AgentChain architecture.

---

## Table of Contents
1. [Current State Analysis](#current-state-analysis)
2. [Design Goals](#design-goals)
3. [Proposed Solutions](#proposed-solutions)
4. [Implementation Patterns](#implementation-patterns)
5. [Integration with Existing AgentChain](#integration-with-existing-agentchain)
6. [Code Examples](#code-examples)

---

## Current State Analysis

### What PromptChain Already Has

1. **AgentChain Orchestration**:
   - Router mode: Selects one agent per turn
   - Pipeline mode: Sequential execution
   - Round-robin mode: Cyclic execution
   - Broadcast mode: Parallel execution

2. **Limited Agent-to-Agent Communication**:
   - `[REROUTE]` markers for agent-to-agent rerouting in router mode
   - History sharing via `auto_include_history`
   - Context passing through pipeline mode

3. **What's Missing**:
   - Direct agent-to-agent messaging
   - Capability discovery between agents
   - Dynamic help requests mid-task
   - Collaborative problem-solving
   - Agent-initiated task delegation

---

## Design Goals

### Core Requirements

1. **Direct Communication**: Agents can send messages directly to other agents
2. **Capability Discovery**: Agents can discover what other agents can do
3. **Help Requests**: Agents can request assistance when stuck
4. **Proactive Assistance**: Agents can offer help to others
5. **Context Sharing**: Agents can share relevant context with each other
6. **Backward Compatibility**: Don't break existing AgentChain functionality

### Design Principles

- **Non-Breaking**: Existing code continues to work
- **Opt-In**: Agents choose to participate in direct communication
- **Transparent**: Communication is logged and observable
- **Flexible**: Multiple communication patterns supported
- **Efficient**: Minimal overhead when not used

---

## Proposed Solutions

### Solution 1: Agent Communication Bus (Recommended)

**Concept**: A lightweight message bus that agents can use to communicate directly.

**Architecture**:
```
Agent A → Communication Bus → Agent B
         ↓
    Event Log
         ↓
    Orchestrator (optional monitoring)
```

**Benefits**:
- Decoupled: Agents don't need to know about orchestrator
- Flexible: Supports multiple communication patterns
- Observable: All communication logged
- Extensible: Easy to add new message types

### Solution 2: Agent Registry with Capability Discovery

**Concept**: Agents register their capabilities, and other agents can query the registry.

**Architecture**:
```
Agent A → Capability Query → Registry → Agent B Info
         ↓
    Request Help → Agent B
```

**Benefits**:
- Self-documenting: Agents declare what they can do
- Dynamic: Capabilities can change over time
- Efficient: No need to query all agents

### Solution 3: Embedded Communication Tools

**Concept**: Add special tools to each agent that enable communication with other agents.

**Architecture**:
```
Agent A (with communication tools)
    ├─ ask_agent(agent_name, question)
    ├─ request_help(agent_name, task_description)
    └─ share_context(agent_name, context_data)
```

**Benefits**:
- Natural: Uses existing tool-calling infrastructure
- LLM-friendly: Agents can reason about when to communicate
- Integrated: Works with existing AgenticStepProcessor

---

## Implementation Patterns

### Pattern 1: Communication Bus Implementation

```python
class AgentCommunicationBus:
    """
    Lightweight message bus for agent-to-agent communication.
    """
    def __init__(self, agent_chain):
        self.agent_chain = agent_chain
        self.message_queue = asyncio.Queue()
        self.subscribers = {}  # agent_name -> callback
        self.message_history = []
        self.enabled = True
    
    async def send_message(self, from_agent: str, to_agent: str, 
                          message_type: str, content: Any, 
                          requires_response: bool = False):
        """
        Send a message from one agent to another.
        
        Args:
            from_agent: Name of sending agent
            to_agent: Name of receiving agent
            message_type: Type of message (query, help_request, context_share, etc.)
            content: Message content
            requires_response: Whether sender expects a response
        """
        if not self.enabled:
            return None
        
        message = {
            "message_id": str(uuid.uuid4()),
            "from_agent": from_agent,
            "to_agent": to_agent,
            "message_type": message_type,
            "content": content,
            "requires_response": requires_response,
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }
        
        # Add to history
        self.message_history.append(message)
        
        # Deliver to recipient if they're subscribed
        if to_agent in self.subscribers:
            response = await self.subscribers[to_agent](message)
            if requires_response:
                message["response"] = response
                message["status"] = "responded"
            return response
        else:
            # Queue for later delivery
            await self.message_queue.put(message)
            return None
    
    def subscribe(self, agent_name: str, callback: Callable):
        """Agent subscribes to receive messages."""
        self.subscribers[agent_name] = callback
    
    async def broadcast(self, from_agent: str, message_type: str, 
                       content: Any, exclude_agents: List[str] = None):
        """Broadcast message to all agents except sender."""
        exclude_agents = exclude_agents or []
        responses = {}
        
        for agent_name in self.agent_chain.agents.keys():
            if agent_name != from_agent and agent_name not in exclude_agents:
                response = await self.send_message(
                    from_agent, agent_name, message_type, content, 
                    requires_response=True
                )
                responses[agent_name] = response
        
        return responses
```

### Pattern 2: Capability Registry

```python
class AgentCapabilityRegistry:
    """
    Registry for agent capabilities and discovery.
    """
    def __init__(self):
        self.capabilities = {}  # agent_name -> capability_info
        self.capability_index = {}  # capability_type -> [agent_names]
    
    def register_agent(self, agent_name: str, capabilities: Dict[str, Any]):
        """
        Register an agent's capabilities.
        
        Args:
            agent_name: Name of the agent
            capabilities: Dict with:
                - "description": Human-readable description
                - "skills": List of skill keywords
                - "task_types": List of task types this agent can handle
                - "examples": Example tasks this agent can do
        """
        self.capabilities[agent_name] = {
            **capabilities,
            "registered_at": datetime.now().isoformat()
        }
        
        # Index by capability type
        for task_type in capabilities.get("task_types", []):
            if task_type not in self.capability_index:
                self.capability_index[task_type] = []
            if agent_name not in self.capability_index[task_type]:
                self.capability_index[task_type].append(agent_name)
    
    def find_agents_for_task(self, task_type: str, 
                             skill_keywords: List[str] = None) -> List[str]:
        """
        Find agents capable of handling a specific task type.
        
        Args:
            task_type: Type of task (e.g., "research", "writing", "analysis")
            skill_keywords: Optional keywords to match against agent skills
        
        Returns:
            List of agent names that can handle this task
        """
        candidates = self.capability_index.get(task_type, [])
        
        if skill_keywords:
            # Filter by skill keywords
            filtered = []
            for agent_name in candidates:
                agent_skills = self.capabilities[agent_name].get("skills", [])
                if any(keyword.lower() in [s.lower() for s in agent_skills] 
                       for keyword in skill_keywords):
                    filtered.append(agent_name)
            return filtered
        
        return candidates
    
    def get_agent_capabilities(self, agent_name: str) -> Dict[str, Any]:
        """Get full capability information for an agent."""
        return self.capabilities.get(agent_name, {})
    
    def list_all_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Get capabilities of all registered agents."""
        return self.capabilities.copy()
```

### Pattern 3: Communication Tools for Agents

```python
def create_communication_tools(agent_chain, comm_bus, capability_registry):
    """
    Create communication tools that can be registered with agents.
    """
    
    async def ask_agent(agent_name: str, question: str) -> str:
        """
        Ask another agent a question.
        
        Args:
            agent_name: Name of agent to ask
            question: The question to ask
        
        Returns:
            Response from the agent
        """
        if agent_name not in agent_chain.agents:
            return f"Error: Agent '{agent_name}' not found."
        
        response = await comm_bus.send_message(
            from_agent="current_agent",  # Will be set by context
            to_agent=agent_name,
            message_type="query",
            content={"question": question},
            requires_response=True
        )
        
        return response.get("content", {}).get("answer", "No response received")
    
    async def request_help(agent_name: str, task_description: str, 
                          context: str = "") -> str:
        """
        Request help from another agent on a specific task.
        
        Args:
            agent_name: Name of agent to request help from
            task_description: Description of the task needing help
            context: Additional context about the current work
        
        Returns:
            Helpful response from the agent
        """
        if agent_name not in agent_chain.agents:
            return f"Error: Agent '{agent_name}' not found."
        
        response = await comm_bus.send_message(
            from_agent="current_agent",
            to_agent=agent_name,
            message_type="help_request",
            content={
                "task_description": task_description,
                "context": context
            },
            requires_response=True
        )
        
        return response.get("content", {}).get("help", "No help received")
    
    async def find_helpful_agents(task_type: str, 
                                  skill_keywords: List[str] = None) -> str:
        """
        Find agents that can help with a specific task type.
        
        Args:
            task_type: Type of task (e.g., "research", "writing")
            skill_keywords: Optional keywords to match
        
        Returns:
            JSON string listing helpful agents and their capabilities
        """
        helpful_agents = capability_registry.find_agents_for_task(
            task_type, skill_keywords
        )
        
        result = []
        for agent_name in helpful_agents:
            capabilities = capability_registry.get_agent_capabilities(agent_name)
            result.append({
                "agent_name": agent_name,
                "description": capabilities.get("description", ""),
                "skills": capabilities.get("skills", [])
            })
        
        return json.dumps(result, indent=2)
    
    async def share_context(agent_name: str, context_data: str) -> str:
        """
        Share context with another agent.
        
        Args:
            agent_name: Name of agent to share with
            context_data: Context information to share
        
        Returns:
            Confirmation message
        """
        await comm_bus.send_message(
            from_agent="current_agent",
            to_agent=agent_name,
            message_type="context_share",
            content={"context": context_data},
            requires_response=False
        )
        
        return f"Context shared with {agent_name}"
    
    # Return tools in OpenAI function format
    return [
        {
            "type": "function",
            "function": {
                "name": "ask_agent",
                "description": "Ask another agent a question directly",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_name": {"type": "string", "description": "Name of agent to ask"},
                        "question": {"type": "string", "description": "The question to ask"}
                    },
                    "required": ["agent_name", "question"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "request_help",
                "description": "Request help from another agent on a specific task",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_name": {"type": "string", "description": "Name of agent to request help from"},
                        "task_description": {"type": "string", "description": "Description of task needing help"},
                        "context": {"type": "string", "description": "Additional context about current work"}
                    },
                    "required": ["agent_name", "task_description"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "find_helpful_agents",
                "description": "Find agents that can help with a specific task type",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_type": {"type": "string", "description": "Type of task (e.g., 'research', 'writing')"},
                        "skill_keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional keywords to match agent skills"
                        }
                    },
                    "required": ["task_type"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "share_context",
                "description": "Share context information with another agent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_name": {"type": "string", "description": "Name of agent to share with"},
                        "context_data": {"type": "string", "description": "Context information to share"}
                    },
                    "required": ["agent_name", "context_data"]
                }
            }
        }
    ]
```

---

## Integration with Existing AgentChain

### Enhanced AgentChain Class

```python
class AgentChain:
    """
    Enhanced AgentChain with agent-to-agent communication support.
    """
    
    def __init__(self, agents, agent_descriptions, 
                 enable_agent_communication: bool = False,
                 **kwargs):
        # ... existing initialization ...
        
        # NEW: Agent communication infrastructure
        self.enable_agent_communication = enable_agent_communication
        if enable_agent_communication:
            self.comm_bus = AgentCommunicationBus(self)
            self.capability_registry = AgentCapabilityRegistry()
            
            # Register all agents with capability registry
            for agent_name, description in agent_descriptions.items():
                # Extract capabilities from description or use defaults
                capabilities = self._extract_capabilities(agent_name, description)
                self.capability_registry.register_agent(agent_name, capabilities)
            
            # Create communication tools
            self.communication_tools = create_communication_tools(
                self, self.comm_bus, self.capability_registry
            )
            
            # Register tools with each agent
            self._register_communication_tools_with_agents()
            
            # Set up message handlers
            self._setup_message_handlers()
    
    def _extract_capabilities(self, agent_name: str, description: str) -> Dict[str, Any]:
        """
        Extract capabilities from agent description.
        Can be enhanced with LLM-based extraction.
        """
        # Simple keyword-based extraction
        description_lower = description.lower()
        
        task_types = []
        if any(word in description_lower for word in ["research", "find", "search"]):
            task_types.append("research")
        if any(word in description_lower for word in ["write", "draft", "compose"]):
            task_types.append("writing")
        if any(word in description_lower for word in ["analyze", "analysis"]):
            task_types.append("analysis")
        if any(word in description_lower for word in ["review", "edit", "validate"]):
            task_types.append("review")
        
        return {
            "description": description,
            "task_types": task_types,
            "skills": task_types  # Simplified
        }
    
    def _register_communication_tools_with_agents(self):
        """Register communication tools with each agent."""
        for agent_name, agent in self.agents.items():
            # Add communication tools to agent
            for tool in self.communication_tools:
                # Create bound versions that know the current agent
                bound_tool = self._create_bound_tool(tool, agent_name)
                agent.add_tools([bound_tool])
    
    def _create_bound_tool(self, tool: Dict, current_agent_name: str) -> Dict:
        """Create a tool bound to a specific agent context."""
        # This would need to be implemented based on how PromptChain handles tools
        # The idea is to inject current_agent_name into tool execution context
        return tool  # Simplified
    
    def _setup_message_handlers(self):
        """Set up handlers for incoming messages."""
        for agent_name, agent in self.agents.items():
            async def create_handler(name):
                async def handler(message):
                    # Process message and generate response
                    response = await self._process_agent_message(name, message)
                    return response
                return handler
            
            self.comm_bus.subscribe(agent_name, await create_handler(agent_name))
    
    async def _process_agent_message(self, agent_name: str, message: Dict) -> Dict:
        """
        Process an incoming message for an agent.
        """
        agent = self.agents[agent_name]
        message_type = message["message_type"]
        content = message["content"]
        
        # Format message for agent processing
        if message_type == "query":
            prompt = f"Another agent is asking you: {content.get('question', '')}"
        elif message_type == "help_request":
            prompt = f"Another agent needs help with: {content.get('task_description', '')}\n\nContext: {content.get('context', '')}\n\nHow can you help?"
        elif message_type == "context_share":
            # Store context for later use
            if not hasattr(agent, '_shared_context'):
                agent._shared_context = []
            agent._shared_context.append(content.get("context", ""))
            return {"status": "received", "message": "Context stored"}
        else:
            prompt = f"Message from {message['from_agent']}: {content}"
        
        # Process with agent
        response = await agent.process_prompt_async(prompt)
        
        return {
            "status": "responded",
            "content": {"answer": response},
            "from_agent": agent_name
        }
```

---

## Code Examples

### Example 1: Basic Agent Communication

```python
from promptchain import PromptChain
from promptchain.utils.agent_chain import AgentChain

# Create specialized agents
researcher = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Research the topic: {input}"]
)

writer = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Write about: {input}"]
)

# Create AgentChain with communication enabled
agent_chain = AgentChain(
    agents={
        "researcher": researcher,
        "writer": writer
    },
    agent_descriptions={
        "researcher": "Researches topics and finds information",
        "writer": "Writes content based on research"
    },
    enable_agent_communication=True,  # NEW: Enable communication
    execution_mode="router"
)

# Now agents can communicate with each other!
# The writer can ask the researcher for help:
# "I need to write about quantum computing. Let me ask the researcher for information."
# The LLM in the writer agent can use the ask_agent tool to query the researcher.
```

### Example 2: Collaborative Problem Solving

```python
# Agent A is stuck and requests help from Agent B
# This happens automatically when Agent A's LLM decides it needs help

# Agent A's internal reasoning:
# "I'm trying to solve this problem but I need expertise in data analysis.
#  Let me find agents that can help with analysis."

# Agent A calls: find_helpful_agents(task_type="analysis")
# Returns: [{"agent_name": "analyst", "description": "...", "skills": [...]}]

# Agent A calls: request_help(
#     agent_name="analyst",
#     task_description="Analyze this dataset and find patterns",
#     context="I'm working on a customer segmentation problem..."
# )

# Analyst agent receives the request and responds with analysis
# Agent A continues with the analysis results
```

### Example 3: Context Sharing Between Agents

```python
# Agent A discovers important context and shares it with Agent B
# Agent A calls: share_context(
#     agent_name="writer",
#     context_data="The user prefers technical writing style and wants code examples"
# )

# Writer agent receives context and stores it
# When writer processes future requests, it uses this shared context
```

---

## Implementation Roadmap

### Phase 1: Core Communication Infrastructure
1. Implement `AgentCommunicationBus`
2. Implement `AgentCapabilityRegistry`
3. Add communication tools
4. Integrate with AgentChain initialization

### Phase 2: Message Handling
1. Implement message handlers for each agent
2. Add message processing logic
3. Implement response generation
4. Add message history tracking

### Phase 3: Advanced Features
1. LLM-based capability extraction
2. Proactive help offering
3. Collaborative workflows
4. Communication analytics

### Phase 4: Integration & Testing
1. Integration tests
2. Performance optimization
3. Documentation
4. Example workflows

---

## Benefits of This Approach

1. **Non-Breaking**: Existing AgentChain code continues to work
2. **Opt-In**: Communication is enabled only when requested
3. **Natural**: Uses existing tool-calling infrastructure
4. **Flexible**: Supports multiple communication patterns
5. **Observable**: All communication is logged
6. **LLM-Friendly**: Agents can reason about when to communicate

---

## Next Steps

1. **Review Design**: Validate approach with team
2. **Prototype**: Build minimal viable implementation
3. **Test**: Create example workflows
4. **Iterate**: Refine based on feedback
5. **Document**: Create user guide and examples

---

**Note**: This design enables agents to help each other naturally through the existing tool-calling mechanism, making it feel like agents are truly collaborating rather than just being orchestrated.

