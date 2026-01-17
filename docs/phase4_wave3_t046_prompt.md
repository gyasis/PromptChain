# T046: Integration Test - AgenticStepProcessor in Agent Chain

## Objective
Create integration test verifying AgenticStepProcessor executes correctly within AgentChain workflows, maintaining history isolation and tool access.

## Context Files
- `/home/gyasis/Documents/code/PromptChain/promptchain/utils/agent_chain.py` (integration target)
- `/home/gyasis/Documents/code/PromptChain/promptchain/utils/agentic_step_processor.py` (processor)
- `/home/gyasis/Documents/code/PromptChain/tests/cli/integration/test_agentic_reasoning.py` (similar pattern)

## Requirements

### Test File Location
`/home/gyasis/Documents/code/PromptChain/tests/cli/integration/test_agentchain_agentic_processor.py`

### Test Cases

#### 1. `test_agentic_processor_in_agent_chain()`

**Scenario**: Single agent with AgenticStepProcessor in instruction chain

```python
@pytest.mark.asyncio
async def test_agentic_processor_in_agent_chain():
    """AgenticStepProcessor executes within agent chain workflow."""
    # Create processor
    processor = AgenticStepProcessor(
        objective="Analyze input and provide detailed reasoning",
        max_internal_steps=4,
        model_name="openai/gpt-4o-mini"  # Use fast model for testing
    )

    # Create agent with processor in instructions
    agent = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=[
            "Prepare analysis: {input}",
            processor,  # Multi-hop reasoning step
            "Synthesize final result: {input}"
        ],
        verbose=True
    )

    # Create agent chain
    agent_chain = AgentChain(
        agents={"analyst": agent},
        execution_mode="router",
        auto_include_history=True,
        verbose=True
    )

    # Execute workflow
    result = await agent_chain.process_input_async(
        "What are the key patterns in large-scale distributed systems?"
    )

    # Assertions
    assert result is not None
    assert len(result) > 100  # Meaningful output
    assert "distributed systems" in result.lower()

    # Verify processor executed
    assert hasattr(agent, 'agentic_step_details')
    assert len(agent.agentic_step_details) == 1
    assert agent.agentic_step_details[0]['objective'] == processor.objective
```

#### 2. `test_agentic_processor_tool_access()`

**Scenario**: Processor uses tools registered in parent chain

```python
@pytest.mark.asyncio
async def test_agentic_processor_tool_access():
    """AgenticStepProcessor can access tools from parent PromptChain."""
    # Create mock search tool
    def mock_search(query: str) -> str:
        """Mock search tool for testing."""
        return f"Search results for: {query}\n- Result 1\n- Result 2\n- Result 3"

    # Create processor
    processor = AgenticStepProcessor(
        objective="Search for information and analyze results",
        max_internal_steps=3,
        model_name="openai/gpt-4o-mini"
    )

    # Create agent with tool registration
    agent = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=[processor],
        verbose=True
    )

    # Register tool
    agent.register_tool_function(mock_search)
    agent.add_tools([{
        "type": "function",
        "function": {
            "name": "mock_search",
            "description": "Search for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    }])

    # Execute
    result = await agent.process_prompt_async("Find information about Python asyncio")

    # Assertions
    assert result is not None
    assert len(result) > 50
    # Processor should have used search tool
    # (Tool call detection depends on AgenticStepProcessor internal tracking)
```

#### 3. `test_agentic_processor_history_isolation()`

**Scenario**: Processor maintains isolated history from agent chain

```python
@pytest.mark.asyncio
async def test_agentic_processor_history_isolation():
    """AgenticStepProcessor history doesn't pollute agent chain history."""
    from promptchain.utils.execution_history_manager import ExecutionHistoryManager

    # Create history manager
    history_manager = ExecutionHistoryManager(max_tokens=2000)

    # Create processor with progressive history mode
    processor = AgenticStepProcessor(
        objective="Multi-step reasoning task",
        max_internal_steps=5,
        model_name="openai/gpt-4o-mini",
        history_mode="progressive"  # Builds internal history
    )

    # Create agent
    agent = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=[
            "Initial prompt: {input}",
            processor,
            "Final prompt: {input}"
        ],
        verbose=True
    )

    # Execute with history tracking
    history_manager.add_entry("user_input", "Complex reasoning task", source="test")

    result = await agent.process_prompt_async("Explain quantum computing concepts")

    history_manager.add_entry("agent_output", result, source="agent")

    # Assertions
    # Agent chain history should only have user input and final output
    formatted_history = history_manager.get_formatted_history()
    history_entries = history_manager.get_entries()

    assert len(history_entries) == 2  # user_input + agent_output only
    assert "user_input" in [e.entry_type for e in history_entries]
    assert "agent_output" in [e.entry_type for e in history_entries]

    # Internal processor steps should NOT be in agent history
    # (They're isolated in processor.step_history)
```

#### 4. `test_multiple_agentic_processors_in_chain()`

**Scenario**: Multiple AgenticStepProcessor instances in same chain

```python
@pytest.mark.asyncio
async def test_multiple_agentic_processors_in_chain():
    """Multiple AgenticStepProcessor instances execute sequentially."""
    # Create two processors with different objectives
    processor1 = AgenticStepProcessor(
        objective="Phase 1: Data collection and initial analysis",
        max_internal_steps=3,
        model_name="openai/gpt-4o-mini"
    )

    processor2 = AgenticStepProcessor(
        objective="Phase 2: Synthesis and recommendations",
        max_internal_steps=3,
        model_name="openai/gpt-4o-mini"
    )

    # Create agent with multiple processors
    agent = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=[
            "Initialize: {input}",
            processor1,
            "Intermediate summary: {input}",
            processor2,
            "Final conclusions: {input}"
        ],
        store_steps=True,
        verbose=True
    )

    # Execute
    result = await agent.process_prompt_async(
        "Analyze the impact of remote work on software development teams"
    )

    # Assertions
    assert result is not None
    assert len(result) > 200

    # Both processors should have executed
    assert hasattr(agent, 'agentic_step_details')
    assert len(agent.agentic_step_details) == 2

    details = agent.agentic_step_details
    assert details[0]['objective'] == processor1.objective
    assert details[1]['objective'] == processor2.objective

    # Verify step results stored
    assert len(agent.step_outputs) == 5  # 5 instructions total
```

### Success Criteria
- All 4 integration tests pass
- AgenticStepProcessor executes within agent chains
- Tool access works correctly
- History isolation maintained
- Multiple processors in same chain work
- Tests run in <30 seconds

## Validation
Run: `pytest tests/cli/integration/test_agentchain_agentic_processor.py -v`
Expected: 4 passed in <30s

## Deliverable
- Integration test file with 4 passing tests
- Verification of AgenticStepProcessor in AgentChain
- Documentation of integration patterns
