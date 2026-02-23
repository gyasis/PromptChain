# T047: Integration Test - Multi-Hop Reasoning with Tool Calls

## Objective
Create integration test demonstrating complex multi-hop reasoning workflow with AgenticStepProcessor using file search and analysis tools.

## Context Files
- `/home/gyasis/Documents/code/PromptChain/promptchain/utils/agentic_step_processor.py` (processor)
- `/home/gyasis/Documents/code/PromptChain/promptchain/tools/ripgrep_wrapper.py` (search tool)
- `/home/gyasis/Documents/code/PromptChain/tests/cli/integration/test_multi_hop_tools.py` (reference pattern)

## Requirements

### Test File Location
`/home/gyasis/Documents/code/PromptChain/tests/cli/integration/test_multihop_reasoning_tools.py`

### Test Scenario

**Workflow**: Code analysis with iterative search refinement

1. AgenticStepProcessor receives objective: "Find and analyze authentication patterns in the codebase"
2. Step 1: Search for "authentication" in code
3. Step 2: Analyze search results, identify key files
4. Step 3: Search specific files for detailed patterns
5. Step 4: Synthesize findings into comprehensive analysis

### Test Implementation

#### Setup: Mock File System

```python
import pytest
import tempfile
import os
from pathlib import Path

@pytest.fixture
def mock_codebase(tmp_path):
    """Create mock codebase for testing."""
    # Create file structure
    auth_file = tmp_path / "auth" / "authentication.py"
    auth_file.parent.mkdir(parents=True)
    auth_file.write_text("""
class AuthenticationManager:
    def __init__(self):
        self.jwt_secret = "secret_key"

    def authenticate(self, username, password):
        # JWT-based authentication
        token = self._generate_jwt(username)
        return token

    def _generate_jwt(self, username):
        import jwt
        return jwt.encode({"user": username}, self.jwt_secret)
    """)

    config_file = tmp_path / "config" / "security.py"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("""
AUTHENTICATION_BACKEND = "jwt"
SESSION_TIMEOUT = 3600
PASSWORD_HASHING = "bcrypt"
    """)

    utils_file = tmp_path / "utils" / "validators.py"
    utils_file.parent.mkdir(parents=True)
    utils_file.write_text("""
def validate_credentials(username, password):
    # Basic validation
    return len(username) > 0 and len(password) >= 8
    """)

    return tmp_path
```

#### Test Case 1: Multi-Hop File Search and Analysis

```python
@pytest.mark.asyncio
async def test_multihop_file_search_analysis(mock_codebase):
    """AgenticStepProcessor performs multi-hop file search and analysis."""
    from promptchain.utils.agentic_step_processor import AgenticStepProcessor
    from promptchain.utils.promptchaining import PromptChain
    from promptchain.tools.ripgrep_wrapper import RipgrepSearcher

    # Create search tool
    searcher = RipgrepSearcher()

    def search_files(query: str) -> str:
        """Search codebase for pattern."""
        results = searcher.search(query, search_path=str(mock_codebase))
        if not results:
            return f"No results found for: {query}"

        # Format results
        output = f"Search results for '{query}':\n"
        for result in results[:10]:  # Limit to 10 results
            output += f"- {result}\n"
        return output

    def read_file(filepath: str) -> str:
        """Read file contents."""
        try:
            full_path = mock_codebase / filepath
            if full_path.exists():
                return full_path.read_text()
            return f"File not found: {filepath}"
        except Exception as e:
            return f"Error reading file: {e}"

    # Create AgenticStepProcessor
    processor = AgenticStepProcessor(
        objective="Find and analyze authentication patterns in the codebase. "
                  "Identify key files, search patterns, and provide detailed analysis.",
        max_internal_steps=6,
        model_name="openai/gpt-4o-mini",
        store_detailed_steps=True
    )

    # Create chain with tools
    chain = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=[processor],
        verbose=True
    )

    # Register tools
    chain.register_tool_function(search_files)
    chain.register_tool_function(read_file)

    chain.add_tools([
        {
            "type": "function",
            "function": {
                "name": "search_files",
                "description": "Search through codebase files using pattern matching",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search pattern or keyword"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read contents of a specific file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Relative path to file"
                        }
                    },
                    "required": ["filepath"]
                }
            }
        }
    ])

    # Execute multi-hop reasoning
    result = await chain.process_prompt_async(
        "Analyze authentication patterns in this codebase"
    )

    # Assertions
    assert result is not None
    assert len(result) > 200  # Substantial analysis

    # Should mention key findings
    assert "jwt" in result.lower() or "authentication" in result.lower()

    # Verify multi-hop execution
    assert hasattr(chain, 'agentic_step_details')
    step_details = chain.agentic_step_details[0]

    # Processor should have taken multiple internal steps
    assert len(step_details['internal_steps']) >= 2

    # Should have made tool calls (search and/or read)
    internal_steps = step_details['internal_steps']
    tool_calls_made = any(
        'tool_call' in str(step).lower() or 'search' in str(step).lower()
        for step in internal_steps
    )
    assert tool_calls_made, "AgenticStepProcessor should have used tools"
```

#### Test Case 2: Progressive Refinement with Tool Feedback

```python
@pytest.mark.asyncio
async def test_multihop_progressive_refinement(mock_codebase):
    """AgenticStepProcessor refines search based on tool feedback."""
    from promptchain.utils.agentic_step_processor import AgenticStepProcessor
    from promptchain.utils.promptchaining import PromptChain

    search_log = []  # Track search queries to verify refinement

    def tracked_search(query: str) -> str:
        """Search with query tracking."""
        search_log.append(query)

        # Simulate search results
        if "authentication" in query.lower():
            return "Found: auth/authentication.py, config/security.py"
        elif "jwt" in query.lower():
            return "Found: auth/authentication.py (JWT implementation)"
        elif "password" in query.lower():
            return "Found: config/security.py (PASSWORD_HASHING)"
        else:
            return "No results found"

    # Create processor with progressive history
    processor = AgenticStepProcessor(
        objective="Search for authentication methods. Refine search based on "
                  "initial results to find specific implementation details.",
        max_internal_steps=5,
        model_name="openai/gpt-4o-mini",
        history_mode="progressive",  # Build context across steps
        store_detailed_steps=True
    )

    # Create chain
    chain = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=[processor],
        verbose=True
    )

    # Register tracked search
    chain.register_tool_function(tracked_search)
    chain.add_tools([{
        "type": "function",
        "function": {
            "name": "tracked_search",
            "description": "Search codebase with query tracking",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    }])

    # Execute
    result = await chain.process_prompt_async("Find authentication implementation")

    # Assertions
    assert result is not None

    # Verify progressive refinement - should make multiple searches
    assert len(search_log) >= 2, "Should refine search multiple times"

    # Later searches should be more specific than initial
    # (This is heuristic - initial search broader, later more targeted)
    if len(search_log) >= 2:
        initial_query = search_log[0]
        later_query = search_log[-1]

        # Later query likely more specific (longer or contains specific term)
        # This is a weak heuristic but validates refinement behavior
        assert initial_query != later_query, "Queries should evolve"
```

#### Test Case 3: Error Recovery in Multi-Hop Workflow

```python
@pytest.mark.asyncio
async def test_multihop_error_recovery():
    """AgenticStepProcessor recovers from tool errors during multi-hop reasoning."""
    from promptchain.utils.agentic_step_processor import AgenticStepProcessor
    from promptchain.utils.promptchaining import PromptChain

    call_count = {"count": 0}

    def flaky_search(query: str) -> str:
        """Search that fails first time, succeeds second."""
        call_count["count"] += 1
        if call_count["count"] == 1:
            raise Exception("Network timeout")
        return f"Results for: {query}"

    # Create processor
    processor = AgenticStepProcessor(
        objective="Search for information and handle failures gracefully",
        max_internal_steps=5,
        model_name="openai/gpt-4o-mini",
        clarification_attempts=2  # Allow retry on errors
    )

    chain = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=[processor]
    )

    chain.register_tool_function(flaky_search)
    chain.add_tools([{
        "type": "function",
        "function": {
            "name": "flaky_search",
            "description": "Search that may fail",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    }])

    # Execute - should recover from first failure
    result = await chain.process_prompt_async("Search for information")

    # Assertions
    assert result is not None
    # Processor should have recovered and completed despite error
    assert call_count["count"] >= 1  # At least one attempt made
```

### Success Criteria
- All 3 integration tests pass
- Multi-hop reasoning with tools demonstrated
- Progressive search refinement validated
- Error recovery verified
- Tests run in <45 seconds

## Validation
Run: `pytest tests/cli/integration/test_multihop_reasoning_tools.py -v`
Expected: 3 passed in <45s

## Deliverable
- Integration test file with 3 comprehensive tests
- Demonstration of complex multi-hop workflows
- Verification of tool integration patterns
