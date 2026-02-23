# T055: Error Handling for Max Steps Exhaustion

## Objective
Implement graceful error handling when AgenticStepProcessor reaches max_internal_steps without completing objective, with user-friendly TUI error display.

## Context Files
- `/home/gyasis/Documents/code/PromptChain/promptchain/utils/agentic_step_processor.py` (add error handling)
- `/home/gyasis/Documents/code/PromptChain/promptchain/cli/tui/app.py` (display errors)
- `/home/gyasis/Documents/code/PromptChain/promptchain/cli/error_handler.py` (error formatting)

## Requirements

### Error Types to Handle

1. **MaxStepsExhaustedError**: Steps exhausted without completion
2. **ToolCallError**: Tool call failed during reasoning
3. **ObjectiveAmbiguityError**: Objective unclear after clarification attempts

### Implementation in AgenticStepProcessor

**Add Custom Exceptions** (`promptchain/utils/agentic_step_processor.py`):

```python
class AgenticStepProcessorError(Exception):
    """Base exception for AgenticStepProcessor errors."""
    pass


class MaxStepsExhaustedError(AgenticStepProcessorError):
    """Raised when max_internal_steps reached without completing objective."""

    def __init__(self, objective: str, steps_taken: int, last_output: str):
        self.objective = objective
        self.steps_taken = steps_taken
        self.last_output = last_output
        super().__init__(
            f"AgenticStepProcessor exhausted {steps_taken} steps without completing "
            f"objective: '{objective}'\n\nLast output: {last_output[:200]}..."
        )


class ToolCallError(AgenticStepProcessorError):
    """Raised when tool call fails during reasoning."""

    def __init__(self, tool_name: str, error_message: str, step_num: int):
        self.tool_name = tool_name
        self.error_message = error_message
        self.step_num = step_num
        super().__init__(
            f"Tool '{tool_name}' failed at step {step_num}: {error_message}"
        )


class ObjectiveAmbiguityError(AgenticStepProcessorError):
    """Raised when objective remains unclear after clarification attempts."""

    def __init__(self, objective: str, clarification_attempts: int):
        self.objective = objective
        self.clarification_attempts = clarification_attempts
        super().__init__(
            f"Objective unclear after {clarification_attempts} clarification attempts: "
            f"'{objective}'"
        )
```

**Add Error Handling in `process_async()`**:

```python
async def process_async(self, user_input: str) -> str:
    """Process with comprehensive error handling."""
    try:
        # Existing reasoning loop
        for step_num in range(1, self.max_internal_steps + 1):
            # Reasoning logic...

            # Check for completion
            if self._is_objective_complete(response):
                # Success path
                return self._extract_final_synthesis(response)

        # If loop completes without return, objective not achieved
        raise MaxStepsExhaustedError(
            objective=self.objective,
            steps_taken=self.max_internal_steps,
            last_output=self.step_history[-1] if self.step_history else ""
        )

    except ToolCallError:
        # Re-raise tool errors
        raise

    except ObjectiveAmbiguityError:
        # Re-raise clarification errors
        raise

    except Exception as e:
        # Wrap unexpected errors
        raise AgenticStepProcessorError(
            f"Unexpected error during reasoning: {str(e)}"
        ) from e
```

**Add Graceful Degradation Option**:

```python
class AgenticStepProcessor:
    def __init__(
        self,
        # ... existing params
        fallback_on_exhaustion: bool = True,  # NEW
        partial_results_acceptable: bool = False  # NEW
    ):
        """
        Args:
            fallback_on_exhaustion: Return partial results instead of error
            partial_results_acceptable: Accept incomplete objective completion
        """
        self.fallback_on_exhaustion = fallback_on_exhaustion
        self.partial_results_acceptable = partial_results_acceptable

    async def process_async(self, user_input: str) -> str:
        """Process with fallback option."""
        try:
            # Reasoning loop...
            for step_num in range(1, self.max_internal_steps + 1):
                # ...

                if self._is_objective_complete(response):
                    return self._extract_final_synthesis(response)

            # Max steps reached
            if self.fallback_on_exhaustion:
                # Return partial results with disclaimer
                last_result = self.step_history[-1] if self.step_history else ""
                return (
                    f"[PARTIAL RESULTS - Max steps ({self.max_internal_steps}) reached]\n\n"
                    f"{last_result}\n\n"
                    f"Note: The objective '{self.objective}' was not fully completed, "
                    f"but here are the findings so far."
                )
            else:
                # Raise error
                raise MaxStepsExhaustedError(
                    objective=self.objective,
                    steps_taken=self.max_internal_steps,
                    last_output=self.step_history[-1] if self.step_history else ""
                )

        except ToolCallError as e:
            # Handle tool failures
            if self.partial_results_acceptable:
                return self._synthesize_partial_results_after_tool_error(e)
            raise

        except Exception as e:
            raise AgenticStepProcessorError(f"Error: {str(e)}") from e
```

### TUI Error Display

**Extend `promptchain/cli/error_handler.py`**:

```python
from promptchain.utils.agentic_step_processor import (
    MaxStepsExhaustedError,
    ToolCallError,
    ObjectiveAmbiguityError
)
from rich.panel import Panel
from rich.text import Text


def format_agentic_error(error: Exception) -> Panel:
    """Format AgenticStepProcessor errors for TUI display."""
    if isinstance(error, MaxStepsExhaustedError):
        text = Text()
        text.append("⚠ Reasoning Steps Exhausted\n\n", style="bold yellow")
        text.append(f"Objective: ", style="bold")
        text.append(f"{error.objective}\n\n")
        text.append(f"Steps Taken: ", style="bold")
        text.append(f"{error.steps_taken}\n\n")
        text.append("Last Output:\n", style="bold")
        text.append(f"{error.last_output[:300]}...\n\n", style="dim")
        text.append("Suggestion: ", style="bold cyan")
        text.append(
            "Try increasing max_internal_steps or breaking objective into "
            "smaller sub-tasks."
        )

        return Panel(
            text,
            title="[yellow]Max Reasoning Steps Reached[/yellow]",
            border_style="yellow"
        )

    elif isinstance(error, ToolCallError):
        text = Text()
        text.append("⚠ Tool Call Failed\n\n", style="bold red")
        text.append(f"Tool: ", style="bold")
        text.append(f"{error.tool_name}\n")
        text.append(f"Step: ", style="bold")
        text.append(f"{error.step_num}\n")
        text.append(f"Error: ", style="bold")
        text.append(f"{error.error_message}\n\n")
        text.append("Suggestion: ", style="bold cyan")
        text.append("Check tool configuration and availability.")

        return Panel(
            text,
            title="[red]Tool Call Error[/red]",
            border_style="red"
        )

    elif isinstance(error, ObjectiveAmbiguityError):
        text = Text()
        text.append("⚠ Objective Unclear\n\n", style="bold magenta")
        text.append(f"Objective: ", style="bold")
        text.append(f"{error.objective}\n")
        text.append(f"Clarification Attempts: ", style="bold")
        text.append(f"{error.clarification_attempts}\n\n")
        text.append("Suggestion: ", style="bold cyan")
        text.append("Rephrase objective with more specific requirements.")

        return Panel(
            text,
            title="[magenta]Ambiguous Objective[/magenta]",
            border_style="magenta"
        )

    else:
        # Generic agentic error
        return Panel(
            str(error),
            title="[red]Reasoning Error[/red]",
            border_style="red"
        )
```

**Update TUI App Error Handling** (`promptchain/cli/tui/app.py`):

```python
from promptchain.cli.error_handler import format_agentic_error
from promptchain.utils.agentic_step_processor import AgenticStepProcessorError

async def _handle_agent_execution(self, user_input: str):
    """Execute agent with agentic error handling."""
    try:
        # Agent execution...
        result = await self.agent_chain.process_input_async(user_input)
        # Display result...

    except AgenticStepProcessorError as e:
        # Format and display agentic error
        error_panel = format_agentic_error(e)
        self.query_one("#chat-area").mount(Static(error_panel))

        # Update reasoning widget to show error
        if self.reasoning_widget:
            self.reasoning_widget.fail_reasoning(str(e))

        # Log error
        self.session_manager.log_error(
            error_type=type(e).__name__,
            error_message=str(e),
            context={"objective": getattr(e, 'objective', None)}
        )

    except Exception as e:
        # Existing generic error handling...
        pass
```

### Unit Tests

**Test File**: `tests/cli/unit/test_agentic_error_handling.py`

```python
import pytest
from promptchain.utils.agentic_step_processor import (
    AgenticStepProcessor,
    MaxStepsExhaustedError,
    ToolCallError
)


@pytest.mark.asyncio
async def test_max_steps_exhaustion_raises_error():
    """Processor raises MaxStepsExhaustedError when steps exhausted."""
    processor = AgenticStepProcessor(
        objective="Impossible task requiring many steps",
        max_internal_steps=2,  # Very low limit
        model_name="openai/gpt-4o-mini",
        fallback_on_exhaustion=False  # Error mode
    )

    with pytest.raises(MaxStepsExhaustedError) as exc_info:
        await processor.process_async("Complete impossible task")

    error = exc_info.value
    assert error.objective == "Impossible task requiring many steps"
    assert error.steps_taken == 2


@pytest.mark.asyncio
async def test_max_steps_fallback_returns_partial():
    """Processor returns partial results when fallback enabled."""
    processor = AgenticStepProcessor(
        objective="Complex task",
        max_internal_steps=2,
        model_name="openai/gpt-4o-mini",
        fallback_on_exhaustion=True  # Fallback mode
    )

    result = await processor.process_async("Test input")

    assert result is not None
    assert "[PARTIAL RESULTS" in result
    assert "Max steps (2) reached" in result
```

### Success Criteria
- Custom exceptions defined and raised appropriately
- Graceful fallback option works
- TUI displays formatted error messages
- Error logging captures context
- Unit tests verify error handling

## Validation
1. Run unit tests: `pytest tests/cli/unit/test_agentic_error_handling.py -v`
2. Manual TUI test with low max_steps
3. Verify error display formatting

## Deliverable
- Error classes in `agentic_step_processor.py`
- Error formatting in `error_handler.py`
- TUI integration in `app.py`
- Unit tests for error scenarios
