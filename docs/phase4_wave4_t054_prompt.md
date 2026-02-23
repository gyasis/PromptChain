# T054: AgenticStepProcessor Completion Detection and Display

## Objective
Implement robust objective completion detection in AgenticStepProcessor and display final synthesis prominently in TUI.

## Context Files
- `/home/gyasis/Documents/code/PromptChain/promptchain/utils/agentic_step_processor.py` (improve detection)
- `/home/gyasis/Documents/code/PromptChain/promptchain/cli/tui/app.py` (display synthesis)
- `/home/gyasis/Documents/code/PromptChain/promptchain/cli/tui/widgets/reasoning_progress.py` (T051 widget)

## Requirements

### Part 1: Improve Completion Detection Logic

**Enhance `promptchain/utils/agentic_step_processor.py`**:

```python
import re
from typing import Optional


class AgenticStepProcessor:
    """Extended with robust completion detection."""

    # Completion markers to look for in LLM responses
    COMPLETION_MARKERS = [
        "objective achieved",
        "task complete",
        "analysis complete",
        "final synthesis",
        "final result",
        "conclusion reached",
        "OBJECTIVE_COMPLETE",
        "[COMPLETE]"
    ]

    def _is_objective_complete(self, llm_response: str) -> bool:
        """Enhanced objective completion detection.

        Checks for:
        1. Explicit completion markers
        2. Structured completion tags
        3. Objective restatement with conclusion
        4. Synthesis indicators

        Args:
            llm_response: LLM response from current reasoning step

        Returns:
            True if objective appears to be complete
        """
        response_lower = llm_response.lower()

        # Check for explicit markers
        for marker in self.COMPLETION_MARKERS:
            if marker.lower() in response_lower:
                return True

        # Check for structured completion tag (ask LLM to use this)
        if re.search(r'<completion>(.*?)</completion>', llm_response, re.DOTALL):
            return True

        # Check for synthesis pattern: "Based on [analysis], [conclusion]"
        synthesis_patterns = [
            r'based on.*analysis.*conclude',
            r'after.*investigation.*determined',
            r'final.*synthesis.*shows',
            r'in conclusion.*findings'
        ]

        for pattern in synthesis_patterns:
            if re.search(pattern, response_lower, re.DOTALL):
                # Additional check: response should be substantial (not just starting)
                if len(llm_response) > 200:
                    return True

        # Check if objective is explicitly mentioned with conclusion
        objective_lower = self.objective.lower()
        objective_words = set(objective_lower.split())

        # Count how many objective keywords appear in response
        objective_word_matches = sum(
            1 for word in objective_words
            if len(word) > 3 and word in response_lower
        )

        # If >50% objective words appear + synthesis indicators, likely complete
        if (objective_word_matches / max(len(objective_words), 1)) > 0.5:
            synthesis_indicators = ['result', 'conclusion', 'finding', 'summary']
            if any(ind in response_lower for ind in synthesis_indicators):
                return True

        return False

    def _extract_final_synthesis(self, llm_response: str) -> str:
        """Extract final synthesis from completed reasoning.

        Looks for:
        1. Content within <completion> tags
        2. Final paragraph after synthesis markers
        3. Complete response if no specific markers

        Args:
            llm_response: LLM response containing final synthesis

        Returns:
            Extracted final synthesis text
        """
        # Check for structured completion tags
        completion_match = re.search(
            r'<completion>(.*?)</completion>',
            llm_response,
            re.DOTALL
        )
        if completion_match:
            return completion_match.group(1).strip()

        # Look for "Final Synthesis:" or similar headers
        synthesis_match = re.search(
            r'(?:final synthesis|final result|conclusion):?\s*(.*)',
            llm_response,
            re.IGNORECASE | re.DOTALL
        )
        if synthesis_match:
            return synthesis_match.group(1).strip()

        # If no specific markers, return full response
        # (It's the final output, so entire response is synthesis)
        return llm_response.strip()

    async def _execute_reasoning_step(
        self,
        user_input: str,
        step_num: int
    ) -> str:
        """Execute reasoning step with completion-aware prompt.

        Modified to ask LLM to explicitly signal completion.
        """
        # Build prompt with completion instructions
        if step_num == 1:
            base_prompt = f"""
You are an AI assistant performing multi-step reasoning to achieve this objective:

OBJECTIVE: {self.objective}

USER INPUT: {user_input}

Perform reasoning step {step_num} of up to {self.max_internal_steps} total steps.

IMPORTANT COMPLETION INSTRUCTIONS:
- If you have fully achieved the objective, wrap your final synthesis in <completion>...</completion> tags
- If you need more steps, explain what you've done and what remains to be investigated
- Use available tools when necessary to gather information

Begin your reasoning:
"""
        else:
            # Subsequent steps include history
            history_context = self._build_history_context()
            base_prompt = f"""
You are continuing multi-step reasoning to achieve this objective:

OBJECTIVE: {self.objective}

PREVIOUS STEPS:
{history_context}

Perform reasoning step {step_num} of up to {self.max_internal_steps} total steps.

IMPORTANT COMPLETION INSTRUCTIONS:
- If you have NOW fully achieved the objective, wrap your final synthesis in <completion>...</completion> tags
- If you need more steps, explain progress and next actions
- Use available tools when necessary

Continue your reasoning:
"""

        # Call LLM with enhanced prompt
        response = await self._call_llm_async(base_prompt)

        return response
```

### Part 2: TUI Display of Final Synthesis

**Extend `promptchain/cli/tui/app.py`**:

```python
from rich.panel import Panel
from rich.text import Text
from textual.widgets import Static


class PromptChainTUI(App):
    def _display_reasoning_synthesis(
        self,
        objective: str,
        steps_taken: int,
        final_synthesis: str
    ):
        """Display final reasoning synthesis prominently.

        Args:
            objective: Original objective
            steps_taken: Number of reasoning steps taken
            final_synthesis: Final synthesis result
        """
        # Create rich panel for synthesis
        synthesis_text = Text()
        synthesis_text.append("🎯 Objective: ", style="bold cyan")
        synthesis_text.append(f"{objective}\n\n", style="cyan")
        synthesis_text.append("📊 Steps Taken: ", style="bold green")
        synthesis_text.append(f"{steps_taken}\n\n", style="green")
        synthesis_text.append("✨ Final Synthesis:\n", style="bold yellow")
        synthesis_text.append(final_synthesis, style="")

        synthesis_panel = Panel(
            synthesis_text,
            title="[bold magenta]Multi-Hop Reasoning Complete[/bold magenta]",
            border_style="magenta",
            padding=(1, 2)
        )

        # Mount panel in chat area
        chat_area = self.query_one("#chat-area")
        chat_area.mount(Static(synthesis_panel))

        # Auto-scroll to show synthesis
        chat_area.scroll_end(animate=True)

    async def _handle_agent_execution(self, user_input: str):
        """Execute agent with synthesis display."""
        try:
            # Set up callbacks
            self._setup_agent_chain_callbacks()

            # Execute
            result = await self.agent_chain.process_input_async(user_input)

            # Check if result came from AgenticStepProcessor
            if self._was_agentic_reasoning():
                # Extract reasoning metadata
                reasoning_data = self._get_reasoning_metadata()

                # Display synthesis prominently
                self._display_reasoning_synthesis(
                    objective=reasoning_data["objective"],
                    steps_taken=reasoning_data["steps_taken"],
                    final_synthesis=result
                )

                # Complete reasoning widget
                if self.reasoning_widget:
                    self.reasoning_widget.complete_reasoning("Synthesis ready")
                    self.set_timer(2.0, self._remove_reasoning_widget)

            else:
                # Regular agent response
                self._display_agent_response(result)

        except Exception as e:
            # Error handling...
            pass

    def _was_agentic_reasoning(self) -> bool:
        """Check if last execution involved AgenticStepProcessor."""
        # Check if any agent has agentic_step_details
        for agent in self.agent_chain.agents.values():
            if hasattr(agent, 'agentic_step_details') and agent.agentic_step_details:
                return True
        return False

    def _get_reasoning_metadata(self) -> dict:
        """Extract reasoning metadata from last execution."""
        for agent in self.agent_chain.agents.values():
            if hasattr(agent, 'agentic_step_details') and agent.agentic_step_details:
                latest = agent.agentic_step_details[-1]
                return {
                    "objective": latest.get("objective", "N/A"),
                    "steps_taken": len(latest.get("internal_steps", [])),
                    "step_details": latest.get("internal_steps", [])
                }

        return {
            "objective": "N/A",
            "steps_taken": 0,
            "step_details": []
        }
```

### Part 3: Enhanced Synthesis Formatting

**Create Synthesis Formatter** (`promptchain/cli/utils/synthesis_formatter.py`):

```python
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown


def format_reasoning_synthesis(
    objective: str,
    steps_taken: int,
    final_synthesis: str,
    step_summaries: list = None
) -> Panel:
    """Format reasoning synthesis for rich display.

    Args:
        objective: Original objective
        steps_taken: Number of steps taken
        final_synthesis: Final synthesis text
        step_summaries: Optional list of step summaries

    Returns:
        Rich Panel with formatted synthesis
    """
    # Build content
    content = Text()

    # Header
    content.append("╔═══════════════════════════════════════╗\n", style="bold magenta")
    content.append("║  Multi-Hop Reasoning Complete  ║\n", style="bold magenta")
    content.append("╚═══════════════════════════════════════╝\n\n", style="bold magenta")

    # Objective
    content.append("🎯 ", style="bold cyan")
    content.append("Objective\n", style="bold cyan")
    content.append(f"{objective}\n\n", style="cyan")

    # Steps summary
    content.append("📊 ", style="bold green")
    content.append("Reasoning Process\n", style="bold green")
    content.append(f"Completed in {steps_taken} steps\n", style="green")

    if step_summaries:
        content.append("\nStep Summary:\n", style="bold")
        for idx, summary in enumerate(step_summaries[:5], 1):  # Limit to 5
            content.append(f"  {idx}. ", style="dim")
            content.append(f"{summary}\n", style="")

    content.append("\n")

    # Final synthesis (with markdown support)
    content.append("✨ ", style="bold yellow")
    content.append("Final Synthesis\n", style="bold yellow")

    # Check if synthesis contains markdown
    if any(marker in final_synthesis for marker in ['##', '**', '```', '- ']):
        # Render as markdown
        md = Markdown(final_synthesis)
        # Note: Can't directly append Markdown to Text, would need Console rendering
        # For now, append as plain text
        content.append(final_synthesis, style="")
    else:
        content.append(final_synthesis, style="")

    return Panel(
        content,
        border_style="magenta",
        padding=(1, 2),
        title="[bold white on magenta] REASONING COMPLETE [/bold white on magenta]"
    )
```

### Success Criteria
- Completion detection works reliably (>90% accuracy)
- Final synthesis extracted cleanly
- TUI displays synthesis prominently
- Formatting handles markdown content
- Step summaries included in display
- Auto-scroll to synthesis panel

## Testing

**Unit Test** (`tests/cli/unit/test_completion_detection.py`):

```python
import pytest
from promptchain.utils.agentic_step_processor import AgenticStepProcessor


def test_completion_detection_explicit_marker():
    """Completion detected from explicit marker."""
    processor = AgenticStepProcessor(
        objective="Test task",
        max_internal_steps=5,
        model_name="openai/gpt-4o-mini"
    )

    response = """
    I have completed the analysis. Based on my investigation,
    I can now provide the final result.

    OBJECTIVE_COMPLETE

    The key findings are...
    """

    assert processor._is_objective_complete(response) is True


def test_completion_detection_structured_tag():
    """Completion detected from structured tag."""
    processor = AgenticStepProcessor(
        objective="Research task",
        max_internal_steps=5,
        model_name="openai/gpt-4o-mini"
    )

    response = """
    After thorough research, here is the synthesis:

    <completion>
    The research shows that X leads to Y because of Z.
    Key findings: 1) ..., 2) ..., 3) ...
    </completion>
    """

    assert processor._is_objective_complete(response) is True


def test_synthesis_extraction():
    """Final synthesis extracted correctly."""
    processor = AgenticStepProcessor(
        objective="Test",
        max_internal_steps=5,
        model_name="openai/gpt-4o-mini"
    )

    response = """
    <completion>
    This is the final synthesis with key findings.
    </completion>
    """

    synthesis = processor._extract_final_synthesis(response)
    assert "final synthesis" in synthesis
    assert "<completion>" not in synthesis  # Tags removed
```

## Validation
- Run unit tests: `pytest tests/cli/unit/test_completion_detection.py -v`
- Manual TUI test with agentic workflows
- Verify synthesis display formatting

## Deliverable
- Enhanced completion detection in `AgenticStepProcessor`
- Synthesis extraction logic
- TUI synthesis display in `app.py`
- Synthesis formatter utility
- Unit tests for detection logic
