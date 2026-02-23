"""Task completion detection and handoff controller for agentic workflows.

This module implements the agentic loop control logic:
- Task status detection (COMPLETE, IN_PROGRESS, BLOCKED, REQUIRES_HANDOFF)
- Handoff routing to other agents or back to user
- Threshold conditions for user input requirements
- Progress tracking and continuation logic

Works with the prompt-level instructions in AGENTIC_LOOP_BLOCK to coordinate
agent behavior and workflow control.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class TaskStatus(Enum):
    """Task completion status signals from agentic agents."""

    COMPLETE = "TASK_COMPLETE"
    IN_PROGRESS = "TASK_IN_PROGRESS"
    BLOCKED = "TASK_BLOCKED"
    REQUIRES_HANDOFF = "TASK_REQUIRES_HANDOFF"
    UNKNOWN = "UNKNOWN"


@dataclass
class TaskState:
    """Tracks the current state of an agentic task."""

    status: TaskStatus = TaskStatus.UNKNOWN
    steps_completed: int = 0
    failed_attempts: int = 0
    steps_without_progress: int = 0
    last_response: str = ""
    handoff_target: Optional[str] = None
    handoff_reason: Optional[str] = None
    user_questions: List[str] = field(default_factory=list)
    partial_result: Optional[str] = None


@dataclass
class TaskControllerConfig:
    """Configuration for task controller thresholds."""

    max_internal_steps: int = 15
    max_failed_attempts: int = 3
    max_steps_without_progress: int = 5
    min_progress_chars: int = 100  # Minimum new chars to count as progress
    handoff_enabled: bool = True
    auto_continue_enabled: bool = True


class TaskController:
    """Controls agentic task flow, completion detection, and handoffs.

    This controller works in conjunction with the AGENTIC_LOOP_BLOCK prompt
    instructions. Agents are instructed to include status signals in their
    responses, which this controller parses to determine next actions.

    Example agent response patterns detected:

        [TASK_COMPLETE] Successfully implemented the feature...
        [TASK_BLOCKED] Need user input: What database should I use?
        [TASK_REQUIRES_HANDOFF:security_auditor] Security review needed...
    """

    # Regex patterns for status detection
    STATUS_PATTERNS = {
        TaskStatus.COMPLETE: re.compile(
            r'\[TASK_COMPLETE\]|\bTASK_COMPLETE\b|'
            r'✅\s*(?:Task|Objective)\s+[Cc]omplete|'
            r'(?:I have|I\'ve)\s+(?:successfully\s+)?completed',
            re.IGNORECASE
        ),
        TaskStatus.BLOCKED: re.compile(
            r'\[TASK_BLOCKED\]|\bTASK_BLOCKED\b|'
            r'(?:Need|Require|Waiting for)\s+(?:user\s+)?(?:input|clarification|decision)|'
            r'(?:Cannot|Can\'t)\s+proceed\s+without',
            re.IGNORECASE
        ),
        TaskStatus.REQUIRES_HANDOFF: re.compile(
            r'\[TASK_REQUIRES_HANDOFF(?::(\w+))?\]|'
            r'(?:Should|Need to)\s+hand\s*off\s+to|'
            r'(?:Recommend|Suggest)(?:ing)?\s+(?:that\s+)?(\w+)\s+agent',
            re.IGNORECASE
        ),
        TaskStatus.IN_PROGRESS: re.compile(
            r'\[TASK_IN_PROGRESS\]|\bTASK_IN_PROGRESS\b|'
            r'(?:Continuing|Still working|Next step|Working on)',
            re.IGNORECASE
        ),
    }

    # Pattern to extract handoff target
    HANDOFF_TARGET_PATTERN = re.compile(
        r'\[TASK_REQUIRES_HANDOFF:(\w+)\]|'
        r'hand\s*off\s+to\s+(?:the\s+)?(\w+)|'
        r'(\w+)\s+agent\s+should\s+(?:handle|review|process)',
        re.IGNORECASE
    )

    # Pattern to extract user questions
    USER_QUESTION_PATTERN = re.compile(
        r'(?:Need|Require)\s+(?:user\s+)?(?:input|answer)\s*(?:for|about|on|to)?[:\s]*([^\n.?]+\??)|'
        r'\?([^\n]+)\n|'
        r'(?:Please\s+)?(?:clarify|specify|confirm|choose)[:\s]*([^\n.]+)',
        re.IGNORECASE
    )

    def __init__(self, config: Optional[TaskControllerConfig] = None):
        """Initialize task controller.

        Args:
            config: Controller configuration. Uses defaults if not provided.
        """
        self.config = config or TaskControllerConfig()
        self.task_state = TaskState()
        self._previous_response_len = 0

    def reset(self) -> None:
        """Reset task state for a new task."""
        self.task_state = TaskState()
        self._previous_response_len = 0

    def detect_status(self, response: str) -> TaskStatus:
        """Detect task status from agent response.

        Parses the response looking for status signals that agents include
        based on the AGENTIC_LOOP_BLOCK instructions.

        Args:
            response: Agent response text

        Returns:
            Detected TaskStatus
        """
        # Check each pattern in priority order
        for status, pattern in self.STATUS_PATTERNS.items():
            if pattern.search(response):
                return status

        return TaskStatus.UNKNOWN

    def extract_handoff_target(self, response: str) -> Optional[str]:
        """Extract handoff target agent from response.

        Args:
            response: Agent response text

        Returns:
            Target agent name if found, None otherwise
        """
        match = self.HANDOFF_TARGET_PATTERN.search(response)
        if match:
            # Return first non-None group
            for group in match.groups():
                if group:
                    return group.lower()
        return None

    def extract_user_questions(self, response: str) -> List[str]:
        """Extract questions needing user input from response.

        Args:
            response: Agent response text

        Returns:
            List of questions/prompts for user
        """
        questions = []
        for match in self.USER_QUESTION_PATTERN.finditer(response):
            for group in match.groups():
                if group and group.strip():
                    questions.append(group.strip())
        return questions

    def check_progress(self, response: str) -> bool:
        """Check if meaningful progress was made in this step.

        Args:
            response: Agent response text

        Returns:
            True if progress detected, False otherwise
        """
        # Calculate new content added
        new_content = len(response) - self._previous_response_len
        self._previous_response_len = len(response)

        # Progress if substantial new content
        return new_content >= self.config.min_progress_chars

    def update_state(self, response: str, step_number: int) -> TaskState:
        """Update task state based on agent response.

        Main entry point for processing agent responses and updating
        the task state machine.

        Args:
            response: Agent response text
            step_number: Current step number in agentic loop

        Returns:
            Updated TaskState
        """
        # Detect status
        status = self.detect_status(response)

        # Update state
        self.task_state.status = status
        self.task_state.steps_completed = step_number
        self.task_state.last_response = response

        # Check progress
        if not self.check_progress(response):
            self.task_state.steps_without_progress += 1
        else:
            self.task_state.steps_without_progress = 0

        # Extract handoff info if needed
        if status == TaskStatus.REQUIRES_HANDOFF:
            self.task_state.handoff_target = self.extract_handoff_target(response)
            # Extract reason from response (first sentence after HANDOFF signal)
            handoff_match = self.STATUS_PATTERNS[TaskStatus.REQUIRES_HANDOFF].search(response)
            if handoff_match:
                start = handoff_match.end()
                reason_end = response.find('.', start) + 1 or start + 100
                self.task_state.handoff_reason = response[start:reason_end].strip()

        # Extract user questions if blocked
        if status == TaskStatus.BLOCKED:
            self.task_state.user_questions = self.extract_user_questions(response)

        # Store partial result for continuation
        self.task_state.partial_result = response

        return self.task_state

    def should_continue(self) -> Tuple[bool, str]:
        """Determine if agentic loop should continue.

        Returns:
            Tuple of (should_continue, reason)
        """
        state = self.task_state

        # Task complete - stop
        if state.status == TaskStatus.COMPLETE:
            return False, "Task completed successfully"

        # Task blocked - return to user
        if state.status == TaskStatus.BLOCKED:
            return False, f"Waiting for user input: {state.user_questions}"

        # Handoff needed - stop current agent
        if state.status == TaskStatus.REQUIRES_HANDOFF:
            return False, f"Handoff to {state.handoff_target}: {state.handoff_reason}"

        # Check thresholds
        if state.steps_completed >= self.config.max_internal_steps:
            return False, f"Max steps ({self.config.max_internal_steps}) reached"

        if state.failed_attempts >= self.config.max_failed_attempts:
            return False, f"Max failed attempts ({self.config.max_failed_attempts}) reached"

        if state.steps_without_progress >= self.config.max_steps_without_progress:
            return False, f"No progress for {self.config.max_steps_without_progress} steps"

        # Continue if in progress or unknown
        if state.status in (TaskStatus.IN_PROGRESS, TaskStatus.UNKNOWN):
            return True, "Task in progress"

        return False, "Unknown condition"

    def get_continuation_context(self) -> str:
        """Generate context for continuing the agentic loop.

        Creates a prompt that helps the agent continue from where it left off.

        Returns:
            Continuation context string
        """
        state = self.task_state

        return f"""Continue working on the task. Current progress:
- Steps completed: {state.steps_completed}/{self.config.max_internal_steps}
- Last action: {state.last_response[:200] if state.last_response else 'N/A'}...

Remember to check task completion after this step:
- TASK_COMPLETE if objective achieved
- TASK_BLOCKED if you need user input
- TASK_REQUIRES_HANDOFF if another agent should handle this
- Continue if more work needed"""

    def get_handoff_context(self) -> Dict[str, Any]:
        """Generate context for agent handoff.

        Returns:
            Dictionary with handoff context for target agent
        """
        state = self.task_state

        return {
            "source_agent_progress": {
                "steps_completed": state.steps_completed,
                "partial_result": state.partial_result,
            },
            "handoff_target": state.handoff_target,
            "handoff_reason": state.handoff_reason,
            "continuation_prompt": (
                f"Continuing task from previous agent. "
                f"Reason for handoff: {state.handoff_reason}\n\n"
                f"Previous progress:\n{state.partial_result[:500] if state.partial_result else 'No prior result'}..."
            ),
        }

    def get_user_return_context(self) -> Dict[str, Any]:
        """Generate context for returning control to user.

        Returns:
            Dictionary with context for user prompt
        """
        state = self.task_state

        return {
            "status": state.status.value,
            "steps_completed": state.steps_completed,
            "questions": state.user_questions,
            "partial_result": state.partial_result,
            "reason": self.should_continue()[1],
        }


# Convenience function for simple status detection
def detect_task_status(response: str) -> TaskStatus:
    """Quick status detection without full controller.

    Args:
        response: Agent response text

    Returns:
        Detected TaskStatus
    """
    controller = TaskController()
    return controller.detect_status(response)
