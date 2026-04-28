"""Legacy TUI prompt generator.

Preserves the exact system prompt that the Herself Health terminal TUI has been
shipping since 2025-09. The prompt body is frozen byte-for-byte against
``tests/fixtures/legacy_tui_prompt.snapshot.txt``; the only runtime
substitution is the objective slot.

This module is a drop-in `BasePromptBuilder` implementation. It is pure,
deterministic, and makes no network or filesystem calls on ``generate``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# The set of tool names the legacy prompt body names explicitly. If the
# registered tools differ, ``generate`` emits a single advisory warning so the
# operator knows the prompt text is out of sync with the actual tool registry.
_EXPECTED_TUI_DEFAULT_TOOL_NAMES = frozenset(
    {
        "task_list_write_tool",
        "ripgrep_search",
        "file_read",
        "file_write",
        "file_edit",
        "terminal_execute",
        "list_directory",
        "create_directory",
    }
)


# Frozen template — byte-identical to
# ``tests/fixtures/legacy_tui_prompt.snapshot.txt``. The ``{self.objective}``
# token is the ONLY runtime slot. Double-braces (``{{...}}``) in the example
# block are intentional: they mirror the original f-string source so the
# rendered output matches the snapshot exactly.
_LEGACY_TEMPLATE = """Your goal is to achieve the following objective: {self.objective}

REACT WORKFLOW (Reason-Act-Observe):
You MUST follow this pattern for EVERY request:

STEP 1 - THINK: Analyze the request
- What is the user actually asking for?
- What type of task is this?
- What information or actions are needed?

STEP 2 - PLAN: Use task_list_write_tool to create your plan
IMMEDIATELY call task_list_write_tool with your planned tasks:
```
task_list_write_tool([
  {{"content": "Task description", "status": "pending", "activeForm": "Doing task..."}},
  {{"content": "Second task", "status": "pending", "activeForm": "Doing second task..."}}
])
```
This MUST be your FIRST tool call - always create a plan before executing.

STEP 3 - ACT: Execute one task at a time
- Mark task as "in_progress" when starting
- Use the appropriate tool for the current task
- Wait for the result

STEP 4 - OBSERVE: Check result and update plan
- Did it succeed? Mark task "completed"
- Error? Add recovery tasks to the list
- New information? Adjust remaining tasks

STEP 5 - REPEAT: Continue until all tasks complete

AVAILABLE TOOLS (choose correctly):
- task_list_write_tool: REQUIRED first - create/update your task plan
- ripgrep_search: Search LOCAL files and code (NOT for web searches!)
- file_read/file_write/file_edit: File operations
- terminal_execute: Run shell commands
- list_directory/create_directory: Directory operations
- sandbox_*: Safe code execution environment

MCP TOOLS (external services via Model Context Protocol):
- mcp_gemini_gemini_research: WEB SEARCH - Use this to search the internet for current information
- mcp_gemini_ask_gemini: Ask Gemini general questions or get a second opinion
- mcp_gemini_gemini_brainstorm: Creative brainstorming and idea generation
- mcp_gemini_gemini_debug: Debug errors with Gemini's help
- mcp_gemini_gemini_code_review: Get code review feedback

IMPORTANT TOOL SELECTION:
- "search the web/internet/online" -> USE mcp_gemini_gemini_research (NOT ripgrep!)
- "find packages/libraries/docs online" -> USE mcp_gemini_gemini_research
- "find in files/code locally" -> Use ripgrep_search
- "read a file" -> Use file_read
- "run a command" -> Use terminal_execute
- "run a script" -> Use terminal_execute('python /path/to/script.py')

EXECUTING SCRIPTS - ALWAYS RUN WHEN ASKED:
When the user asks you to "run", "execute", or "create and run" a script:
1. Create the script file with file_write
2. THEN run it with terminal_execute('python /absolute/path/to/script.py')
3. Show the actual OUTPUT from the script execution
4. DO NOT just create the script and tell the user to run it themselves
5. If the script generates files (HTML, images, etc.), report the output paths

Example workflow for "create a Plotly chart":
1. file_write('/path/to/chart.py', code_content)
2. terminal_execute('python /path/to/chart.py')
3. Show the script output AND confirm the generated file exists

PATH HANDLING - ABSOLUTE PATHS REQUIRED:
ALWAYS use and return FULL ABSOLUTE PATHS (e.g., /home/user/project/file.py), NEVER relative paths (e.g., ../file.py, ./src).

Tools for path operations:
- resolve_path: Convert ANY path to absolute path (use before reporting paths to user)
- find_paths: Find files/dirs and get their absolute paths
- get_cwd: Get current working directory
- path_info: Check if path exists and get full info

SECURITY MODES - Path boundary handling depends on session security mode:
- STRICT: All outside-directory paths require explicit confirmation. Tool returns `requires_confirmation: true`.
- DEFAULT: First access shows warning, then auto-allows subsequent access. Tool returns warning message once.
- TRUSTED: No boundary warnings - all paths accessible without confirmation.

When you receive a path warning or `requires_confirmation: true`:
1. Inform user the path is outside working directory
2. In STRICT mode: Wait for explicit confirmation before proceeding
3. In DEFAULT/TRUSTED mode: Proceed but mention the location to user

When reporting file/directory locations to the user:
- WRONG: "Found it at ../hybridrag"
- CORRECT: "Found it at /home/user/Documents/code/hybridrag"

CRITICAL:
- ALWAYS call task_list_write_tool FIRST to show your plan
- Update task status as you work (pending -> in_progress -> completed)
- If a task fails, add a recovery task and continue
- Only give final answer after completing all tasks with real tool results

FINAL ANSWER REQUIREMENTS:
- Your final answer MUST include the FULL content/information from tool results
- DO NOT just say "I have explained" or "I have provided" - actually SHOW the information
- If a tool returns documentation, code examples, or explanations - include that content in your response
- If you used mcp_gemini_ask_gemini or similar - include the actual answer Gemini gave, not a summary
- The user cannot see tool results directly - they only see YOUR final response
- WRONG: "I have explained how to add MCP servers" (user sees nothing useful)
- CORRECT: Include the actual steps, code, and explanations from the tool response
"""


def _extract_tool_names(tools: List[Dict[str, Any]]) -> List[str]:
    """Pull tool names from OpenAI-format tool schemas.

    Supports both wrapped (``{"type": "function", "function": {"name": ...}}``)
    and flat (``{"name": ...}``) shapes. Missing/malformed entries are silently
    skipped — mismatch warnings are advisory, not fatal.
    """
    names: List[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            names.append(fn["name"])
            continue
        name = tool.get("name")
        if isinstance(name, str):
            names.append(name)
    return names


class LegacyTUIPromptGenerator:
    """Frozen copy of the legacy Herself Health TUI system prompt.

    The prompt body is byte-identical to the on-disk snapshot; only the
    ``objective`` argument is interpolated. ``tools`` is accepted for Protocol
    conformance but not used to render content — it is only inspected to warn
    when the caller's tool set diverges from the names named in the prompt.

    Example:
        >>> gen = LegacyTUIPromptGenerator()
        >>> prompt = gen.generate("Summarize this file", tools=[])
        >>> prompt.startswith("Your goal is to achieve the following objective:")
        True
    """

    def __init__(self) -> None:
        # Nothing to wire — the template is a module constant so instances are
        # cheap and share state. Kept as an explicit init for Protocol clarity.
        pass

    def generate(
        self,
        objective: str,
        tools: List[Dict[str, Any]],
        context: Optional[str] = None,
    ) -> str:
        """Render the legacy system prompt with ``objective`` interpolated.

        Args:
            objective: Non-empty string describing the agent's goal. Inlined
                verbatim into the template's ``{self.objective}`` slot.
            tools: OpenAI-format tool schemas. Accepted for Protocol
                conformance; not used to shape content. If non-empty and its
                names diverge from :data:`_EXPECTED_TUI_DEFAULT_TOOL_NAMES`,
                a single :func:`logging.Logger.warning` is emitted.
            context: Optional prior scratchpad. When provided, appended as a
                ``PRIOR CONTEXT:\\n<context>`` block after the frozen body.

        Returns:
            The fully rendered system prompt.
        """
        # Warn-only diagnostic — never blocks rendering.
        if tools:
            provided = set(_extract_tool_names(tools))
            if provided and provided != _EXPECTED_TUI_DEFAULT_TOOL_NAMES:
                unexpected = sorted(provided - _EXPECTED_TUI_DEFAULT_TOOL_NAMES)
                missing = sorted(_EXPECTED_TUI_DEFAULT_TOOL_NAMES - provided)
                logger.warning(
                    "LegacyTUIPromptGenerator: registered tools diverge from "
                    "the legacy prompt body. unexpected=%s missing=%s. The "
                    "prompt text is frozen and will not reflect these tools.",
                    unexpected,
                    missing,
                )

        # Direct str.replace keeps the template opaque — no format-spec parsing,
        # no escaped-brace surprises. The token is unique in the template.
        body = _LEGACY_TEMPLATE.replace("{self.objective}", objective)

        if context is not None:
            return f"{body}\n\nPRIOR CONTEXT:\n{context}"
        return body

    def get_token_estimate(
        self,
        objective: str,
        tools: List[Dict[str, Any]],
    ) -> int:
        """Estimate the token count of ``generate(objective, tools)``.

        Uses ``tiktoken`` (cl100k_base) when available, otherwise falls back to
        ``len(prompt) // 4`` — a reasonable English-text rule of thumb. Always
        non-negative.
        """
        rendered = self.generate(objective, tools)
        try:
            import tiktoken  # type: ignore[import-untyped]

            encoding = tiktoken.get_encoding("cl100k_base")
            return max(0, len(encoding.encode(rendered)))
        except Exception:
            # tiktoken missing, encoding unavailable, or runtime error — use
            # the crude character-based fallback rather than propagating.
            return max(0, len(rendered) // 4)
