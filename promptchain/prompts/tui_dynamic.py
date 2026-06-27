"""Dynamic TUI prompt generator — static foundation + live tool inventory.

Design (per the TUI prompt-grounding work, 2026-06-27):

- ``TUI_FOUNDATION_PROMPT`` is the **static** system prompt — the curated,
  hand-authored guidance the TUI agent always reasons from (REACT discipline,
  script-execution rules, absolute-path handling, security modes, final-answer
  requirements). This is the source of truth; everything else builds on it.
- The **dynamic** layer only *adds*: the ``AVAILABLE TOOLS`` and ``MCP TOOLS``
  sections are rendered at ``generate`` time from the tools **actually
  registered/loaded** for the turn — so the prompt can never again advertise
  tools that don't exist, nor hide tools that do.

Contrast with siblings:
- ``LegacyTUIPromptGenerator`` — frozen v0.5.0 prompt with a *hardcoded* tool
  list (drifts from the registry; names MCP tools that may not be loaded).
- ``DynamicPromptGenerator`` — fully dynamic + minimal (drops the curated
  guidance entirely).

This generator is the middle path: static curated base, dynamic tool tail.
It implements the ``BasePromptBuilder`` Protocol.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# STATIC FOUNDATION — the canonical hand-authored base. The ONLY runtime slot
# is ``{objective}``. Tool inventory is intentionally NOT here; it is appended
# dynamically by ``DynamicTUIPromptGenerator.generate`` from the live registry.
# =============================================================================
TUI_FOUNDATION_PROMPT = """Your goal is to achieve the following objective: {objective}

REACT WORKFLOW (Reason-Act-Observe):
You MUST follow this pattern for EVERY request:

STEP 1 - THINK: Analyze the request — what is the user actually asking for, what
type of task is this, and what information or actions are needed?

STEP 2 - PLAN: Call task_list_write_tool FIRST with your planned tasks (each with
status "pending"). This MUST be your first tool call — always plan before acting:
```
task_list_write_tool([
  {"content": "Task description", "status": "pending", "activeForm": "Doing task..."}
])
```

STEP 3 - ACT: Execute ONE task at a time — mark it "in_progress", use the correct
tool for that task, and wait for the result.

STEP 4 - OBSERVE: Check the result — success → mark "completed"; error → add a
recovery task and continue; new information → adjust the remaining tasks.

STEP 5 - REPEAT until every task is complete.

EXECUTING SCRIPTS — ALWAYS RUN WHEN ASKED:
When the user asks you to "run", "execute", or "create and run" a script:
1. Create the file with file_write
2. Run it with terminal_execute('python /absolute/path/to/script.py')
3. Show the actual OUTPUT — do NOT just create it and tell the user to run it
4. If it generates files (HTML, images, etc.), report their absolute paths

PATH HANDLING — ABSOLUTE PATHS REQUIRED:
ALWAYS use and report FULL ABSOLUTE paths (e.g. /home/user/project/file.py), NEVER
relative paths (../file, ./src). Use resolve_path / find_paths / get_cwd / path_info
to resolve and verify paths before reporting them.

SECURITY MODES — path-boundary handling depends on the session security mode:
- STRICT: outside-directory paths require explicit confirmation (the tool returns
  requires_confirmation: true) — wait for the user before proceeding.
- DEFAULT: first access warns, then auto-allows — proceed but mention the location.
- TRUSTED: no boundary warnings.

CRITICAL:
- ALWAYS call task_list_write_tool FIRST to show your plan.
- Update task status as you work (pending → in_progress → completed).
- If a task fails, add a recovery task and continue.
- Only give your final answer after completing all tasks with real tool results.

FINAL ANSWER REQUIREMENTS:
- Your final answer MUST include the FULL content/information from tool results —
  actually SHOW it; do not say "I have explained" or "I have provided".
- The user cannot see tool results directly; they only see YOUR final response."""


def _tool_name(tool: Dict[str, Any]) -> Optional[str]:
    """Pull the tool name from an OpenAI-format schema (wrapped or flat)."""
    if not isinstance(tool, dict):
        return None
    fn = tool.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
        return fn["name"]
    name = tool.get("name")
    return name if isinstance(name, str) else None


def _tool_desc(tool: Dict[str, Any]) -> str:
    """Pull a short one-line description from an OpenAI-format schema."""
    fn = tool.get("function") if isinstance(tool, dict) else None
    desc = ""
    if isinstance(fn, dict):
        desc = fn.get("description") or ""
    if not desc:
        desc = tool.get("description", "") if isinstance(tool, dict) else ""
    first = (desc or "").strip().splitlines()[0] if desc else ""
    return first[:97] + "..." if len(first) > 100 else first


def _render_block(header: str, tools: List[Dict[str, Any]]) -> str:
    """Render a '- name: desc' bullet block from tool schemas (sorted by name)."""
    lines: List[str] = [header]
    rows = []
    for t in tools:
        n = _tool_name(t)
        if not n:
            continue
        d = _tool_desc(t)
        rows.append(f"- {n}: {d}" if d else f"- {n}")
    lines.extend(sorted(rows))
    return "\n".join(lines)


class DynamicTUIPromptGenerator:
    """Static TUI foundation + a dynamically-rendered, registry-accurate tool list.

    Example:
        >>> gen = DynamicTUIPromptGenerator()
        >>> p = gen.generate("Summarize file", tools=[{"function": {"name": "file_read", "description": "Read a file"}}])
        >>> "AVAILABLE TOOLS" in p and "file_read" in p
        True
    """

    def generate(
        self,
        objective: str,
        tools: List[Dict[str, Any]],
        context: Optional[str] = None,
    ) -> str:
        """Render the foundation with ``objective`` + a live tool inventory.

        Args:
            objective: Non-empty goal string, inlined into the foundation.
            tools: OpenAI-format schemas of the tools ACTUALLY registered/loaded
                for this turn. Split into local (``AVAILABLE TOOLS``) and
                ``mcp_*`` (``MCP TOOLS``); the MCP section is omitted entirely
                when no MCP tools are loaded (so the prompt never promises a
                tool that isn't there).
            context: Optional prior scratchpad, appended verbatim.

        Returns:
            The fully rendered system prompt.
        """
        parts: List[str] = [TUI_FOUNDATION_PROMPT.replace("{objective}", objective)]

        local = [t for t in tools if (_tool_name(t) or "") and not (_tool_name(t) or "").startswith("mcp_")]
        mcp = [t for t in tools if (_tool_name(t) or "").startswith("mcp_")]

        if local:
            parts.append(_render_block("AVAILABLE TOOLS (choose correctly):", local))

        if mcp:
            parts.append(_render_block(
                "MCP TOOLS (external services via Model Context Protocol):", mcp))
            # If a web-search-capable MCP tool is loaded, steer web queries to it
            # (ripgrep_search is LOCAL only). Only emitted when such a tool exists.
            web = next(
                (_tool_name(t) for t in mcp
                 if any(k in (_tool_name(t) or "") for k in ("research", "search", "web"))),
                None,
            )
            if web:
                parts.append(
                    f"WEB SEARCH: for internet/online queries use {web} "
                    f"(NOT ripgrep_search — that searches LOCAL files only)."
                )

        body = "\n\n".join(p for p in parts if p)
        if context is not None:
            return f"{body}\n\nPRIOR CONTEXT:\n{context}"
        return body

    def get_token_estimate(
        self,
        objective: str,
        tools: List[Dict[str, Any]],
    ) -> int:
        """Estimate token count of ``generate(objective, tools)`` (tiktoken or //4)."""
        rendered = self.generate(objective, tools)
        try:
            import tiktoken  # type: ignore[import-untyped]

            return max(0, len(tiktoken.get_encoding("cl100k_base").encode(rendered)))
        except Exception:
            return max(0, len(rendered) // 4)
