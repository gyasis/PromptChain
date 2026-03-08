"""Agent template definitions for PromptChain CLI.

This module provides pre-configured agent templates for common use cases:

BASIC TEMPLATES:
- Researcher: Deep research with AgenticStepProcessor and web search
- Coder: Code generation, execution, and validation
- Analyst: Data analysis and interpretation
- Terminal: Fast execution agent with no history (token-efficient)

SUPER AGENT TEMPLATES (Enhanced with leaked prompt patterns):
- super_coder: Elite code generation following Cursor IDE patterns
- linux_expert: System administration and shell mastery
- super_debugger: Root cause analysis with bounded fix attempts
- super_researcher: 8-step research with planning methodology
- security_auditor: OWASP-aligned defensive security review
- default_enhanced: Comprehensive general-purpose super agent

Templates can be instantiated via `create_from_template()`.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from promptchain.utils.agentic_step_processor import AgenticStepProcessor

from ..models.agent_config import Agent, HistoryConfig
from .prompt_templates import (AGENTIC_LOOP_BLOCK, CODER_SPECIALIZATION_BLOCK,
                               COMMUNICATION_BLOCK, DEBUGGING_BLOCK,
                               IDENTITY_BLOCK, LINUX_EXPERT_BLOCK,
                               MAKING_CODE_CHANGES_BLOCK, RESEARCHER_BLOCK,
                               SAFETY_SECURITY_BLOCK, SANDBOX_EXECUTION_BLOCK,
                               SEARCHING_AND_READING_BLOCK,
                               SECURITY_AUDITOR_BLOCK, SUPER_AGENT_PROMPT,
                               TERMINAL_EXECUTION_BLOCK, TOOL_CALLING_BLOCK,
                               WEB_SEARCH_BLOCK, build_instruction_chain,
                               get_default_mandatory_tools,
                               get_default_tool_registry)


@dataclass
class AgentTemplate:
    """Base agent template definition.

    Attributes:
        name: Template name (e.g., "researcher", "coder")
        display_name: Human-readable display name
        description: Template purpose and capabilities
        model: Default LLM model
        instruction_chain: Prompt templates and processing steps
        tools: List of tool identifiers
        history_config: History management configuration
        metadata: Additional template-specific metadata
    """

    name: str
    display_name: str
    description: str
    model: str
    instruction_chain: List[Union[str, Dict]]
    tools: List[str]
    history_config: HistoryConfig
    metadata: Optional[Dict] = None


# T023: Researcher Template with AgenticStepProcessor
RESEARCHER_TEMPLATE = AgentTemplate(
    name="researcher",
    display_name="Researcher",
    description="Deep research specialist with multi-hop reasoning and web search capabilities. "
    "Uses AgenticStepProcessor for 8-step autonomous research workflows.",
    model="openai/gpt-4.1-mini-2025-04-14",
    instruction_chain=[
        "Analyze research query: {input}",
        {
            "type": "agentic_step",
            "objective": "Conduct comprehensive research using multiple sources and reasoning steps",
            "max_internal_steps": 8,
            "tools": ["web_search", "mcp_web_browser", "mcp_filesystem_read"],
        },
        "Synthesize findings into comprehensive report: {input}",
    ],
    tools=["web_search"],
    history_config=HistoryConfig.for_agent_type(
        "researcher"
    ),  # T076: Use factory method
    metadata={
        "category": "research",
        "complexity": "high",
        "token_usage": "high",
    },
)


# T024: Coder Template with file ops + code execution + validation
CODER_TEMPLATE = AgentTemplate(
    name="coder",
    display_name="Coder",
    description="Code generation specialist with file operations, code execution, and validation. "
    "Supports test-driven development and iterative refinement.",
    model="openai/gpt-4.1-mini-2025-04-14",
    instruction_chain=[
        "Analyze coding request and identify requirements: {input}",
        "Generate implementation with proper error handling and documentation: {input}",
        {
            "type": "agentic_step",
            "objective": "Validate code through execution and testing, iterate until passing",
            "max_internal_steps": 5,
            "tools": ["mcp_filesystem_write", "execute_code", "run_tests"],
        },
    ],
    tools=["mcp_filesystem_read", "mcp_filesystem_write", "execute_code"],
    history_config=HistoryConfig.for_agent_type("coder"),  # T076: Use factory method
    metadata={
        "category": "development",
        "complexity": "high",
        "token_usage": "medium",
    },
)


# T025: Analyst Template with data analysis instruction chain
ANALYST_TEMPLATE = AgentTemplate(
    name="analyst",
    display_name="Analyst",
    description="Data analysis specialist for interpreting datasets, generating insights, "
    "and creating visualizations. Supports statistical analysis and reporting.",
    model="openai/gpt-4.1-mini-2025-04-14",
    instruction_chain=[
        "Understand data analysis request and objectives: {input}",
        "Load and explore dataset structure, identify patterns: {input}",
        "Perform statistical analysis and generate insights: {input}",
        "Create visualizations and comprehensive report: {input}",
    ],
    tools=["mcp_filesystem_read", "data_analysis"],
    history_config=HistoryConfig.for_agent_type("analyst"),  # T076: Use factory method
    metadata={
        "category": "analysis",
        "complexity": "medium",
        "token_usage": "medium-high",
    },
)


# T026: Terminal Template with history disabled + fast model
TERMINAL_TEMPLATE = AgentTemplate(
    name="terminal",
    display_name="Terminal",
    description="Fast execution agent for command-line operations and quick tasks. "
    "History disabled for maximum token efficiency (~60% savings). "
    "Uses fast model (gpt-3.5-turbo) for sub-second responses.",
    model="openai/gpt-3.5-turbo",  # Fast, cost-effective model
    instruction_chain=["{input}"],  # Direct pass-through, no additional processing
    tools=["execute_shell", "mcp_filesystem_read", "mcp_filesystem_write"],
    history_config=HistoryConfig.for_agent_type(
        "terminal"
    ),  # T076: Use factory method (~60% token savings)
    metadata={
        "category": "execution",
        "complexity": "low",
        "token_usage": "minimal",
        "response_time": "fast",
    },
)


# =============================================================================
# SUPER AGENT TEMPLATES (Enhanced with leaked prompt patterns)
# =============================================================================

# Super Coder: Elite code generation following Cursor IDE patterns
SUPER_CODER_TEMPLATE = AgentTemplate(
    name="super_coder",
    display_name="Super Coder",
    description="Elite code generation agent using Cursor IDE patterns. "
    "Execution-first philosophy: writes code via tools, never outputs raw code. "
    "Bounded linter loops (max 3), read-before-edit enforcement, file_edit tool preference.",
    model="openai/gpt-4.1-mini-2025-04-14",
    instruction_chain=[
        # Build comprehensive instruction using prompt_templates
        IDENTITY_BLOCK.format(
            agent_name="Super Coder",
            agent_description="Elite code generation specialist with execution-first philosophy",
        )
        + "\n\n"
        + TOOL_CALLING_BLOCK
        + "\n\n"
        + MAKING_CODE_CHANGES_BLOCK
        + "\n\n"
        + SEARCHING_AND_READING_BLOCK
        + "\n\n"
        + DEBUGGING_BLOCK
        + "\n\n"
        + SANDBOX_EXECUTION_BLOCK
        + "\n\n"
        + CODER_SPECIALIZATION_BLOCK
        + "\n\n"
        + "Now process this request:\n{input}",
    ],
    tools=[
        "file_read",
        "file_write",
        "file_edit",
        "file_append",
        "read_file_range",
        "insert_at_line",
        "replace_lines",
        "insert_after_pattern",
        "insert_before_pattern",
        "ripgrep_search",
        "list_directory",
        "create_directory",
        "sandbox_provision_uv",
        "sandbox_execute",
        "sandbox_cleanup",
        "terminal_execute",
    ],
    history_config=HistoryConfig.for_agent_type("coder"),
    metadata={
        "category": "development",
        "complexity": "expert",
        "token_usage": "high",
        "patterns": ["cursor_ide", "execution_first", "bounded_loops"],
    },
)

# Linux Expert: System administration and shell mastery
LINUX_EXPERT_TEMPLATE = AgentTemplate(
    name="linux_expert",
    display_name="Linux Expert",
    description="System administration specialist with deep Linux/Unix expertise. "
    "Shell command mastery, process management, permissions, networking, and troubleshooting. "
    "Uses terminal_execute with | cat for pagers, understands systemd, networking, and security.",
    model="openai/gpt-4.1-mini-2025-04-14",
    instruction_chain=[
        IDENTITY_BLOCK.format(
            agent_name="Linux Expert",
            agent_description="System administration and shell mastery specialist",
        )
        + "\n\n"
        + TOOL_CALLING_BLOCK
        + "\n\n"
        + TERMINAL_EXECUTION_BLOCK
        + "\n\n"
        + LINUX_EXPERT_BLOCK
        + "\n\n"
        + SAFETY_SECURITY_BLOCK
        + "\n\n"
        + "Now process this request:\n{input}",
    ],
    tools=[
        "terminal_execute",
        "file_read",
        "file_write",
        "file_edit",
        "list_directory",
        "create_directory",
        "ripgrep_search",
    ],
    history_config=HistoryConfig.for_agent_type("terminal"),
    metadata={
        "category": "system_admin",
        "complexity": "expert",
        "token_usage": "medium",
        "patterns": ["shell_mastery", "system_administration"],
    },
)

# Super Debugger: Root cause analysis with bounded fix attempts
SUPER_DEBUGGER_TEMPLATE = AgentTemplate(
    name="super_debugger",
    display_name="Super Debugger",
    description="Root cause analysis specialist with bounded fix attempts. "
    "Follows debugging workflow: reproduce -> isolate -> hypothesize -> verify -> fix. "
    "Maximum 3 fix attempts before escalating. Uses <thinking> tags for analysis.",
    model="openai/gpt-4.1-mini-2025-04-14",
    instruction_chain=[
        IDENTITY_BLOCK.format(
            agent_name="Super Debugger",
            agent_description="Root cause analysis specialist with systematic debugging methodology",
        )
        + "\n\n"
        + TOOL_CALLING_BLOCK
        + "\n\n"
        + DEBUGGING_BLOCK
        + "\n\n"
        + SEARCHING_AND_READING_BLOCK
        + "\n\n"
        + MAKING_CODE_CHANGES_BLOCK
        + "\n\n"
        + """
DEBUGGING WORKFLOW:
1. REPRODUCE: First, reproduce the issue to confirm it exists
2. ISOLATE: Use ripgrep_search to find relevant code paths
3. HYPOTHESIZE: Use <thinking> tags to reason about root cause
4. VERIFY: Add logging/assertions to test hypothesis
5. FIX: Apply minimal, targeted fix addressing root cause
6. TEST: Verify fix works and doesn't introduce regressions

BOUNDED ATTEMPTS:
- Maximum 3 fix attempts per issue
- After 3 failures, stop and report what was tried
- Never blindly guess - each attempt must be hypothesis-driven

Now debug this issue:
{input}""",
    ],
    tools=[
        "file_read",
        "file_edit",
        "read_file_range",
        "ripgrep_search",
        "sandbox_provision_uv",
        "sandbox_execute",
        "sandbox_cleanup",
        "terminal_execute",
        "list_directory",
    ],
    history_config=HistoryConfig.for_agent_type(
        "researcher"
    ),  # Full history for debugging context
    metadata={
        "category": "debugging",
        "complexity": "expert",
        "token_usage": "high",
        "patterns": ["root_cause_analysis", "bounded_attempts", "thinking_tags"],
    },
)

# Super Researcher: 8-step research with planning methodology
SUPER_RESEARCHER_TEMPLATE = AgentTemplate(
    name="super_researcher",
    display_name="Super Researcher",
    description="Advanced research agent with planning methodology and web search. "
    "8-step autonomous workflow with planskill-style difficulty assessment. "
    "Multi-source verification, progressive depth, comprehensive synthesis.",
    model="openai/gpt-4.1-mini-2025-04-14",
    instruction_chain=[
        IDENTITY_BLOCK.format(
            agent_name="Super Researcher",
            agent_description="Advanced research specialist with multi-hop reasoning and synthesis",
        )
        + "\n\n"
        + TOOL_CALLING_BLOCK
        + "\n\n"
        + WEB_SEARCH_BLOCK
        + "\n\n"
        + RESEARCHER_BLOCK
        + "\n\n"
        + SEARCHING_AND_READING_BLOCK
        + "\n\n"
        + """
RESEARCH METHODOLOGY (8-Step Workflow):

STEP 1 - QUERY ANALYSIS:
- Assess difficulty_level (1-5)
- Identify possible_vague_parts_of_query
- Determine if clarification needed

STEP 2 - PLANNING:
- Create research strategy
- Identify required sources
- Set depth and breadth targets

STEP 3 - PRIMARY RESEARCH:
- Use ripgrep_search for codebase exploration
- Use file_read for deep-dives
- Document findings progressively

STEP 4 - SECONDARY RESEARCH:
- Cross-reference multiple sources
- Verify claims with evidence
- Note conflicting information

STEP 5 - SYNTHESIS:
- Combine findings into coherent narrative
- Identify patterns and themes
- Note gaps in knowledge

STEP 6 - VERIFICATION:
- Fact-check key claims
- Test hypotheses where possible
- Acknowledge uncertainty

STEP 7 - DOCUMENTATION:
- Structure findings clearly
- Include source citations
- Highlight actionable insights

STEP 8 - DELIVERY:
- Summarize key findings
- Provide recommendations
- Offer next steps

Now research this topic:
{input}""",
    ],
    tools=[
        "ripgrep_search",
        "file_read",
        "read_file_range",
        "list_directory",
        "terminal_execute",
        # MCP Gemini tools for web research
        "mcp_gemini_gemini_research",
        "mcp_gemini_ask_gemini",
    ],
    history_config=HistoryConfig.for_agent_type("researcher"),
    metadata={
        "category": "research",
        "complexity": "expert",
        "token_usage": "high",
        "patterns": ["8_step_workflow", "planskill", "multi_source"],
        "requires_mcp": ["gemini"],  # Indicates this template needs gemini MCP server
    },
)

# Security Auditor: OWASP-aligned defensive security review
SECURITY_AUDITOR_TEMPLATE = AgentTemplate(
    name="security_auditor",
    display_name="Security Auditor",
    description="OWASP-aligned defensive security review specialist. "
    "Vulnerability assessment, secure code review, threat modeling. "
    "Focus on actionable remediation, not exploitation.",
    model="openai/gpt-4.1-mini-2025-04-14",
    instruction_chain=[
        IDENTITY_BLOCK.format(
            agent_name="Security Auditor",
            agent_description="OWASP-aligned defensive security review specialist",
        )
        + "\n\n"
        + TOOL_CALLING_BLOCK
        + "\n\n"
        + SECURITY_AUDITOR_BLOCK
        + "\n\n"
        + SAFETY_SECURITY_BLOCK
        + "\n\n"
        + SEARCHING_AND_READING_BLOCK
        + "\n\n"
        + """
SECURITY AUDIT WORKFLOW:

1. THREAT MODELING:
   - Identify attack surface
   - Map trust boundaries
   - Enumerate potential threats

2. VULNERABILITY SCAN:
   Use ripgrep_search to find:
   - SQL injection patterns: exec, query, cursor, %s, format(
   - XSS patterns: innerHTML, document.write, eval(
   - Command injection: subprocess, os.system, exec
   - Hardcoded secrets: password, api_key, secret, token
   - Insecure crypto: md5, sha1, DES

3. CODE REVIEW:
   - Authentication/authorization logic
   - Input validation and sanitization
   - Error handling and information disclosure
   - Session management
   - Access control

4. FINDINGS REPORT:
   Format: [SEVERITY] [CATEGORY] Description
   - CRITICAL: Immediate exploitation possible
   - HIGH: Significant risk, exploit likely
   - MEDIUM: Moderate risk, conditional exploit
   - LOW: Minor issue, defense-in-depth
   - INFO: Best practice recommendation

5. REMEDIATION:
   - Provide specific, actionable fixes
   - Include secure code examples
   - Reference OWASP guidelines

DEFENSIVE FOCUS:
- Goal is REMEDIATION, not exploitation
- Assist with authorized testing, CTF, educational contexts
- Refuse: DoS attacks, mass targeting, supply chain compromise

Now audit this:
{input}""",
    ],
    tools=[
        "ripgrep_search",
        "file_read",
        "read_file_range",
        "list_directory",
        "terminal_execute",
    ],
    history_config=HistoryConfig.for_agent_type("researcher"),
    metadata={
        "category": "security",
        "complexity": "expert",
        "token_usage": "high",
        "patterns": ["owasp", "defensive_security", "threat_modeling"],
    },
)

# Default Enhanced: Comprehensive general-purpose super agent
DEFAULT_ENHANCED_TEMPLATE = AgentTemplate(
    name="default_enhanced",
    display_name="Default Enhanced",
    description="Comprehensive general-purpose super agent combining all patterns. "
    "Execution-first philosophy with full tool access including web search. "
    "Suitable for any coding, file manipulation, research, or Linux task.",
    model="openai/gpt-4.1-mini-2025-04-14",
    instruction_chain=[
        SUPER_AGENT_PROMPT
        + "\n\n"
        + WEB_SEARCH_BLOCK
        + "\n\nNow process this request:\n{input}",
    ],
    tools=[
        # All 19 local tools available
        "sandbox_provision_uv",
        "sandbox_provision_docker",
        "sandbox_execute",
        "sandbox_list",
        "sandbox_cleanup",
        "file_read",
        "file_write",
        "file_edit",
        "file_append",
        "file_delete",
        "list_directory",
        "create_directory",
        "read_file_range",
        "insert_at_line",
        "replace_lines",
        "insert_after_pattern",
        "insert_before_pattern",
        "ripgrep_search",
        "terminal_execute",
        # MCP Gemini tools for web research (auto-connected)
        "mcp_gemini_gemini_research",
        "mcp_gemini_ask_gemini",
        "mcp_gemini_gemini_code_review",
        "mcp_gemini_gemini_brainstorm",
    ],
    history_config=HistoryConfig.for_agent_type("coder"),
    metadata={
        "category": "general",
        "complexity": "expert",
        "token_usage": "high",
        "patterns": ["all_patterns", "execution_first", "full_access"],
        "requires_mcp": ["gemini"],  # Auto-connect gemini MCP
    },
)


# Template registry for lookup
AGENT_TEMPLATES: Dict[str, AgentTemplate] = {
    # Basic templates
    "researcher": RESEARCHER_TEMPLATE,
    "coder": CODER_TEMPLATE,
    "analyst": ANALYST_TEMPLATE,
    "terminal": TERMINAL_TEMPLATE,
    # Super agent templates (enhanced with leaked prompt patterns)
    "super_coder": SUPER_CODER_TEMPLATE,
    "linux_expert": LINUX_EXPERT_TEMPLATE,
    "super_debugger": SUPER_DEBUGGER_TEMPLATE,
    "super_researcher": SUPER_RESEARCHER_TEMPLATE,
    "security_auditor": SECURITY_AUDITOR_TEMPLATE,
    "default_enhanced": DEFAULT_ENHANCED_TEMPLATE,
}


def create_from_template(
    template_name: str,
    agent_name: str,
    model_override: Optional[str] = None,
    description_override: Optional[str] = None,
) -> Agent:
    """Create Agent instance from template (T027).

    Args:
        template_name: Name of template. Available templates:
            BASIC: "researcher", "coder", "analyst", "terminal"
            SUPER AGENTS: "super_coder", "linux_expert", "super_debugger",
                         "super_researcher", "security_auditor", "default_enhanced"
        agent_name: Unique name for this agent instance
        model_override: Optional model to override template default
        description_override: Optional description to override template default

    Returns:
        Agent: Configured agent instance

    Raises:
        ValueError: If template_name not found

    Examples:
        >>> agent = create_from_template("researcher", "my-researcher")
        >>> agent = create_from_template("super_coder", "elite-dev")
        >>> agent = create_from_template("linux_expert", "sysadmin", model_override="claude-3-opus")
        >>> agent = create_from_template("default_enhanced", "super-agent")
    """
    if template_name not in AGENT_TEMPLATES:
        available = ", ".join(AGENT_TEMPLATES.keys())
        raise ValueError(
            f"Template '{template_name}' not found. Available templates: {available}"
        )

    template = AGENT_TEMPLATES[template_name]

    # Create agent from template
    agent = Agent(
        name=agent_name,
        model_name=model_override or template.model,
        description=description_override or template.description,
        instruction_chain=template.instruction_chain.copy(),
        tools=template.tools.copy(),
        history_config=template.history_config,
        created_at=datetime.now().timestamp(),
        metadata={
            "template": template_name,
            "template_metadata": template.metadata,
        },
    )

    return agent


def list_templates() -> List[Dict[str, str]]:
    """List all available agent templates.

    Returns:
        List[Dict[str, str]]: List of template info dicts with name, display_name, description
    """
    return [
        {
            "name": template.name,
            "display_name": template.display_name,
            "description": template.description,
            "model": template.model,
            "complexity": template.metadata.get("complexity", "unknown"),
            "token_usage": template.metadata.get("token_usage", "unknown"),
        }
        for template in AGENT_TEMPLATES.values()
    ]


def get_template_info(template_name: str) -> Optional[Dict]:
    """Get detailed information about a specific template.

    Args:
        template_name: Name of template

    Returns:
        Optional[Dict]: Template info dictionary or None if not found
    """
    if template_name not in AGENT_TEMPLATES:
        return None

    template = AGENT_TEMPLATES[template_name]

    return {
        "name": template.name,
        "display_name": template.display_name,
        "description": template.description,
        "model": template.model,
        "instruction_steps": len(template.instruction_chain),
        "tools": template.tools,
        "history_enabled": template.history_config.enabled,
        "max_tokens": template.history_config.max_tokens,
        "metadata": template.metadata,
    }


def validate_template_name(template_name: str) -> bool:
    """Check if template name is valid.

    Args:
        template_name: Name to validate

    Returns:
        bool: True if template exists
    """
    return template_name in AGENT_TEMPLATES


# T101: Template Tool Validation

# Standard tool registry - the 19 tools actually available in CLI
STANDARD_TOOLS = {
    # Sandbox (Isolated Code Execution)
    "sandbox_provision_uv": "Fast Python venv with UV package manager",
    "sandbox_provision_docker": "Docker container with full OS isolation",
    "sandbox_execute": "Run code in provisioned sandbox",
    "sandbox_list": "List active sandboxes",
    "sandbox_cleanup": "Destroy sandbox and free resources",
    # File Operations
    "file_read": "Read complete file contents",
    "file_write": "Write/create file (overwrites existing)",
    "file_edit": "Replace text using sed-like operations",
    "file_append": "Append content to file",
    "file_delete": "Delete file (with confirmation)",
    # Directory Operations
    "list_directory": "List with permissions, size, dates",
    "create_directory": "Create with parents (mkdir -p)",
    # Advanced File Editing
    "read_file_range": "Read specific line range (efficient for large files)",
    "insert_at_line": "Insert content at line number",
    "replace_lines": "Replace range of lines",
    "insert_after_pattern": "Insert after regex match",
    "insert_before_pattern": "Insert before regex match",
    # Search
    "ripgrep_search": "Fast regex search across files (like rg/grep)",
    # Terminal
    "terminal_execute": "Run shell commands with security guardrails",
}


def validate_template_tools(
    template_name: str, available_tools: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Validate that all tools referenced in a template exist (T101).

    Args:
        template_name: Template to validate
        available_tools: Optional list of available tool names (defaults to STANDARD_TOOLS)

    Returns:
        Dict with validation results:
            - valid: bool (True if all tools available)
            - missing_tools: List[str] (tools that don't exist)
            - available_tools: List[str] (tools that exist)
            - warnings: List[str] (non-critical warnings)

    Example:
        >>> result = validate_template_tools("researcher")
        >>> if not result["valid"]:
        >>>     print(f"Missing tools: {result['missing_tools']}")
    """
    if template_name not in AGENT_TEMPLATES:
        return {
            "valid": False,
            "missing_tools": [],
            "available_tools": [],
            "warnings": [f"Template '{template_name}' not found"],
            "error": f"Invalid template name: {template_name}",
        }

    template = AGENT_TEMPLATES[template_name]

    # Use provided available_tools or default to STANDARD_TOOLS
    if available_tools is None:
        available_tools = list(STANDARD_TOOLS.keys())

    # Check which tools are missing
    missing_tools = [tool for tool in template.tools if tool not in available_tools]
    available_tool_list = [tool for tool in template.tools if tool in available_tools]

    # Generate warnings for MCP tools (they require server connection)
    warnings = []
    for tool in template.tools:
        if tool.startswith("mcp_"):
            warnings.append(f"Tool '{tool}' requires MCP server connection")

    valid = len(missing_tools) == 0

    return {
        "valid": valid,
        "missing_tools": missing_tools,
        "available_tools": available_tool_list,
        "warnings": warnings,
        "template_name": template_name,
        "total_tools": len(template.tools),
    }


def validate_all_templates(
    available_tools: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """Validate all templates for tool availability (T101).

    Args:
        available_tools: Optional list of available tool names

    Returns:
        Dict mapping template names to validation results

    Example:
        >>> results = validate_all_templates()
        >>> for template_name, result in results.items():
        >>>     if not result["valid"]:
        >>>         print(f"{template_name}: Missing {result['missing_tools']}")
    """
    results = {}
    for template_name in AGENT_TEMPLATES.keys():
        results[template_name] = validate_template_tools(template_name, available_tools)
    return results
