#!/usr/bin/env python3
"""
Agentic Chat Team with Gemini MCP Integration
==============================================

A 5-agent conversational system with:
- Research, Analysis, Terminal, Documentation, and Synthesis agents
- Gemini MCP server integration for information searches
- Complete logging and history management
- Interactive chat sessions with persistent context

Usage:
    python agentic_team_chat.py                    # Default: verbose logging
    python agentic_team_chat.py --quiet            # Suppress all logging
    python agentic_team_chat.py --no-logging       # Disable file logging
    python agentic_team_chat.py --log-level INFO   # Set specific log level
"""

import asyncio
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from promptchain import PromptChain
from promptchain.utils.agent_chain import AgentChain
from promptchain.utils.execution_history_manager import ExecutionHistoryManager
from promptchain.utils.logging_utils import RunLogger
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

# Import visual output system
from visual_output import ChatVisualizer
from promptchain.tools.terminal.terminal_tool import TerminalTool

# Load environment variables
load_dotenv()


def setup_gemini_mcp_config():
    """Configure Gemini MCP server integration"""
    return [{
        "id": "gemini",
        "type": "stdio",
        "command": "/home/gyasis/Documents/code/claude_code-gemini-mcp/.venv/bin/python",
        "args": ["/home/gyasis/Documents/code/claude_code-gemini-mcp/server.py"],
        "cwd": "/home/gyasis/Documents/code/claude_code-gemini-mcp",
        "env": {}
    }]


def create_research_agent(mcp_config, verbose=False):
    """
    Research Agent: Expert at finding information and conducting research
    Uses Gemini MCP for advanced search capabilities with AgenticStepProcessor
    """
    # Create agentic step processor with Gemini MCP access
    research_step = AgenticStepProcessor(
        objective="""You are the Research Agent with access to Gemini search capabilities.

OBJECTIVE: Conduct comprehensive research using available Gemini search tools, verify information, and provide detailed findings with sources.

Your expertise:
- Deep web research using Gemini tools (gemini_research, ask_gemini)
- Fact verification from multiple sources
- Comprehensive information gathering
- Trend analysis and pattern recognition
- Source citation and validation

Research methodology:
1. Use gemini_research for broad topics requiring Google Search grounding
2. Use ask_gemini for direct questions and clarifications
3. Verify critical facts from multiple angles
4. Organize findings logically with proper citations
5. Highlight confidence levels and uncertainties

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: MARKDOWN FORMATTING GUIDELINES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USE HEADERS, NOT NUMBERED LISTS:
✅ GOOD:
## Candle Library Overview
**Candle** is a minimalist ML framework for Rust...

### Key Features
Candle provides **pure Rust implementation** with no Python dependencies...

### Basic Example
```rust
use candle_core::{Tensor, Device};
```

❌ BAD (DON'T DO THIS):
1. Candle Library Overview
   • Candle is a minimalist ML framework
   • Key features:
      1. Pure Rust implementation
      2. No Python dependencies

FORMATTING RULES:
- Use ## for main sections (h2)
- Use ### for subsections (h3)
- Use **bold** for emphasis and key terms
- Use bullet points SPARINGLY - only for lists of 3+ related items
- Write in paragraphs with proper flow
- Use code blocks for examples
- Use tables for comparisons
- Avoid nested numbered lists

STRUCTURE YOUR RESPONSE:
## [Main Topic]
Brief intro paragraph with **key terms bold**.

### [Subtopic 1]
Paragraph explaining this aspect...

### [Subtopic 2]
Another paragraph...

### Quick Comparison (if relevant)
| Feature | Option A | Option B |
|---------|----------|----------|
| ...     | ...      | ...      |

### Summary
Final thoughts paragraph...

Always provide: findings, sources, confidence assessment, and recommendations for further research if needed.""",
        max_internal_steps=8,
        model_name="openai/gpt-4.1-mini",  # Latest model with 1M token context
        history_mode="progressive"  # Better multi-hop reasoning with full history
    )

    return PromptChain(
        models=[],  # Empty - AgenticStepProcessor has its own model_name
        instructions=[research_step],
        mcp_servers=mcp_config,  # Only research agent has Gemini access
        verbose=verbose,  # Configurable verbosity
        store_steps=True
    )


def create_analysis_agent(verbose=False):
    """
    Analysis Agent: Expert at analyzing data and extracting insights
    """
    analysis_step = AgenticStepProcessor(
        objective="""You are the Analysis Agent, an expert at analytical reasoning and insight extraction.

OBJECTIVE: Perform deep analytical reasoning to extract insights, identify patterns, and provide evidence-based conclusions.

Your expertise:
- Critical analysis of complex data and information
- Pattern recognition and trend identification
- Logical reasoning and inference
- Hypothesis formation and testing
- Risk assessment and opportunity evaluation
- Causal relationship analysis

Analytical methodology:
1. Decompose complex information into analyzable components
2. Identify key patterns, correlations, and anomalies
3. Apply logical reasoning to draw evidence-based conclusions
4. Assess confidence levels and identify uncertainties
5. Highlight actionable insights and strategic implications
6. Consider alternative explanations and counterarguments

Always provide: key insights, supporting evidence, confidence levels, and strategic recommendations.""",
        max_internal_steps=6,
        model_name="anthropic/claude-sonnet-4-5-20250929",  # Latest Sonnet 4.5 with 1M token context
        history_mode="progressive"  # Better multi-hop reasoning with full history
    )

    return PromptChain(
        models=[],  # Empty - AgenticStepProcessor has its own model_name
        instructions=[analysis_step],
        verbose=verbose,  # Configurable verbosity
        store_steps=True
    )


def create_terminal_agent(scripts_dir: Path = None, verbose=False):
    """
    Terminal Agent: Expert at executing terminal commands and system operations
    Has access to TerminalTool for safe command execution
    Can execute scripts written by the Coding Agent
    """
    # Create terminal tool with security guardrails
    terminal = TerminalTool(
        timeout=120,
        require_permission=False,  # Let agent make decisions autonomously
        verbose=verbose,  # Configurable verbosity
        use_persistent_session=True,
        session_name="agentic_terminal_session",
        visual_debug=False
    )

    scripts_note = f"\n\nScripts workspace: {scripts_dir}\nYou can execute scripts written by the Coding Agent from this directory." if scripts_dir else ""

    terminal_step = AgenticStepProcessor(
        objective=f"""You are the Terminal Agent, an expert at system operations and command-line execution.

OBJECTIVE: Execute terminal commands and scripts safely and efficiently to accomplish system tasks and gather information.

Your expertise:
- Safe terminal command execution
- System administration and diagnostics
- File system operations and navigation
- Process management and monitoring
- Package management and installation
- Environment configuration
- Log analysis and debugging
- Executing scripts from the Coding Agent{scripts_note}

Available tool:
- execute_terminal_command(command: str) -> str: Execute terminal commands with security guardrails

Command execution methodology:
1. Analyze what needs to be accomplished
2. Plan safe, efficient command sequences
3. Execute commands or scripts one at a time
4. Verify results and handle errors
5. Provide clear output interpretation
6. Suggest follow-up actions if needed

Script execution:
- Scripts from Coding Agent are in: {scripts_dir if scripts_dir else 'scripts directory'}
- Execute with: bash <script>.sh or python <script>.py
- Review script contents before execution if needed

Security guidelines:
- Avoid destructive operations without explicit user request
- Use safe flags (e.g., -i for interactive confirmations)
- Verify paths before operations
- Check command availability before execution
- Handle errors gracefully

Always provide: command explanations, execution results, output interpretation, and recommendations.""",
        max_internal_steps=7,
        model_name="openai/gpt-4.1-mini",  # Latest model with 1M token context
        history_mode="progressive"  # Better multi-hop reasoning with full history
    )

    # Create agent with terminal tool registered
    agent = PromptChain(
        models=[],  # Empty - AgenticStepProcessor has its own model_name
        instructions=[terminal_step],
        verbose=verbose,  # Configurable verbosity
        store_steps=True
    )

    # Create wrapper function with correct name for registration
    def execute_terminal_command(command: str) -> str:
        """Execute terminal command - wrapper for TerminalTool"""
        return terminal(command)

    # Register terminal tool
    agent.register_tool_function(execute_terminal_command)

    # Add tool schema for LLM
    agent.add_tools([{
        "type": "function",
        "function": {
            "name": "execute_terminal_command",
            "description": "Execute a terminal command safely with security guardrails. Returns command output (stdout/stderr). Use for system operations, file management, running programs, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The terminal command to execute (e.g., 'ls -la', 'python script.py', 'git status')"
                    }
                },
                "required": ["command"]
            }
        }
    }])

    return agent


def create_documentation_agent(verbose=False):
    """
    Documentation Agent: Expert at creating clear, comprehensive documentation
    """
    documentation_step = AgenticStepProcessor(
        objective="""You are the Documentation Agent, an expert technical writer and educator.

OBJECTIVE: Create clear, comprehensive, and user-friendly documentation that explains concepts effectively.

Your expertise:
- Clear technical writing and explanation
- User guide and tutorial creation
- Complex concept simplification
- Information architecture and organization
- Example creation and illustration
- Troubleshooting and FAQ development

Documentation methodology:
1. Identify target audience and their knowledge level
2. Structure information hierarchically (overview → details)
3. Use clear, concise, jargon-free language
4. Provide concrete, practical examples
5. Anticipate common questions and pain points
6. Include troubleshooting guides and best practices
7. Add visual aids descriptions when beneficial

Always provide: clear explanations, practical examples, common pitfalls, and quick-start guides.""",
        max_internal_steps=5,
        model_name="anthropic/claude-sonnet-4-5-20250929",  # Latest Sonnet 4.5 with 1M token context
        history_mode="progressive"  # Better multi-hop reasoning with full history
    )

    return PromptChain(
        models=[],  # Empty - AgenticStepProcessor has its own model_name
        instructions=[documentation_step],
        verbose=verbose,  # Configurable verbosity
        store_steps=True
    )


def create_synthesis_agent(verbose=False):
    """
    Synthesis Agent: Expert at combining insights and creating cohesive final outputs
    """
    synthesis_step = AgenticStepProcessor(
        objective="""You are the Synthesis Agent, an expert at integrating diverse information into unified strategic outputs.

OBJECTIVE: Synthesize insights from multiple sources into cohesive, actionable recommendations with strategic clarity.

Your expertise:
- Multi-source insight integration
- Narrative construction and storytelling
- Contradiction resolution and harmonization
- Strategic recommendation development
- Holistic thinking and systems analysis
- Actionable roadmap creation

Synthesis methodology:
1. Gather and review all available information comprehensively
2. Identify common themes, patterns, and unique insights
3. Resolve contradictions using logical analysis
4. Integrate perspectives into a unified framework
5. Develop clear, prioritized, actionable recommendations
6. Create strategic roadmaps with next steps
7. Highlight synergies and opportunities

Always provide: integrated insights, strategic recommendations, prioritized action items, and implementation considerations.""",
        max_internal_steps=6,
        model_name="openai/gpt-4.1-mini",  # Latest model with 1M token context
        history_mode="progressive"  # Better multi-hop reasoning with full history
    )

    return PromptChain(
        models=[],  # Empty - AgenticStepProcessor has its own model_name
        instructions=[synthesis_step],
        verbose=verbose,  # Configurable verbosity
        store_steps=True
    )


def create_coding_agent(scripts_dir: Path, verbose=False):
    """
    Coding Agent: Expert at writing scripts and code for terminal execution
    Writes scripts to shared workspace for terminal agent to execute
    """
    # Create write_script tool
    def write_script(filename: str, code: str) -> str:
        """Write a script to the scripts workspace"""
        try:
            # Ensure filename is safe (no path traversal)
            safe_filename = Path(filename).name
            script_path = scripts_dir / safe_filename

            # Write script
            with open(script_path, 'w') as f:
                f.write(code)

            # Make executable if it's a shell script
            if safe_filename.endswith(('.sh', '.bash', '.py')):
                import os
                os.chmod(script_path, 0o755)

            # Provide clear execution command
            if safe_filename.endswith('.py'):
                exec_cmd = f"python {script_path}"
            elif safe_filename.endswith(('.sh', '.bash')):
                exec_cmd = f"bash {script_path}"
            else:
                exec_cmd = f"{script_path}"

            return f"✅ Script written to: {script_path}\n✅ Terminal agent can execute it with: {exec_cmd}"
        except Exception as e:
            return f"❌ Error writing script: {e}"

    coding_step = AgenticStepProcessor(
        objective=f"""You are the Coding Agent, an expert at writing clean, efficient scripts and code.

OBJECTIVE: Write scripts, automation code, and utilities that the Terminal agent can execute.

Your expertise:
- Python scripting and automation
- Bash/shell scripting
- Data processing scripts
- System administration scripts
- API integration scripts
- File manipulation utilities

Scripts workspace: {scripts_dir}

Available tool:
- write_script(filename: str, code: str) -> str: Write a script to the workspace

Coding methodology:
1. Understand the task requirements clearly
2. Choose appropriate language (Python, Bash, etc.)
3. Write clean, well-commented code
4. Include error handling and validation
5. Make scripts executable and robust
6. Provide clear usage instructions
7. Follow best practices and security guidelines

Security guidelines:
- Validate all inputs
- Avoid hardcoded credentials
- Use safe file operations
- Include proper error handling
- Add helpful comments and documentation

Always provide: script filename, code with comments, usage instructions, and any dependencies needed.""",
        max_internal_steps=6,
        model_name="openai/gpt-4.1-mini",  # Latest model with 1M token context
        history_mode="progressive"  # Better multi-hop reasoning with full history
    )

    # Create agent with write_script tool
    agent = PromptChain(
        models=[],  # Empty - AgenticStepProcessor has its own model_name
        instructions=[coding_step],
        verbose=verbose,
        store_steps=True
    )

    # Register the write_script tool
    agent.register_tool_function(write_script)
    agent.add_tools([{
        "type": "function",
        "function": {
            "name": "write_script",
            "description": "Write a script to the scripts workspace for terminal agent execution",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Script filename (e.g., process_data.py, backup.sh)"
                    },
                    "code": {
                        "type": "string",
                        "description": "Complete script code with proper syntax"
                    }
                },
                "required": ["filename", "code"]
            }
        }
    }])

    return agent


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Agentic Chat Team - 6-Agent Collaborative System with Coding + Terminal Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                        # Run with default verbose logging
  %(prog)s --quiet                # Suppress all console output
  %(prog)s --no-logging           # Disable file logging
  %(prog)s --log-level ERROR      # Only show errors
  %(prog)s --quiet --no-logging   # Minimal mode (no logging at all)
        """
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all console logging and verbose output'
    )

    parser.add_argument(
        '--dev',
        action='store_true',
        help='Dev mode: quiet terminal + full debug logs to file (recommended for development)'
    )

    parser.add_argument(
        '--no-logging',
        action='store_true',
        help='Disable file logging completely'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level for terminal output (default: INFO)'
    )

    parser.add_argument(
        '--no-history-management',
        action='store_true',
        help='Disable automatic history truncation'
    )

    parser.add_argument(
        '--max-history-tokens',
        type=int,
        default=180000,  # 90% of 200k standard context window for optimal model usage
        help='Maximum tokens for history (default: 180000 - 90%% of context limit)'
    )

    parser.add_argument(
        '--session-name',
        type=str,
        default=None,
        help='Custom session name for logs and cache'
    )

    return parser.parse_args()


def create_router_config():
    """Configure intelligent autonomous routing logic with multi-agent planning capability"""
    from datetime import datetime
    current_date = datetime.now().strftime("%Y-%m-%d")

    return {
        "models": ["openai/gpt-4.1-mini"],  # Latest model with 1M token context for better routing decisions
        "instructions": [None, "{input}"],
        "decision_prompt_templates": {
            "static_plan": f"""You are the Master Orchestrator, responsible for creating multi-agent execution plans.

CURRENT DATE: {current_date}
KNOWLEDGE CUTOFF: Most LLM training data is from early 2024 or earlier

USER REQUEST: {{user_input}}

CONVERSATION HISTORY:
{{history}}

AVAILABLE AGENTS & TOOLS:
{{agent_details}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AGENT TOOL CAPABILITIES (CRITICAL - USE CORRECT TOOLS FOR EACH TASK):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. RESEARCH AGENT ✅ [HAS TOOLS]
   Tools: gemini_research, ask_gemini (Gemini MCP with Google Search grounding)
   When to use: ANY question about current events, recent tech, latest updates, post-2024 info
   Capabilities: Real-time web search, fact verification, current information retrieval
   Example: "What is Candle library?" → MUST use Research (current docs/examples from web)

2. ANALYSIS AGENT ❌ [NO TOOLS - INTERNAL KNOWLEDGE ONLY]
   Tools: None (uses only training knowledge + provided data)
   When to use: Analyzing data/text already provided, pattern recognition, logical reasoning
   Limitations: Cannot get external info, stuck with training data cutoff

3. CODING AGENT ✅ [HAS TOOLS]
   Tools: write_script (creates Python/Bash files in workspace)
   When to use: Need to CREATE scripts/code files
   Capabilities: Writes executable scripts to disk

4. TERMINAL AGENT ✅ [HAS TOOLS]
   Tools: execute_terminal_command (runs shell commands)
   When to use: Need to EXECUTE commands or scripts
   Capabilities: Run scripts, check system, file operations

5. DOCUMENTATION AGENT ❌ [NO TOOLS - INTERNAL KNOWLEDGE ONLY]
   Tools: None (uses only training knowledge)
   When to use: Explaining concepts from training data, writing tutorials
   Limitations: CANNOT verify current info, knowledge cutoff applies

6. SYNTHESIS AGENT ❌ [NO TOOLS - INTERNAL KNOWLEDGE ONLY]
   Tools: None (uses only training knowledge + provided context)
   When to use: Combining/synthesizing information already gathered
   Limitations: Cannot get new info, only works with what's provided

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL ROUTING RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚨 KNOWLEDGE BOUNDARY DETECTION:
- If query is about NEW tech/libraries/events (post-2024) → MUST START with RESEARCH
- If query asks "what is X?" for unknown/recent tech → MUST START with RESEARCH
- If query needs CURRENT information → MUST START with RESEARCH
- If query can be answered from training data alone → Can use agents without tools
- If unsure whether info is current → SAFER to use RESEARCH first

🚨 TOOL REQUIREMENT DETECTION:
- Need web search/current info? → RESEARCH (has Gemini MCP)
- Need to write code files? → CODING (has write_script)
- Need to execute commands? → TERMINAL (has execute_terminal_command)
- Just explaining known concepts? → DOCUMENTATION or ANALYSIS (no tools needed)

🚨 MULTI-STEP PLANNING:
When task needs multiple capabilities, create a PLAN (sequence of agents):
- "What is X library? Show examples" → ["research", "documentation"]
  (research gets current info, documentation formats it clearly)
- "Research X and analyze findings" → ["research", "analysis"]
  (research gathers data, analysis processes it)
- "Create backup script and run it" → ["coding", "terminal"]
  (coding writes script, terminal executes it)
- "Find latest AI trends and create strategy" → ["research", "synthesis"]
  (research gets current trends, synthesis creates strategy)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES OF CORRECT PLANS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Query: "What is Rust's Candle library?"
✅ CORRECT: {{"plan": ["research", "documentation"], "initial_input": "Research Rust Candle library - current features, examples, and documentation"}}
❌ WRONG: {{"plan": ["documentation"], ...}} ← Documentation has NO tools, will hallucinate!

Query: "Analyze this dataset: [data]"
✅ CORRECT: {{"plan": ["analysis"], "initial_input": "Analyze the provided dataset..."}}
❌ No research needed - data already provided

Query: "Create a log cleanup script and run it"
✅ CORRECT: {{"plan": ["coding", "terminal"], "initial_input": "Create log cleanup script, then execute it"}}

Query: "What are latest GPT-5 features and how to use them?"
✅ CORRECT: {{"plan": ["research", "documentation"], "initial_input": "Research GPT-5 latest features (current as of {current_date}), then create usage guide"}}

Query: "Explain how neural networks work"
✅ CORRECT: {{"plan": ["documentation"], "initial_input": "Explain neural networks fundamentals"}}
❌ No research needed - well-established concept in training data

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RETURN JSON ONLY (no other text):
{{
    "plan": ["agent1", "agent2", ...],  // Ordered list of agents to execute sequentially
    "initial_input": "refined task description for first agent",
    "reasoning": "why this plan and these specific agents with their tools"
}}

If only ONE agent needed, plan can have single agent: {{"plan": ["agent_name"], ...}}
""",
            "single_agent_dispatch": """You are the Orchestrator, responsible for routing tasks to the most capable agent.

USER REQUEST: {user_input}

CONVERSATION HISTORY:
{history}

AVAILABLE AGENTS:
{agent_details}

AGENT CAPABILITIES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. RESEARCH AGENT
   - Has exclusive access to Gemini MCP tools (gemini_research, ask_gemini)
   - Expert at: web research, fact-finding, information gathering, source verification
   - Best for: "find information about X", "research Y", "what are the latest trends in Z"
   - Uses: AgenticStepProcessor with up to 8 reasoning steps

2. ANALYSIS AGENT
   - Expert at: data analysis, pattern recognition, critical thinking, insight extraction
   - Best for: "analyze this data", "what patterns exist", "evaluate X vs Y"
   - Uses: AgenticStepProcessor with up to 6 reasoning steps

3. CODING AGENT
   - Has write_script tool to create Python/Bash scripts saved to scripts workspace
   - Expert at: writing scripts, automation code, utilities, data processing scripts
   - Best for: "write a script to X", "create a Python program for Y", "automate Z task"
   - Scripts saved to workspace for Terminal agent execution
   - Uses: AgenticStepProcessor with up to 6 reasoning steps

4. TERMINAL AGENT
   - Has access to TerminalTool for safe command execution (execute_terminal_command)
   - Can execute scripts from Coding Agent's workspace
   - Expert at: terminal commands, script execution, file operations, system administration
   - Best for: "execute this script", "run commands", "check system status", "install package"
   - Uses: AgenticStepProcessor with up to 7 reasoning steps

5. DOCUMENTATION AGENT
   - Expert at: technical writing, tutorials, concept explanation, user guides
   - Best for: "explain how X works", "create guide for Y", "document Z"
   - Uses: AgenticStepProcessor with up to 5 reasoning steps

6. SYNTHESIS AGENT
   - Expert at: integrating insights, strategic planning, actionable recommendations
   - Best for: "combine insights", "create strategy", "final recommendations"
   - Uses: AgenticStepProcessor with up to 6 reasoning steps

ROUTING LOGIC:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- If request needs web search or fact-finding → RESEARCH (has Gemini tools)
- If request needs data analysis or pattern recognition → ANALYSIS
- If request needs writing a script or code → CODING (has write_script tool)
- If request needs executing commands or scripts → TERMINAL (has TerminalTool)
- If request needs terminal commands or system operations → TERMINAL (has TerminalTool)
- If request needs explanation or documentation → DOCUMENTATION
- If request needs synthesis or strategic planning → SYNTHESIS

Consider conversation history for context-aware routing.
Each agent uses AgenticStepProcessor for multi-step reasoning autonomy.

RETURN JSON ONLY:
{{"chosen_agent": "agent_name", "refined_query": "optimally refined query for selected agent", "reasoning": "brief explanation of routing choice"}}
"""
        }
    }


def create_agentic_orchestrator(agent_descriptions: dict, log_event_callback=None):
    """
    Create AgenticStepProcessor-based orchestrator - returns async wrapper function for AgentChain

    Args:
        agent_descriptions: Dict of agent names -> descriptions
        log_event_callback: Optional callback for logging orchestrator decisions
    """
    from datetime import datetime

    # Format agent descriptions for the objective
    agents_formatted = "\n".join([f"- {name}: {desc}" for name, desc in agent_descriptions.items()])

    # Create the orchestrator PromptChain once (closure captures this)
    orchestrator_step = AgenticStepProcessor(
        objective=f"""You are the Master Orchestrator. Analyze the user request and choose the best agent.

CURRENT DATE: {{current_date}}
KNOWLEDGE CUTOFF: Most LLM training data is from early 2024 or earlier

AVAILABLE AGENTS:
{agents_formatted}

AGENT TOOL CAPABILITIES:
1. RESEARCH ✅ [HAS TOOLS] - gemini_research, ask_gemini (Google Search)
   → Use for: Current events, recent tech, latest updates, "What is X?" queries

2. ANALYSIS ❌ [NO TOOLS] - Training knowledge only
   → Use for: Analyzing provided data, pattern recognition

3. CODING ✅ [HAS TOOLS] - write_script
   → Use for: Creating script files

4. TERMINAL ✅ [HAS TOOLS] - execute_terminal_command
   → Use for: Executing commands/scripts

5. DOCUMENTATION ❌ [NO TOOLS] - Training knowledge only
   → Use for: Explaining well-known concepts from training

6. SYNTHESIS ❌ [NO TOOLS] - Training knowledge only
   → Use for: Combining already-gathered information

CRITICAL ROUTING RULES:
🔍 Query about NEW/UNKNOWN tech (post-2024)? → RESEARCH (web search required)
🔍 Query "what is X library/tool"? → RESEARCH (get current docs/examples)
📚 Query about WELL-KNOWN concepts? → DOCUMENTATION (use training knowledge)
💻 Need to CREATE code/scripts? → CODING (write_script tool)
⚡ Need to EXECUTE commands? → TERMINAL (execute tool)

EXAMPLES:
- "What is Candle library?" → {{"chosen_agent": "research", "reasoning": "Unknown library needs web research"}}
- "Explain neural networks" → {{"chosen_agent": "documentation", "reasoning": "Well-known concept in training"}}
- "Create backup script" → {{"chosen_agent": "coding", "reasoning": "Need write_script tool"}}
- "Run ls command" → {{"chosen_agent": "terminal", "reasoning": "Need execute tool"}}

YOUR OUTPUT MUST BE VALID JSON (nothing else):
{{"chosen_agent": "agent_name", "reasoning": "brief explanation"}}
""",
        max_internal_steps=5,  # Multi-hop reasoning
        model_name="openai/gpt-4.1-mini",
        history_mode="progressive"  # Context accumulation - CRITICAL for multi-hop reasoning!
    )

    orchestrator_chain = PromptChain(
        models=[],  # AgenticStepProcessor has its own model
        instructions=[orchestrator_step],
        verbose=False,
        store_steps=True
    )

    # Return async wrapper function that AgentChain expects
    async def agentic_router_wrapper(
        user_input: str,
        history: list,  # AgentChain passes 'history', not 'conversation_history'
        agent_descriptions: dict
    ) -> str:
        """Async wrapper for AgenticStepProcessor orchestrator with decision logging"""
        # Inject current date dynamically
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Log the orchestrator input
        if log_event_callback:
            log_event_callback("orchestrator_input", {
                "user_query": user_input,
                "history_size": len(history),
                "current_date": current_date
            }, level="DEBUG")

        # Use the orchestrator chain with progressive history
        result = await orchestrator_chain.process_prompt_async(user_input)

        # Parse and log the orchestrator decision
        try:
            import json
            decision = json.loads(result)
            chosen_agent = decision.get("chosen_agent", "unknown")
            reasoning = decision.get("reasoning", "no reasoning provided")

            # Get step count from orchestrator (if available)
            step_count = len(orchestrator_chain.step_outputs) if hasattr(orchestrator_chain, 'step_outputs') else 0

            # Log the complete orchestrator decision
            if log_event_callback:
                log_event_callback("orchestrator_decision", {
                    "chosen_agent": chosen_agent,
                    "reasoning": reasoning,
                    "raw_output": result,
                    "internal_steps": step_count,
                    "user_query": user_input
                }, level="INFO")

            logging.info(f"🎯 ORCHESTRATOR DECISION: Agent={chosen_agent} | Steps={step_count} | Reasoning: {reasoning}")

        except json.JSONDecodeError as e:
            logging.warning(f"Failed to parse orchestrator output as JSON: {result}")
            if log_event_callback:
                log_event_callback("orchestrator_parse_error", {
                    "error": str(e),
                    "raw_output": result
                }, level="WARNING")

        return result  # Returns JSON string with chosen_agent

    return agentic_router_wrapper


async def main():
    """Main function to run the agentic chat team"""

    # Parse command-line arguments
    args = parse_arguments()

    # Setup logging with proper date-based organization FIRST (needed for file handler)
    today = datetime.now().strftime('%Y-%m-%d')
    log_dir = Path("./agentic_team_logs") / today
    log_dir.mkdir(parents=True, exist_ok=True)

    # Determine session name early (needed for log file)
    session_name = args.session_name or f"session_{datetime.now().strftime('%H%M%S')}"
    session_log_path = log_dir / f"{session_name}.log"  # .log extension for detailed debug logs
    session_jsonl_path = log_dir / f"{session_name}.jsonl"  # .jsonl for structured events

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LOGGING ARCHITECTURE: Separate Terminal and File Logging
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Terminal Handler: Controlled by --quiet or --dev flags
    # File Handler: ALWAYS DEBUG level (captures everything)
    # This allows "pretty dev mode" where terminal is quiet but file has full details
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Clear any existing handlers
    logging.root.handlers = []

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(levelname)s - %(message)s')

    # 1. File Handler - ALWAYS DEBUG level (captures everything)
    if not args.no_logging:
        file_handler = logging.FileHandler(session_log_path, mode='a')
        file_handler.setLevel(logging.DEBUG)  # Capture EVERYTHING to file
        file_handler.setFormatter(detailed_formatter)
        logging.root.addHandler(file_handler)

    # 2. Terminal Handler - Controlled by flags
    terminal_handler = logging.StreamHandler()
    if args.dev or args.quiet:
        # Dev mode or quiet mode: suppress terminal output
        terminal_handler.setLevel(logging.ERROR)
    else:
        # Normal mode: show based on log level argument
        terminal_handler.setLevel(getattr(logging, args.log_level))
    terminal_handler.setFormatter(simple_formatter)
    logging.root.addHandler(terminal_handler)

    # Set root logger to DEBUG (handlers control what's actually shown/saved)
    logging.root.setLevel(logging.DEBUG)

    # Configure specific loggers
    logging.getLogger("promptchain").setLevel(logging.DEBUG)  # Capture all promptchain events
    logging.getLogger("httpx").setLevel(logging.WARNING)  # Reduce HTTP noise
    logging.getLogger("LiteLLM").setLevel(logging.INFO)  # Show LLM calls

    # Always filter out the "No models defined" warning (expected behavior with AgenticStepProcessor)
    import warnings
    warnings.filterwarnings("ignore", message=".*No models defined in PromptChain.*")

    # Configure output based on quiet/dev flags
    verbose = not (args.quiet or args.dev)

    # Log the logging configuration
    logging.debug(f"━━━ LOGGING CONFIGURATION ━━━")
    logging.debug(f"Mode: {'DEV (quiet terminal + full debug file)' if args.dev else 'QUIET (minimal output)' if args.quiet else 'VERBOSE'}")
    logging.debug(f"Terminal Level: {terminal_handler.level} ({logging.getLevelName(terminal_handler.level)})")
    logging.debug(f"File Level: DEBUG (all events)" if not args.no_logging else "File Logging: DISABLED")
    logging.debug(f"Session Log: {session_log_path}")
    logging.debug(f"Event Log: {session_jsonl_path}")
    logging.debug(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # Initialize visual output system
    viz = ChatVisualizer()

    if verbose:
        viz.render_header(
            "AGENTIC CHAT TEAM",
            "6-Agent Collaborative System with Multi-Hop Reasoning"
        )
        viz.render_system_message("Initializing agents and systems...", "info")

    # Cache organized by date (log_dir already created earlier)
    cache_dir = Path("./agentic_team_logs/cache") / today
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Scripts directory for coding agent (shared workspace for generated scripts)
    scripts_dir = Path("./agentic_team_logs/scripts") / today
    scripts_dir.mkdir(parents=True, exist_ok=True)

    if verbose and not args.no_logging:
        print(f"✅ Debug Log: {session_log_path}")
        print(f"✅ Event Log: {session_jsonl_path}")
        print(f"✅ Scripts workspace: {scripts_dir}")

    # Initialize history manager with configurable settings
    history_manager = ExecutionHistoryManager(
        max_tokens=args.max_history_tokens,
        max_entries=100,
        truncation_strategy="oldest_first"
    )

    if verbose:
        print(f"✅ History manager: {args.max_history_tokens} max tokens")

    # Setup MCP configuration for Gemini
    mcp_config = setup_gemini_mcp_config()

    # Create the 6 specialized agents with AgenticStepProcessor
    # NOTE: All agents are created with verbose=False for clean terminal output
    # All detailed logging goes to the session log file
    if verbose:
        print("🔧 Creating 6 specialized agents with agentic reasoning capabilities...\n")

    agents = {
        "research": create_research_agent(mcp_config, verbose=False),  # Only agent with Gemini MCP access
        "analysis": create_analysis_agent(verbose=False),
        "coding": create_coding_agent(scripts_dir, verbose=False),  # Coding agent with script writing capability
        "terminal": create_terminal_agent(scripts_dir, verbose=False),  # Terminal agent with TerminalTool and scripts access
        "documentation": create_documentation_agent(verbose=False),
        "synthesis": create_synthesis_agent(verbose=False)
    }

    agent_descriptions = {
        "research": "Expert researcher with exclusive Gemini MCP access (gemini_research, ask_gemini) for web search and fact-finding. Uses AgenticStepProcessor with 8-step reasoning.",
        "analysis": "Expert analyst for data analysis, pattern recognition, and insight extraction. Uses AgenticStepProcessor with 6-step reasoning.",
        "coding": f"Expert coder who writes Python/Bash scripts and saves them to {scripts_dir} for Terminal agent execution. Has write_script tool. Uses AgenticStepProcessor with 6-step reasoning.",
        "terminal": f"Expert system operator with TerminalTool access for executing commands and scripts from {scripts_dir}. Uses AgenticStepProcessor with 7-step reasoning.",
        "documentation": "Expert technical writer for clear guides, tutorials, and documentation. Uses AgenticStepProcessor with 5-step reasoning.",
        "synthesis": "Expert synthesizer for integrating insights and strategic planning. Uses AgenticStepProcessor with 6-step reasoning."
    }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ENHANCED EVENT LOGGING SYSTEM
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Structured events (JSONL) for orchestrator decisions, tool calls, agent actions
    # NO TRUNCATION - Capture complete details for debugging
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def log_event(event_type: str, data: dict, level: str = "INFO"):
        """
        Enhanced event logging - captures FULL details without truncation

        Args:
            event_type: Type of event (orchestrator_decision, tool_call, agent_response, etc.)
            data: Complete event data (not truncated)
            level: Log level (INFO, DEBUG, WARNING, ERROR)
        """
        if not args.no_logging:
            try:
                import json
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "level": level,
                    "event": event_type,
                    "session": session_name,
                    **data
                }
                # Write to JSONL with pretty formatting for readability (NO truncation)
                with open(session_jsonl_path, 'a') as f:
                    f.write(json.dumps(log_entry, indent=None, ensure_ascii=False) + "\n")

                # Also log to Python logging system for .log file
                log_msg = f"[{event_type}] " + " | ".join([f"{k}={v}" for k, v in data.items()])
                getattr(logging, level.lower(), logging.info)(log_msg)

            except Exception as e:
                logging.error(f"Event logging failed: {e}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ENHANCED TOOL CALL LOGGING HANDLER
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Captures tool calls with FULL query/args for debugging
    # Logs to both terminal (if verbose) and file (always)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    class ToolCallHandler(logging.Handler):
        """Enhanced handler to capture and log tool calls with full arguments"""

        def __init__(self, visualizer, log_event_fn):
            super().__init__()
            self.viz = visualizer
            self.log_event = log_event_fn
            self.last_tool = None

        def emit(self, record):
            import re
            msg = record.getMessage()

            # ━━━ MCP TOOL CALLS ━━━
            # Try to extract tool name and arguments from MCP logs
            if "CallToolRequest" in msg or "Tool call:" in msg:
                tool_name = "unknown_mcp_tool"
                tool_args = ""

                # Extract tool name and args from message
                # Pattern: "Calling MCP tool: tool_name with args: {json}"
                match = re.search(r'(?:Calling|Tool call:)\s+(?:MCP\s+)?tool:?\s+(\w+)(?:\s+with args:?\s+(.+))?', msg, re.IGNORECASE)
                if match:
                    tool_name = match.group(1)
                    tool_args = match.group(2) if match.group(2) else ""

                # Detect specific tools
                if "gemini_research" in msg.lower():
                    tool_name = "gemini_research"
                    # Try to extract query
                    query_match = re.search(r'(?:query|topic|prompt)["\']?\s*:\s*["\']([^"\']+)["\']', msg, re.IGNORECASE)
                    if query_match:
                        tool_args = query_match.group(1)
                elif "ask_gemini" in msg.lower():
                    tool_name = "ask_gemini"
                    query_match = re.search(r'(?:query|prompt)["\']?\s*:\s*["\']([^"\']+)["\']', msg, re.IGNORECASE)
                    if query_match:
                        tool_args = query_match.group(1)

                # Display in terminal if verbose
                self.viz.render_tool_call(tool_name, "mcp", tool_args)

                # Log to file with FULL details
                if self.log_event:
                    self.log_event("tool_call_mcp", {
                        "tool_name": tool_name,
                        "tool_type": "mcp",
                        "arguments": tool_args,
                        "raw_log": msg
                    }, level="INFO")

            # ━━━ LOCAL TOOL CALLS ━━━
            elif "Calling tool:" in msg or "write_script" in msg.lower() or "execute_terminal_command" in msg.lower():
                tool_name = "unknown_local_tool"
                tool_args = ""

                if "write_script" in msg.lower():
                    tool_name = "write_script"
                    # Try to extract filename
                    filename_match = re.search(r'filename["\']?\s*:\s*["\']([^"\']+)["\']', msg, re.IGNORECASE)
                    if filename_match:
                        tool_args = f"filename={filename_match.group(1)}"

                elif "execute_terminal_command" in msg.lower():
                    tool_name = "execute_terminal_command"
                    # Try to extract command
                    cmd_match = re.search(r'command["\']?\s*:\s*["\']([^"\']+)["\']', msg, re.IGNORECASE)
                    if cmd_match:
                        tool_args = cmd_match.group(1)

                # Display in terminal if verbose
                self.viz.render_tool_call(tool_name, "local", tool_args)

                # Log to file with FULL details
                if self.log_event:
                    self.log_event("tool_call_local", {
                        "tool_name": tool_name,
                        "tool_type": "local",
                        "arguments": tool_args,
                        "raw_log": msg
                    }, level="INFO")

    # Always add tool handler - tool calls should be visible even in quiet/dev mode
    tool_handler = ToolCallHandler(viz, log_event)
    tool_handler.setLevel(logging.DEBUG)  # Capture all tool activity
    logging.getLogger("promptchain").addHandler(tool_handler)
    # Also capture MCP server logs (from server.py)
    logging.getLogger().addHandler(tool_handler)

    # Create the agent chain with router mode
    if verbose:
        print("🚀 Creating AgentChain with intelligent routing...\n")

    agent_chain = AgentChain(
        agents=agents,
        agent_descriptions=agent_descriptions,
        execution_mode="router",
        router=create_agentic_orchestrator(agent_descriptions, log_event_callback=log_event),  # AgenticStepProcessor with decision logging
        cache_config={
            "name": session_name,
            "path": str(cache_dir)
        },
        verbose=False  # Clean terminal - all logs go to file
    )

    # Log initialization
    log_event("system_initialized", {
        "agents": list(agents.keys()),
        "mcp_servers": ["gemini"],
        "cache_location": str(cache_dir),
        "max_history_tokens": args.max_history_tokens
    })

    viz.render_system_message("System initialized successfully!", "success")

    # Display team roster with visual styling
    viz.render_team_roster(agents)

    # Display available commands
    viz.render_commands_help()

    # Track session stats
    stats = {
        "total_queries": 0,
        "agent_usage": {name: 0 for name in agents.keys()},
        "start_time": datetime.now(),
        "history_truncations": 0,
        "total_tokens_processed": 0
    }

    # Helper function for automatic history management
    def manage_history_automatically():
        """Automatically manage and truncate history if needed"""
        try:
            # Check current history size
            current_size = history_manager.current_token_count
            max_size = history_manager.max_tokens

            if current_size > max_size * 0.9:  # 90% threshold
                print(f"\n⚠️  History approaching limit ({current_size}/{max_size} tokens)")
                print("    Automatically truncating oldest entries...")

                # Truncate using built-in strategy
                history_manager.truncate_to_limit()
                stats["history_truncations"] += 1

                new_size = history_manager.current_token_count
                print(f"    ✅ History truncated: {current_size} → {new_size} tokens\n")

                # Log the truncation event
                log_event("history_truncated", {
                    "old_size": current_size,
                    "new_size": new_size,
                    "truncation_count": stats["history_truncations"]
                })

            return current_size
        except Exception as e:
            log_event("history_management_error", {"error": str(e)})
            return 0

    # Helper function to display history summary
    def show_history_summary():
        """Display a summary of current history"""
        try:
            total_entries = history_manager.history_size
            total_tokens = history_manager.current_token_count
            max_tokens = history_manager.max_tokens

            print("\n📊 HISTORY SUMMARY:")
            print("=" * 80)
            print(f"Total Entries: {total_entries}")
            print(f"Total Tokens: {total_tokens} / {max_tokens} ({(total_tokens/max_tokens*100):.1f}% full)")
            print(f"Truncations: {stats['history_truncations']}")
            print(f"Queries Processed: {stats['total_queries']}")
            print("=" * 80)

            # Show entry type breakdown
            entry_types = {}
            for entry in history_manager.history:
                entry_type = entry.get('type', 'unknown')
                entry_types[entry_type] = entry_types.get(entry_type, 0) + 1

            print("\nEntry Breakdown:")
            for entry_type, count in sorted(entry_types.items()):
                print(f"  - {entry_type}: {count}")
            print("=" * 80)

        except Exception as e:
            print(f"⚠️  Error displaying history: {e}")

    try:
        # Interactive chat loop
        while True:
            # Use Claude Code-style input box
            user_input = viz.get_input_with_box().strip()

            if not user_input:
                continue

            if user_input.lower() in ['exit', 'quit']:
                viz.render_system_message("Ending session...", "info")
                break

            if user_input.lower() == 'help':
                viz.render_commands_help()
                continue

            if user_input.lower() == 'clear':
                viz.clear_screen()
                viz.render_header(
                    "AGENTIC CHAT TEAM",
                    "6-Agent Collaborative System with Multi-Hop Reasoning"
                )
                continue

            if user_input.lower() == 'history':
                print("\n📜 CONVERSATION HISTORY:")
                print("=" * 80)
                formatted_history = history_manager.get_formatted_history(
                    format_style='chat',
                    max_tokens=4000
                )
                print(formatted_history)
                print("=" * 80)

                # Also show summary
                show_history_summary()
                continue

            if user_input.lower() == 'history-summary':
                show_history_summary()
                continue

            if user_input.lower() == 'stats':
                duration = datetime.now() - stats["start_time"]
                current_tokens = history_manager.current_token_count
                max_tokens = history_manager.max_tokens

                viz.render_stats({
                    "Session Duration": str(duration).split('.')[0],
                    "Total Queries": stats['total_queries'],
                    "History Truncations": stats['history_truncations'],
                    "History Size": f"{current_tokens} / {max_tokens} tokens",
                    "History Usage": f"{(current_tokens/max_tokens*100):.1f}%",
                    "Total Entries": history_manager.history_size
                })

                print("\nAgent Usage:")
                for agent, count in stats['agent_usage'].items():
                    print(f"  - {agent}: {count} queries")

                # Logging info
                print(f"\nLogs Directory: {log_dir}")
                print("=" * 80)
                continue

            if user_input.lower() == 'save':
                summary_path = log_dir / f"session_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(summary_path, 'w') as f:
                    f.write("AGENTIC TEAM SESSION SUMMARY\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"Session Duration: {datetime.now() - stats['start_time']}\n")
                    f.write(f"Total Queries: {stats['total_queries']}\n\n")
                    f.write("CONVERSATION HISTORY:\n")
                    f.write("=" * 80 + "\n")
                    f.write(history_manager.get_formatted_history(format_style='chat'))
                print(f"\n💾 Session saved to: {summary_path}")
                continue

            # Process user query
            stats['total_queries'] += 1

            # Automatic history management - check before processing
            current_history_size = manage_history_automatically()

            # Add user input to history
            history_manager.add_entry("user_input", user_input, source="user")

            # Comprehensive logging
            log_event("user_query", {
                "query": user_input,
                "history_size_before": current_history_size,
                "history_entries": history_manager.history_size
            })

            try:
                # Let the agent chain handle routing and execution
                # Clean terminal: just show processing indicator
                if verbose:
                    print("\n⏳ Processing...\n")

                # Execute agent chain
                response = await agent_chain.process_input(user_input)

                # Track which agent was used
                history_manager.add_entry("agent_output", response, source="agent_chain")

                # Render response with rich markdown formatting
                viz.render_agent_response("Agent", response, show_banner=False)

                # Log detailed agent response info to file (NO TRUNCATION)
                log_event("agent_response", {
                    "response": response,  # Full response, not truncated
                    "response_length": len(response),
                    "response_word_count": len(response.split()),
                    "history_size_after": history_manager.current_token_count
                }, level="INFO")

                # Try to extract step count from the last used agent's AgenticStepProcessor
                # Note: This is best-effort - AgentChain doesn't expose which agent ran
                # The orchestrator decision log already has this info
                logging.info(f"✅ Agent execution completed | Response length: {len(response)} chars")

            except Exception as e:
                error_msg = f"Error processing query: {str(e)}"
                viz.render_system_message(error_msg, "error")
                history_manager.add_entry("error", error_msg, source="system")

                # Log error details to file
                log_event("error", {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "query": user_input
                })

    finally:
        # Save final session summary
        if verbose:
            print("\n💾 Saving final session summary...")

        # Final log entry with complete session summary (all in one file)
        log_event("session_ended", {
            "total_queries": stats['total_queries'],
            "history_truncations": stats['history_truncations'],
            "duration": str(datetime.now() - stats['start_time']),
            "final_history_size": history_manager.current_token_count,
            "agent_usage": stats['agent_usage'],
            "log_file": str(session_log_path)
        })

        if verbose:
            print("\n👋 Goodbye!\n")


if __name__ == "__main__":
    asyncio.run(main())
