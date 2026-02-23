"""
OrchestratorSupervisor - Master orchestrator with oversight capabilities

Combines AgenticStepProcessor's multi-hop reasoning with supervisory control,
logging, event emissions, and decision tracking.
"""

from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain.utils.promptchaining import PromptChain
from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType
import json
import logging

logger = logging.getLogger(__name__)


class OrchestratorSupervisor:
    """
    Master orchestrator with multi-hop reasoning and oversight capabilities.

    Features:
    - AgenticStepProcessor for deep multi-step reasoning
    - Decision tracking and pattern analysis
    - Event emission for observability
    - Logging integration
    - Dev mode terminal output
    - Supervisory oversight and metrics
    """

    def __init__(
        self,
        agent_descriptions: Dict[str, str],
        log_event_callback: Optional[Callable] = None,
        dev_print_callback: Optional[Callable] = None,
        max_reasoning_steps: int = 8,
        model_name: str = "openai/gpt-4.1-mini",
        strategy_preference: str = "adaptive"  # ✅ NEW: Strategy guidance
    ):
        """
        Initialize the orchestrator supervisor.

        Args:
            agent_descriptions: Dict mapping agent names to descriptions
            log_event_callback: Callback for logging events
            dev_print_callback: Callback for --dev mode terminal output
            max_reasoning_steps: Max internal reasoning steps for AgenticStepProcessor
            model_name: Model to use for reasoning
            strategy_preference: Strategy to guide orchestrator behavior:
                - "adaptive": Let orchestrator decide (multi-hop reasoning, default)
                - "prefer_single": Bias toward single-agent dispatch when possible
                - "prefer_multi": Bias toward multi-agent plans for complex tasks
                - "always_research_first": Always start with research for unknown topics
                - "conservative": Only use multi-agent when absolutely necessary
                - "aggressive": Use multi-agent plans liberally
        """
        self.agent_descriptions = agent_descriptions
        self.log_event_callback = log_event_callback
        self.dev_print_callback = dev_print_callback
        self.strategy_preference = strategy_preference  # ✅ Store preference

        # Oversight metrics
        self.metrics = {
            "total_decisions": 0,
            "single_agent_dispatches": 0,
            "multi_agent_plans": 0,
            "total_reasoning_steps": 0,
            "total_tools_called": 0,
            "average_plan_length": 0.0,
            "decision_history": []
        }

        # Build objective with multi-hop reasoning guidance
        objective = self._build_multi_hop_objective()

        # Create AgenticStepProcessor as reasoning engine with reasoning step tool
        self.reasoning_engine = AgenticStepProcessor(
            objective=objective,
            max_internal_steps=max_reasoning_steps,
            model_name=model_name,
            history_mode="progressive"  # Critical for multi-hop reasoning
        )

        # Wrap in PromptChain for execution (PromptChain handles tool registration)
        self.orchestrator_chain = PromptChain(
            models=[],  # AgenticStepProcessor has its own model
            instructions=[self.reasoning_engine],
            verbose=False,
            store_steps=True
        )

        # Register reasoning step tool to enable visible multi-step analysis
        def output_reasoning_step(step_number: int, step_name: str, analysis: str) -> str:
            """Output a reasoning step during multi-hop analysis. Use this to show your thinking process.

            Args:
                step_number: The step number (1-4)
                step_name: Name of the step (e.g., "CLASSIFY TASK", "IDENTIFY CAPABILITIES")
                analysis: Your detailed analysis for this step

            Returns:
                Confirmation to continue to next step
            """
            # Display in terminal for real-time observability
            if self.dev_print_callback:
                self.dev_print_callback(f"🧠 Orchestrator Step {step_number}: {step_name}", "")
                # Show analysis with indentation (no truncation - user wants full output)
                for line in analysis.split('\n'):
                    self.dev_print_callback(f"   {line}", "")

            # Log the reasoning step for observability
            if self.log_event_callback:
                self.log_event_callback("orchestrator_reasoning_step", {
                    "step_number": step_number,
                    "step_name": step_name,
                    "analysis": analysis[:500]  # Truncate for logging
                }, level="INFO")

            # Provide step-specific guidance to enforce sequential execution
            if step_number == 1:
                return "✅ Step 1 (CLASSIFY TASK) recorded. REQUIRED: Now call output_reasoning_step with step_number=2 (IDENTIFY CAPABILITIES)."
            elif step_number == 2:
                return "✅ Step 2 (IDENTIFY CAPABILITIES) recorded. REQUIRED: Now call output_reasoning_step with step_number=3 (CHECK SUFFICIENCY)."
            elif step_number == 3:
                # Check if analysis indicates multi-agent is needed
                if "NEED_MULTI" in analysis.upper() or "INSUFFICIENT" in analysis.upper() or "MULTI-AGENT" in analysis.upper():
                    return "✅ Step 3 (CHECK SUFFICIENCY) recorded. Multi-agent needed. REQUIRED: Now call output_reasoning_step with step_number=4 (DESIGN COMBINATION)."
                else:
                    return "✅ Step 3 (CHECK SUFFICIENCY) recorded. Single agent sufficient. You may now provide the final plan JSON."
            elif step_number == 4:
                return "✅ Step 4 (DESIGN COMBINATION) recorded. All reasoning steps complete. Now provide the final plan JSON."
            else:
                return f"✅ Step {step_number} ({step_name}) recorded. Continue to next step or provide final plan JSON."

        # Register tool on the PromptChain (not AgenticStepProcessor)
        self.orchestrator_chain.register_tool_function(output_reasoning_step)
        self.orchestrator_chain.add_tools([{
            "type": "function",
            "function": {
                "name": "output_reasoning_step",
                "description": "Output a reasoning step during multi-hop analysis to show your thinking process explicitly. Call this for STEP 1, STEP 2, STEP 3, STEP 4, and STEP 5 (if needed) before providing the final plan JSON.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "step_number": {
                            "type": "integer",
                            "description": "The step number (1 for CLASSIFY, 2 for KNOWLEDGE_VS_EXECUTION, 3 for IDENTIFY, 4 for CHECK, 5 for DESIGN)"
                        },
                        "step_name": {
                            "type": "string",
                            "description": "Name of the step (e.g., 'CLASSIFY TASK', 'IDENTIFY CAPABILITIES', 'CHECK SUFFICIENCY', 'DESIGN COMBINATION')"
                        },
                        "analysis": {
                            "type": "string",
                            "description": "Your detailed analysis for this reasoning step"
                        }
                    },
                    "required": ["step_number", "step_name", "analysis"]
                }
            }
        }])

        # Register event callback to capture reasoning metadata
        self.orchestrator_chain.register_callback(self._orchestrator_event_callback)

        logger.info(f"OrchestratorSupervisor initialized with {len(agent_descriptions)} agents, max_steps={max_reasoning_steps}, strategy={strategy_preference}")

    def _get_strategy_guidance(self) -> str:
        """Get strategy-specific prompting guidance"""
        strategies = {
            "adaptive": """
✅ Use your best judgment with multi-hop reasoning
- Analyze task complexity thoroughly
- Choose single agent when sufficient
- Choose multi-agent when beneficial
- No bias - let reasoning determine optimal approach
            """,

            "prefer_single": """
⚡ PREFERENCE: Single-agent dispatch when possible
- Default to single agent unless task CLEARLY needs multiple capabilities
- Only use multi-agent if one agent absolutely cannot handle the task
- Examples:
  - "What is X? Give examples" → Try single agent first (research can handle both)
  - "Create and test script" → Must use multi-agent (coding → terminal)
            """,

            "prefer_multi": """
🔀 PREFERENCE: Multi-agent plans for thorough results
- Break complex tasks into specialized agent subtasks
- Use multi-agent even when single agent might work
- Prioritize specialization over simplicity
- Examples:
  - "What is X?" → Use ["research", "documentation"] for thorough coverage
  - "Explain Y" → Use ["analysis", "documentation"] for deeper insight
            """,

            "always_research_first": """
🔍 PREFERENCE: Research-first for unknown topics
- ANY query about tech/libraries/events → Start with research
- Use pattern: ["research", "documentation"] or ["research", "analysis"]
- Only skip research if topic is DEFINITELY in training data
- Examples:
  - "What is Zig?" → ["research", "documentation"]
  - "Explain neural networks" → Can use ["documentation"] (well-known)
            """,

            "conservative": """
💼 PREFERENCE: Conservative multi-agent usage
- Use single agent unless task has MULTIPLE distinct phases
- Require strong evidence for multi-agent necessity
- Examples of when multi-agent IS needed:
  - "Research X and create code" → ["research", "coding"]
  - "Create script and run it" → ["coding", "terminal"]
- Examples of when single agent suffices:
  - "What is X? Give examples" → ["research"] (one phase)
            """,

            "aggressive": """
🚀 PREFERENCE: Aggressive multi-agent orchestration
- Liberally use multi-agent plans for quality
- Break tasks into fine-grained subtasks
- Use 2-3+ agents even for moderate complexity
- Examples:
  - "What is X?" → ["research", "analysis", "documentation"]
  - "Create script" → ["research", "coding", "terminal"]
            """
        }

        return strategies.get(self.strategy_preference, strategies["adaptive"])

    def _build_multi_hop_objective(self) -> str:
        """Build comprehensive objective with multi-hop reasoning guidance"""
        agents_formatted = "\n".join([f"- {name}: {desc}" for name, desc in self.agent_descriptions.items()])

        # ✅ Strategy preference guidance
        strategy_guidance = self._get_strategy_guidance()

        return f"""You are the Master Orchestrator. Use multi-step reasoning to create the best execution plan.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRATEGY PREFERENCE: {self.strategy_preference.upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{strategy_guidance}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CURRENT DATE: {{current_date}}
KNOWLEDGE CUTOFF: Most LLM training data is from early 2024 or earlier

AVAILABLE AGENTS:
{agents_formatted}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  CRITICAL: CHECK FOR PENDING QUESTIONS FROM PREVIOUS AGENT (STEP 0)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BEFORE STARTING THE 5-STEP ANALYSIS, YOU MUST CHECK THE CONVERSATION HISTORY:

1. Look at the LAST agent response in the conversation history
2. Check if it ended with a QUESTION or REQUEST for user input
3. Check if the current user input is a SHORT RESPONSE (yes/no/ok/sure/single word/number)

If ALL THREE are true:
→ The user is RESPONDING to the agent's question
→ Route back to THE SAME AGENT that asked the question
→ SKIP the 5-step analysis entirely
→ Return plan with the previous agent immediately

Examples of PENDING QUESTION scenarios:

🔹 EXAMPLE 1: Installation Confirmation
Last Agent Response: "The library is not installed. Would you like me to install it?"
User Input: "yes"
Analysis: User is answering Terminal agent's question
Decision: Return ["terminal"] immediately (skip 5-step analysis)

🔹 EXAMPLE 2: Choice Selection
Last Agent Response: "I found 3 options. Which one should I use? (1/2/3)"
User Input: "2"
Analysis: User is selecting from options presented by previous agent
Decision: Return to that same agent immediately

🔹 EXAMPLE 3: Confirmation
Last Agent Response: "Script created. Should I run it now?"
User Input: "ok"
Analysis: User is confirming action from Coding agent
Decision: Return ["coding"] or ["coding", "terminal"] to complete workflow

🔹 EXAMPLE 4: NOT a response (proceed with 5-step analysis)
Last Agent Response: "Here is the analysis of the data."
User Input: "create a chart"
Analysis: User is making a NEW request, not answering a question
Decision: Proceed with normal 5-step analysis

⚠️ DETECTION RULES:

Question indicators in last agent response:
- Ends with "?"
- Contains phrases: "Would you like", "Should I", "Do you want", "Which one", "Select"
- Contains options: "(yes/no)", "(1/2/3)", "Choose"

Response indicators in user input:
- Short length (< 20 chars)
- Matches: yes/no/ok/sure/yeah/nope/confirm/cancel/proceed
- Matches: numbers (1, 2, 3)
- Matches: single word affirmative/negative

IF PENDING QUESTION DETECTED:
- Extract which agent asked the question from history
- SKIP all 5 reasoning steps
- Return JSON immediately: {{"plan": ["that_agent"], "initial_input": "{{user_input}}", "reasoning": "User responding to [agent]'s question: [question summary]"}}

IF NO PENDING QUESTION:
- Proceed with normal 5-step analysis below

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AGENT TOOL CAPABILITIES (CRITICAL - USE MULTI-HOP REASONING TO DETERMINE NEEDS):
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
🧠 MULTI-HOP REASONING PROCESS (USE ALL STEPS):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1: CLASSIFY TASK COMPLEXITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Determine if task is SIMPLE or COMPLEX:

✅ SIMPLE (likely single agent):
   - Single question: "What is X?", "Explain Y", "How does Z work?"
   - Single action: "Analyze this data", "Write documentation for X"
   - One primary goal with no distinct phases

❌ COMPLEX (likely multiple agents):
   - Multiple phases: "Research X and create tutorial"
   - Create + execute: "Write script and run it"
   - Gather + process + output: "Find latest X, analyze it, create report"

STEP 2: DISTINGUISH KNOWLEDGE vs EXECUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ CRITICAL: Even if you have system context (like current_date), the user may want VERIFICATION via command execution!

Knowledge-based queries (NO TOOLS NEEDED):
- "What year is it?" → Can answer from system context
- "Explain neural networks" → Training knowledge sufficient
- "Tell me about X concept" → Documentation/analysis sufficient

Execution-based queries (TOOLS REQUIRED):
- "Check the date" → TERMINAL needed (must run 'date' command)
- "Verify system time" → TERMINAL needed (execute actual check)
- "Run date command" → TERMINAL needed (explicit command request)
- "Get current timestamp" → TERMINAL needed (system execution)

🔴 IF QUERY CONTAINS EXECUTION VERBS, USE TOOLS:
- "check", "verify", "run", "execute", "test", "show me actual"
- These indicate user wants REAL SYSTEM OUTPUT, not just your knowledge

STEP 3: IDENTIFY REQUIRED CAPABILITIES & TOOLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What does the task ACTUALLY need to accomplish?

🔍 Need current/real-time info from web?
   → REQUIRES: Research agent ✅ (gemini_research, ask_gemini)

📝 Need to create code/script files?
   → REQUIRES: Coding agent ✅ (write_script)

⚡ Need to execute commands/scripts?
   → REQUIRES: Terminal agent ✅ (execute_terminal_command)

🧮 Need to analyze existing data/information?
   → OPTIONAL: Analysis agent ❌ (no tools, uses reasoning only)

📖 Need to explain/document known concepts?
   → OPTIONAL: Documentation agent ❌ (no tools, uses training knowledge)

🔗 Need to combine/synthesize gathered information?
   → OPTIONAL: Synthesis agent ❌ (no tools, uses reasoning only)

STEP 4: CHECK IF SINGLE AGENT HAS SUFFICIENT TOOLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compare task requirements to agent capabilities:

Does ONE agent have ALL the tools needed?
   → YES: Use single-agent plan ["agent_name"]
   → NO: Proceed to STEP 5 for multi-agent combination

Example checks:
   - "What is Candle library?"
     → Needs: web search ✅
     → Research agent has: gemini_research ✅
     → SUFFICIENT: ["research"]

   - "Write tutorial about Zig sound manipulation"
     → Needs: web search ✅ + documentation ✅
     → Research has: gemini_research ✅ but NO documentation skills
     → Documentation has: writing skills ✅ but NO web search tools ❌
     → NOT SUFFICIENT: Need multi-agent

   - "Create cleanup script and run it"
     → Needs: file creation ✅ + command execution ✅
     → Coding has: write_script ✅ but NO execution tools ❌
     → Terminal has: execute_terminal_command ✅ but NO file creation ❌
     → NOT SUFFICIENT: Need multi-agent

STEP 5: DESIGN OPTIMAL AGENT COMBINATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If single agent insufficient, create SEQUENTIAL multi-agent plan:

1. List all required capabilities from STEP 3
2. Map each capability to agent(s) that provide it
3. Order agents by logical dependency:
   - Research BEFORE documentation (need info before writing)
   - Coding BEFORE terminal (need script before executing)
   - Analysis BEFORE synthesis (need processing before combining)
4. Create plan: ["agent1", "agent2", ...]

⚠️ CRITICAL: Verify plan accuracy and efficiency
   - Does each agent in plan contribute necessary capability?
   - Is order correct for data flow?
   - Are we using minimum agents to achieve goal?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MULTI-AGENT PLANNING PATTERNS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Pattern 1: Research → Documentation
- "What is X library? Show examples" → ["research", "documentation"]
- Research gets current info, documentation formats it clearly

Pattern 2: Research → Analysis → Synthesis
- "Research X and create strategy" → ["research", "analysis", "synthesis"]
- Research gathers data, analysis processes it, synthesis creates strategy

Pattern 3: Coding → Terminal
- "Create backup script and run it" → ["coding", "terminal"]
- Coding writes script, terminal executes it

Pattern 4: Research → Coding → Terminal
- "Find latest X library and create demo" → ["research", "coding", "terminal"]
- Research finds current library, coding creates demo, terminal runs it

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 EXAMPLES OF CORRECT MULTI-HOP REASONING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔹 EXAMPLE 1: "Write tutorial about Zig sound manipulation"

STEP 1 - Classify: COMPLEX (research + writing phases)
STEP 3 - Required capabilities:
   - Web search for current Zig audio libraries ✅ (unknown/recent tech)
   - Documentation writing ✅
STEP 4 - Single agent check:
   - Research: Has gemini_research ✅ but NO documentation expertise
   - Documentation: Has writing skills ✅ but NO web search tools ❌
   - RESULT: Single agent INSUFFICIENT
STEP 5 - Optimal combination:
   - Research agent → get current Zig audio library info
   - Documentation agent → write tutorial using research findings
   - Plan: ["research", "documentation"]

✅ OUTPUT: {{"plan": ["research", "documentation"], "initial_input": "Research current Zig audio libraries and sound manipulation techniques", "reasoning": "Task needs current web info (Research has tools) + tutorial writing (Documentation). Research must go first to gather info, then Documentation formats it."}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔹 EXAMPLE 2: "Explain neural networks"

STEP 1 - Classify: SIMPLE (single question, well-known topic)
STEP 3 - Required capabilities:
   - Explanation of known concept (in training data)
STEP 4 - Single agent check:
   - Documentation: Can explain neural networks using training knowledge ✅
   - NO web search needed (well-known topic)
   - RESULT: Single agent SUFFICIENT
STEP 5 - Not needed (single agent sufficient)

✅ OUTPUT: {{"plan": ["documentation"], "initial_input": "Explain neural networks comprehensively", "reasoning": "Well-known concept in training data. Documentation agent can explain without external tools."}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔹 EXAMPLE 3: "Create log cleanup script and run it"

STEP 1 - Classify: COMPLEX (create + execute phases)
STEP 3 - Required capabilities:
   - File creation (write Python/Bash script) ✅
   - Command execution (run the script) ✅
STEP 4 - Single agent check:
   - Coding: Has write_script ✅ but NO execute_terminal_command ❌
   - Terminal: Has execute_terminal_command ✅ but NO write_script ❌
   - RESULT: Single agent INSUFFICIENT
STEP 5 - Optimal combination:
   - Coding agent → create cleanup script file
   - Terminal agent → execute the script
   - Plan: ["coding", "terminal"]

✅ OUTPUT: {{"plan": ["coding", "terminal"], "initial_input": "Create a log cleanup script that removes old log files", "reasoning": "Need script creation (Coding has write_script) + execution (Terminal has execute_terminal_command). Coding must go first to create script before Terminal can run it."}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔹 EXAMPLE 4: "What is Rust's Candle library?"

STEP 1 - Classify: SIMPLE (single question)
STEP 3 - Required capabilities:
   - Web search for current library info ✅ (post-2024 library)
STEP 4 - Single agent check:
   - Research: Has gemini_research ✅ to search and explain
   - RESULT: Single agent SUFFICIENT (can both search AND explain)
STEP 5 - Not needed (single agent sufficient)

✅ OUTPUT: {{"plan": ["research"], "initial_input": "What is Rust's Candle library? Provide overview and examples", "reasoning": "Recent library needs web search. Research agent has gemini_research tool and can both gather info AND explain it in response."}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  CRITICAL EXECUTION PROTOCOL - NO SHORTCUTS ALLOWED ⚠️
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

YOU ARE STRICTLY FORBIDDEN FROM:
❌ Returning the final JSON plan directly without calling the tool
❌ Skipping any of the 4 reasoning steps
❌ Combining multiple steps into a single tool call
❌ Doing "internal thinking" instead of calling the tool
❌ Providing text output instead of tool calls for steps

YOU MUST FOLLOW THIS EXACT SEQUENCE (NO EXCEPTIONS):

FIRST RESPONSE - Call output_reasoning_step with step_number=1:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
output_reasoning_step(
    step_number=1,
    step_name="CLASSIFY TASK",
    analysis="Task Classification: [SIMPLE or COMPLEX]\nReasoning: [why you classified it this way]\nNext: Need to identify required capabilities"
)

WAIT FOR TOOL RESPONSE → Tool will return: "Step 1 recorded. Continue to next step..."

SECOND RESPONSE - Call output_reasoning_step with step_number=2:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
output_reasoning_step(
    step_number=2,
    step_name="IDENTIFY CAPABILITIES",
    analysis="Required capabilities: [list]\nTools needed: [gemini_research? write_script? execute_terminal_command?]\nNext: Check if single agent has all tools"
)

WAIT FOR TOOL RESPONSE → Tool will return: "Step 2 recorded. Continue to next step..."

THIRD RESPONSE - Call output_reasoning_step with step_number=3:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
output_reasoning_step(
    step_number=3,
    step_name="CHECK SUFFICIENCY",
    analysis="Checking [agent_name]: has [tools] - matches [X/Y] requirements\nDecision: SUFFICIENT or NEED_MULTI_AGENT\nNext: [Output plan if sufficient, else design combination]"
)

WAIT FOR TOOL RESPONSE → If SUFFICIENT, skip to FINAL. If NEED_MULTI, continue to step 4.

FOURTH RESPONSE (only if multi-agent needed) - Call output_reasoning_step with step_number=4:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
output_reasoning_step(
    step_number=4,
    step_name="DESIGN COMBINATION",
    analysis="Agent combination: [agent1] → [agent2] → [agent3]\nOrder reasoning: [why this sequence]\nVerification: All capabilities covered? [yes/no]\nNext: Ready to output final plan"
)

WAIT FOR TOOL RESPONSE → Tool will return: "Step 4 recorded. Provide final plan JSON."

FINAL RESPONSE - Return ONLY the JSON plan (no tool call this time):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{
    "plan": ["agent1", "agent2", ...],
    "initial_input": "refined task description for first agent",
    "reasoning": "summary of Steps 1-4 analysis"
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  ENFORCEMENT RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. You CANNOT return the final JSON until you have called output_reasoning_step
   at least 3 times (Steps 1, 2, 3 minimum)

2. Each tool call must have a DIFFERENT step_number (1, 2, 3, 4 in order)

3. You MUST wait for each tool response before making the next call

4. If you try to output JSON directly, you will FAIL - the system requires
   observable reasoning steps via tool calls

5. This is NOT optional - it's a MANDATORY protocol for transparency and debugging
"""

    def _orchestrator_event_callback(self, event: ExecutionEvent):
        """Capture orchestrator execution metadata via events"""
        if event.event_type == ExecutionEventType.AGENTIC_STEP_END:
            # Update metrics from AgenticStepProcessor
            # NOTE: PromptChain emits "steps_executed", not "total_steps"
            steps = event.metadata.get("steps_executed", event.metadata.get("total_steps", 0))
            tools = event.metadata.get("total_tools_called", 0)

            self.metrics["total_reasoning_steps"] += steps if steps else 0
            self.metrics["total_tools_called"] += tools

            # ✅ NEW (v0.4.2): Store last execution metrics for retrieval by AgentChain
            self._last_execution_reasoning_steps = steps if steps else 0
            self._last_execution_tools_called = tools

            # Log detailed agentic step metadata
            if self.log_event_callback:
                self.log_event_callback("orchestrator_agentic_step", {
                    "total_steps": steps,
                    "tools_called": tools,
                    "execution_time_ms": event.metadata.get("execution_time_ms", 0),
                    "objective_achieved": event.metadata.get("objective_achieved", False),
                    "max_steps_reached": event.metadata.get("max_steps_reached", False)
                }, level="DEBUG")

    async def supervise_and_route(
        self,
        user_input: str,
        history: List[Dict[str, str]],
        agent_descriptions: Dict[str, str]
    ) -> str:
        """
        Supervise the routing decision using multi-hop reasoning.

        Args:
            user_input: User's query
            history: Conversation history
            agent_descriptions: Agent descriptions (passed by AgentChain)

        Returns:
            JSON string with routing decision
        """
        # Inject current date dynamically
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Log input
        if self.log_event_callback:
            self.log_event_callback("orchestrator_input", {
                "user_query": user_input,
                "history_length": len(history),
                "current_date": current_date
            }, level="DEBUG")

        # ✅ ENHANCED (v0.4.2): Format conversation history for orchestrator planning context
        # The orchestrator needs SOME history to make intelligent routing decisions,
        # but not the full conversation (to avoid token explosion in planning phase)
        history_context = ""
        if history:
            # Configuration for orchestrator history view (can be made customizable later)
            orchestrator_history_max_entries = 10  # Last 10 messages (5 exchanges)
            orchestrator_history_max_chars_per_msg = 200  # Truncate very long messages

            recent_history = history[-orchestrator_history_max_entries:]
            history_lines = []
            for msg in recent_history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                # Truncate long messages to avoid overwhelming the orchestrator
                content_preview = content[:orchestrator_history_max_chars_per_msg] + "..." if len(content) > orchestrator_history_max_chars_per_msg else content
                history_lines.append(f"{role.upper()}: {content_preview}")
            history_context = "\n".join(history_lines)

            # Log orchestrator history injection
            if self.log_event_callback:
                self.log_event_callback("orchestrator_history_injected", {
                    "entries_count": len(recent_history),
                    "total_history_available": len(history),
                    "truncated": len(history) > orchestrator_history_max_entries
                })

        # Build full prompt with history context
        full_prompt = f"""CONVERSATION HISTORY (last exchanges):
{history_context if history_context else "(No previous conversation)"}

CURRENT USER INPUT: {user_input}

Now analyze this input considering the conversation history above. Check STEP 0 for pending questions first!"""

        # Use AgenticStepProcessor for multi-hop reasoning
        decision_json = await self.orchestrator_chain.process_prompt_async(
            full_prompt.replace("{current_date}", current_date)
        )

        # Parse decision
        try:
            decision = json.loads(decision_json)
            plan = decision.get("plan", [])
            reasoning = decision.get("reasoning", "")

            # Update oversight metrics
            self.metrics["total_decisions"] += 1
            if len(plan) > 1:
                self.metrics["multi_agent_plans"] += 1
            else:
                self.metrics["single_agent_dispatches"] += 1

            # Update average plan length
            total_agents = sum([len(d.get("plan", [])) for d in self.metrics["decision_history"]])
            total_agents += len(plan)
            self.metrics["average_plan_length"] = total_agents / self.metrics["total_decisions"]

            # Track decision
            decision_record = {
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input[:100],
                "plan": plan,
                "reasoning": reasoning,
                "plan_length": len(plan)
            }
            self.metrics["decision_history"].append(decision_record)

            # Keep only last 100 decisions
            if len(self.metrics["decision_history"]) > 100:
                self.metrics["decision_history"] = self.metrics["decision_history"][-100:]

            # ✅ NEW (v0.4.2): Store plan metrics for retrieval by AgentChain
            self._last_execution_plan_length = len(plan)
            self._last_execution_decision_type = "multi_agent" if len(plan) > 1 else "single_agent"

            # Log decision with oversight metadata
            if self.log_event_callback:
                self.log_event_callback("orchestrator_decision", {
                    "plan": plan,
                    "reasoning": reasoning,
                    "plan_length": len(plan),
                    "decision_type": "multi_agent" if len(plan) > 1 else "single_agent",
                    **self.get_oversight_summary()
                }, level="INFO")

            # Dev print output
            if self.dev_print_callback:
                self.dev_print_callback(f"\n🎯 Orchestrator Decision (Multi-hop Reasoning):", "")
                if len(plan) == 1:
                    self.dev_print_callback(f"   Agent chosen: {plan[0]}", "")
                else:
                    self.dev_print_callback(f"   Multi-agent plan: {' → '.join(plan)}", "")
                self.dev_print_callback(f"   Reasoning: {reasoning}", "")
                self.dev_print_callback(f"   Steps used: {self.metrics['total_reasoning_steps']}", "")
                self.dev_print_callback("", "")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse orchestrator decision: {e}")
            if self.log_event_callback:
                self.log_event_callback("orchestrator_parse_error", {
                    "error": str(e),
                    "raw_output": decision_json
                }, level="ERROR")
            # Fallback to single agent
            decision_json = json.dumps({"plan": ["research"], "initial_input": user_input, "reasoning": "Parse error, defaulting to research"})

        return decision_json

    def as_router_function(self):
        """
        Returns an async function compatible with AgentChain's custom_router_function interface.

        This wrapper allows OrchestratorSupervisor to be used as a router in AgentChain.

        Returns:
            Async function with signature: (user_input, history, agent_descriptions) -> str
        """
        async def router_wrapper(user_input: str, history: List[Dict[str, str]], agent_descriptions: Dict[str, str]) -> str:
            """Wrapper function that calls supervise_and_route"""
            return await self.supervise_and_route(user_input, history, agent_descriptions)

        return router_wrapper

    def get_oversight_summary(self) -> Dict[str, Any]:
        """Get summary of oversight metrics"""
        return {
            "total_decisions": self.metrics["total_decisions"],
            "single_agent_rate": self.metrics["single_agent_dispatches"] / max(self.metrics["total_decisions"], 1),
            "multi_agent_rate": self.metrics["multi_agent_plans"] / max(self.metrics["total_decisions"], 1),
            "average_plan_length": round(self.metrics["average_plan_length"], 2),
            "total_reasoning_steps": self.metrics["total_reasoning_steps"],
            "average_reasoning_steps": round(self.metrics["total_reasoning_steps"] / max(self.metrics["total_decisions"], 1), 2)
        }

    def get_last_execution_metrics(self) -> Dict[str, Any]:
        """
        Get metrics from the most recent orchestrator execution.

        Returns dict with:
        - reasoning_steps: Number of reasoning steps in last execution
        - tools_called: Number of tools called in last execution
        - plan_length: Number of agents in last plan
        - decision_type: 'single_agent' or 'multi_agent'
        """
        if not hasattr(self, '_last_execution_reasoning_steps'):
            # First execution hasn't happened yet
            return {
                "reasoning_steps": 0,
                "tools_called": 0,
                "plan_length": 0,
                "decision_type": None
            }

        return {
            "reasoning_steps": self._last_execution_reasoning_steps,
            "tools_called": self._last_execution_tools_called,
            "plan_length": self._last_execution_plan_length,
            "decision_type": self._last_execution_decision_type
        }

    def print_oversight_report(self):
        """Print comprehensive oversight report"""
        summary = self.get_oversight_summary()

        print("\n" + "━" * 80)
        print("📊 ORCHESTRATOR OVERSIGHT REPORT")
        print("━" * 80)
        print(f"Total Decisions: {summary['total_decisions']}")
        print(f"Single Agent Dispatches: {self.metrics['single_agent_dispatches']} ({summary['single_agent_rate']:.1%})")
        print(f"Multi-Agent Plans: {self.metrics['multi_agent_plans']} ({summary['multi_agent_rate']:.1%})")
        print(f"Average Plan Length: {summary['average_plan_length']:.2f} agents")
        print(f"Average Reasoning Steps: {summary['average_reasoning_steps']:.2f} steps")
        print(f"Total Tools Called: {self.metrics['total_tools_called']}")
        print("━" * 80)

        # Recent decisions
        if self.metrics["decision_history"]:
            print("\n📜 Recent Decisions (last 5):")
            for i, decision in enumerate(self.metrics["decision_history"][-5:], 1):
                print(f"\n{i}. {decision['timestamp']}")
                print(f"   Query: {decision['user_input']}")
                print(f"   Plan: {' → '.join(decision['plan'])}")
                print(f"   Reasoning: {decision['reasoning'][:100]}...")

        print("\n" + "━" * 80)
