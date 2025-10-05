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

        # Create AgenticStepProcessor as reasoning engine
        self.reasoning_engine = AgenticStepProcessor(
            objective=objective,
            max_internal_steps=max_reasoning_steps,
            model_name=model_name,
            history_mode="progressive"  # Critical for multi-hop reasoning
        )

        # Wrap in PromptChain for execution
        self.orchestrator_chain = PromptChain(
            models=[],  # AgenticStepProcessor has its own model
            instructions=[self.reasoning_engine],
            verbose=False,
            store_steps=True
        )

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

STEP 2: IDENTIFY REQUIRED CAPABILITIES & TOOLS
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

STEP 3: CHECK IF SINGLE AGENT HAS SUFFICIENT TOOLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compare task requirements to agent capabilities:

Does ONE agent have ALL the tools needed?
   → YES: Use single-agent plan ["agent_name"]
   → NO: Proceed to STEP 4 for multi-agent combination

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

STEP 4: DESIGN OPTIMAL AGENT COMBINATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If single agent insufficient, create SEQUENTIAL multi-agent plan:

1. List all required capabilities from STEP 2
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
STEP 2 - Required capabilities:
   - Web search for current Zig audio libraries ✅ (unknown/recent tech)
   - Documentation writing ✅
STEP 3 - Single agent check:
   - Research: Has gemini_research ✅ but NO documentation expertise
   - Documentation: Has writing skills ✅ but NO web search tools ❌
   - RESULT: Single agent INSUFFICIENT
STEP 4 - Optimal combination:
   - Research agent → get current Zig audio library info
   - Documentation agent → write tutorial using research findings
   - Plan: ["research", "documentation"]

✅ OUTPUT: {{"plan": ["research", "documentation"], "initial_input": "Research current Zig audio libraries and sound manipulation techniques", "reasoning": "Task needs current web info (Research has tools) + tutorial writing (Documentation). Research must go first to gather info, then Documentation formats it."}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔹 EXAMPLE 2: "Explain neural networks"

STEP 1 - Classify: SIMPLE (single question, well-known topic)
STEP 2 - Required capabilities:
   - Explanation of known concept (in training data)
STEP 3 - Single agent check:
   - Documentation: Can explain neural networks using training knowledge ✅
   - NO web search needed (well-known topic)
   - RESULT: Single agent SUFFICIENT
STEP 4 - Not needed (single agent sufficient)

✅ OUTPUT: {{"plan": ["documentation"], "initial_input": "Explain neural networks comprehensively", "reasoning": "Well-known concept in training data. Documentation agent can explain without external tools."}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔹 EXAMPLE 3: "Create log cleanup script and run it"

STEP 1 - Classify: COMPLEX (create + execute phases)
STEP 2 - Required capabilities:
   - File creation (write Python/Bash script) ✅
   - Command execution (run the script) ✅
STEP 3 - Single agent check:
   - Coding: Has write_script ✅ but NO execute_terminal_command ❌
   - Terminal: Has execute_terminal_command ✅ but NO write_script ❌
   - RESULT: Single agent INSUFFICIENT
STEP 4 - Optimal combination:
   - Coding agent → create cleanup script file
   - Terminal agent → execute the script
   - Plan: ["coding", "terminal"]

✅ OUTPUT: {{"plan": ["coding", "terminal"], "initial_input": "Create a log cleanup script that removes old log files", "reasoning": "Need script creation (Coding has write_script) + execution (Terminal has execute_terminal_command). Coding must go first to create script before Terminal can run it."}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔹 EXAMPLE 4: "What is Rust's Candle library?"

STEP 1 - Classify: SIMPLE (single question)
STEP 2 - Required capabilities:
   - Web search for current library info ✅ (post-2024 library)
STEP 3 - Single agent check:
   - Research: Has gemini_research ✅ to search and explain
   - RESULT: Single agent SUFFICIENT (can both search AND explain)
STEP 4 - Not needed (single agent sufficient)

✅ OUTPUT: {{"plan": ["research"], "initial_input": "What is Rust's Candle library? Provide overview and examples", "reasoning": "Recent library needs web search. Research agent has gemini_research tool and can both gather info AND explain it in response."}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USE YOUR INTERNAL REASONING STEPS TO ANALYZE THE QUERY, THEN OUTPUT:

RETURN JSON ONLY (no other text):
{{
    "plan": ["agent1", "agent2", ...],  // Ordered list of agents (can be single agent ["agent_name"])
    "initial_input": "refined task description for first agent",
    "reasoning": "multi-step reasoning summary of why this plan"
}}
"""

    def _orchestrator_event_callback(self, event: ExecutionEvent):
        """Capture orchestrator execution metadata via events"""
        if event.event_type == ExecutionEventType.AGENTIC_STEP_END:
            # Update metrics from AgenticStepProcessor
            # NOTE: PromptChain emits "steps_executed", not "total_steps"
            steps = event.metadata.get("steps_executed", event.metadata.get("total_steps", 0))
            self.metrics["total_reasoning_steps"] += steps if steps else 0
            self.metrics["total_tools_called"] += event.metadata.get("total_tools_called", 0)

            # Log detailed agentic step metadata
            if self.log_event_callback:
                self.log_event_callback("orchestrator_agentic_step", {
                    "total_steps": steps,
                    "tools_called": event.metadata.get("total_tools_called", 0),
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

        # Use AgenticStepProcessor for multi-hop reasoning
        decision_json = await self.orchestrator_chain.process_prompt_async(
            user_input.replace("{current_date}", current_date)
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
