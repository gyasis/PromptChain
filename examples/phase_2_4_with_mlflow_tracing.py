"""
Phase 2-4 Demo with PROPER MLflow Observability

Uses MLflow's native @mlflow.trace decorator to capture:
- THINK phases (reasoning before actions)
- ACT phases (tool executions)
- OBSERVE phases (results analysis)
- Blackboard state evolution
- All intermediate steps

NO LangChain required - pure MLflow native tracing.
"""
import mlflow
import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from dotenv import load_dotenv
load_dotenv()

from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain import PromptChain

# Initialize MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("phase_2-4_tracing")


# Mock tools with @mlflow.trace decorators
@mlflow.trace(span_type="TOOL", name="search_files")
def search_files(query: str) -> str:
    """Search for files - traced as TOOL span."""
    print(f"\n🔍 EXECUTING: search_files('{query}')")
    results = {
        "authentication": "Found:\n- login.py (JWT authentication)\n- auth_middleware.py (token validation)\n- session_manager.py (session handling)",
        "database": "Found:\n- user_model.py (User ORM)\n- connection.py (DB connection pool)",
    }
    result = results.get(query.lower(), f"No files found for '{query}'")

    print(f"   → {result[:80]}...")
    return result


@mlflow.trace(span_type="TOOL", name="read_code")
def read_code(filename: str) -> str:
    """Read code file - traced as TOOL span."""
    print(f"\n📖 EXECUTING: read_code('{filename}')")
    content = f"""# {filename}
def authenticate(username, password):
    user = db.get_user(username)
    if not user or not bcrypt.check(password, user.hash):
        return None
    return create_jwt_token(user.id)
"""

    print(f"   → Returned {len(content)} characters")
    return content


@mlflow.trace(span_type="TOOL", name="analyze_security")
def analyze_security(code: str) -> str:
    """Analyze for security issues - traced as TOOL span."""
    print(f"\n🔐 EXECUTING: analyze_security(<code>)")
    analysis = """Issues found:
✅ Uses bcrypt for hashing
⚠️ Missing rate limiting
⚠️ No MFA support
💡 Recommend: Add rate limiting, enable MFA"""

    print(f"   → {analysis[:60]}...")
    return analysis


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search files by topic",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_code",
            "description": "Read code file",
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string"}},
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_security",
            "description": "Analyze code security",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"]
            }
        }
    }
]


@mlflow.trace(name="Phase_2-4_Workflow")
async def run_with_full_tracing():
    """
    Main workflow with full MLflow tracing.

    Every phase (THINK → ACT → OBSERVE) will be captured as spans.
    """
    print("=" * 80)
    print("🚀 Phase 2-4 Demo with FULL MLflow Tracing")
    print("=" * 80)
    print("\nEnabled Features:")
    print("  ✅ Phase 2: Blackboard Architecture (token optimization)")
    print("  ✅ Phase 3: Chain of Verification (safety)")
    print("  ✅ Phase 3: Checkpointing (stuck detection)")
    print("  ✅ Phase 4: TAO Loop (THINK → ACT → OBSERVE)")
    print("  ✅ Phase 4: Dry Run (prediction)")
    print("  ✅ MLflow Tracing: FULL observability")
    print("\n" + "=" * 80)

    # Create processor with ALL features
    processor = AgenticStepProcessor(
        objective="""Use the available tools to:
1. Search for authentication-related files
2. Read the authentication code
3. Analyze the code for security issues
4. Provide recommendations based on the analysis""",
        max_internal_steps=8,
        model_name="openai/gpt-4o-mini",  # Using OpenAI for reliable tool calling
        model_params={
            "tool_choice": "auto"  # Explicitly enable tool calling
        },

        # All phases enabled
        enable_blackboard=True,
        enable_cove=True,
        enable_checkpointing=True,
        enable_tao_loop=True,
        enable_dry_run=True,
        cove_confidence_threshold=0.7,
    )

    # Log processor configuration to MLflow
    mlflow.log_params({
        "objective": processor.objective,
        "max_internal_steps": processor.max_internal_steps,
        "model_name": processor.model_name,
        "phase_2_blackboard": processor.enable_blackboard,
        "phase_3_cove": processor.enable_cove,
        "phase_3_checkpointing": processor.enable_checkpointing,
        "phase_4_tao_loop": processor.enable_tao_loop,
        "phase_4_dry_run": processor.enable_dry_run,
    })

    # Create chain
    chain = PromptChain(
        models=["openai/gpt-4o-mini"],  # Match processor model
        instructions=[processor]
    )

    # Register tools
    chain.register_tool_function(search_files)
    chain.register_tool_function(read_code)
    chain.register_tool_function(analyze_security)
    chain.add_tools(TOOLS)

    print("\n🎯 Starting analysis with FULL tracing...")
    print("Watch for THINK → ACT → OBSERVE phases:\n")

    # Execute - all steps will be traced
    with mlflow.start_span(name="AgenticStepProcessor_Execution", span_type="CHAIN") as main_span:
        result = await chain.process_prompt_async("Start the analysis")

        # Log final metrics
        mlflow.log_metrics({
            "total_tokens": processor.total_prompt_tokens + processor.total_completion_tokens,
            "prompt_tokens": processor.total_prompt_tokens,
            "completion_tokens": processor.total_completion_tokens,
        })

        # Log Blackboard state as artifact
        if processor.blackboard:
            blackboard_state = processor.blackboard.get_state()
            mlflow.log_dict(blackboard_state, "blackboard_final_state.json")

            # Add Blackboard metrics to span
            main_span.set_attribute("blackboard_facts_count", len(blackboard_state['facts_discovered']))
            main_span.set_attribute("blackboard_observations_count", len(blackboard_state['observations']))
            main_span.set_attribute("blackboard_plan_steps", len(blackboard_state['current_plan']))

            print(f"\n📋 Blackboard captured: {len(blackboard_state['facts_discovered'])} facts")

    print("\n" + "=" * 80)
    print("✅ COMPLETE - Check MLflow UI!")
    print("=" * 80)
    print(f"\n📊 Result: {result[:200]}...")
    print(f"\n🎯 View traces at: http://localhost:5000")
    print("   Navigate to: Experiments → phase_2-4_tracing → Traces tab")
    print("\n" + "=" * 80)

    return result


if __name__ == "__main__":
    # Start MLflow run with tracing enabled
    with mlflow.start_run(run_name="Phase_2-4_Full_Tracing"):
        asyncio.run(run_with_full_tracing())
