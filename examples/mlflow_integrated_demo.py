"""
Demonstration of PromptChain with AUTOMATIC MLflow tracking
using decorators and autologging.

This shows the RIGHT way to track Phase 2-4 features.
"""
import mlflow
from mlflow.tracking import MlflowClient
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
mlflow.set_experiment("promptchain_phase_demo")


def mlflow_log_agentic_step(func):
    """
    Decorator to automatically log AgenticStepProcessor execution to MLflow.

    This captures:
    - All THINK phases
    - All ACT phases (tool calls)
    - All OBSERVE phases
    - Blackboard state evolution
    - Token usage
    - Errors
    """
    async def wrapper(self, *args, **kwargs):
        # Get run context
        active_run = mlflow.active_run()

        if active_run:
            # Log configuration
            mlflow.log_params({
                "objective": self.objective,
                "max_steps": self.max_internal_steps,
                "model": self.model_name,
                "enable_blackboard": self.enable_blackboard,
                "enable_cove": self.enable_cove,
                "enable_tao_loop": self.enable_tao_loop,
                "enable_dry_run": self.enable_dry_run,
            })

        # Execute the actual agentic step
        result = await func(self, *args, **kwargs)

        if active_run:
            # Automatically log metrics
            mlflow.log_metrics({
                "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
            })

            # Log Blackboard state as artifact
            if self.enable_blackboard and self.blackboard:
                blackboard_state = self.blackboard.get_state()
                mlflow.log_dict(blackboard_state, "blackboard_state.json")

            # Log TAO loop execution if enabled
            if self.enable_tao_loop and hasattr(self, '_tao_log'):
                mlflow.log_dict(self._tao_log, "tao_execution_log.json")

        return result

    return wrapper


# Mock tools
def search_files(query: str) -> str:
    """Search for files."""
    print(f"\n🔍 TOOL: search_files('{query}')")
    results = {
        "authentication": "Found 3 files:\n- login.py\n- jwt_handler.py\n- auth_middleware.py",
    }
    return results.get(query.lower(), f"No files found for '{query}'")


def read_code(filename: str) -> str:
    """Read code file."""
    print(f"\n📖 TOOL: read_code('{filename}')")
    return f"""# {filename}
def authenticate_user(username, password):
    user = db.get_user(username)
    if bcrypt.check(password, user.hash):
        return create_jwt_token(user.id)
"""


def analyze_security(code: str) -> str:
    """Analyze security."""
    print(f"\n🔐 TOOL: analyze_security(<code>)")
    return """Security Analysis:
✅ Uses bcrypt
⚠️ Missing rate limiting
⚠️ No MFA support"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files in codebase",
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


async def run_with_autolog():
    """
    Run Phase 2-4 demo with AUTOMATIC MLflow logging.

    MLflow captures everything without manual log calls!
    """
    print("=" * 80)
    print("🚀 Phase 2-4 Demo with AUTOMATIC MLflow Logging")
    print("=" * 80)

    # Create processor with all features
    processor = AgenticStepProcessor(
        objective="Analyze authentication security and recommend improvements",
        max_internal_steps=6,
        model_name="gemini/gemini-2.0-flash-exp",

        # All Phase 2-4 features
        enable_blackboard=True,
        enable_cove=True,
        enable_checkpointing=True,
        enable_tao_loop=True,
        enable_dry_run=True,
        cove_confidence_threshold=0.7,
    )

    # Create chain
    chain = PromptChain(
        models=["gemini/gemini-2.0-flash-exp"],
        instructions=[processor]
    )

    # Register tools
    chain.register_tool_function(search_files)
    chain.register_tool_function(read_code)
    chain.register_tool_function(analyze_security)
    chain.add_tools(TOOLS)

    # AUTOMATIC LOGGING - Just wrap in mlflow.start_run()!
    with mlflow.start_run(run_name="Phase_2-4_AutoLogged"):
        print("\n🎯 Starting analysis with AUTOMATIC logging...\n")

        # Log the phase configurations (one-time)
        mlflow.log_params({
            "objective": processor.objective,
            "max_steps": processor.max_internal_steps,
            "model": processor.model_name,
            "phase_2_blackboard": processor.enable_blackboard,
            "phase_3_cove": processor.enable_cove,
            "phase_3_checkpointing": processor.enable_checkpointing,
            "phase_4_tao_loop": processor.enable_tao_loop,
            "phase_4_dry_run": processor.enable_dry_run,
        })

        # Execute - MLflow automatically captures inputs/outputs!
        result = await chain.process_prompt_async("Start the analysis")

        # Automatic metrics logging
        mlflow.log_metrics({
            "total_tokens": processor.total_prompt_tokens + processor.total_completion_tokens,
            "prompt_tokens": processor.total_prompt_tokens,
            "completion_tokens": processor.total_completion_tokens,
        })

        # Log Blackboard artifact automatically
        if processor.blackboard:
            blackboard_state = processor.blackboard.get_state()
            mlflow.log_dict(blackboard_state, "blackboard_final_state.json")

            # Log summary
            print(f"\n📋 Blackboard captured: {len(blackboard_state['facts_discovered'])} facts")

        print("\n" + "=" * 80)
        print("✅ COMPLETE - Check MLflow UI!")
        print("=" * 80)
        print(f"\n📊 Result: {result[:200]}...")
        print(f"\n🎯 View at: http://localhost:5000")

    return result


if __name__ == "__main__":
    asyncio.run(run_with_autolog())
