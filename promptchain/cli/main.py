"""PromptChain CLI entry point.

This module provides the command-line interface for launching the PromptChain
interactive terminal UI.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import click

from promptchain.observability import (init_mlflow, shutdown_mlflow,
                                       track_session)

from . import __version__
from .models import Config
from .tui.app import PromptChainApp

# Global flag for dev mode (set by CLI)
_DEV_MODE = False


def setup_logging(
    dev_mode: bool = False,
    session_name: str = "default",
    sessions_dir: Optional[Path] = None,
):
    """Configure logging based on mode.

    Args:
        dev_mode: If True, enable DEBUG logging to file
        session_name: Session name for log file path
        sessions_dir: Base sessions directory
    """
    global _DEV_MODE
    _DEV_MODE = dev_mode

    # ALWAYS suppress LiteLLM's own console logging (it uses env var, not Python logging)
    os.environ["LITELLM_LOG"] = "CRITICAL"

    if dev_mode:
        # DEV MODE: Write ALL logs to a single debug file (NO console output)
        if sessions_dir is None:
            sessions_dir = Path.home() / ".promptchain" / "sessions"

        log_dir = sessions_dir / session_name
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped debug log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_log_path = log_dir / f"debug_{timestamp}.log"

        # Configure file handler with detailed format
        file_handler = logging.FileHandler(debug_log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
                datefmt="%H:%M:%S",
            )
        )

        # Set root logger to DEBUG with ONLY file handler (no console)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Remove any existing handlers to prevent console leakage
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add only file handler
        root_logger.addHandler(file_handler)

        # Suppress all console output from known noisy libraries
        # (they still go to file via root logger)
        for noisy_logger in [
            "litellm",
            "httpx",
            "httpcore",
            "LiteLLM",
            "openai",
            "urllib3",
            "asyncio",
            "anyio",
            "mcp",
        ]:
            logging.getLogger(noisy_logger).handlers = []  # Remove their handlers
            logging.getLogger(noisy_logger).propagate = (
                True  # Propagate to root (file only)
            )

        # Log startup info to file
        root_logger.info("=" * 80)
        root_logger.info(f"PromptChain DEV MODE - Debug Log Started")
        root_logger.info(f"Timestamp: {datetime.now().isoformat()}")
        root_logger.info(f"Session: {session_name}")
        root_logger.info(f"Log file: {debug_log_path}")
        root_logger.info(f"Python: {sys.version}")
        root_logger.info("=" * 80)

        # Tell user where logs are (this is the only console output)
        click.echo(f"[DEV MODE] Debug logs: {debug_log_path}")

    else:
        # PRODUCTION MODE: Suppress ALL logging that spills into TUI
        # (LITELLM_LOG already set to CRITICAL above)

        # Suppress all WARNING and INFO from root logger (only CRITICAL errors will show)
        logging.basicConfig(level=logging.CRITICAL, format="%(message)s")

        # Additional catch-all: Suppress any logger that wasn't explicitly named
        logging.getLogger().setLevel(logging.CRITICAL)

        # External libraries
        logging.getLogger("litellm").setLevel(logging.CRITICAL)
        logging.getLogger("httpx").setLevel(logging.CRITICAL)
        logging.getLogger("httpcore").setLevel(logging.CRITICAL)
        logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)

        # PromptChain internal modules
        logging.getLogger("promptchain.utils.execution_history_manager").setLevel(
            logging.CRITICAL
        )
        logging.getLogger("promptchain.utils.agentic_step_processor").setLevel(
            logging.CRITICAL
        )
        logging.getLogger("promptchain.utils.agent_chain").setLevel(logging.CRITICAL)
        logging.getLogger("promptchain.utils.logging_utils").setLevel(logging.CRITICAL)
        logging.getLogger("promptchain.utils.preprompt").setLevel(logging.CRITICAL)
        logging.getLogger("promptchain.utils.mcp_helpers").setLevel(logging.CRITICAL)
        logging.getLogger("promptchain.cli.activity_logger").setLevel(logging.CRITICAL)
        logging.getLogger("promptchain.cli.utils.mcp_manager").setLevel(
            logging.CRITICAL
        )
        logging.getLogger("terminal_tool").setLevel(logging.CRITICAL)

        # MCP library loggers
        logging.getLogger("mcp").setLevel(logging.CRITICAL)
        logging.getLogger("mcp.server").setLevel(logging.CRITICAL)
        logging.getLogger("mcp.client").setLevel(logging.CRITICAL)
        logging.getLogger("anyio").setLevel(logging.CRITICAL)


@click.group(invoke_without_command=True)
@click.option(
    "--session",
    "-s",
    default="default",
    help="Session name to load or create (default: 'default')",
)
@click.option(
    "--sessions-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for session storage (default: ~/.promptchain/sessions)",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to YAML configuration file (overrides .promptchain.yml)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose mode to see detailed internal steps, tool calls, and results",
)
@click.option(
    "--dev",
    is_flag=True,
    default=False,
    help="Enable dev mode: writes ALL debug logs to a timestamped file in session directory",
)
@click.version_option(version=__version__, prog_name="promptchain")
@click.pass_context
def main(
    ctx,
    session: str,
    sessions_dir: Optional[Path],
    config: Optional[Path],
    verbose: bool,
    dev: bool,
):
    """PromptChain - Interactive terminal interface for LLM conversations.

    Launch an interactive chat interface with persistent sessions, multi-agent
    support, and full conversation history management.

    Examples:
        promptchain                         # Start with default session
        promptchain --session my-proj       # Load/create 'my-proj' session
        promptchain --config custom.yml     # Use custom YAML config
        promptchain --verbose               # Enable verbose observability mode
        promptchain --dev                   # Enable dev mode with full debug logs
        promptchain query "your question"   # Single query without TUI
        promptchain patterns branch "..."   # Run branching pattern
        promptchain patterns expand "..."   # Run query expansion
        promptchain --help                  # Show this help message

    Configuration Precedence:
        1. CLI arguments (--config, --sessions-dir)
        2. Project-level .promptchain.yml (in current directory)
        3. User-level ~/.promptchain/config.yml
        4. Built-in defaults
    """
    # Store context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["session"] = session
    ctx.obj["sessions_dir"] = sessions_dir
    ctx.obj["config"] = config
    ctx.obj["verbose"] = verbose
    ctx.obj["dev"] = dev

    # If no subcommand invoked, launch the TUI (default behavior)
    if ctx.invoked_subcommand is None:
        _launch_tui(session, sessions_dir, config, verbose, dev)


@track_session()
def _launch_tui(
    session: str,
    sessions_dir: Optional[Path],
    config: Optional[Path],
    verbose: bool,
    dev: bool,
):
    """Launch the interactive TUI application."""
    # Initialize MLflow observability
    init_mlflow()

    try:
        # Setup logging first (dev mode enables debug file logging)
        setup_logging(dev_mode=dev, session_name=session, sessions_dir=sessions_dir)
        # T021: Load YAML configuration with precedence
        yaml_config = None
        try:
            from .config import load_config_with_precedence

            yaml_config = load_config_with_precedence(
                cli_config_path=str(config) if config else None
            )

            if yaml_config:
                click.echo(f"✓ Loaded configuration from YAML")

        except ImportError:
            # YAML config not available, continue with defaults
            pass
        except Exception as e:
            click.echo(f"⚠ Warning: Failed to load YAML config: {e}", err=True)

        # Load base configuration (T151-T153)
        base_config = Config.load_or_create_default()

        # Override sessions_dir from CLI if provided (highest precedence)
        if sessions_dir:
            base_config.sessions_dir = str(sessions_dir)
        # Override from YAML config if not set by CLI
        elif yaml_config and yaml_config.session.working_directory:
            base_config.sessions_dir = yaml_config.session.working_directory

        # Create and run the Textual app (T037)
        app = PromptChainApp(
            session_name=session,
            sessions_dir=(
                Path(base_config.sessions_dir) if base_config.sessions_dir else None
            ),
            config=base_config,  # Config already includes settings from YAML
            verbose_mode=verbose,  # T118: Verbose observability mode
        )

        # Run the app (Textual handles async event loop internally)
        app.run()

    finally:
        # Shutdown MLflow observability (flush background queue, close runs)
        shutdown_mlflow()


@main.command()
@click.argument("query", required=True)
@click.option(
    "--session",
    "-s",
    default="default",
    help="Session name to use (default: 'default')",
)
@click.option(
    "--sessions-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for session storage (default: ~/.promptchain/sessions)",
)
@click.option(
    "--agent",
    "-a",
    default=None,
    help="Specific agent to use (default: session's active agent)",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="Model to use (overrides agent model, e.g., 'openai/gpt-4')",
)
def query(
    query: str,
    session: str,
    sessions_dir: Optional[Path],
    agent: Optional[str],
    model: Optional[str],
):
    """Execute a single query without launching the TUI.

    Runs a query against the specified session and returns the response.
    Perfect for scripting and command-line workflows.

    QUERY: The question or task to execute

    Examples:
        promptchain query "What is the capital of France?"
        promptchain query "Analyze the code in main.py" --session my-project
        promptchain query "Graph the Dow Jones" --model openai/gpt-4o
        promptchain query "Summarize README.md" --agent researcher
    """
    import asyncio
    import os
    import re
    import subprocess
    import tempfile

    from promptchain.utils.agentic_step_processor import AgenticStepProcessor
    from promptchain.utils.promptchaining import PromptChain

    from .session_manager import SessionManager
    from .utils.mcp_manager import MCPManager

    # Setup basic logging (suppress verbose output)
    setup_logging(dev_mode=False, session_name=session, sessions_dir=sessions_dir)

    def extract_and_run_scripts(response: str, work_dir: str) -> list:
        """Extract shell scripts from response and execute them.

        Returns list of (script_path, success, output) tuples.
        """
        results: List[Any] = []

        # Pattern to find bash/shell code blocks
        patterns = [
            r"```(?:bash|sh|shell)\n(.*?)```",  # ```bash ... ```
            r"```\n#!/bin/bash\n(.*?)```",  # ``` #!/bin/bash ... ```
        ]

        scripts = []
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            scripts.extend(matches)

        if not scripts:
            return results

        for i, script_content in enumerate(scripts):
            # Add shebang if not present
            if not script_content.strip().startswith("#!"):
                script_content = "#!/bin/bash\nset -e\n" + script_content

            # Save script to temp file
            script_path = os.path.join(work_dir, f"auto_script_{i}.sh")
            with open(script_path, "w") as f:
                f.write(script_content)
            os.chmod(script_path, 0o755)

            # Execute the script
            click.echo(f"\n[AUTO-EXEC] Running script {i+1}...", err=True)
            try:
                result = subprocess.run(
                    ["bash", script_path],
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                )
                success = result.returncode == 0
                output = result.stdout + result.stderr
                results.append((script_path, success, output))

                if success:
                    click.echo(
                        f"[AUTO-EXEC] Script {i+1} completed successfully", err=True
                    )
                    if result.stdout:
                        click.echo(result.stdout)
                else:
                    click.echo(
                        f"[AUTO-EXEC] Script {i+1} failed (exit {result.returncode})",
                        err=True,
                    )
                    click.echo(result.stderr, err=True)

            except subprocess.TimeoutExpired:
                results.append((script_path, False, "Script timed out after 5 minutes"))
                click.echo(f"[AUTO-EXEC] Script {i+1} timed out", err=True)
            except Exception as e:
                results.append((script_path, False, str(e)))
                click.echo(f"[AUTO-EXEC] Script {i+1} error: {e}", err=True)

        return results

    async def run_query():
        # Initialize session manager
        sessions_path = sessions_dir or Path.home() / ".promptchain" / "sessions"
        session_manager = SessionManager(sessions_dir=sessions_path)

        # Load or create session
        session_obj = session_manager.load_session(session)
        if not session_obj:
            session_obj = session_manager.create_session(session)

        # Get agent to use
        agent_name = agent or session_obj.active_agent or "default"
        agent_obj = session_obj.agents.get(agent_name)

        if not agent_obj:
            # Create default agent if doesn't exist
            from .models.agent_config import Agent

            model_name = model or session_obj.default_model
            agent_obj = Agent(name=agent_name, model_name=model_name)
            session_obj.agents[agent_name] = agent_obj

        # Override model if specified
        model_name = model or agent_obj.model_name

        # Initialize PrePrompt to load from both project and user prompts directories
        from promptchain.utils.preprompt import PrePrompt

        user_prompts_dir = Path.home() / ".promptchain" / "prompts" / "agents"
        user_custom_dir = Path.home() / ".promptchain" / "prompts" / "custom"

        # PrePrompt will search additional dirs first, then standard (project root) prompts
        preprompt = PrePrompt(
            additional_prompt_dirs=[str(user_prompts_dir), str(user_custom_dir)]
        )

        # Load strategy from project strategies folder
        project_strategies_dir = (
            Path(__file__).parent.parent.parent / "prompts" / "strategies"
        )

        # Try to load autonomous_executor prompt with code_execution strategy
        try:
            # Load base prompt
            base_prompt = preprompt.load("autonomous_executor")

            # Load and prepend strategy
            strategy_file = project_strategies_dir / "code_execution.json"
            if strategy_file.exists():
                import json

                strategy_data = json.loads(strategy_file.read_text())
                strategy_prompt = strategy_data.get("prompt", "")
                base_prompt = f"{strategy_prompt}\n\n{base_prompt}"

            # Substitute placeholders
            loaded_prompt = (
                base_prompt.replace("{context}", "")
                .replace("{instructions}", f"USER TASK: {query}")
                .replace("{input}", query)
            )
            click.echo(
                f"[{agent_name}] Loaded prompt: autonomous_executor:code_execution",
                err=True,
            )
        except FileNotFoundError:
            # Fallback to inline prompt if prompt files not found
            click.echo(f"[{agent_name}] Using fallback autonomous prompt", err=True)
            loaded_prompt = f"""AUTONOMOUS SINGLE-QUERY MODE - Provide executable solution

You cannot ask questions. The user needs a complete, executable solution NOW.

CRITICAL: If this task requires code generation, you MUST generate a complete executable shell script (.sh).
The script will be automatically executed after you provide it.

DELIVERY FORMAT:
1. Make reasonable assumptions about ambiguous details
2. For code tasks: Generate a COMPLETE bash script that sets up environment, installs deps, and runs
3. Wrap executable code in ```bash ... ``` code blocks
4. Include ALL setup steps (uv, pip, conda, etc.)
5. Do NOT ask "which do you want?" - pick the most likely and document your assumption

USER TASK: {query}

Provide the complete, executable solution now."""

        # Create PromptChain for execution (AgenticStepProcessor available for future agentic workflows)
        chain = PromptChain(
            models=[
                {
                    "name": model_name,
                    "params": {
                        "max_completion_tokens": agent_obj.max_completion_tokens
                    },
                }
            ],
            instructions=[loaded_prompt],
            verbose=True,
        )

        # Create working directory for script execution
        work_dir = tempfile.mkdtemp(prefix="promptchain_exec_")

        # Execute query
        try:
            click.echo(f"[{agent_name}] Processing query autonomously...", err=True)
            response = await chain.process_prompt_async(loaded_prompt)

            # Print response to stdout
            click.echo("\n--- AGENT RESPONSE ---\n")
            click.echo(response)

            # Auto-execute any shell scripts found in response
            click.echo("\n--- AUTO-EXECUTION ---", err=True)
            exec_results = extract_and_run_scripts(response, work_dir)

            if exec_results:
                click.echo(
                    f"\n[AUTO-EXEC] Executed {len(exec_results)} script(s)", err=True
                )
                for script_path, success, output in exec_results:
                    status = "SUCCESS" if success else "FAILED"
                    click.echo(
                        f"  - {os.path.basename(script_path)}: {status}", err=True
                    )
            else:
                click.echo(
                    "[AUTO-EXEC] No executable scripts found in response", err=True
                )

            # Update session usage
            agent_obj.update_usage()
            session_manager.save_session(session_obj)

        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            import traceback

            traceback.print_exc()
            sys.exit(1)

    # Run async query
    asyncio.run(run_query())


# Register subcommand groups
try:
    from .commands.patterns import patterns

    main.add_command(patterns)
except ImportError:
    # Patterns commands not available (missing dependencies)
    pass

try:
    from .commands.chains import chain

    main.add_command(chain)
except ImportError:
    # Chain commands not available
    pass


if __name__ == "__main__":
    main()
