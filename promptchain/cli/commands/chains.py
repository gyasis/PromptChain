"""
Chain management CLI commands.

Provides commands for creating, listing, running, and managing chains
via the ChainFactory system.

Usage:
    promptchain chain list              # List all available chains
    promptchain chain show <model>      # Show chain details
    promptchain chain run <model> <input>  # Execute a chain
    promptchain chain create <model>    # Create new chain interactively
    promptchain chain delete <model>    # Delete a chain
"""

import click
import json
import sys
from pathlib import Path
from typing import Optional


@click.group()
def chain():
    """Chain management commands for the ChainFactory system.

    Chains are strict, guardrailed LLM workflows that can be versioned,
    nested, and managed via the VIN/Model system.
    """
    pass


@chain.command("list")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed info")
def list_chains(as_json: bool, verbose: bool):
    """List all available chains.

    Examples:
        promptchain chain list
        promptchain chain list --json
        promptchain chain list -v
    """
    try:
        from promptchain.utils.chain_factory import ChainFactory

        factory = ChainFactory()
        chains = factory.list_chains()

        if as_json:
            output = []
            for name, info in chains.items():
                output.append({
                    "model": name,
                    "latest": info.get("latest", "v1.0"),
                    "versions": info.get("versions", []),
                    "tags": info.get("tags", [])
                })
            click.echo(json.dumps(output, indent=2))
        else:
            if not chains:
                click.echo("No chains found. Create one with: promptchain chain create <name>")
                return

            click.echo("\n📦 Available Chains:\n")
            for name, info in chains.items():
                latest = info.get("latest", "v1.0")
                versions = info.get("versions", [latest])
                tags = info.get("tags", [])

                click.echo(f"  {name}")
                click.echo(f"    Latest: {latest}")
                if verbose:
                    click.echo(f"    Versions: {', '.join(versions)}")
                    if tags:
                        click.echo(f"    Tags: {', '.join(tags)}")
                click.echo()

            click.echo(f"Total: {len(chains)} chain(s)\n")

    except Exception as e:
        click.echo(f"Error listing chains: {e}", err=True)
        sys.exit(1)


@chain.command("show")
@click.argument("model")
@click.option("--version", "-v", default=None, help="Specific version (default: latest)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def show_chain(model: str, version: Optional[str], as_json: bool):
    """Show details of a specific chain.

    Examples:
        promptchain chain show query-optimizer
        promptchain chain show query-optimizer --version v1.0
        promptchain chain show query-optimizer --json
    """
    try:
        from promptchain.utils.chain_factory import ChainFactory

        factory = ChainFactory()

        # Build reference
        chain_ref = f"{model}:{version}" if version else model
        chain_def = factory.resolve(chain_ref)

        if as_json:
            click.echo(chain_def.model_dump_json(indent=2))
        else:
            click.echo(f"\n🔗 Chain: {chain_def.model} (v{chain_def.version.lstrip('v')})\n")
            click.echo(f"  VIN: {chain_def.vin}")
            click.echo(f"  Mode: {chain_def.mode.value}")
            click.echo(f"  LLM: {chain_def.llm_model}")

            if chain_def.description:
                click.echo(f"  Description: {chain_def.description}")

            click.echo(f"\n  Steps ({len(chain_def.steps)}):")
            for i, step in enumerate(chain_def.steps, 1):
                step_desc = step.description or step.id
                if step.type.value == "prompt":
                    detail = step.prompt_id or (step.content[:40] + "..." if step.content and len(step.content) > 40 else step.content)
                elif step.type.value == "chain":
                    detail = f"→ {step.chain_id}"
                elif step.type.value == "function":
                    detail = f"fn:{step.function_name}"
                elif step.type.value == "agentic":
                    detail = f"objective:{step.objective[:30]}..." if step.objective else "agentic"
                else:
                    detail = ""

                click.echo(f"    {i}. [{step.type.value}] {step_desc}")
                if detail:
                    click.echo(f"       {detail}")

            click.echo(f"\n  Guardrails:")
            click.echo(f"    Max steps: {chain_def.guardrails.max_steps}")
            click.echo(f"    Timeout: {chain_def.guardrails.timeout_seconds}s")
            click.echo(f"    Max nested depth: {chain_def.guardrails.max_nested_depth}")
            if chain_def.guardrails.forbidden_patterns:
                click.echo(f"    Forbidden patterns: {len(chain_def.guardrails.forbidden_patterns)}")

            click.echo()

    except Exception as e:
        click.echo(f"Error showing chain '{model}': {e}", err=True)
        sys.exit(1)


@chain.command("run")
@click.argument("model")
@click.argument("input_text", required=False)
@click.option("--version", "-v", default=None, help="Specific version (default: latest)")
@click.option("--file", "-f", type=click.Path(exists=True), help="Read input from file")
@click.option("--verbose", is_flag=True, help="Show execution details")
def run_chain(model: str, input_text: Optional[str], version: Optional[str], file: Optional[str], verbose: bool):
    """Execute a chain with input.

    Examples:
        promptchain chain run query-optimizer "SELECT * FROM users"
        promptchain chain run query-optimizer -f query.sql
        promptchain chain run query-optimizer --version v1.0 "input"
    """
    import asyncio

    # Get input
    if file:
        with open(file, 'r') as f:
            input_data = f.read()
    elif input_text:
        input_data = input_text
    else:
        # Read from stdin if no input provided
        if not sys.stdin.isatty():
            input_data = sys.stdin.read()
        else:
            click.echo("Error: No input provided. Use argument, --file, or pipe input.", err=True)
            sys.exit(1)

    async def execute():
        try:
            from promptchain.utils.chain_factory import ChainFactory
            from promptchain.utils.chain_executor import ChainExecutor

            factory = ChainFactory()
            executor = ChainExecutor(factory=factory, verbose=verbose)

            # Build reference
            chain_ref = f"{model}:{version}" if version else model

            if verbose:
                click.echo(f"Executing chain: {chain_ref}", err=True)
                click.echo(f"Input length: {len(input_data)} chars", err=True)

            result = await executor.execute_by_id_async(chain_ref, input_data)

            # Output result
            click.echo(result)

            if verbose:
                records = executor.get_execution_records()
                if records:
                    last = records[-1]
                    click.echo(f"\nExecution time: {last.execution_time_ms}ms", err=True)
                    click.echo(f"Steps executed: {last.steps_executed}", err=True)

        except Exception as e:
            click.echo(f"Error executing chain: {e}", err=True)
            if verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)

    asyncio.run(execute())


@chain.command("create")
@click.argument("model")
@click.option("--description", "-d", default=None, help="Chain description")
@click.option("--mode", type=click.Choice(["strict", "hybrid"]), default="strict", help="Execution mode")
@click.option("--llm", default="openai/gpt-4.1-mini-2025-04-14", help="LLM model for prompt steps")
@click.option("--interactive", "-i", is_flag=True, help="Interactive step creation")
def create_chain(model: str, description: Optional[str], mode: str, llm: str, interactive: bool):
    """Create a new chain.

    Examples:
        promptchain chain create my-chain -d "My custom chain"
        promptchain chain create my-chain --mode hybrid
        promptchain chain create my-chain -i  # Interactive mode
    """
    try:
        from promptchain.utils.chain_factory import ChainFactory
        from promptchain.utils.chain_models import (
            ChainDefinition, ChainStepDefinition, StepType, ChainMode
        )

        factory = ChainFactory()

        steps = []

        if interactive:
            click.echo(f"\n🔧 Creating chain: {model}\n")
            click.echo("Add steps (type 'done' when finished):\n")

            step_num = 1
            while True:
                click.echo(f"Step {step_num}:")
                step_type = click.prompt(
                    "  Type",
                    type=click.Choice(["prompt", "chain", "function", "done"]),
                    default="prompt"
                )

                if step_type == "done":
                    break

                step_id = click.prompt("  ID", default=f"step_{step_num}")

                if step_type == "prompt":
                    content = click.prompt("  Prompt content")
                    steps.append(ChainStepDefinition(
                        id=step_id,
                        type=StepType.PROMPT,
                        content=content
                    ))
                elif step_type == "chain":
                    chain_id = click.prompt("  Chain reference (model:version)")
                    steps.append(ChainStepDefinition(
                        id=step_id,
                        type=StepType.CHAIN,
                        chain_id=chain_id
                    ))
                elif step_type == "function":
                    func_name = click.prompt("  Function name")
                    steps.append(ChainStepDefinition(
                        id=step_id,
                        type=StepType.FUNCTION,
                        function_name=func_name
                    ))

                step_num += 1
                click.echo()

        else:
            # Create with default single prompt step
            steps = [
                ChainStepDefinition(
                    id="main",
                    type=StepType.PROMPT,
                    content="Process the following input: {input}"
                )
            ]

        if not steps:
            click.echo("Error: Chain must have at least one step", err=True)
            sys.exit(1)

        # Create chain definition
        chain_def = ChainDefinition(
            vin="",  # Auto-generated
            model=model,
            version="v1.0",
            description=description,
            mode=ChainMode(mode),
            llm_model=llm,
            steps=steps
        )

        # Save chain
        saved_path = factory.save(chain_def)

        click.echo(f"\n✓ Chain '{model}' created successfully!")
        click.echo(f"  VIN: {chain_def.vin}")
        click.echo(f"  Path: {saved_path}")
        click.echo(f"  Steps: {len(steps)}")
        click.echo(f"\nRun with: promptchain chain run {model} \"your input\"\n")

    except Exception as e:
        click.echo(f"Error creating chain: {e}", err=True)
        sys.exit(1)


@chain.command("delete")
@click.argument("model")
@click.option("--version", "-v", default=None, help="Specific version (default: all versions)")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
def delete_chain(model: str, version: Optional[str], force: bool):
    """Delete a chain.

    Examples:
        promptchain chain delete my-chain
        promptchain chain delete my-chain --version v1.0
        promptchain chain delete my-chain -f  # No confirmation
    """
    try:
        from promptchain.utils.chain_factory import ChainFactory

        factory = ChainFactory()

        if version:
            target = f"{model}:{version}"
        else:
            target = model

        if not force:
            confirm = click.confirm(f"Delete chain '{target}'?", default=False)
            if not confirm:
                click.echo("Cancelled.")
                return

        factory.delete(target)
        click.echo(f"✓ Chain '{target}' deleted.")

    except Exception as e:
        click.echo(f"Error deleting chain: {e}", err=True)
        sys.exit(1)


@chain.command("validate")
@click.argument("model")
@click.option("--version", "-v", default=None, help="Specific version (default: latest)")
def validate_chain(model: str, version: Optional[str]):
    """Validate a chain definition.

    Examples:
        promptchain chain validate query-optimizer
        promptchain chain validate query-optimizer --version v1.0
    """
    try:
        from promptchain.utils.chain_factory import ChainFactory

        factory = ChainFactory()

        chain_ref = f"{model}:{version}" if version else model
        chain_def = factory.resolve(chain_ref)

        result = factory.validate(chain_def)

        if result.passed:
            click.echo(f"✓ Chain '{chain_ref}' is valid")
        else:
            click.echo(f"✗ Chain '{chain_ref}' has issues:\n")
            for issue in result.issues:
                icon = "❌" if issue.severity == "error" else "⚠️" if issue.severity == "warning" else "ℹ️"
                step_info = f" (step: {issue.step_id})" if issue.step_id else ""
                click.echo(f"  {icon} [{issue.severity}]{step_info} {issue.message}")
                if issue.suggestion:
                    click.echo(f"     → {issue.suggestion}")

        # Show warnings even if passed
        warnings = [i for i in result.issues if i.severity == "warning"]
        if result.passed and warnings:
            click.echo(f"\nWarnings ({len(warnings)}):")
            for w in warnings:
                click.echo(f"  ⚠️ {w.message}")

    except Exception as e:
        click.echo(f"Error validating chain: {e}", err=True)
        sys.exit(1)


@chain.command("export")
@click.argument("model")
@click.option("--version", "-v", default=None, help="Specific version (default: latest)")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def export_chain(model: str, version: Optional[str], output: Optional[str]):
    """Export a chain definition to JSON.

    Examples:
        promptchain chain export query-optimizer > chain.json
        promptchain chain export query-optimizer -o chain.json
    """
    try:
        from promptchain.utils.chain_factory import ChainFactory

        factory = ChainFactory()

        chain_ref = f"{model}:{version}" if version else model
        chain_def = factory.resolve(chain_ref)

        json_output = chain_def.model_dump_json(indent=2)

        if output:
            with open(output, 'w') as f:
                f.write(json_output)
            click.echo(f"✓ Exported to {output}")
        else:
            click.echo(json_output)

    except Exception as e:
        click.echo(f"Error exporting chain: {e}", err=True)
        sys.exit(1)


@chain.command("import")
@click.argument("file", type=click.Path(exists=True))
@click.option("--force", "-f", is_flag=True, help="Overwrite existing chain")
def import_chain(file: str, force: bool):
    """Import a chain definition from JSON file.

    Examples:
        promptchain chain import chain.json
        promptchain chain import chain.json --force
    """
    try:
        from promptchain.utils.chain_factory import ChainFactory
        from promptchain.utils.chain_models import ChainDefinition

        factory = ChainFactory()

        with open(file, 'r') as f:
            data = json.load(f)

        chain_def = ChainDefinition(**data)

        # Check if exists
        try:
            existing = factory.resolve(f"{chain_def.model}:{chain_def.version}")
            if not force:
                click.echo(f"Chain '{chain_def.model}:{chain_def.version}' already exists. Use --force to overwrite.", err=True)
                sys.exit(1)
        except:
            pass  # Chain doesn't exist, proceed

        saved_path = factory.save(chain_def)

        click.echo(f"✓ Imported chain '{chain_def.model}:{chain_def.version}'")
        click.echo(f"  VIN: {chain_def.vin}")
        click.echo(f"  Path: {saved_path}")

    except Exception as e:
        click.echo(f"Error importing chain: {e}", err=True)
        sys.exit(1)
