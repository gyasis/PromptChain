#!/usr/bin/env python3
"""
Test script for PromptManager functionality.

This script demonstrates and tests the core functionality of the
PromptManager class including prompt discovery, loading, and template management.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from promptchain.cli.prompt_manager import PromptManager
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def test_prompt_listing():
    """Test listing available prompts."""
    console.print("\n[bold cyan]Test 1: List Available Prompts[/bold cyan]")

    prompts_dir = Path.home() / ".promptchain" / "prompts"
    pm = PromptManager(prompts_dir)

    prompts = pm.list_prompts()

    if not prompts:
        console.print("[yellow]No prompts found[/yellow]")
        return

    table = Table(title="Available Prompts")
    table.add_column("ID", style="cyan")
    table.add_column("Category", style="magenta")
    table.add_column("Description", style="green")

    for prompt in prompts:
        desc = prompt.description or "No description"
        table.add_row(prompt.id, prompt.category, desc[:50] + "..." if len(desc) > 50 else desc)

    console.print(table)
    console.print(f"\n[green]✓ Found {len(prompts)} prompts[/green]")


def test_prompt_loading():
    """Test loading prompts with and without strategies."""
    console.print("\n[bold cyan]Test 2: Load Prompts[/bold cyan]")

    prompts_dir = Path.home() / ".promptchain" / "prompts"
    pm = PromptManager(prompts_dir)

    # Test loading basic prompt
    try:
        content = pm.load_prompt("researcher")
        console.print(Panel(
            content[:200] + "..." if len(content) > 200 else content,
            title="[green]Researcher Prompt (Basic)[/green]",
            border_style="green"
        ))
        console.print("[green]✓ Loaded researcher prompt[/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed to load researcher prompt: {e}[/red]")

    # Test loading prompt with strategy
    try:
        content = pm.load_prompt("researcher", strategy="concise")
        console.print(Panel(
            content[:200] + "..." if len(content) > 200 else content,
            title="[green]Researcher Prompt (with Concise Strategy)[/green]",
            border_style="green"
        ))
        console.print("[green]✓ Loaded researcher prompt with concise strategy[/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed to load researcher prompt with strategy: {e}[/red]")


def test_strategy_listing():
    """Test listing available strategies."""
    console.print("\n[bold cyan]Test 3: List Strategies[/bold cyan]")

    prompts_dir = Path.home() / ".promptchain" / "prompts"
    pm = PromptManager(prompts_dir)

    strategies = pm.list_strategies()

    if not strategies:
        console.print("[yellow]No strategies found[/yellow]")
        return

    table = Table(title="Available Strategies")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Description", style="green")

    for strategy in strategies:
        desc = strategy.description or "No description"
        table.add_row(strategy.id, strategy.name, desc)

    console.print(table)
    console.print(f"\n[green]✓ Found {len(strategies)} strategies[/green]")


def test_template_listing():
    """Test listing agent templates."""
    console.print("\n[bold cyan]Test 4: List Agent Templates[/bold cyan]")

    prompts_dir = Path.home() / ".promptchain" / "prompts"
    pm = PromptManager(prompts_dir)

    templates = pm.list_templates()

    if not templates:
        console.print("[yellow]No templates found[/yellow]")
        return

    table = Table(title="Available Agent Templates")
    table.add_column("Name", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Prompt", style="yellow")
    table.add_column("Strategy", style="green")
    table.add_column("Tags", style="blue")

    for template in templates:
        tags_str = ", ".join(template.tags) if template.tags else "none"
        table.add_row(
            template.name,
            template.model,
            template.prompt_id,
            template.strategy or "none",
            tags_str
        )

    console.print(table)
    console.print(f"\n[green]✓ Found {len(templates)} templates[/green]")


def test_save_custom_prompt():
    """Test saving a custom prompt."""
    console.print("\n[bold cyan]Test 5: Save Custom Prompt[/bold cyan]")

    prompts_dir = Path.home() / ".promptchain" / "prompts"
    pm = PromptManager(prompts_dir)

    # Save a test prompt
    test_content = """# Test Agent

This is a test agent prompt for demonstration purposes.

## Capabilities
- Test capability 1
- Test capability 2

{context}
"""

    try:
        path = pm.save_prompt(
            "test_agent",
            test_content,
            category="custom",
            description="Test agent for demonstration"
        )
        console.print(f"[green]✓ Saved custom prompt to: {path}[/green]")

        # Verify it can be loaded
        loaded = pm.load_prompt("test_agent")
        console.print("[green]✓ Successfully loaded saved prompt[/green]")

        # Clean up
        path.unlink()
        console.print("[green]✓ Cleaned up test prompt[/green]")

    except Exception as e:
        console.print(f"[red]✗ Failed to save/load custom prompt: {e}[/red]")


def test_prompt_search():
    """Test searching prompts."""
    console.print("\n[bold cyan]Test 6: Search Prompts[/bold cyan]")

    prompts_dir = Path.home() / ".promptchain" / "prompts"
    pm = PromptManager(prompts_dir)

    # Search for "code" related prompts
    results = pm.list_prompts(search="code")

    if results:
        console.print(f"[green]✓ Found {len(results)} prompts matching 'code':[/green]")
        for prompt in results:
            console.print(f"  - {prompt.id} ({prompt.category})")
    else:
        console.print("[yellow]No prompts matching 'code' found[/yellow]")

    # Filter by category
    agent_prompts = pm.list_prompts(category="agents")
    console.print(f"\n[green]✓ Found {len(agent_prompts)} prompts in 'agents' category[/green]")


def main():
    """Run all tests."""
    console.print("[bold magenta]PromptManager Test Suite[/bold magenta]")
    console.print("=" * 60)

    try:
        test_prompt_listing()
        test_prompt_loading()
        test_strategy_listing()
        test_template_listing()
        test_save_custom_prompt()
        test_prompt_search()

        console.print("\n" + "=" * 60)
        console.print("[bold green]All Tests Complete![/bold green]")

    except Exception as e:
        console.print(f"\n[bold red]Test suite failed: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
