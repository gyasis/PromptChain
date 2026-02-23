"""CLI commands for advanced agentic patterns.

Provides command-line interface for executing LightRAG patterns:
- branch: Generate branching hypotheses
- expand: Query expansion
- multihop: Multi-hop retrieval
- hybrid: Hybrid search fusion
- sharded: Sharded retrieval
- speculate: Speculative execution
"""

import asyncio
import functools
import sys
from typing import List, Optional, Tuple

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

# Issue #2 Fix: Import centralized event loop manager
from ..utils.event_loop_manager import run_async_in_context

console = Console()


def run_async(coro):
    """Execute async coroutine in sync context.

    FIXED: Uses centralized event loop management to handle both:
    - CLI context (no running loop) - creates new loop via asyncio.run()
    - TUI context (running loop) - raises clear error to use await instead

    This fixes Issue #2: Event Loop Race Conditions in TUI Pattern Commands.
    """
    return run_async_in_context(coro)


def handle_errors(func):
    """Decorator for consistent error handling."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ImportError as e:
            console.print(f"[red]Error:[/red] Missing dependency: {e}")
            console.print("[yellow]Install LightRAG integration:[/yellow] pip install deeplake")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Error:[/red] {type(e).__name__}: {e}")
            sys.exit(1)
    return wrapper


@click.group()
def patterns():
    """Advanced agentic pattern commands.

    Execute sophisticated retrieval and reasoning patterns powered by LightRAG.
    Each pattern solves specific challenges in multi-step reasoning, search,
    and information synthesis.
    """
    pass


@patterns.command()
@click.argument('problem')
@click.option(
    '--count',
    default=3,
    type=int,
    help='Number of hypotheses to generate (default: 3)'
)
@click.option(
    '--mode',
    type=click.Choice(['local', 'global', 'hybrid'], case_sensitive=False),
    default='hybrid',
    help='Search mode: local (entity-focused), global (theme-focused), or hybrid (both)'
)
@click.option(
    '--deeplake-path',
    default=None,
    help='Path to DeepLake dataset (default: ~/.lightrag/default)'
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Show detailed execution information'
)
@handle_errors
def branch(problem: str, count: int, mode: str, deeplake_path: Optional[str], verbose: bool):
    """Generate branching hypotheses for a problem.

    Uses LightRAG's branching thoughts pattern to explore multiple solution
    paths in parallel. Ideal for complex problems with multiple valid approaches.

    PROBLEM: The question or problem to analyze

    Example:
        promptchain patterns branch "How can we reduce carbon emissions?" --count 5
    """
    from promptchain.patterns.executors import execute_branch

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Generating {count} hypotheses...", total=None)

        result = run_async(execute_branch(
            query=problem,
            count=count,
            mode=mode,
            deeplake_path=deeplake_path,
            verbose=verbose,
        ))
        progress.update(task, completed=True)

    # Display results
    tree = Tree(f"[bold cyan]Branching Hypotheses[/bold cyan] for: {problem}")

    if result.get('success') and result.get('hypotheses'):
        for i, hypothesis in enumerate(result['hypotheses'], 1):
            if isinstance(hypothesis, dict):
                branch_node = tree.add(f"[yellow]Hypothesis {i}[/yellow]")
                branch_node.add(f"[green]Approach:[/green] {hypothesis.get('approach', 'N/A')}")
                branch_node.add(f"[blue]Rationale:[/blue] {hypothesis.get('rationale', 'N/A')}")
                if 'confidence' in hypothesis:
                    branch_node.add(f"[magenta]Confidence:[/magenta] {hypothesis['confidence']:.2%}")
            else:
                tree.add(f"[yellow]{i}.[/yellow] {hypothesis}")
    elif result.get('error'):
        tree.add(f"[red]Error:[/red] {result['error']}")
    else:
        tree.add(f"[white]{result.get('result', 'No results')}[/white]")

    console.print(tree)
    console.print(f"[dim]Execution time: {result.get('execution_time_ms', 0):.0f}ms[/dim]")

    if verbose:
        console.print("\n[dim]Raw result:[/dim]")
        console.print(result)


@patterns.command()
@click.argument('query')
@click.option(
    '--strategies',
    multiple=True,
    default=['semantic'],
    type=click.Choice(['synonym', 'semantic', 'acronym', 'contextual'], case_sensitive=False),
    help='Expansion strategies to use (can specify multiple)'
)
@click.option(
    '--max-expansions',
    default=5,
    type=int,
    help='Maximum query variations to generate (default: 5)'
)
@click.option(
    '--deeplake-path',
    default=None,
    help='Path to DeepLake dataset'
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Show detailed execution information'
)
@handle_errors
def expand(query: str, strategies: Tuple[str], max_expansions: int, deeplake_path: Optional[str], verbose: bool):
    """Expand a query using multiple strategies.

    Generates alternative query formulations to improve retrieval coverage.
    Useful when a single query might miss relevant documents due to vocabulary
    mismatch or ambiguous terminology.

    QUERY: The search query to expand

    Example:
        promptchain patterns expand "ML optimization" --strategies semantic --strategies synonym
    """
    from promptchain.patterns.executors import execute_expand

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Expanding query with {len(strategies)} strategies...", total=None)

        result = run_async(execute_expand(
            query=query,
            strategies=list(strategies),
            max_expansions=max_expansions,
            deeplake_path=deeplake_path,
            verbose=verbose,
        ))
        progress.update(task, completed=True)

    # Display results
    console.print(Panel(f"[bold]Original Query:[/bold] {query}", border_style="cyan"))

    table = Table(title="Query Expansions", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Strategy", style="cyan")
    table.add_column("Expanded Query", style="green")

    if result.get('success') and result.get('expansions'):
        for i, expansion_item in enumerate(result['expansions'], 1):
            if isinstance(expansion_item, dict):
                table.add_row(
                    str(i),
                    expansion_item.get('strategy', 'N/A'),
                    expansion_item.get('query', 'N/A')
                )
            else:
                table.add_row(str(i), "unknown", str(expansion_item))
    elif result.get('error'):
        console.print(f"[red]Error:[/red] {result['error']}")
    else:
        console.print(f"[yellow]Result:[/yellow] {result.get('result', 'No results')}")

    console.print(table)
    console.print(f"[dim]Execution time: {result.get('execution_time_ms', 0):.0f}ms[/dim]")


@patterns.command()
@click.argument('question')
@click.option(
    '--max-hops',
    default=5,
    type=int,
    help='Maximum reasoning hops (default: 5)'
)
@click.option(
    '--objective',
    default=None,
    help='Custom objective for reasoning'
)
@click.option(
    '--deeplake-path',
    default=None,
    help='Path to DeepLake dataset'
)
@click.option(
    '--mode',
    type=click.Choice(['local', 'global', 'hybrid'], case_sensitive=False),
    default='hybrid',
    help='Search mode for retrieval'
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Show detailed execution information'
)
@handle_errors
def multihop(question: str, max_hops: int, objective: Optional[str], deeplake_path: Optional[str],
             mode: str, verbose: bool):
    """Answer a complex question via multi-hop retrieval.

    Performs iterative reasoning across multiple retrieval steps. Each step
    refines the search based on previous findings, enabling complex question
    answering that requires synthesizing information from multiple sources.

    QUESTION: The complex question requiring multi-hop reasoning

    Example:
        promptchain patterns multihop "What are the environmental impacts of electric vehicles?" --max-hops 3
    """
    from promptchain.patterns.executors import execute_multihop

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Reasoning across {max_hops} hops...", total=None)

        result = run_async(execute_multihop(
            query=question,
            max_hops=max_hops,
            objective=objective,
            mode=mode,
            deeplake_path=deeplake_path,
            verbose=verbose,
        ))
        progress.update(task, completed=True)

    # Display results
    console.print(Panel(f"[bold]Question:[/bold] {question}", border_style="cyan"))

    if result.get('success'):
        if result.get('hops'):
            tree = Tree("[bold magenta]Reasoning Hops[/bold magenta]")
            for i, hop in enumerate(result['hops'], 1):
                hop_node = tree.add(f"[yellow]Hop {i}[/yellow]")
                if isinstance(hop, dict):
                    hop_node.add(f"[green]Query:[/green] {hop.get('query', 'N/A')}")
                    hop_node.add(f"[blue]Finding:[/blue] {hop.get('finding', 'N/A')}")
                else:
                    hop_node.add(str(hop))
            console.print(tree)

        if result.get('answer'):
            console.print(Panel(
                f"[bold green]Final Answer:[/bold green]\n{result['answer']}",
                border_style="green"
            ))
    elif result.get('error'):
        console.print(f"[red]Error:[/red] {result['error']}")
    else:
        console.print(f"[yellow]Result:[/yellow] {result.get('result', 'No results')}")

    console.print(f"[dim]Execution time: {result.get('execution_time_ms', 0):.0f}ms[/dim]")


@patterns.command()
@click.argument('query')
@click.option(
    '--fusion',
    type=click.Choice(['rrf', 'linear', 'borda'], case_sensitive=False),
    default='rrf',
    help='Fusion algorithm: rrf (Reciprocal Rank Fusion), linear, or borda'
)
@click.option(
    '--weights',
    nargs=2,
    type=float,
    default=(0.5, 0.5),
    help='Weights for [local, global] search (default: 0.5 0.5)'
)
@click.option(
    '--deeplake-path',
    default=None,
    help='Path to DeepLake dataset'
)
@click.option(
    '--top-k',
    default=10,
    type=int,
    help='Number of results to return (default: 10)'
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Show detailed execution information'
)
@handle_errors
def hybrid(query: str, fusion: str, weights: Tuple[float, float], deeplake_path: Optional[str],
           top_k: int, verbose: bool):
    """Execute hybrid search with fusion.

    Combines local (entity-focused) and global (theme-focused) search results
    using advanced fusion algorithms. Provides more comprehensive retrieval
    than single-mode search.

    QUERY: The search query

    Example:
        promptchain patterns hybrid "machine learning best practices" --fusion rrf --weights 0.6 0.4
    """
    from promptchain.patterns.executors import execute_hybrid

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Executing hybrid search with {fusion.upper()} fusion...", total=None)

        result = run_async(execute_hybrid(
            query=query,
            fusion=fusion,
            weights=weights,
            top_k=top_k,
            deeplake_path=deeplake_path,
            verbose=verbose,
        ))
        progress.update(task, completed=True)

    # Display results
    console.print(Panel(f"[bold]Query:[/bold] {query}", border_style="cyan"))
    console.print(f"[dim]Fusion: {fusion.upper()} | Weights: Local={weights[0]}, Global={weights[1]}[/dim]\n")

    table = Table(title="Hybrid Search Results", show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Source", style="cyan", width=10)
    table.add_column("Content", style="green")
    table.add_column("Score", style="yellow", width=8)

    if result.get('success') and result.get('results'):
        for i, res in enumerate(result['results'][:top_k], 1):
            if isinstance(res, dict):
                table.add_row(
                    str(i),
                    res.get('source', 'N/A'),
                    res.get('content', 'N/A')[:100] + "...",
                    f"{res.get('score', 0):.4f}"
                )
    elif result.get('error'):
        console.print(f"[red]Error:[/red] {result['error']}")

    console.print(table)
    console.print(f"[dim]Execution time: {result.get('execution_time_ms', 0):.0f}ms[/dim]")


@patterns.command()
@click.argument('query')
@click.option(
    '--shards',
    multiple=True,
    required=True,
    help='Shard names/paths to query (can specify multiple)'
)
@click.option(
    '--aggregation',
    type=click.Choice(['merge', 'weighted', 'rrf'], case_sensitive=False),
    default='rrf',
    help='Aggregation method: merge (simple concat), weighted, or rrf (default: rrf)'
)
@click.option(
    '--mode',
    type=click.Choice(['local', 'global', 'hybrid'], case_sensitive=False),
    default='hybrid',
    help='Search mode for each shard'
)
@click.option(
    '--top-k',
    default=10,
    type=int,
    help='Number of results per shard (default: 10)'
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Show detailed execution information'
)
@handle_errors
def sharded(query: str, shards: Tuple[str], aggregation: str, mode: str, top_k: int, verbose: bool):
    """Query across multiple LightRAG shards.

    Executes parallel queries across multiple dataset shards and aggregates
    results. Useful for distributed knowledge bases or domain-specific datasets.

    QUERY: The search query to execute across all shards

    Example:
        promptchain patterns sharded "AI ethics" --shards tech_shard --shards academic_shard --aggregation rrf
    """
    from promptchain.patterns.executors import execute_sharded

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Querying {len(shards)} shards...", total=None)

        result = run_async(execute_sharded(
            query=query,
            shards=list(shards),
            aggregation=aggregation,
            mode=mode,
            top_k=top_k,
            verbose=verbose,
        ))
        progress.update(task, completed=True)

    # Display results
    console.print(Panel(f"[bold]Query:[/bold] {query}", border_style="cyan"))
    console.print(f"[dim]Shards: {len(shards)} | Aggregation: {aggregation.upper()} | Mode: {mode}[/dim]\n")

    tree = Tree("[bold magenta]Sharded Results[/bold magenta]")

    if result.get('success'):
        if result.get('shard_results'):
            for shard_name, shard_result in result['shard_results'].items():
                shard_node = tree.add(f"[yellow]Shard: {shard_name}[/yellow]")
                if isinstance(shard_result, list):
                    for i, res in enumerate(shard_result[:5], 1):  # Show top 5 per shard
                        if isinstance(res, dict):
                            shard_node.add(f"[green]{i}.[/green] {res.get('content', 'N/A')[:80]}...")

        if result.get('aggregated'):
            agg_node = tree.add("[bold cyan]Aggregated Results[/bold cyan]")
            for i, res in enumerate(result['aggregated'][:top_k], 1):
                if isinstance(res, dict):
                    agg_node.add(
                        f"[green]{i}.[/green] {res.get('content', 'N/A')[:80]}... "
                        f"[dim](score: {res.get('score', 0):.4f})[/dim]"
                    )
    elif result.get('error'):
        tree.add(f"[red]Error:[/red] {result['error']}")

    console.print(tree)
    console.print(f"[dim]Execution time: {result.get('execution_time_ms', 0):.0f}ms[/dim]")


@patterns.command()
@click.argument('context')
@click.option(
    '--min-confidence',
    default=0.7,
    type=float,
    help='Minimum prediction confidence threshold (default: 0.7)'
)
@click.option(
    '--prefetch-count',
    default=3,
    type=int,
    help='Number of queries to prefetch (default: 3)'
)
@click.option(
    '--deeplake-path',
    default=None,
    help='Path to DeepLake dataset'
)
@click.option(
    '--mode',
    type=click.Choice(['local', 'global', 'hybrid'], case_sensitive=False),
    default='hybrid',
    help='Search mode for speculative queries'
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Show detailed execution information'
)
@handle_errors
def speculate(context: str, min_confidence: float, prefetch_count: int, deeplake_path: Optional[str],
              mode: str, verbose: bool):
    """Speculatively execute likely next queries.

    Predicts and pre-fetches results for likely follow-up queries based on
    conversation context. Reduces latency for multi-turn interactions.

    CONTEXT: Current conversation context or user intent

    Example:
        promptchain patterns speculate "User is researching renewable energy" --prefetch-count 5
    """
    from promptchain.patterns.executors import execute_speculate

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Predicting {prefetch_count} likely queries...", total=None)

        result = run_async(execute_speculate(
            context=context,
            min_confidence=min_confidence,
            prefetch_count=prefetch_count,
            mode=mode,
            deeplake_path=deeplake_path,
            verbose=verbose,
        ))
        progress.update(task, completed=True)

    # Display results
    console.print(Panel(f"[bold]Context:[/bold] {context}", border_style="cyan"))
    console.print(f"[dim]Min Confidence: {min_confidence:.0%} | Prefetch Count: {prefetch_count}[/dim]\n")

    table = Table(title="Speculative Queries", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Predicted Query", style="green")
    table.add_column("Confidence", style="yellow", width=12)
    table.add_column("Status", style="cyan", width=12)

    if result.get('success') and result.get('predictions'):
        for i, prediction in enumerate(result['predictions'], 1):
            if isinstance(prediction, dict):
                confidence = prediction.get('confidence', 0)
                status = "✓ Prefetched" if confidence >= min_confidence else "⊗ Skipped"
                table.add_row(
                    str(i),
                    prediction.get('query', 'N/A'),
                    f"{confidence:.1%}",
                    status
                )
    elif result.get('error'):
        console.print(f"[red]Error:[/red] {result['error']}")

    console.print(table)
    console.print(f"[dim]Execution time: {result.get('execution_time_ms', 0):.0f}ms[/dim]")

    if verbose and result.get('cache_info'):
        console.print("\n[dim]Cache Statistics:[/dim]")
        console.print(result['cache_info'])


if __name__ == '__main__':
    patterns()
