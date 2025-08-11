"""
Research Agent CLI

Rich command-line interface using Typer for the Research Agent system.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, List
from enum import Enum

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich.live import Live
from rich.layout import Layout
from rich.syntax import Syntax
from rich.tree import Tree

from .core.config import ResearchConfig
from .core.orchestrator import AdvancedResearchOrchestrator
from .core.session import ResearchSession, SessionStatus
from .utils.chat_interface import InteractiveChatInterface, ChatMode

# Initialize Typer app
app = typer.Typer(
    name="research-agent",
    help="🔬 Advanced Research Agent with 3-tier RAG processing",
    add_completion=True,
    rich_markup_mode="rich"
)

console = Console()


class OutputFormat(str, Enum):
    """Output format options"""
    json = "json"
    markdown = "markdown"
    text = "text"
    html = "html"


class ProcessingMode(str, Enum):
    """Processing mode options"""
    cloud = "cloud"
    ollama = "ollama"
    hybrid = "hybrid"


@app.command()
def research(
    topic: str = typer.Argument(..., help="Research topic to investigate"),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to configuration YAML file"
    ),
    max_papers: int = typer.Option(
        100, "--max-papers", "-p",
        help="Maximum number of papers to retrieve"
    ),
    max_iterations: int = typer.Option(
        5, "--iterations", "-i",
        help="Maximum research iterations"
    ),
    mode: ProcessingMode = typer.Option(
        ProcessingMode.cloud, "--mode", "-m",
        help="Processing mode (cloud/ollama/hybrid)"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output directory for results"
    ),
    interactive: bool = typer.Option(
        True, "--interactive/--no-interactive",
        help="Enable interactive chat after research"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Enable verbose output"
    )
):
    """
    🔍 Conduct comprehensive research on a topic
    
    This command will:
    - Generate 8-12 research queries
    - Search literature from Sci-Hub, ArXiv, and PubMed
    - Process papers through 3-tier RAG system
    - Perform iterative refinement with gap analysis
    - Generate comprehensive literature review
    - Optionally start interactive chat
    """
    console.print(f"\n[bold cyan]🔬 Research Agent[/bold cyan]")
    console.print(f"[dim]Topic:[/dim] [bold]{topic}[/bold]\n")
    
    # Load configuration
    config = _load_configuration(config_file, mode, max_papers, max_iterations, verbose)
    
    # Create output directory
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(f"./research_output/{topic.replace(' ', '_')}")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize orchestrator
    with console.status("[bold green]Initializing research system...") as status:
        orchestrator = AdvancedResearchOrchestrator(config)
        status.update("[bold green]Research system ready!")
    
    # Progress callback
    def progress_callback(message: str):
        console.print(f"[cyan]►[/cyan] {message}")
    
    # Run research session
    try:
        console.print(Panel.fit(
            f"[bold]Starting Research Session[/bold]\n\n"
            f"• Max Papers: {max_papers}\n"
            f"• Max Iterations: {max_iterations}\n"
            f"• Mode: {mode.value}\n"
            f"• Output: {output_dir}",
            title="Configuration"
        ))
        
        # Run async research
        session = asyncio.run(
            orchestrator.conduct_research_session(
                research_topic=topic,
                callbacks=[progress_callback]
            )
        )
        
        # Display results
        _display_research_results(session, console)
        
        # Save results
        _save_research_results(session, output_dir, console)
        
        # Start interactive chat if requested
        if interactive and session.status == SessionStatus.COMPLETED:
            console.print("\n[bold cyan]Starting Interactive Chat...[/bold cyan]")
            asyncio.run(_interactive_chat_session(session, orchestrator, console))
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Research interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)
    finally:
        asyncio.run(orchestrator.shutdown())


@app.command()
def chat(
    session_file: Path = typer.Argument(..., help="Path to saved research session JSON"),
    mode: ChatMode = typer.Option(
        ChatMode.QUESTION_ANSWERING, "--mode", "-m",
        help="Chat mode (qa/deep_dive/compare/summary/custom)"
    )
):
    """
    💬 Start interactive chat with saved research session
    """
    console.print(f"\n[bold cyan]💬 Research Chat[/bold cyan]")
    
    # Load session
    try:
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        # Reconstruct session (simplified - would need proper deserialization)
        session = ResearchSession(
            topic=session_data['topic'],
            session_id=session_data['session_id']
        )
        
        # Initialize minimal orchestrator for chat
        config = ResearchConfig()
        orchestrator = AdvancedResearchOrchestrator(config)
        
        # Start chat
        asyncio.run(_interactive_chat_session(session, orchestrator, console, initial_mode=mode))
        
    except Exception as e:
        console.print(f"[bold red]Error loading session:[/bold red] {e}")
        sys.exit(1)


@app.command()
def analyze(
    papers_dir: Path = typer.Argument(..., help="Directory containing papers"),
    query: str = typer.Argument(..., help="Analysis query"),
    output_format: OutputFormat = typer.Option(
        OutputFormat.markdown, "--format", "-f",
        help="Output format"
    )
):
    """
    📊 Analyze papers with a specific query
    """
    console.print(f"\n[bold cyan]📊 Paper Analysis[/bold cyan]")
    console.print(f"[dim]Query:[/dim] {query}")
    console.print(f"[dim]Papers:[/dim] {papers_dir}\n")
    
    # This would implement direct paper analysis
    console.print("[yellow]Direct analysis mode not yet implemented[/yellow]")


@app.command()
def list_sessions(
    sessions_dir: Path = typer.Option(
        Path("./research_output"), "--dir", "-d",
        help="Directory containing research sessions"
    )
):
    """
    📋 List saved research sessions
    """
    console.print(f"\n[bold cyan]📋 Research Sessions[/bold cyan]\n")
    
    # Create table
    table = Table(title="Available Sessions")
    table.add_column("Session ID", style="cyan")
    table.add_column("Topic", style="green")
    table.add_column("Date", style="yellow")
    table.add_column("Papers", justify="right")
    table.add_column("Status")
    
    # Find session files
    session_files = list(sessions_dir.glob("*/session.json"))
    
    for session_file in session_files:
        try:
            with open(session_file, 'r') as f:
                data = json.load(f)
            
            table.add_row(
                data.get('session_id', 'unknown')[:8],
                data.get('topic', 'unknown')[:30],
                data.get('created_at', 'unknown')[:10],
                str(data.get('total_papers', 0)),
                data.get('status', 'unknown')
            )
        except:
            continue
    
    console.print(table)


@app.command()
def stats(
    session_file: Path = typer.Argument(..., help="Path to research session")
):
    """
    📈 Show detailed statistics for a research session
    """
    console.print(f"\n[bold cyan]📈 Session Statistics[/bold cyan]\n")
    
    try:
        with open(session_file, 'r') as f:
            data = json.load(f)
        
        # Create statistics panels
        stats_layout = Layout()
        stats_layout.split_column(
            Layout(name="overview"),
            Layout(name="details")
        )
        
        # Overview statistics
        overview = Panel(
            f"[bold]Research Topic:[/bold] {data.get('topic', 'N/A')}\n"
            f"[bold]Session ID:[/bold] {data.get('session_id', 'N/A')[:8]}\n"
            f"[bold]Status:[/bold] {data.get('status', 'N/A')}\n"
            f"[bold]Created:[/bold] {data.get('created_at', 'N/A')}",
            title="Overview"
        )
        
        # Detailed statistics table
        details_table = Table(title="Research Metrics")
        details_table.add_column("Metric", style="cyan")
        details_table.add_column("Value", justify="right")
        
        stats = data.get('statistics', {})
        details_table.add_row("Total Papers", str(stats.get('total_papers', 0)))
        details_table.add_row("Queries Processed", str(stats.get('queries_completed', 0)))
        details_table.add_row("Iterations", str(stats.get('iterations_completed', 0)))
        details_table.add_row("Success Rate", f"{stats.get('success_rate', 0):.1%}")
        details_table.add_row("Processing Time", f"{stats.get('total_time', 0):.1f}s")
        
        console.print(overview)
        console.print(details_table)
        
        # Show paper distribution
        if 'paper_statistics' in data:
            paper_stats = data['paper_statistics']
            
            # Year distribution
            year_tree = Tree("📅 Papers by Year")
            for year, count in paper_stats.get('papers_by_year', {}).items():
                year_tree.add(f"{year}: {count} papers")
            
            # Source distribution
            source_tree = Tree("📚 Papers by Source")
            for source, count in paper_stats.get('papers_by_source', {}).items():
                source_tree.add(f"{source}: {count} papers")
            
            console.print(year_tree)
            console.print(source_tree)
        
    except Exception as e:
        console.print(f"[bold red]Error loading statistics:[/bold red] {e}")


@app.command()
def export(
    session_file: Path = typer.Argument(..., help="Path to research session"),
    output_file: Path = typer.Argument(..., help="Output file path"),
    format: OutputFormat = typer.Option(
        OutputFormat.markdown, "--format", "-f",
        help="Export format"
    ),
    include_citations: bool = typer.Option(
        True, "--citations/--no-citations",
        help="Include bibliography"
    )
):
    """
    📤 Export research results to different formats
    """
    console.print(f"\n[bold cyan]📤 Exporting Research[/bold cyan]")
    console.print(f"[dim]Format:[/dim] {format.value}")
    console.print(f"[dim]Output:[/dim] {output_file}\n")
    
    try:
        with open(session_file, 'r') as f:
            data = json.load(f)
        
        if format == OutputFormat.markdown:
            content = _export_to_markdown(data, include_citations)
        elif format == OutputFormat.json:
            content = json.dumps(data, indent=2)
        elif format == OutputFormat.html:
            content = _export_to_html(data, include_citations)
        else:
            content = _export_to_text(data, include_citations)
        
        with open(output_file, 'w') as f:
            f.write(content)
        
        console.print(f"[green]✓[/green] Exported to {output_file}")
        
    except Exception as e:
        console.print(f"[bold red]Export failed:[/bold red] {e}")


# Helper functions

def _load_configuration(
    config_file: Optional[Path],
    mode: ProcessingMode,
    max_papers: int,
    max_iterations: int,
    verbose: bool
) -> ResearchConfig:
    """Load and merge configuration"""
    if config_file and config_file.exists():
        config = ResearchConfig.from_yaml(str(config_file))
    else:
        config = ResearchConfig()
    
    # Override with CLI options
    config.research_session.max_papers_total = max_papers
    config.research_session.max_iterations = max_iterations
    config.general.verbose = verbose
    
    # Set processing mode
    if mode == ProcessingMode.ollama:
        config.set_ollama_mode()
    elif mode == ProcessingMode.hybrid:
        config.set_hybrid_mode()
    
    return config


def _display_research_results(session: ResearchSession, console: Console):
    """Display research results in rich format"""
    
    # Summary panel
    summary = Panel(
        f"[bold green]Research Complete![/bold green]\n\n"
        f"• Papers Analyzed: {len(session.papers)}\n"
        f"• Queries Processed: {len(session.completed_queries)}/{len(session.queries)}\n"
        f"• Iterations: {session.current_iteration}\n"
        f"• Completion Score: {session.calculate_completion_score():.1%}",
        title="Summary"
    )
    console.print(summary)
    
    # Key findings
    if session.literature_review:
        exec_summary = session.literature_review.get('literature_review', {}).get('executive_summary', {})
        
        if exec_summary.get('key_findings'):
            findings_tree = Tree("🔍 Key Findings")
            for finding in exec_summary['key_findings'][:5]:
                findings_tree.add(finding)
            console.print(findings_tree)
        
        if exec_summary.get('research_gaps'):
            gaps_tree = Tree("🎯 Research Gaps")
            for gap in exec_summary['research_gaps'][:5]:
                gaps_tree.add(gap)
            console.print(gaps_tree)


def _save_research_results(session: ResearchSession, output_dir: Path, console: Console):
    """Save research results to files"""
    
    with console.status("[bold green]Saving results...") as status:
        # Save session data
        session_file = output_dir / "session.json"
        with open(session_file, 'w') as f:
            json.dump(session.to_dict(), f, indent=2, default=str)
        
        # Save literature review
        if session.literature_review:
            review_file = output_dir / "literature_review.json"
            with open(review_file, 'w') as f:
                json.dump(session.literature_review, f, indent=2)
            
            # Also save as markdown
            md_file = output_dir / "literature_review.md"
            md_content = _export_to_markdown(session.literature_review, include_citations=True)
            with open(md_file, 'w') as f:
                f.write(md_content)
        
        status.update("[bold green]Results saved!")
    
    console.print(f"\n[green]✓[/green] Results saved to {output_dir}")


async def _interactive_chat_session(
    session: ResearchSession,
    orchestrator: AdvancedResearchOrchestrator,
    console: Console,
    initial_mode: ChatMode = ChatMode.QUESTION_ANSWERING
):
    """Run interactive chat session"""
    
    # Initialize chat interface
    chat_interface = InteractiveChatInterface(
        session=session,
        config=orchestrator.config.__dict__,
        orchestrator=orchestrator
    )
    
    chat_interface.set_mode(initial_mode)
    
    # Display welcome message
    console.print(Panel(
        "[bold]Interactive Research Chat[/bold]\n\n"
        "Commands:\n"
        "• /mode - Change chat mode\n"
        "• /suggest - Get question suggestions\n"
        "• /history - Show chat history\n"
        "• /stats - Show session statistics\n"
        "• /export - Export chat session\n"
        "• /help - Show help\n"
        "• /quit - Exit chat",
        title="Welcome"
    ))
    
    # Show suggested questions
    suggestions = chat_interface.get_suggested_questions()
    console.print("\n[dim]Suggested questions:[/dim]")
    for i, suggestion in enumerate(suggestions[:5], 1):
        console.print(f"[dim]{i}.[/dim] {suggestion}")
    
    console.print("\n[cyan]Type your question or use a command:[/cyan]\n")
    
    # Chat loop
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                if user_input == '/quit':
                    if Confirm.ask("Exit chat session?"):
                        break
                
                elif user_input == '/mode':
                    modes = chat_interface.get_available_modes()
                    console.print("\n[bold]Available Modes:[/bold]")
                    for mode_info in modes:
                        console.print(f"• {mode_info['name']} ({mode_info['mode']}): {mode_info['description']}")
                    
                    mode_choice = Prompt.ask("Select mode", 
                                            choices=[m['mode'] for m in modes])
                    chat_interface.set_mode(ChatMode(mode_choice))
                    console.print(f"[green]Mode changed to: {mode_choice}[/green]")
                
                elif user_input == '/suggest':
                    suggestions = chat_interface.get_suggested_questions()
                    console.print("\n[bold]Suggested Questions:[/bold]")
                    for i, suggestion in enumerate(suggestions, 1):
                        console.print(f"{i}. {suggestion}")
                
                elif user_input == '/history':
                    history = chat_interface.get_chat_history(limit=10)
                    console.print("\n[bold]Chat History:[/bold]")
                    for entry in history:
                        role_color = "cyan" if entry['role'] == 'user' else "green"
                        console.print(f"[{role_color}]{entry['role'].title()}:[/{role_color}] {entry['message'][:200]}...")
                
                elif user_input == '/stats':
                    stats = session.get_session_statistics()
                    console.print(Panel(
                        f"Papers: {stats.get('total_papers', 0)}\n"
                        f"Queries: {stats.get('queries_completed', 0)}\n"
                        f"Completion: {stats.get('completion_score', 0):.1%}",
                        title="Session Statistics"
                    ))
                
                elif user_input == '/export':
                    export_file = Path(f"chat_export_{session.session_id[:8]}.json")
                    export_data = chat_interface.export_chat_session()
                    with open(export_file, 'w') as f:
                        json.dump(export_data, f, indent=2, default=str)
                    console.print(f"[green]Chat exported to {export_file}[/green]")
                
                elif user_input == '/help':
                    console.print(Panel(
                        "Commands:\n"
                        "• /mode - Change chat mode\n"
                        "• /suggest - Get question suggestions\n"
                        "• /history - Show chat history\n"
                        "• /stats - Show session statistics\n"
                        "• /export - Export chat session\n"
                        "• /help - Show this help\n"
                        "• /quit - Exit chat",
                        title="Help"
                    ))
                
                else:
                    console.print(f"[yellow]Unknown command: {user_input}[/yellow]")
                
                continue
            
            # Process regular message
            with console.status("[dim]Thinking...[/dim]"):
                response = await chat_interface.process_message(user_input)
            
            # Display response
            console.print(f"\n[bold green]Assistant:[/bold green]")
            console.print(Markdown(response))
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Chat interrupted[/yellow]")
            if Confirm.ask("Exit chat session?"):
                break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    # Generate final summary
    console.print("\n[dim]Generating chat summary...[/dim]")
    summary = await chat_interface.generate_final_summary()
    console.print(Panel(summary, title="Chat Summary"))


def _export_to_markdown(data: dict, include_citations: bool) -> str:
    """Export data to markdown format"""
    md_lines = []
    
    # Handle different data structures
    if 'literature_review' in data:
        # Full synthesis data
        lit_review = data['literature_review']
        exec_summary = lit_review.get('executive_summary', {})
        
        md_lines.append(f"# Literature Review: {data.get('topic', 'Research Topic')}\n")
        md_lines.append(f"## Executive Summary\n")
        md_lines.append(exec_summary.get('overview', 'No overview available') + '\n')
        
        md_lines.append("### Key Findings\n")
        for finding in exec_summary.get('key_findings', []):
            md_lines.append(f"- {finding}")
        
        md_lines.append("\n### Research Gaps\n")
        for gap in exec_summary.get('research_gaps', []):
            md_lines.append(f"- {gap}")
        
        # Add sections
        sections = lit_review.get('sections', {})
        for section_name, section_content in sections.items():
            md_lines.append(f"\n## {section_name.replace('_', ' ').title()}\n")
            if isinstance(section_content, dict):
                md_lines.append(section_content.get('content', '') + '\n')
            else:
                md_lines.append(str(section_content) + '\n')
        
        # Add citations if requested
        if include_citations and 'citations' in data:
            md_lines.append("\n## Bibliography\n")
            for entry in data['citations'].get('bibliography', []):
                authors = ', '.join(entry.get('authors', [])[:3])
                md_lines.append(f"- {authors} ({entry.get('year', 'N/A')}). "
                              f"**{entry.get('title', 'Untitled')}**. "
                              f"*{entry.get('venue', 'Unknown')}*.")
    
    else:
        # Simple data export
        md_lines.append(f"# Research Data Export\n")
        md_lines.append(f"```json\n{json.dumps(data, indent=2)}\n```")
    
    return '\n'.join(md_lines)


def _export_to_html(data: dict, include_citations: bool) -> str:
    """Export data to HTML format"""
    # Convert markdown to HTML (simplified)
    md_content = _export_to_markdown(data, include_citations)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Research Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; }}
        h3 {{ color: #999; }}
    </style>
</head>
<body>
    <pre>{md_content}</pre>
</body>
</html>"""
    
    return html


def _export_to_text(data: dict, include_citations: bool) -> str:
    """Export data to plain text"""
    # Similar to markdown but without formatting
    md_content = _export_to_markdown(data, include_citations)
    # Remove markdown syntax
    text = md_content.replace('#', '').replace('*', '').replace('`', '')
    return text


if __name__ == "__main__":
    app()