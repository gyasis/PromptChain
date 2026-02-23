#!/usr/bin/env python3
"""Verification script for Phase 2 (Foundational) completion.

This script verifies that all Phase 2 tasks (T006-T031) are complete
and the foundational infrastructure is ready for user story work.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.table import Table

console = Console()


def verify_data_models():
    """Verify T006-T011: All data models exist and are importable."""
    console.print("\n[bold cyan]Verifying Data Models (T006-T011)[/bold cyan]")

    checks = []

    # T006-T007: Agent + HistoryConfig (AgentConfig was renamed to Agent)
    try:
        from promptchain.cli.models.agent_config import Agent, HistoryConfig
        checks.append(("Agent (T006)", True, "Importable"))
        checks.append(("HistoryConfig (T007)", True, "Importable"))
    except Exception as e:
        checks.append(("Agent (T006)", False, str(e)))
        checks.append(("HistoryConfig (T007)", False, str(e)))

    # T008: MCPServerConfig
    try:
        from promptchain.cli.models.mcp_config import MCPServerConfig
        checks.append(("MCPServerConfig (T008)", True, "Importable"))
    except Exception as e:
        checks.append(("MCPServerConfig (T008)", False, str(e)))

    # T009: WorkflowState + WorkflowStep
    try:
        from promptchain.cli.models.workflow import WorkflowState, WorkflowStep
        checks.append(("WorkflowState (T009)", True, "Importable"))
        checks.append(("WorkflowStep (T009)", True, "Importable"))
    except Exception as e:
        checks.append(("WorkflowState (T009)", False, str(e)))
        checks.append(("WorkflowStep (T009)", False, str(e)))

    # T010: OrchestrationConfig + RouterConfig
    try:
        from promptchain.cli.models.orchestration_config import OrchestrationConfig, RouterConfig
        checks.append(("OrchestrationConfig (T010)", True, "Importable"))
        checks.append(("RouterConfig (T010)", True, "Importable"))
    except Exception as e:
        checks.append(("OrchestrationConfig (T010)", False, str(e)))
        checks.append(("RouterConfig (T010)", False, str(e)))

    # T011: Session model with V2 fields
    try:
        from promptchain.cli.models.session import Session
        session_fields = Session.__dataclass_fields__.keys()

        required_v2_fields = [
            'orchestration_config',
            'mcp_servers',
            'workflow_state',
            'schema_version'
        ]

        missing_fields = [f for f in required_v2_fields if f not in session_fields]

        if missing_fields:
            checks.append(("Session V2 fields (T011)", False, f"Missing: {missing_fields}"))
        else:
            checks.append(("Session V2 fields (T011)", True, "All V2 fields present"))
    except Exception as e:
        checks.append(("Session V2 fields (T011)", False, str(e)))

    return checks


def verify_database_migration():
    """Verify T012-T015: V2 schema migration exists."""
    console.print("\n[bold cyan]Verifying Database Migration (T012-T015)[/bold cyan]")

    checks = []

    # Check migration file exists
    migration_path = Path("promptchain/cli/migrations/v2_schema.py")
    if migration_path.exists():
        checks.append(("V2 migration file (T012)", True, str(migration_path)))

        # Check migration functions exist
        try:
            from promptchain.cli.migrations.v2_schema import (
                migrate_v1_to_v2,
                verify_migration,
                get_migration_sql
            )
            checks.append(("migrate_v1_to_v2() (T013)", True, "Function exists"))
            checks.append(("verify_migration() (T014)", True, "Function exists"))
            checks.append(("get_migration_sql() (T015)", True, "Function exists"))
        except Exception as e:
            checks.append(("Migration functions (T013-T015)", False, str(e)))
    else:
        checks.append(("V2 migration file (T012)", False, "File not found"))

    return checks


def verify_yaml_infrastructure():
    """Verify T016-T021: YAML configuration infrastructure."""
    console.print("\n[bold cyan]Verifying YAML Infrastructure (T016-T021)[/bold cyan]")

    checks = []

    # T016-T019: YAMLConfigTranslator
    try:
        from promptchain.cli.config.yaml_translator import (
            YAMLConfigTranslator,
            load_config_with_precedence
        )

        translator = YAMLConfigTranslator()

        # Check class methods exist
        class_methods = [
            'build_agents',
            'build_agent_chain',
            'build_mcp_servers'
        ]

        for method in class_methods:
            if hasattr(translator, method):
                checks.append((f"YAMLConfigTranslator.{method}()", True, "Method exists"))
            else:
                checks.append((f"YAMLConfigTranslator.{method}()", False, "Method missing"))

        # Check module-level function (T021)
        if callable(load_config_with_precedence):
            checks.append(("load_config_with_precedence() (T021)", True, "Function exists"))
        else:
            checks.append(("load_config_with_precedence() (T021)", False, "Function missing"))

    except Exception as e:
        checks.append(("YAMLConfigTranslator (T016-T019)", False, str(e)))

    # T020: YAML validator
    try:
        from promptchain.cli.config.yaml_validator import YAMLConfigValidator
        checks.append(("YAMLConfigValidator (T020)", True, "Importable"))
    except Exception as e:
        checks.append(("YAMLConfigValidator (T020)", False, str(e)))

    return checks


def verify_agent_templates():
    """Verify T022-T027: Agent templates system."""
    console.print("\n[bold cyan]Verifying Agent Templates (T022-T027)[/bold cyan]")

    checks = []

    try:
        from promptchain.cli.utils.agent_templates import (
            AGENT_TEMPLATES,
            create_from_template
        )

        checks.append(("Agent templates module (T022)", True, "Importable"))

        # Check template definitions exist
        required_templates = ['researcher', 'coder', 'analyst', 'terminal']

        for template_name in required_templates:
            if template_name in AGENT_TEMPLATES:
                checks.append((f"{template_name} template", True, "Template defined"))
            else:
                checks.append((f"{template_name} template", False, "Template missing"))

        # Check create_from_template function
        if callable(create_from_template):
            checks.append(("create_from_template() (T027)", True, "Function exists"))
        else:
            checks.append(("create_from_template() (T027)", False, "Function missing"))

    except Exception as e:
        checks.append(("Agent templates (T022-T027)", False, str(e)))

    return checks


def verify_session_manager_extensions():
    """Verify T028-T031: SessionManager V2 extensions."""
    console.print("\n[bold cyan]Verifying SessionManager Extensions (T028-T031)[/bold cyan]")

    checks = []

    try:
        from promptchain.cli.session_manager import SessionManager

        # Check V2 methods exist
        v2_methods = [
            'save_agent_configs',      # T028
            'load_agent_configs',      # T028
            'save_mcp_servers',        # T029
            'load_mcp_servers',        # T029
            'save_workflow',           # T030
            'load_workflow',           # T030
            'resume_workflow',         # T030
        ]

        for method in v2_methods:
            if hasattr(SessionManager, method):
                checks.append((f"SessionManager.{method}()", True, "Method exists"))
            else:
                checks.append((f"SessionManager.{method}()", False, "Method missing"))

    except Exception as e:
        checks.append(("SessionManager extensions (T028-T031)", False, str(e)))

    return checks


def print_results(checks, title):
    """Print verification results as a table."""
    table = Table(title=title)
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details", style="dim")

    passed = 0
    failed = 0

    for component, status, details in checks:
        if status:
            table.add_row(component, "[green]✓ PASS[/green]", details)
            passed += 1
        else:
            table.add_row(component, "[red]✗ FAIL[/red]", details)
            failed += 1

    console.print(table)
    return passed, failed


def main():
    """Run all Phase 2 verification checks."""
    console.print("[bold magenta]Phase 2 (Foundational) Verification[/bold magenta]")
    console.print("=" * 70)

    all_checks = []

    # Run all verification checks
    all_checks.extend(verify_data_models())
    all_checks.extend(verify_database_migration())
    all_checks.extend(verify_yaml_infrastructure())
    all_checks.extend(verify_agent_templates())
    all_checks.extend(verify_session_manager_extensions())

    # Print summary
    console.print("\n[bold]Phase 2 Verification Summary[/bold]")
    passed, failed = print_results(all_checks, "All Phase 2 Components")

    console.print(f"\n[bold]Total: {passed} passed, {failed} failed[/bold]")

    if failed == 0:
        console.print("\n[bold green]✓ Phase 2 (Foundational) COMPLETE![/bold green]")
        console.print("\n[dim]All foundational infrastructure is in place.[/dim]")
        console.print("[dim]User story work (Phase 3-8) can now begin.[/dim]")
        return 0
    else:
        console.print(f"\n[bold red]✗ Phase 2 incomplete: {failed} component(s) missing[/bold red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
