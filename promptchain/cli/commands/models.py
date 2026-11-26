"""``promptchain models``: show and initialise the per-role models file."""

import sys

import click

from promptchain.utils import model_roles


@click.group(name="models")
def models() -> None:
    """Show or create the models file that names a model per role."""


@models.command("show")
def show_cmd() -> None:
    """Print each role, its resolved model, and where it came from."""
    try:
        click.echo(model_roles.show_text())
    except model_roles.ModelConfigError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@models.command("init")
@click.option("--force", is_flag=True, help="Overwrite an existing file.")
def init_cmd(force: bool) -> None:
    """Write a template with every role and no model names."""
    try:
        path = model_roles.init_file(force=force)
    except FileExistsError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    click.echo(f"wrote {path}\nEdit it and name a model per role.")
