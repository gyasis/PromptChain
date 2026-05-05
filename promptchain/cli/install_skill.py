"""`promptchain install-skill` — install the PromptChain Claude Code skill.

Copies (or symlinks) the bundled `promptchain.md` skill file from the
installed package into the agent harness skills directory (default
`~/.claude/skills/`). This makes the package's harness-side routing
available to ANY LLM coding agent (Claude Code, Cursor, etc.) without
requiring the user to clone the repo first.

Discovery contract:
- The skill source-of-truth lives INSIDE the package at
  `promptchain/data/skills/promptchain.md` (so it's shipped by
  `pip install promptchain`).
- The harness expects skills at `~/.claude/skills/<name>.md` (or the
  user-specified path).
- Default install: SYMLINK from `~/.claude/skills/promptchain.md` →
  the bundled file. Updates to the package automatically flow through.
- `--copy` opts into a hard copy if the user wants to edit locally
  without affecting the package.

Backup behaviour:
- If the target already exists and is NOT already a symlink to the
  bundled file, it is renamed to `<target>.bak.<YYYY-MM-DD>` before
  the new symlink/copy is created.
"""
from __future__ import annotations

import shutil
import sys
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Optional

import click


SKILL_NAME = "promptchain.md"


def _bundled_skill_path() -> Path:
    """Return the path to the bundled skill file inside the installed package."""
    # `importlib.resources.files` works for both wheel-installed and editable
    # installs (PEP 660); returns a Traversable that's a real path on disk.
    pkg = resources.files("promptchain") / "data" / "skills" / SKILL_NAME
    p = Path(str(pkg))
    if not p.exists():
        raise click.ClickException(
            f"Bundled skill file missing at {p}. Reinstall the package: "
            f"`pip install --force-reinstall promptchain`."
        )
    return p


def _backup_existing(target: Path) -> Optional[Path]:
    """Rename an existing target to `.bak.<date>`. Returns the backup path."""
    if not target.exists() and not target.is_symlink():
        return None

    suffix = date.today().isoformat()
    backup = target.with_suffix(target.suffix + f".bak.{suffix}")
    # If a backup with today's date already exists, append a counter
    counter = 1
    while backup.exists() or backup.is_symlink():
        backup = target.with_suffix(target.suffix + f".bak.{suffix}.{counter}")
        counter += 1
    target.rename(backup)
    return backup


@click.command("install-skill")
@click.option(
    "--target",
    type=click.Path(path_type=Path),
    default=None,
    help="Where to install the skill. Default: ~/.claude/skills/promptchain.md",
)
@click.option(
    "--copy",
    is_flag=True,
    default=False,
    help="Copy the file instead of symlinking. Use if you want to edit locally "
    "without package updates flowing through.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite without backing up the existing target.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print what would happen without doing it.",
)
def install_skill(
    target: Optional[Path],
    copy: bool,
    force: bool,
    dry_run: bool,
) -> None:
    """Install the PromptChain skill into your Claude Code harness.

    By default, symlinks the bundled skill into ~/.claude/skills/promptchain.md
    so the harness picks it up at session start. Use --copy for a hard copy.
    """
    src = _bundled_skill_path()

    if target is None:
        target = Path.home() / ".claude" / "skills" / SKILL_NAME
    target = target.expanduser()

    click.echo(f"Source: {src}")
    click.echo(f"Target: {target}")
    click.echo(f"Mode:   {'copy' if copy else 'symlink'}")

    if dry_run:
        click.echo("(--dry-run set; not making changes)")
        return

    target.parent.mkdir(parents=True, exist_ok=True)

    # If the target already symlinks to our source, nothing to do.
    if target.is_symlink() and target.resolve() == src.resolve():
        click.echo("✓ Already linked to the bundled skill — nothing to do.")
        return

    # Backup any existing target (file, symlink to elsewhere, etc.)
    if (target.exists() or target.is_symlink()) and not force:
        backup = _backup_existing(target)
        if backup:
            click.echo(f"Backed up existing target to: {backup}")
    elif force and (target.exists() or target.is_symlink()):
        target.unlink()
        click.echo("Removed existing target (--force).")

    if copy:
        shutil.copy2(src, target)
        click.echo(f"✓ Copied skill to {target}")
    else:
        # Symlink — store relative if both live under the user's home for
        # robustness across home-dir relocation; otherwise use absolute.
        try:
            rel = src.relative_to(target.parent.resolve(strict=False))
            target.symlink_to(rel)
            click.echo(f"✓ Linked {target} → {rel} (relative)")
        except ValueError:
            target.symlink_to(src)
            click.echo(f"✓ Linked {target} → {src}")

    click.echo("\nNext: in your Claude Code session, the 'promptchain' skill ")
    click.echo("will auto-load when triggered. To uninstall, just delete ")
    click.echo(f"{target} (the package itself remains installed).")


if __name__ == "__main__":
    install_skill()
