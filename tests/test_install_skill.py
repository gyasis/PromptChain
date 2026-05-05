"""Tests for the `promptchain install-skill` CLI subcommand.

Covers:
- --dry-run does not create anything
- Default install creates a symlink to the bundled skill
- Existing target is backed up as <path>.bak.<YYYY-MM-DD>
- --copy creates a real file (not a symlink) with the same content
- Idempotent: re-running on an already-correct symlink is a no-op
- --force overwrites without backup
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest
from click.testing import CliRunner

from promptchain.cli.install_skill import install_skill, _bundled_skill_path


def test_bundled_skill_path_resolves():
    """The bundled skill file must exist inside the installed package."""
    p = _bundled_skill_path()
    assert p.is_file(), f"Bundled skill not found at {p}"
    content = p.read_text()
    assert "name: promptchain" in content, "Skill front-matter missing"


def test_dry_run_creates_nothing(tmp_path: Path):
    """--dry-run must not create or modify any file at the target."""
    target = tmp_path / "skills" / "promptchain.md"
    runner = CliRunner()
    result = runner.invoke(install_skill, ["--target", str(target), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "(--dry-run set; not making changes)" in result.output
    assert not target.exists(), "dry-run created the target!"
    assert not target.is_symlink(), "dry-run created a symlink!"


def test_default_install_creates_symlink(tmp_path: Path):
    """Default mode: target should be a symlink to the bundled skill."""
    target = tmp_path / "skills" / "promptchain.md"
    runner = CliRunner()
    result = runner.invoke(install_skill, ["--target", str(target)])
    assert result.exit_code == 0, result.output
    assert target.is_symlink(), f"target is not a symlink: {target}"
    assert target.resolve() == _bundled_skill_path().resolve()
    assert target.read_text() == _bundled_skill_path().read_text()


def test_copy_mode_creates_real_file(tmp_path: Path):
    """--copy mode produces a real file (not a symlink) with identical content."""
    target = tmp_path / "skills" / "promptchain.md"
    runner = CliRunner()
    result = runner.invoke(install_skill, ["--target", str(target), "--copy"])
    assert result.exit_code == 0, result.output
    assert target.exists()
    assert not target.is_symlink(), "--copy should NOT create a symlink"
    assert target.read_text() == _bundled_skill_path().read_text()


def test_existing_target_is_backed_up(tmp_path: Path):
    """Pre-existing target gets renamed to <name>.bak.<YYYY-MM-DD> before overwrite."""
    target = tmp_path / "skills" / "promptchain.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("PRIOR USER CONTENT — DO NOT LOSE")

    runner = CliRunner()
    result = runner.invoke(install_skill, ["--target", str(target)])
    assert result.exit_code == 0, result.output

    today = date.today().isoformat()
    backup = target.with_suffix(target.suffix + f".bak.{today}")
    assert backup.exists(), f"backup file missing: {backup}"
    assert backup.read_text() == "PRIOR USER CONTENT — DO NOT LOSE"
    assert target.is_symlink(), "new target should be a symlink"
    assert target.resolve() == _bundled_skill_path().resolve()


def test_idempotent_when_already_linked(tmp_path: Path):
    """Re-running install when the target already symlinks to the bundle is a no-op."""
    target = tmp_path / "skills" / "promptchain.md"
    runner = CliRunner()
    # First install
    r1 = runner.invoke(install_skill, ["--target", str(target)])
    assert r1.exit_code == 0
    # Second install — should detect existing correct symlink
    r2 = runner.invoke(install_skill, ["--target", str(target)])
    assert r2.exit_code == 0
    assert "Already linked" in r2.output
    # Make sure no spurious backup was created
    today = date.today().isoformat()
    spurious = target.with_suffix(target.suffix + f".bak.{today}")
    assert not spurious.exists(), "idempotent re-run created an unnecessary backup"


def test_force_overwrites_without_backup(tmp_path: Path):
    """--force removes any existing target without creating a .bak.<date> file."""
    target = tmp_path / "skills" / "promptchain.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("OLD CONTENT")

    runner = CliRunner()
    result = runner.invoke(install_skill, ["--target", str(target), "--force"])
    assert result.exit_code == 0, result.output
    assert target.is_symlink()
    today = date.today().isoformat()
    no_backup = target.with_suffix(target.suffix + f".bak.{today}")
    assert not no_backup.exists(), "--force should NOT create a backup"
