"""Per-role model selection from a user config file.

PromptChain code names no model. The user names them in one YAML file, and each
component asks for its model by role::

    default: <provider/model>        # used by required roles only
    roles:
      supervisor: <provider/model>
      summarizer: <provider/model>

Resolution order for :func:`resolve_model`:

1. the explicit argument the caller passed;
2. ``roles[<role>]`` from the file;
3. ``default`` from the file, only when the model is *required*;
4. a required model raises :class:`ModelNotConfiguredError`; an optional one
   returns ``None`` and its feature stays off.

An optional feature never turns on from ``default`` alone, so setting
``default`` cannot start extra model calls.

The file lives at ``$PROMPTCHAIN_MODELS_CONFIG`` or ``~/.promptchain/models.yaml``.
A missing file is an empty config; a malformed file raises with its path.

Command line::

    python -m promptchain.utils.model_roles show
    python -m promptchain.utils.model_roles init [--force]
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

ENV_VAR = "PROMPTCHAIN_MODELS_CONFIG"


@dataclass(frozen=True)
class Role:
    """One model role: the component it serves and whether a model is required."""

    name: str
    required: bool
    component: str


ROLES: Tuple[Role, ...] = (
    Role("agent", True, "agent_comms LLMAgent / agent() / create_from_template"),
    Role("orchestrator", True, "agent_comms orchestrator() / captain()"),
    Role("supervisor", True, "AgentChain supervisor_model, OrchestratorSupervisor model_name"),
    Role("synthesizer", True, "AgentChain synthesizer_config['model']"),
    Role("history_summarizer", True, "HistorySummarizer model"),
    Role("state_agent", True, "StateAgent / create_state_agent model_name"),
    Role("prompt_engineer", True, "PromptEngineer evaluator_model"),
    Role("chain_builder", True, "DynamicChainBuilder model step"),
    Role("chain", True, "ChainExecutor / ChainFactory / ChainBuilder llm_model"),
    Role("agentic_step", True, "PromptChain agentic step with no models or model_name"),
    Role("router", True, "YAML orchestration.router.model (router mode)"),
    Role("lightrag", True, "LightRAGConfig.model_name"),
    Role("lightrag_embedding", True, "LightRAGConfig.embedding_model"),
    Role("summarizer", False, "AgenticStepProcessor / ContextManagementConfig summarizer_model"),
    Role("prompt_improver", False, "PromptEngineer improver_model (falls back to the evaluator)"),
    Role("judge", False, "BranchingConfig.judge_model"),
    Role("multi_hop_synthesizer", False, "MultiHopConfig.synthesizer_model"),
)

ROLE_NAMES = tuple(r.name for r in ROLES)


class ModelConfigError(ValueError):
    """The models config file cannot be read or has a wrong shape."""


class ModelNotConfiguredError(ValueError):
    """A required model was not passed and the config file does not supply it."""


@dataclass(frozen=True)
class ModelsConfig:
    """Parsed models file."""

    path: Path
    default: Optional[str] = None
    roles: Optional[Dict[str, str]] = None

    def role(self, name: str) -> Optional[str]:
        return (self.roles or {}).get(name) or None


_cache: Optional[ModelsConfig] = None


def config_path() -> Path:
    """Return the models file path: the env override, else ~/.promptchain/models.yaml."""
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".promptchain" / "models.yaml"


def _clean(value: object, where: str, path: Path) -> Optional[str]:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ModelConfigError(f"{path}: {where} must be a string, got {type(value).__name__}")
    return value.strip() or None


def _parse(path: Path) -> ModelsConfig:
    if not path.is_file():
        return ModelsConfig(path=path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError, UnicodeDecodeError) as exc:
        raise ModelConfigError(f"Cannot read models config {path}: {exc}") from exc
    if data is None:
        return ModelsConfig(path=path)
    if not isinstance(data, dict):
        raise ModelConfigError(f"{path}: top level must be a mapping with 'default' and/or 'roles'")
    unknown = set(data) - {"default", "roles"}
    if unknown:
        raise ModelConfigError(
            f"{path}: unknown key(s) {sorted(unknown)}; allowed keys are 'default' and 'roles'"
        )
    raw_roles = data.get("roles") or {}
    if not isinstance(raw_roles, dict):
        raise ModelConfigError(f"{path}: 'roles' must be a mapping of role name to model")
    bad = sorted(set(raw_roles) - set(ROLE_NAMES))
    if bad:
        raise ModelConfigError(
            f"{path}: unknown role(s) {bad}; known roles: {', '.join(ROLE_NAMES)}"
        )
    roles = {}
    for name, value in raw_roles.items():
        cleaned = _clean(value, f"roles.{name}", path)
        if cleaned:
            roles[name] = cleaned
    return ModelsConfig(path=path, default=_clean(data.get("default"), "default", path), roles=roles)


def load() -> ModelsConfig:
    """Return the cached config, reading the file on first use."""
    global _cache
    if _cache is None or _cache.path != config_path():
        _cache = _parse(config_path())
    return _cache


def reload() -> ModelsConfig:
    """Drop the cache and read the file again."""
    global _cache
    _cache = None
    return load()


def configured_default() -> Optional[str]:
    """Return the file's ``default`` model, or None."""
    return load().default


def resolve_model(
    role: str,
    explicit: Optional[str] = None,
    *,
    required: bool,
    param: str,
) -> Optional[str]:
    """Return the model for ``role``.

    Args:
        role: A name from :data:`ROLE_NAMES`.
        explicit: The model the caller passed; wins over the file.
        required: True when the component cannot run without a model. Only a
            required role may fall back to the file's ``default``.
        param: The parameter name, used in the error message.

    Raises:
        ModelNotConfiguredError: ``required`` and no source supplies a model.
        ModelConfigError: the models file is malformed.
    """
    if role not in ROLE_NAMES:
        raise KeyError(f"unknown model role {role!r}; known roles: {', '.join(ROLE_NAMES)}")
    if explicit:
        return explicit
    from_role = load().role(role)
    if from_role:
        return from_role
    if not required:
        return None
    default = load().default
    if default:
        return default
    path = config_path()
    raise ModelNotConfiguredError(
        f"No model for `{param}`: pass {param}=<provider/model>, or set roles.{role} "
        f"(or default) in {path}."
    )


def describe() -> list:
    """Return (role, required, model, source) for every role, as the file resolves them."""
    cfg = load()
    rows = []
    for r in ROLES:
        if cfg.role(r.name):
            rows.append((r.name, r.required, cfg.role(r.name), "role"))
        elif r.required and cfg.default:
            rows.append((r.name, r.required, cfg.default, "default"))
        else:
            rows.append((r.name, r.required, None, "unset"))
    return rows


def show_text() -> str:
    """Render the resolved table for ``show``."""
    cfg = load()
    lines = [
        f"models config: {cfg.path} ({'found' if cfg.path.is_file() else 'not found'})",
        f"default: {cfg.default or '(unset)'}",
        "",
        f"{'role':<24}{'need':<10}{'model':<40}source",
    ]
    for name, required, model, source in describe():
        lines.append(
            f"{name:<24}{'required' if required else 'optional':<10}{model or '-':<40}{source}"
        )
    lines += [
        "",
        "arg = an explicit argument wins over everything and is not shown here.",
        "Optional roles ignore `default`; they stay off until their own role is set.",
    ]
    return "\n".join(lines)


def template_text() -> str:
    """Return the commented template written by ``init``: every role, no models."""
    lines = [
        "# PromptChain model roles. Name the models yourself; PromptChain ships none.",
        "# Precedence: explicit argument > roles.<role> > default (required roles only).",
        "# Format: <provider/model>. Leave a line empty to leave that role unset.",
        "",
        "# Used by every REQUIRED role that has no role entry below.",
        "# Optional roles never read it, so it cannot switch extra features on.",
        "default:",
        "",
        "roles:",
    ]
    for r in ROLES:
        lines.append(f"  # {'required' if r.required else 'optional (feature off when empty)'}: {r.component}")
        lines.append(f"  {r.name}:")
    return "\n".join(lines) + "\n"


def init_file(force: bool = False) -> Path:
    """Write the template to :func:`config_path`; refuse to overwrite without ``force``."""
    path = config_path()
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists (use --force to overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template_text(), encoding="utf-8")
    return path


def main(argv: Optional[list] = None) -> int:
    """Entry point for ``python -m promptchain.utils.model_roles``."""
    args = list(sys.argv[1:] if argv is None else argv)
    cmd = args[0] if args else "show"
    try:
        if cmd == "show":
            print(show_text())
        elif cmd == "init":
            path = init_file(force="--force" in args[1:])
            print(f"wrote {path}\nEdit it and name a model per role.")
        else:
            print("usage: model_roles [show | init [--force]]", file=sys.stderr)
            return 2
    except (ModelConfigError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
