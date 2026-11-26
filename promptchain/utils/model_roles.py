"""Per-role model selection from a user config file.

PromptChain code names no model. The user names them in one YAML file, and each
component asks for its model by role::

    default: <provider/model>        # used by required roles only
    endpoints:                       # optional: where a model runs
      <name>: {api_base: <url>, provider: <litellm provider>}
    roles:
      supervisor: <provider/model>
      summarizer: {model: <name>, endpoint: <name>, think: false}

A role value (and ``default``) is either a model string or a mapping with
``model``, an optional ``endpoint`` naming an entry of ``endpoints``, and an
optional ``think`` (true/false) that switches a thinking model's reasoning on or
off with the switch that endpoint takes. An
endpoint turns into litellm call kwargs (``api_base``, and
``custom_llm_provider`` when it names a ``provider``). :func:`resolve_model_spec`
returns them with the model; :func:`resolve_model` returns the model string only.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

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
class ModelSpec:
    """A resolved model and where it runs.

    ``source`` is ``arg``, ``role`` or ``default``. ``extra_kwargs`` carries
    litellm kwargs other than ``api_base`` (``custom_llm_provider`` and the
    thinking switch). ``think`` is the configured thinking setting, or None.
    """

    model: str
    source: str
    api_base: Optional[str] = None
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
    endpoint: Optional[str] = None
    think: Optional[bool] = None

    @property
    def params(self) -> Dict[str, Any]:
        """The litellm call kwargs this spec adds."""
        out = dict(self.extra_kwargs)
        if self.api_base:
            out["api_base"] = self.api_base
        return out

    def as_model(self) -> Union[str, Dict[str, Any]]:
        """Return a value for PromptChain's ``models=`` list: the string, or a
        ``{"name", "params"}`` dict when the spec carries call kwargs."""
        return {"name": self.model, "params": self.params} if self.params else self.model


@dataclass(frozen=True)
class ModelsConfig:
    """Parsed models file."""

    path: Path
    default: Optional[ModelSpec] = None
    roles: Optional[Dict[str, ModelSpec]] = None
    endpoints: Optional[Dict[str, Dict[str, Any]]] = None

    def role(self, name: str) -> Optional[ModelSpec]:
        return (self.roles or {}).get(name)


_cache: Optional[ModelsConfig] = None
_ENDPOINT_KEYS = ("api_base", "provider")


def config_path() -> Path:
    """Return the models file path: the env override, else ~/.promptchain/models.yaml."""
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".promptchain" / "models.yaml"


def _text(value: object, where: str, path: Path) -> Optional[str]:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ModelConfigError(f"{path}: {where} must be a string, got {type(value).__name__}")
    return value.strip() or None


def _parse_endpoints(raw: object, path: Path) -> Dict[str, Dict[str, Any]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ModelConfigError(f"{path}: 'endpoints' must be a mapping of name to settings")
    out: Dict[str, Dict[str, Any]] = {}
    for name, body in raw.items():
        if not isinstance(body, dict):
            raise ModelConfigError(f"{path}: endpoints.{name} must be a mapping with api_base")
        extra = sorted(set(body) - set(_ENDPOINT_KEYS))
        if extra:
            raise ModelConfigError(
                f"{path}: endpoints.{name} has unsupported key(s) {extra}; "
                f"allowed: {', '.join(_ENDPOINT_KEYS)}"
            )
        api_base = _text(body.get("api_base"), f"endpoints.{name}.api_base", path)
        if not api_base:
            raise ModelConfigError(f"{path}: endpoints.{name}.api_base is required")
        out[str(name)] = {"api_base": api_base, "provider": _text(body.get("provider"), f"endpoints.{name}.provider", path)}
    return out


def _parse_spec(
    value: object, where: str, source: str, endpoints: Dict[str, Dict[str, Any]], path: Path
) -> Optional[ModelSpec]:
    think: Optional[bool] = None
    if isinstance(value, dict):
        extra = sorted(set(value) - {"model", "endpoint", "think"})
        if extra:
            raise ModelConfigError(
                f"{path}: {where} has unsupported key(s) {extra}; allowed: model, endpoint, think"
            )
        model = _text(value.get("model"), f"{where}.model", path)
        endpoint = _text(value.get("endpoint"), f"{where}.endpoint", path)
        if "think" in value:
            think = value["think"]
            if not isinstance(think, bool):
                raise ModelConfigError(f"{path}: {where}.think must be true or false")
    else:
        model, endpoint = _text(value, where, path), None
    if not model:
        if endpoint:
            raise ModelConfigError(f"{path}: {where} names endpoint {endpoint!r} but no model")
        return None
    if not endpoint:
        return ModelSpec(model=model, source=source, extra_kwargs=_think_kwargs(think, None), think=think)
    if endpoint not in endpoints:
        known = ", ".join(sorted(endpoints)) or "none defined"
        raise ModelConfigError(
            f"{path}: {where} uses unknown endpoint {endpoint!r}; endpoints defined: {known}"
        )
    ep = endpoints[endpoint]
    extra_kwargs = {"custom_llm_provider": ep["provider"]} if ep["provider"] else {}
    extra_kwargs.update(_think_kwargs(think, ep["provider"]))
    return ModelSpec(
        model=model, source=source, api_base=ep["api_base"], extra_kwargs=extra_kwargs,
        endpoint=endpoint, think=think,
    )


def _think_kwargs(think: Optional[bool], provider: Optional[str]) -> Dict[str, Any]:
    """litellm kwargs that switch thinking on or off for one call.

    Each endpoint takes its own switch and ignores the other at HTTP 200: Ollama
    takes ``think``; an OpenAI-compatible llama-server takes
    ``chat_template_kwargs.enable_thinking``. ``None`` sends nothing, so the
    model's own behaviour applies."""
    if think is None:
        return {}
    if provider == "openai":
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": think}}}
    return {"think": think}


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
        raise ModelConfigError(f"{path}: top level must be a mapping with 'default', 'endpoints' and/or 'roles'")
    unknown = set(data) - {"default", "roles", "endpoints"}
    if unknown:
        raise ModelConfigError(
            f"{path}: unknown key(s) {sorted(unknown)}; allowed keys are 'default', 'endpoints' and 'roles'"
        )
    endpoints = _parse_endpoints(data.get("endpoints"), path)
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
        spec = _parse_spec(value, f"roles.{name}", "role", endpoints, path)
        if spec:
            roles[name] = spec
    default = _parse_spec(data.get("default"), "default", "default", endpoints, path)
    return ModelsConfig(path=path, default=default, roles=roles, endpoints=endpoints)


def load() -> ModelsConfig:
    """Return the cached config, reading the file on first use."""
    global _cache
    if _cache is None or _cache.path != config_path():
        _cache = _parse(config_path())
    return _cache


def endpoint(name: str) -> Optional[Dict[str, Any]]:
    """The settings of ``endpoints.<name>`` ({api_base, provider}), or None when it is not defined."""
    return (load().endpoints or {}).get(name)


def reload() -> ModelsConfig:
    """Drop the cache and read the file again."""
    global _cache
    _cache = None
    return load()


def configured_default() -> Optional[str]:
    """Return the file's ``default`` model string, or None."""
    spec = load().default
    return spec.model if spec else None


def resolve_model_spec(
    role: str,
    explicit: Optional[str] = None,
    *,
    required: bool,
    param: str,
) -> Optional[ModelSpec]:
    """Return the :class:`ModelSpec` for ``role``.

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
        return ModelSpec(model=explicit, source="arg")
    cfg = load()
    spec = cfg.role(role)
    if spec:
        return spec
    if not required:
        return None
    if cfg.default:
        return cfg.default
    raise ModelNotConfiguredError(
        f"No model for `{param}`: pass {param}=<provider/model>, or set roles.{role} "
        f"(or default) in {config_path()}."
    )


def resolve_model(
    role: str,
    explicit: Optional[str] = None,
    *,
    required: bool,
    param: str,
) -> Optional[str]:
    """Return only the model string of :func:`resolve_model_spec`."""
    spec = resolve_model_spec(role, explicit, required=required, param=param)
    return spec.model if spec else None


def describe() -> list:
    """Return (role, required, spec-or-None) for every role, as the file resolves them."""
    cfg = load()
    rows = []
    for r in ROLES:
        spec = cfg.role(r.name) or (cfg.default if r.required else None)
        rows.append((r.name, r.required, spec))
    return rows


def show_text() -> str:
    """Render the resolved table for ``show``."""
    cfg = load()
    d = cfg.default
    lines = [
        f"models config: {cfg.path} ({'found' if cfg.path.is_file() else 'not found'})",
        f"default: {d.model if d else '(unset)'}" + (f" @ {d.api_base}" if d and d.api_base else ""),
        "",
        f"{'role':<24}{'need':<10}{'model':<34}{'source':<9}{'think':<7}endpoint",
    ]
    for name, required, spec in describe():
        where = ""
        if spec and spec.api_base:
            where = f"{spec.endpoint or ''} {spec.api_base}".strip()
        lines.append(
            f"{name:<24}{'required' if required else 'optional':<10}"
            f"{spec.model if spec else '-':<34}{spec.source if spec else 'unset':<9}"
            f"{_think_label(spec):<7}{where}"
        )
    lines += [
        "",
        "arg = an explicit argument wins over everything and is not shown here.",
        "Optional roles ignore `default`; they stay off until their own role is set.",
    ]
    return "\n".join(lines)


def _think_label(spec: Optional[ModelSpec]) -> str:
    if spec is None or spec.think is None:
        return "-"
    return "on" if spec.think else "off"


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
        "# Where a model runs (optional). A role may then be {model: <name>, endpoint: <name>}.",
        "# Add think: false to a mapping to turn a thinking model's reasoning off for that role.",
        "# endpoints:",
        "#   my-ollama:   {api_base: \"http://<host>:<port>/<path>\"}",
        "#   my-llamacpp: {api_base: \"http://<host>:<port>/<path>/v1\", provider: openai}",
        "",
        "roles:",
        "  # Example, uncomment and fill in:",
        "  # supervisor: {model: <name>, endpoint: my-llamacpp}",
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
