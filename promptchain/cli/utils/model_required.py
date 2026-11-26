"""Model selection helpers for the CLI and TUI.

PromptChain ships no default model. A session, agent or config can hold no model
yet; the code that needs one calls :func:`require_model`, which raises
:class:`ModelNotSelectedError` with a message that tells the user how to pick one.
"""

from typing import Optional

NO_MODEL_MESSAGE = (
    "No model selected. Pick one with Ctrl+N or /model, start with "
    "--model <provider/model>, or set default_model in ~/.promptchain/config.json."
)

NOT_SET_LABEL = "(not set - pick one with /model)"


class ModelNotSelectedError(ValueError):
    """Raised when an operation needs a model and none has been chosen."""


def require_model(*candidates: Optional[str], context: str = "") -> str:
    """Return the first non-empty candidate, else raise ModelNotSelectedError.

    Args:
        candidates: Model names in priority order (None and "" are skipped).
        context: What needed the model, added to the error message.
    """
    for candidate in candidates:
        if candidate:
            return candidate
    prefix = f"{context}: " if context else ""
    raise ModelNotSelectedError(prefix + NO_MODEL_MESSAGE)


def display_model(model: Optional[str]) -> str:
    """Return the model name for display, or a label that says none is set."""
    return model if model else NOT_SET_LABEL
