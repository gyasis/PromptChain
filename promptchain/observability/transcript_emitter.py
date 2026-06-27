"""Opt-in JSONL transcript emitter — writes one append-only JSONL transcript per PromptChain run for SIO mining."""

import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..utils.execution_events import ExecutionEvent, ExecutionEventType


@dataclass
class TranscriptEmitterConfig:
    """Configuration for the JSONL transcript emitter.

    Emission is OFF by default (opt-in). Set ``enabled=True`` or the
    ``PROMPTCHAIN_TRANSCRIPTS_ENABLED`` environment variable to activate.
    """

    enabled: bool = False
    base_dir: Path = field(
        default_factory=lambda: Path.home() / ".promptchain" / "transcripts"
    )
    project: Optional[str] = None
    max_files: int = 500
    max_bytes: int = 200 * 1024 * 1024
    max_value_len: int = 8192


class TranscriptEmitter:
    """Observer that appends execution events to a per-run JSONL transcript file."""

    def __init__(
        self, config: Optional[TranscriptEmitterConfig] = None, **kwargs: Any
    ) -> None:
        """Initialise with an optional config; extra kwargs are accepted for forward-compatibility."""
        self.config = config or TranscriptEmitterConfig()

    async def handle_event(self, event: ExecutionEvent) -> None:
        """Receive an execution event (no-op skeleton — writing not yet implemented)."""
        return
