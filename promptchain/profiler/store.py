"""store — JSON profile store for the Model Profiler (F2).

Persists ``CapabilityProfile`` records to a single JSON file with the on-disk shape::

    {"schema_version": 1, "profiles": {"<model_id>": <CapabilityProfile.to_dict()>, ...}}

Writes are ATOMIC (temp file + ``os.replace``) so a crash never leaves a half-written
store, and ``upsert`` is idempotent. See specs/013-model-profiler/contracts/profile-schema.md.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from promptchain.profiler.jacket import CapabilityProfile

SCHEMA_VERSION = 1


def ewma_update(old: float, observed: float, lam: float = 0.2) -> float:
    """Exponentially-weighted moving-average update (research D5): new = (1-λ)·old + λ·observed."""
    return (1 - lam) * old + lam * observed


class ProfileStore:
    """A small, atomic, idempotent JSON store of CapabilityProfiles keyed by model id."""

    def __init__(self, path) -> None:
        self.path = Path(path)

    def load(self) -> dict:
        """Return the parsed store, or a fresh empty store if missing/empty/corrupt."""
        empty = {"schema_version": SCHEMA_VERSION, "profiles": {}}
        try:
            text = self.path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return empty
        if not text.strip():
            return empty
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            return empty
        # Normalise shape defensively.
        if not isinstance(data, dict):
            return empty
        data.setdefault("schema_version", SCHEMA_VERSION)
        data.setdefault("profiles", {})
        return data

    def save(self, data: dict) -> None:
        """Atomically write *data* to disk (temp file in the same dir, then os.replace)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=self.path.name + ".", suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, default=str)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def upsert(self, profile: CapabilityProfile) -> None:
        """Insert or replace *profile* (keyed by model_id) and persist."""
        data = self.load()
        data.setdefault("profiles", {})[profile.model_id] = profile.to_dict()
        self.save(data)

    def get(self, model_id: str) -> Optional[CapabilityProfile]:
        """Return the stored CapabilityProfile for *model_id*, or None if absent."""
        data = self.load()
        raw = data.get("profiles", {}).get(model_id)
        if raw is None:
            return None
        return CapabilityProfile.from_dict(raw)
