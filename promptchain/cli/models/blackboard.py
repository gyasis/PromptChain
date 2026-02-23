"""
Blackboard Entry Model for Shared Data.

Provides key-value storage for agent collaboration.

FR-011 to FR-015: Blackboard Collaboration
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
import json


@dataclass
class BlackboardEntry:
    """
    Represents a shared data entry on the blackboard.

    Attributes:
        key: Unique identifier for the entry
        value: Stored value (JSON-serializable)
        written_by: Agent that wrote/updated the entry
        written_at: Last write timestamp
        version: Version number for optimistic locking
    """

    key: str
    value: Any
    written_by: str
    written_at: float = field(default_factory=lambda: datetime.now().timestamp())
    version: int = 1

    @classmethod
    def create(
        cls,
        key: str,
        value: Any,
        written_by: str
    ) -> "BlackboardEntry":
        """Create a new blackboard entry."""
        return cls(
            key=key,
            value=value,
            written_by=written_by
        )

    def update(self, value: Any, written_by: str) -> None:
        """Update the entry value, incrementing version."""
        self.value = value
        self.written_by = written_by
        self.written_at = datetime.now().timestamp()
        self.version += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "value": self.value,
            "written_by": self.written_by,
            "written_at": self.written_at,
            "version": self.version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BlackboardEntry":
        """Create from dictionary."""
        return cls(
            key=data["key"],
            value=data["value"],
            written_by=data["written_by"],
            written_at=data.get("written_at", datetime.now().timestamp()),
            version=data.get("version", 1)
        )

    @classmethod
    def from_db_row(cls, row: tuple) -> "BlackboardEntry":
        """Create from database row.

        BUG-013 fix: Validates column count to detect schema changes.

        IMPORTANT: This method assumes the following SELECT column order:
            SELECT session_id, key, value_json, written_by, written_at, version

        If the SELECT query uses a different column order, this method will fail.
        Callers MUST ensure consistent column ordering in SELECT statements.

        Args:
            row: Database row tuple with 6 columns in the order above

        Returns:
            BlackboardEntry instance

        Raises:
            ValueError: If row doesn't have expected 6 columns
        """
        # BUG-013 fix: Validate column count to catch schema changes early
        expected_columns = 6
        if len(row) != expected_columns:
            raise ValueError(
                f"BlackboardEntry.from_db_row expects {expected_columns} columns "
                f"(session_id, key, value_json, written_by, written_at, version) "
                f"but got {len(row)} columns. Check SELECT statement column order."
            )

        # Columns: session_id, key, value_json, written_by, written_at, version
        return cls(
            key=row[1],  # Skip session_id at row[0]
            value=json.loads(row[2]) if row[2] else None,
            written_by=row[3],
            written_at=row[4],
            version=row[5]
        )

    def __repr__(self) -> str:
        value_preview = str(self.value)[:30]
        if len(str(self.value)) > 30:
            value_preview += "..."
        return (
            f"BlackboardEntry(key='{self.key}', "
            f"value={value_preview}, "
            f"v{self.version})"
        )
