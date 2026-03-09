"""
Semantic Memo Store for Long-Term Memory (AG2 Pattern)

Issue #7 Enhancement: Implements AG2's "Teachability" pattern for storing
and retrieving "Lessons Learned" from past task executions.

Architecture:
- SQLite database for persistent storage
- Vector embeddings for semantic search
- Cosine similarity for retrieving relevant memos
- Integration with AgenticStepProcessor for context injection

Benefits:
- Learn from past successes and failures
- Reduce repeated mistakes
- Speed up similar task resolution
- Build institutional knowledge over time
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Memo:
    """Represents a stored lesson learned.

    Attributes:
        memo_id: Unique identifier
        task_description: Description of the task that was performed
        solution: The solution or approach that worked
        outcome: Result of applying the solution (success/failure)
        embedding: Vector embedding of task description
        timestamp: When the memo was created
        metadata: Additional contextual information
        relevance_score: Similarity score when retrieved (populated during search)
    """

    memo_id: int
    task_description: str
    solution: str
    outcome: str
    embedding: Optional[np.ndarray] = None
    timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    relevance_score: Optional[float] = None


class MemoStore:
    """Persistent store for task/solution memos with vector search.

    Implements AG2's Teachability pattern for long-term memory.

    Features:
    - SQLite persistence with automatic schema creation
    - Vector embeddings for semantic similarity
    - Cosine similarity search for relevant memos
    - Metadata filtering and deduplication
    - Integration ready for AgenticStepProcessor
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        embedding_function: Optional[Any] = None,
        max_memos: int = 1000,
        similarity_threshold: float = 0.7,
    ):
        """Initialize memo store.

        Args:
            db_path: Path to SQLite database (default: ~/.promptchain/memos.db)
            embedding_function: Function that takes text and returns vector embedding.
                               If None, uses simple bag-of-words approximation.
            max_memos: Maximum memos to store (oldest removed when exceeded)
            similarity_threshold: Minimum cosine similarity for retrieval (0.0-1.0)
        """
        # Setup database path
        if db_path is None:
            db_path = str(Path.home() / ".promptchain" / "memos.db")

        self.db_path = db_path
        self.embedding_function = embedding_function
        self.max_memos = max_memos
        self.similarity_threshold = similarity_threshold

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        logger.info(
            f"MemoStore initialized: db={db_path}, "
            f"max_memos={max_memos}, threshold={similarity_threshold}"
        )

    def _init_database(self) -> None:
        """Create database schema if not exists."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memos (
                    memo_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_description TEXT NOT NULL,
                    solution TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    embedding BLOB,
                    timestamp REAL NOT NULL,
                    metadata TEXT
                )
            """
            )

            # Create index on timestamp for efficient cleanup
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_timestamp ON memos(timestamp DESC)
            """
            )

            conn.commit()
            logger.debug("Database schema initialized")

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate vector embedding for text.

        Args:
            text: Text to embed

        Returns:
            Numpy array representing embedding vector
        """
        if self.embedding_function is not None:
            try:
                return self.embedding_function(text)
            except Exception as e:
                logger.warning(f"Embedding function failed: {e}. Using fallback.")

        # Fallback: Simple bag-of-words with TF weighting
        words = text.lower().split()
        word_counts: Dict[str, int] = {}
        for word in words:
            if len(word) > 3:  # Filter short words
                word_counts[word] = word_counts.get(word, 0) + 1

        # Create fixed-size vector (100 dimensions)
        # Use hash of word to determine position
        vector = np.zeros(100)
        for word, count in word_counts.items():
            idx = hash(word) % 100
            vector[idx] += count

        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score between 0.0 and 1.0
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def store_memo(
        self,
        task_description: str,
        solution: str,
        outcome: str = "success",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Store a new memo with task/solution pair.

        Args:
            task_description: Description of the task performed
            solution: The solution or approach that was used
            outcome: Result of the solution (success/failure/partial)
            metadata: Additional contextual information

        Returns:
            ID of the stored memo
        """
        # Generate embedding
        embedding = self._generate_embedding(task_description)
        embedding_bytes = embedding.tobytes()

        # Store in database
        timestamp = time.time()
        metadata_json = json.dumps(metadata) if metadata else None

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO memos (task_description, solution, outcome, embedding, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    task_description,
                    solution,
                    outcome,
                    embedding_bytes,
                    timestamp,
                    metadata_json,
                ),
            )
            memo_id = cursor.lastrowid
            conn.commit()

        # Enforce max_memos limit
        self._cleanup_old_memos()

        logger.info(f"Stored memo #{memo_id}: {task_description[:50]}...")
        return memo_id or 0

    def _cleanup_old_memos(self) -> None:
        """Remove oldest memos if max_memos exceeded."""
        with sqlite3.connect(self.db_path) as conn:
            # Count current memos
            cursor = conn.execute("SELECT COUNT(*) FROM memos")
            count = cursor.fetchone()[0]

            if count > self.max_memos:
                # Delete oldest memos
                to_delete = count - self.max_memos
                conn.execute(
                    """
                    DELETE FROM memos
                    WHERE memo_id IN (
                        SELECT memo_id FROM memos
                        ORDER BY timestamp ASC
                        LIMIT ?
                    )
                    """,
                    (to_delete,),
                )
                conn.commit()
                logger.debug(
                    f"Cleaned up {to_delete} old memos (limit: {self.max_memos})"
                )

    def retrieve_relevant_memos(
        self,
        task_description: str,
        top_k: int = 3,
        outcome_filter: Optional[str] = None,
    ) -> List[Memo]:
        """Retrieve memos most relevant to the given task.

        Args:
            task_description: Description of the current task
            top_k: Number of top similar memos to retrieve
            outcome_filter: Filter by outcome (e.g., "success" only)

        Returns:
            List of Memo objects sorted by relevance (highest first)
        """
        # Generate query embedding
        query_embedding = self._generate_embedding(task_description)

        # Retrieve all memos from database
        with sqlite3.connect(self.db_path) as conn:
            if outcome_filter:
                cursor = conn.execute(
                    "SELECT memo_id, task_description, solution, outcome, embedding, timestamp, metadata "
                    "FROM memos WHERE outcome = ?",
                    (outcome_filter,),
                )
            else:
                cursor = conn.execute(
                    "SELECT memo_id, task_description, solution, outcome, embedding, timestamp, metadata "
                    "FROM memos"
                )

            rows = cursor.fetchall()

        # Calculate similarities
        memo_scores = []
        for row in rows:
            (
                memo_id,
                task_desc,
                solution,
                outcome,
                embedding_bytes,
                timestamp,
                metadata_json,
            ) = row

            # Reconstruct embedding
            if embedding_bytes:
                embedding = np.frombuffer(embedding_bytes, dtype=np.float64)
                similarity = self._cosine_similarity(query_embedding, embedding)

                # Only include if above threshold
                if similarity >= self.similarity_threshold:
                    metadata = json.loads(metadata_json) if metadata_json else None
                    memo = Memo(
                        memo_id=memo_id,
                        task_description=task_desc,
                        solution=solution,
                        outcome=outcome,
                        embedding=embedding,
                        timestamp=timestamp,
                        metadata=metadata,
                        relevance_score=similarity,
                    )
                    memo_scores.append((similarity, memo))

        # Sort by similarity and return top_k
        memo_scores.sort(key=lambda x: x[0], reverse=True)
        relevant_memos = [memo for _, memo in memo_scores[:top_k]]

        logger.info(
            f"Retrieved {len(relevant_memos)} relevant memos for: {task_description[:50]}... "
            f"(threshold: {self.similarity_threshold})"
        )

        return relevant_memos

    def get_all_memos(self, limit: int = 100) -> List[Memo]:
        """Retrieve all memos ordered by timestamp (most recent first).

        Args:
            limit: Maximum number of memos to retrieve

        Returns:
            List of Memo objects
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT memo_id, task_description, solution, outcome, embedding, timestamp, metadata
                FROM memos
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()

        memos = []
        for row in rows:
            (
                memo_id,
                task_desc,
                solution,
                outcome,
                embedding_bytes,
                timestamp,
                metadata_json,
            ) = row
            embedding = (
                np.frombuffer(embedding_bytes, dtype=np.float64)
                if embedding_bytes
                else None
            )
            metadata = json.loads(metadata_json) if metadata_json else None

            memo = Memo(
                memo_id=memo_id,
                task_description=task_desc,
                solution=solution,
                outcome=outcome,
                embedding=embedding,
                timestamp=timestamp,
                metadata=metadata,
            )
            memos.append(memo)

        return memos

    def delete_memo(self, memo_id: int) -> bool:
        """Delete a specific memo by ID.

        Args:
            memo_id: ID of memo to delete

        Returns:
            True if deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM memos WHERE memo_id = ?", (memo_id,))
            conn.commit()
            deleted = cursor.rowcount > 0

        if deleted:
            logger.info(f"Deleted memo #{memo_id}")
        else:
            logger.warning(f"Memo #{memo_id} not found for deletion")

        return deleted

    def clear_all_memos(self) -> int:
        """Delete all memos from the store.

        Returns:
            Number of memos deleted
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM memos")
            count = cursor.rowcount
            conn.commit()

        logger.info(f"Cleared all memos ({count} deleted)")
        return count

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the memo store.

        Returns:
            Dictionary with statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            # Total count
            cursor = conn.execute("SELECT COUNT(*) FROM memos")
            total_count = cursor.fetchone()[0]

            # Count by outcome
            cursor = conn.execute(
                "SELECT outcome, COUNT(*) FROM memos GROUP BY outcome"
            )
            outcome_counts = {row[0]: row[1] for row in cursor.fetchall()}

            # Oldest and newest timestamps
            cursor = conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM memos")
            min_ts, max_ts = cursor.fetchone()

        return {
            "total_memos": total_count,
            "outcome_counts": outcome_counts,
            "oldest_timestamp": min_ts,
            "newest_timestamp": max_ts,
            "max_memos": self.max_memos,
            "similarity_threshold": self.similarity_threshold,
            "db_path": self.db_path,
        }

    def format_memos_for_context(self, memos: List[Memo], max_memos: int = 3) -> str:
        """Format retrieved memos for injection into LLM context.

        Args:
            memos: List of memos to format
            max_memos: Maximum number of memos to include

        Returns:
            Formatted string ready for context injection
        """
        if not memos:
            return ""

        lines = [
            "=== Relevant Lessons Learned from Past Tasks ===\n",
            f"Found {len(memos[:max_memos])} relevant memo(s) that may help:\n",
        ]

        for i, memo in enumerate(memos[:max_memos], 1):
            lines.append(f"\n**Lesson {i}** (relevance: {memo.relevance_score:.2f}):")
            lines.append(f"  Task: {memo.task_description}")
            lines.append(f"  Solution: {memo.solution}")
            lines.append(f"  Outcome: {memo.outcome}")

            if memo.metadata:
                lines.append(f"  Context: {json.dumps(memo.metadata, indent=2)}")

        lines.append("\n=== End of Lessons Learned ===\n")

        return "\n".join(lines)


# ============================================================================
# Integration with AgenticStepProcessor
# ============================================================================


def inject_relevant_memos(
    memo_store: MemoStore,
    task_description: str,
    existing_context: str = "",
    top_k: int = 3,
) -> str:
    """Helper function to inject relevant memos into context.

    Designed for integration with AgenticStepProcessor._execute_step().

    Args:
        memo_store: MemoStore instance
        task_description: Description of the current task
        existing_context: Existing context to prepend memos to
        top_k: Number of memos to retrieve

    Returns:
        Context string with memos prepended
    """
    relevant_memos = memo_store.retrieve_relevant_memos(task_description, top_k=top_k)

    if relevant_memos:
        memo_context = memo_store.format_memos_for_context(
            relevant_memos, max_memos=top_k
        )
        return f"{memo_context}\n\n{existing_context}"

    return existing_context
