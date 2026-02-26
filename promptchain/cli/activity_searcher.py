"""
Search and query interface for agent activity logs.

This module provides the ActivitySearcher class that enables grep-style text search
and SQL queries across agent activity logs, allowing users and agents to find
specific interactions without loading full conversation history.
"""

import json
import logging
import sqlite3
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ActivitySearcher:
    """Search and query interface for agent activity logs.

    Provides grep-style text search and SQL queries for finding specific
    agent interactions, reasoning chains, tool calls, and errors without
    loading full conversation history into memory.

    Attributes:
        session_name: Current session identifier
        log_dir: Directory containing JSONL activity logs
        db_path: Path to SQLite database
    """

    def __init__(self, session_name: str, log_dir: Path, db_path: Path):
        """Initialize activity searcher.

        Args:
            session_name: Current session identifier
            log_dir: Directory containing JSONL activity logs
            db_path: Path to SQLite database
        """
        self.session_name = session_name
        self.log_dir = Path(log_dir)
        self.db_path = Path(db_path)
        self.activity_log_path = self.log_dir / "activities.jsonl"

        logger.info(f"ActivitySearcher initialized for session '{session_name}'")

    def grep_logs(
        self,
        pattern: str,
        agent_name: Optional[str] = None,
        activity_type: Optional[str] = None,
        since: Optional[datetime] = None,
        max_results: int = 100,
        context_lines: int = 0,
    ) -> List[Dict[str, Any]]:
        """Grep through activity logs with filters.

        Uses ripgrep (rg) if available, falls back to Python search.

        Args:
            pattern: Search pattern (regex supported)
            agent_name: Optional agent name filter
            activity_type: Optional activity type filter
            since: Optional datetime filter (only activities after this time)
            max_results: Maximum number of results to return
            context_lines: Number of context lines to include (for grep -C)

        Returns:
            List of matching activity dictionaries
        """
        results = []

        try:
            # Check if ripgrep is available
            use_ripgrep = self._check_ripgrep_available()

            if use_ripgrep:
                results = self._grep_with_ripgrep(pattern, context_lines, max_results)
            else:
                results = self._grep_with_python(pattern, max_results)

            # Apply filters
            results = self._apply_filters(
                results, agent_name=agent_name, activity_type=activity_type, since=since
            )

            logger.debug(
                f"Grep search for pattern '{pattern}' returned {len(results)} results"
            )

        except Exception as e:
            logger.error(f"Grep search failed: {e}")

        return results[:max_results]

    def _check_ripgrep_available(self) -> bool:
        """Check if ripgrep (rg) is available on system."""
        try:
            result = subprocess.run(["rg", "--version"], capture_output=True, timeout=1)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _grep_with_ripgrep(
        self, pattern: str, context_lines: int, max_results: int
    ) -> List[Dict[str, Any]]:
        """Use ripgrep for fast text search."""
        if not self.activity_log_path.exists():
            return []

        try:
            cmd = [
                "rg",
                pattern,
                str(self.activity_log_path),
                "--json",  # JSON output for parsing
            ]

            if context_lines > 0:
                cmd.extend(["-C", str(context_lines)])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode not in [0, 1]:  # 1 means no matches
                logger.warning(f"Ripgrep returned code {result.returncode}")
                return []

            # Parse ripgrep JSON output
            matches = []
            for line in result.stdout.split("\n"):
                if not line.strip():
                    continue
                try:
                    rg_entry = json.loads(line)
                    if rg_entry.get("type") == "match":
                        # Extract the matched line (full JSON activity)
                        matched_line = rg_entry["data"]["lines"]["text"]
                        activity = json.loads(matched_line)
                        matches.append(activity)
                except json.JSONDecodeError:
                    continue

                if len(matches) >= max_results:
                    break

            return matches

        except subprocess.TimeoutExpired:
            logger.warning("Ripgrep search timed out, falling back to Python search")
            return self._grep_with_python(pattern, max_results)
        except Exception as e:
            logger.error(f"Ripgrep search failed: {e}")
            return []

    def _grep_with_python(self, pattern: str, max_results: int) -> List[Dict[str, Any]]:
        """Fallback Python-based text search."""
        if not self.activity_log_path.exists():
            return []

        matches = []

        try:
            import re

            regex = re.compile(pattern, re.IGNORECASE)

            with open(self.activity_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue

                    # Search in the raw line
                    if regex.search(line):
                        try:
                            activity = json.loads(line)
                            matches.append(activity)

                            if len(matches) >= max_results:
                                break
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.error(f"Python grep failed: {e}")

        return matches

    def _apply_filters(
        self,
        activities: List[Dict[str, Any]],
        agent_name: Optional[str] = None,
        activity_type: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Apply filters to activity list."""
        filtered = activities

        if agent_name:
            filtered = [a for a in filtered if a.get("agent_name") == agent_name]

        if activity_type:
            filtered = [a for a in filtered if a.get("activity_type") == activity_type]

        if since:
            since_iso = since.isoformat()
            filtered = [a for a in filtered if a.get("timestamp", "") >= since_iso]

        return filtered

    def sql_query(
        self, query: str, params: Optional[Tuple] = None
    ) -> List[Dict[str, Any]]:
        """Execute custom SQL query on activity database.

        Args:
            query: SQL query string
            params: Optional query parameters (for prepared statements)

        Returns:
            List of result dictionaries
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            results = [dict(row) for row in cursor.fetchall()]
            conn.close()

            logger.debug(f"SQL query returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"SQL query failed: {e}")
            return []

    def get_interaction_chain(
        self, chain_id: str, include_nested: bool = True, include_content: bool = True
    ) -> Dict[str, Any]:
        """Get complete interaction chain with nested structure.

        Args:
            chain_id: Interaction chain ID
            include_nested: Whether to include nested activities
            include_content: Whether to load full content from JSONL

        Returns:
            Dictionary with chain info and activities
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get chain metadata
            cursor.execute(
                """
                SELECT *
                FROM interaction_chains
                WHERE chain_id = ?
            """,
                (chain_id,),
            )

            chain_row = cursor.fetchone()
            if not chain_row:
                conn.close()
                return {}

            chain_info = dict(chain_row)

            # Get all activities in chain
            cursor.execute(
                """
                SELECT *
                FROM agent_activities
                WHERE interaction_chain_id = ?
                ORDER BY timestamp ASC
            """,
                (chain_id,),
            )

            activities = [dict(row) for row in cursor.fetchall()]
            conn.close()

            # Enrich with full content if requested
            if include_content and activities:
                activities = self._enrich_with_full_content(activities)

            # Build nested structure if requested
            if include_nested:
                activities = self._build_nested_structure(activities)

            chain_info["activities"] = activities

            return chain_info

        except Exception as e:
            logger.error(f"Failed to get interaction chain: {e}")
            return {}

    def _enrich_with_full_content(
        self, activities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Load full content from JSONL for activities."""
        activity_map = {}

        try:
            with open(self.activity_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    activity_map[entry["activity_id"]] = entry
        except Exception as e:
            logger.error(f"Failed to load JSONL for enrichment: {e}")
            return activities

        enriched = []
        for activity in activities:
            activity_id = activity["activity_id"]
            if activity_id in activity_map:
                full_entry = activity_map[activity_id]
                activity["content"] = full_entry.get("content", {})
                activity["metadata"] = full_entry.get("metadata", {})
                activity["searchable_text"] = full_entry.get("searchable_text", "")
                activity["tags"] = full_entry.get("tags", [])
            enriched.append(activity)

        return enriched

    def _build_nested_structure(
        self, activities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build nested parent-child activity structure."""
        # Create lookup map
        activity_map = {a["activity_id"]: a for a in activities}

        # Add children list to each activity
        for activity in activities:
            activity["children"] = []

        # Build parent-child relationships
        root_activities = []
        for activity in activities:
            parent_id = activity.get("parent_activity_id")
            if parent_id and parent_id in activity_map:
                activity_map[parent_id]["children"].append(activity)
            else:
                root_activities.append(activity)

        return root_activities

    def find_reasoning_chains(
        self, agent_name: Optional[str] = None, min_depth: int = 2, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find complex reasoning chains (multi-hop) by agent.

        Args:
            agent_name: Optional agent name filter
            min_depth: Minimum depth level to consider as "complex reasoning"
            limit: Maximum number of chains to return

        Returns:
            List of interaction chains with reasoning activities
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = """
                SELECT ic.*
                FROM interaction_chains ic
                WHERE ic.session_name = ?
                  AND ic.max_depth_level >= ?
            """
            params = [self.session_name, min_depth]

            if agent_name:
                query += """
                  AND EXISTS (
                    SELECT 1
                    FROM agent_activities aa
                    WHERE aa.interaction_chain_id = ic.chain_id
                      AND aa.agent_name = ?
                  )
                """
                params.append(agent_name)

            query += """
                ORDER BY ic.started_at DESC
                LIMIT ?
            """
            params.append(limit)

            cursor.execute(query, tuple(params))

            chains = [dict(row) for row in cursor.fetchall()]
            conn.close()

            return chains

        except Exception as e:
            logger.error(f"Failed to find reasoning chains: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get activity statistics for current session.

        Returns:
            Dictionary with statistics:
                - total_activities: Total number of activities
                - total_chains: Total interaction chains
                - active_chains: Number of active chains
                - activities_by_type: Count by activity type
                - activities_by_agent: Count by agent
                - avg_chain_depth: Average chain depth
                - total_errors: Number of error activities
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Total activities
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM agent_activities
                WHERE session_name = ?
            """,
                (self.session_name,),
            )
            total_activities = cursor.fetchone()[0]

            # Total chains
            cursor.execute(
                """
                SELECT COUNT(*), SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END)
                FROM interaction_chains
                WHERE session_name = ?
            """,
                (self.session_name,),
            )
            total_chains, active_chains = cursor.fetchone()

            # Activities by type
            cursor.execute(
                """
                SELECT activity_type, COUNT(*)
                FROM agent_activities
                WHERE session_name = ?
                GROUP BY activity_type
            """,
                (self.session_name,),
            )
            activities_by_type = {row[0]: row[1] for row in cursor.fetchall()}

            # Activities by agent
            cursor.execute(
                """
                SELECT agent_name, COUNT(*)
                FROM agent_activities
                WHERE session_name = ?
                  AND agent_name IS NOT NULL
                GROUP BY agent_name
            """,
                (self.session_name,),
            )
            activities_by_agent = {row[0]: row[1] for row in cursor.fetchall()}

            # Average chain depth
            cursor.execute(
                """
                SELECT AVG(max_depth_level)
                FROM interaction_chains
                WHERE session_name = ?
            """,
                (self.session_name,),
            )
            avg_chain_depth = cursor.fetchone()[0] or 0

            # Total errors
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM agent_activities
                WHERE session_name = ?
                  AND activity_type = 'error'
            """,
                (self.session_name,),
            )
            total_errors = cursor.fetchone()[0]

            conn.close()

            return {
                "total_activities": total_activities,
                "total_chains": total_chains,
                "active_chains": active_chains,
                "activities_by_type": activities_by_type,
                "activities_by_agent": activities_by_agent,
                "avg_chain_depth": round(avg_chain_depth, 2),
                "total_errors": total_errors,
            }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    def search_by_timerange(
        self,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        agent_name: Optional[str] = None,
        activity_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search activities by time range.

        Args:
            start_time: Start of time range
            end_time: End of time range (default: now)
            agent_name: Optional agent name filter
            activity_type: Optional activity type filter
            limit: Maximum number of results

        Returns:
            List of matching activities (most recent first)
        """
        if end_time is None:
            end_time = datetime.now()

        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = """
                SELECT *
                FROM agent_activities
                WHERE session_name = ?
                  AND timestamp >= ?
                  AND timestamp <= ?
            """
            params = [self.session_name, start_time.isoformat(), end_time.isoformat()]

            if agent_name:
                query += " AND agent_name = ?"
                params.append(agent_name)

            if activity_type:
                query += " AND activity_type = ?"
                params.append(activity_type)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)  # type: ignore[arg-type]

            cursor.execute(query, tuple(params))

            activities = [dict(row) for row in cursor.fetchall()]
            conn.close()

            return activities

        except Exception as e:
            logger.error(f"Time range search failed: {e}")
            return []
