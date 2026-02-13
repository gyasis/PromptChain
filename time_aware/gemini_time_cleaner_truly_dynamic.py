#!/usr/bin/env python3
"""
Truly Dynamic Time-Aware Cleaner (NO HARDCODED YEARS)

Simple principle: Never replace years, just inject current context.
Let the search engine figure out what's recent.

Works in ANY year, with ANY knowledge cutoff, forever.

Claude Code Hook Format:
- Input: {"tool_input": {...}} via stdin
- Output: {"updatedInput": {...}} via stdout
"""

import sys
import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

# Parameters to process (moved to module level as constant)
PARAMS_TO_PROCESS = ('prompt', 'query', 'topic', 'question', 'context')

# Compile regex once at module level for performance
# Uses (19|20)\d{2} to match years 1900-2099, avoiding false positives like "port 8080"
TIMEFRAME_RE = re.compile(
    r'\b(19|20)\d{2}\b|'                          # Years 1900-2099
    r'(last|past)\s+\d+\s+(month|week|day|year)s?|'  # "last 3 months"
    r'\b(recent|latest|current)\b|'               # Time keywords
    r'since\s+(19|20)\d{2}|'                      # "since 2023"
    r'from\s+(19|20)\d{2}|'                       # "from 2022"
    r'(19|20)\d{2}-\d{2}-\d{2}|'                  # Dates (2023-01-15)
    r'\bas\s+of\b',                               # "as of"
    re.IGNORECASE
)


def has_explicit_timeframe(text: str) -> bool:
    """
    Detect if user/LLM already specified timing.

    Uses compiled regex for performance. Only matches years 1900-2099
    to avoid false positives on port numbers, error codes, etc.

    Examples:
        "last 3 months" -> True
        "recent tutorials" -> True
        "since 2023" -> True
        "2024 guide" -> True (year mentioned)
        "Python async" -> False (no timing)
        "Port 8080" -> False (not a year)
        "Error 4040" -> False (not a year)
    """
    if not text or not isinstance(text, str):
        return False
    return bool(TIMEFRAME_RE.search(text))


def inject_time_context(text: str, time_context: str) -> str:
    """
    Inject time context ONLY if no timing already specified.

    NO year replacement. NO hardcoding. Just context injection.
    """
    if not text or not isinstance(text, str):
        return text

    # Only inject if timing not already specified
    if not has_explicit_timeframe(text):
        return f'{text} ({time_context})'

    return text


def main() -> None:
    """Main hook execution (Claude Code format)."""
    # Read and parse input with error handling
    try:
        raw_data = sys.stdin.read()
        if not raw_data:
            # Empty input - return empty output
            print(json.dumps({'updatedInput': {}}))
            return
        data: Dict[str, Any] = json.loads(raw_data)
    except json.JSONDecodeError:
        # Malformed JSON - exit gracefully without corrupting output
        sys.exit(1)

    # Get tool_input with type validation
    tool_input = data.get('tool_input')

    # If tool_input is not a dict, pass through unchanged
    if not isinstance(tool_input, dict):
        print(json.dumps({'updatedInput': tool_input}))
        return

    # Get current time
    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')
    six_months_ago = now - timedelta(days=180)

    # Simple time context - no year replacement needed
    time_context = (
        f'Current date: {current_date}. '
        f'Seeking most recent information from {six_months_ago.strftime("%B %Y")} onwards.'
    )

    # Process each parameter
    for param in PARAMS_TO_PROCESS:
        if param in tool_input and tool_input[param]:
            original = str(tool_input[param])
            processed = inject_time_context(original, time_context)

            if processed != original:
                tool_input[param] = processed

    # Return modified input
    output = {'updatedInput': tool_input}
    print(json.dumps(output))


if __name__ == '__main__':
    main()
