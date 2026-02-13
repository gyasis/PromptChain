#!/usr/bin/env python3
"""
Smart Time-Aware Gemini MCP Cleaner (DYNAMIC - works in any year)

Intelligent time handling that adapts as time moves forward:
1. Historical queries (>2 years old) - Protected, user wants specific historical info
2. Outdated recent years (1-2 years old) - Replaced with current year
3. No year mentioned - Injects time context (good for recency/"project aging context")
4. User timeframes - Respected ("last 3 months", "recent", etc.)

Based on DeepLake RAG recency_weight concept - emphasizes recent info when no time specified.
"""

import sys
import json
import re
from datetime import datetime, timedelta

def get_year_categories(current_year: int):
    """
    Dynamically calculate year categories based on current year.

    Returns:
        tuple: (historical_years, outdated_recent_years, current_year)

    Examples:
        2026 → historical: ≤2023, outdated: [2024, 2025], current: 2026
        2027 → historical: ≤2024, outdated: [2025, 2026], current: 2027
        2030 → historical: ≤2027, outdated: [2028, 2029], current: 2030
    """
    # Historical: More than 2 years old
    historical_cutoff = current_year - 2  # 2026: cutoff=2024, so ≤2023 is historical

    # Outdated recent: Last 1-2 years (but not current)
    outdated_recent_years = [
        current_year - 1,  # 2026: 2025
        current_year - 2,  # 2026: 2024
    ]

    return historical_cutoff, outdated_recent_years, current_year


def is_historical_query(text: str, historical_cutoff: int) -> bool:
    """
    Detect if query is asking about specific historical years.

    Args:
        text: Query text
        historical_cutoff: Year threshold (e.g., 2024 in 2026)

    Returns:
        bool: True if query mentions years ≤ cutoff

    Examples (in 2026, cutoff=2024):
        "What happened in 2023?" → True (historical)
        "2022 Python features" → True (historical)
        "Find 2024 tutorial" → False (not historical, outdated recent)
        "Find tutorial" → False (no year mentioned)
    """
    # Look for any year ≤ historical_cutoff
    year_pattern = r'\b(\d{4})\b'
    years_found = re.findall(year_pattern, text)

    for year_str in years_found:
        year = int(year_str)
        if 2000 <= year <= historical_cutoff:  # Valid historical year
            return True

    return False


def has_user_timeframe(text: str) -> bool:
    """
    Detect if user already specified a timeframe.
    If yes, don't add time context (respect user intent).

    Examples:
        "last 3 months" → True
        "past week" → True
        "recent" → True
        "since 2023" → True
    """
    timeframe_patterns = [
        r'last\s+\d+\s+(month|week|day|year)s?',
        r'past\s+\d+\s+(month|week|day|year)s?',
        r'recent',
        r'latest',
        r'current',
        r'since\s+\d{4}',
        r'from\s+\d{4}',
        r'\d{4}-\d{2}-\d{2}',  # Specific dates
        r'as of',
    ]

    for pattern in timeframe_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def clean_text(text: str, current_year: int, current_date: str, time_context: str) -> str:
    """
    Smart text cleaning with dynamic time-aware logic.

    Rules (adapts as time moves forward):
    1. If historical query (>2 years old) → Don't inject time context
    2. Replace outdated recent years (1-2 years old) → Current year
    3. If NO year AND NO user timeframe → Inject time context (recency)
    4. If user specified timeframe → Don't inject (respect intent)
    """
    if not text or not isinstance(text, str):
        return text

    cleaned = text

    # Get dynamic year categories
    historical_cutoff, outdated_recent_years, _ = get_year_categories(current_year)

    # Rule 1: Check if this is a historical query
    is_historical = is_historical_query(text, historical_cutoff)

    # Rule 2: Replace outdated recent years (1-2 years old)
    for year in outdated_recent_years:
        year_str = str(year)
        current_year_str = str(current_year)

        # Replace standalone year references
        cleaned = re.sub(r'\b' + re.escape(year_str) + r'\b', current_year_str, cleaned)

        # Replace year + keywords (e.g., "2024 tutorial" -> "2026 tutorial")
        cleaned = re.sub(
            r'\b' + re.escape(year_str) + r'\s+(tutorial|guide|article|news|update|release|version|best practices|implementation|research|study|paper|development|advancement|trend|breakthrough|framework|library|tool)\b',
            f'{current_year_str} \\1',
            cleaned,
            flags=re.IGNORECASE
        )

    # Rule 3 & 4: Smart time context injection
    # Only inject if:
    # - NOT a historical query AND
    # - User hasn't specified timeframe
    if not is_historical and not has_user_timeframe(text):
        cleaned = f'{cleaned} ({time_context})'

    return cleaned


def main():
    """Main hook execution (Claude Code 2026 format)."""
    # Read input from stdin
    data = json.load(sys.stdin)
    tool_input = data.get('tool_input', {})

    # Get current time
    now = datetime.now()
    current_year = now.year
    current_date = now.strftime('%Y-%m-%d')
    six_months_ago = now - timedelta(days=180)

    # Create time context (similar to DeepLake RAG recency_weight)
    time_context = (
        f'Current date: {current_date}. '
        f'Focus on recent information from {six_months_ago.strftime("%B %Y")} onwards.'
    )

    # Parameters to clean (varies by Gemini tool)
    params_to_clean = ['prompt', 'query', 'topic', 'question', 'context']

    # Clean each parameter
    for param in params_to_clean:
        if param in tool_input and tool_input[param]:
            original = str(tool_input[param])
            cleaned = clean_text(original, current_year, current_date, time_context)

            if cleaned != original:
                tool_input[param] = cleaned

    # Return modified input (Claude Code 2026 format)
    output = {'updatedInput': tool_input}
    print(json.dumps(output))


if __name__ == '__main__':
    main()
