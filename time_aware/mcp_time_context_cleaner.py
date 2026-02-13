#!/usr/bin/env python3
"""
MCP Time Context Cleaner for Claude Code Hooks

This script:
1. Detects and removes outdated year references (2024, 2025, etc.)
2. Replaces them with current year
3. Injects current date/time context into queries
4. Adds relative timeframes for "recent" searches
"""

import sys
import json
import re
from datetime import datetime, timedelta

def clean_and_update_query(text: str, current_year: str, current_date: str, time_context: str) -> str:
    """Clean query by removing outdated years and adding time context."""
    if not text or not isinstance(text, str):
        return text
    
    cleaned = text
    
    # List of potentially outdated years (excluding current year)
    outdated_years = ['2024', '2025']
    
    # Remove/replace outdated year references
    for year in outdated_years:
        if year != current_year:
            # Replace standalone year references
            pattern = r'\b' + re.escape(year) + r'\b'
            cleaned = re.sub(pattern, current_year, cleaned)
            
            # Replace year + common keywords (e.g., "2024 tutorial" -> "2025 tutorial")
            cleaned = re.sub(
                r'\b' + re.escape(year) + r'\s+(tutorial|guide|article|news|update|release|version|best practices|implementation)\b',
                f'{current_year} \\1',
                cleaned,
                flags=re.IGNORECASE
            )
    
    # Add time context if not already present
    has_time_context = re.search(
        r'current date|as of|recent|latest|\d{4}-\d{2}-\d{2}|since \d{4}',
        cleaned,
        re.IGNORECASE
    )
    
    if not has_time_context:
        cleaned = f'{cleaned} ({time_context})'
    
    return cleaned


def main():
    """Main hook execution function."""
    # Read input from stdin
    data = json.load(sys.stdin)
    tool_input = data.get('tool_input', {})
    
    # Get current time information
    now = datetime.now()
    current_year = str(now.year)
    current_date = now.strftime('%Y-%m-%d')
    six_months_ago = now - timedelta(days=180)
    
    # Create time context message
    time_context = (
        f'Current date: {current_date}. '
        f'Focus on recent information from {six_months_ago.strftime("%B %Y")} onwards.'
    )
    
    # Parameters that might contain queries/prompts
    query_params = ['query', 'topic', 'prompt', 'question', 'search_query', 'text']
    
    modified = False
    
    # Process each potential query parameter
    for key in query_params:
        if key in tool_input and tool_input[key]:
            original = str(tool_input[key])
            cleaned = clean_and_update_query(original, current_year, current_date, time_context)
            
            if cleaned != original:
                tool_input[key] = cleaned
                modified = True
    
    # Return modified tool input (Claude Code 2026 format)
    output = {'updatedInput': tool_input}
    print(json.dumps(output))


if __name__ == '__main__':
    main()

