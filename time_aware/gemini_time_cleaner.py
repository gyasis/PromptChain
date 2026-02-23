#!/usr/bin/env python3
"""
Gemini MCP Time Context Cleaner for Claude Code Hooks

Cleans queries/prompts for Gemini MCP tools by:
1. Removing outdated year references (2024, 2025)
2. Replacing with current year
3. Injecting current date/time context
"""

import sys
import json
import re
from datetime import datetime, timedelta

def clean_text(text: str, current_year: str, current_date: str, time_context: str) -> str:
    """Clean text by removing outdated years and adding time context."""
    if not text or not isinstance(text, str):
        return text
    
    cleaned = text
    outdated_years = ['2024', '2025'] if current_year not in ['2024', '2025'] else []
    
    # Remove/replace outdated year references
    for year in outdated_years:
        if year != current_year:
            # Replace standalone year references
            cleaned = re.sub(r'\b' + re.escape(year) + r'\b', current_year, cleaned)
            
            # Replace year + keywords (e.g., "2024 tutorial" -> "2026 tutorial")
            cleaned = re.sub(
                r'\b' + re.escape(year) + r'\s+(tutorial|guide|article|news|update|release|version|best practices|implementation|research|study|paper|development|advancement|trend|breakthrough|framework|library|tool)\b',
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
    """Main hook execution."""
    # Read input from stdin
    data = json.load(sys.stdin)
    tool_input = data.get('tool_input', {})
    
    # Get current time
    now = datetime.now()
    current_year = str(now.year)
    current_date = now.strftime('%Y-%m-%d')
    six_months_ago = now - timedelta(days=180)
    
    # Create time context
    time_context = (
        f'Current date: {current_date}. '
        f'Focus on recent information from {six_months_ago.strftime("%B %Y")} onwards.'
    )
    
    # Parameters to clean (varies by Gemini tool)
    params_to_clean = ['prompt', 'query', 'topic', 'question', 'context']
    
    modified = False
    
    # Clean each parameter
    for param in params_to_clean:
        if param in tool_input and tool_input[param]:
            original = str(tool_input[param])
            cleaned = clean_text(original, current_year, current_date, time_context)
            
            if cleaned != original:
                tool_input[param] = cleaned
                modified = True
    
    # Return modified input (Claude Code 2026 format)
    output = {'updatedInput': tool_input}
    print(json.dumps(output))


if __name__ == '__main__':
    main()

