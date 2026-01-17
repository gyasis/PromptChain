#!/usr/bin/env python3
"""
Simple environment test to check if variables are loaded correctly
"""

import os
import sys
sys.path.insert(0, 'src')

# Force load environment variables
from dotenv import load_dotenv
load_dotenv(override=True)

print("🔍 Environment Variable Check")
print("=" * 40)

print(f"OPENAI_API_KEY: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Missing'}")
print(f"SERPER_API_KEY: {'✅ Set' if os.getenv('SERPER_API_KEY') else '❌ Missing'}")

if os.getenv('SERPER_API_KEY'):
    key = os.getenv('SERPER_API_KEY')
    print(f"SERPER_API_KEY value: {key[:10]}...{key[-4:]}")
else:
    print("SERPER_API_KEY not found in environment")

# Test web search tool initialization
try:
    from research_agent.tools.web_search import web_search_tool
    print(f"Web search available: {web_search_tool.is_available()}")
    print(f"Serper key from tool: {'✅ Set' if web_search_tool.serper_api_key else '❌ Missing'}")
except Exception as e:
    print(f"Error importing web search tool: {e}")