"""
Research Agent Tools Module
Contains tools and utilities for research agent functionality
"""

from .web_search import WebSearchTool, web_search_tool, web_search, ENV_SETUP_INSTRUCTIONS

# Web search availability
WEB_SEARCH_AVAILABLE = web_search_tool.is_available()

# Try to import advanced page scraper
try:
    from .singlepage_advanced import AdvancedPageScraper
    ADVANCED_SCRAPER_AVAILABLE = True
except ImportError:
    ADVANCED_SCRAPER_AVAILABLE = False

__all__ = [
    'WebSearchTool',
    'web_search_tool', 
    'web_search',
    'ENV_SETUP_INSTRUCTIONS',
    'WEB_SEARCH_AVAILABLE',
    'ADVANCED_SCRAPER_AVAILABLE'
]

if ADVANCED_SCRAPER_AVAILABLE:
    __all__.append('AdvancedPageScraper')