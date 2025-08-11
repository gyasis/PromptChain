"""
Search Strategist Agent

Determines optimal search strategies and paper selection based on current findings and gaps.
"""

import json
import logging
from typing import Dict, List, Any, Optional

from promptchain import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

logger = logging.getLogger(__name__)


class SearchStrategistAgent:
    """
    Agent that optimizes search strategies based on queries and current research state
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize PromptChain with agentic processing
        self.chain = PromptChain(
            models=[config.get('model', 'openai/gpt-4o-mini')],
            instructions=[
                "You are a literature search strategist. Optimize search strategy for queries: {queries}",
                AgenticStepProcessor(
                    objective=config.get('processor', {}).get('objective', 
                               "Optimize search strategy based on queries and current findings"),
                    max_internal_steps=config.get('processor', {}).get('max_internal_steps', 3)
                ),
                self._format_strategy_output_instruction()
            ],
            verbose=True
        )
        
        # Register tools
        self._register_tools()
        
        logger.info("SearchStrategistAgent initialized")
    
    def _format_strategy_output_instruction(self) -> str:
        """Instruction for formatting strategy output"""
        return """
        Format your output as a JSON object with the following structure:
        {
            "search_strategy": {
                "primary_keywords": ["keyword1", "keyword2", "keyword3"],
                "secondary_keywords": ["related1", "related2"],
                "boolean_queries": [
                    {
                        "database": "sci_hub",
                        "query": "(machine learning OR deep learning) AND drug discovery",
                        "priority": 1.0
                    }
                ],
                "filters": {
                    "publication_years": [2020, 2024],
                    "paper_types": ["research", "review"],
                    "languages": ["en"]
                }
            },
            "database_allocation": {
                "sci_hub": {
                    "priority": 1.0,
                    "max_papers": 40,
                    "search_terms": ["primary keywords focused search"],
                    "rationale": "Best for getting full papers"
                },
                "arxiv": {
                    "priority": 0.8,
                    "max_papers": 30,
                    "search_terms": ["recent preprints and methods"],
                    "rationale": "Latest research before publication"
                },
                "pubmed": {
                    "priority": 0.7,
                    "max_papers": 20,
                    "search_terms": ["medical/biological applications"],
                    "rationale": "Medical domain expertise"
                }
            },
            "search_optimization": {
                "iteration_focus": "What to focus on for this iteration",
                "gap_targeting": ["Specific gaps to address"],
                "expansion_areas": ["Areas to expand search"],
                "exclusion_criteria": ["What to avoid or deprioritize"]
            }
        }
        """
    
    def _register_tools(self):
        """Register tools for search optimization"""
        
        def search_optimization(context: str) -> str:
            """Optimize search terms and strategies based on context"""
            optimization_techniques = {
                'keyword_expansion': 'Use synonyms, related terms, and domain-specific vocabulary',
                'boolean_logic': 'Combine terms with AND, OR, NOT for precise queries',
                'field_specific': 'Target title, abstract, or full-text fields',
                'temporal_filtering': 'Focus on recent publications or historical analysis',
                'source_diversification': 'Balance across different publication types and venues'
            }
            
            result = "Search Optimization Techniques:\n\n"
            for technique, description in optimization_techniques.items():
                result += f"{technique.upper()}: {description}\n"
            
            return result
        
        def keyword_generation(queries: str) -> str:
            """Generate effective keywords from research queries"""
            keyword_strategies = [
                "Extract core concepts and technical terms",
                "Include synonyms and alternative terminology", 
                "Add domain-specific jargon and acronyms",
                "Consider broader and narrower terms",
                "Include methodology and technique names",
                "Add application domain keywords",
                "Include evaluation metric terms"
            ]
            
            return "Keyword Generation Strategies:\n" + "\n".join(f"- {s}" for s in keyword_strategies)
        
        def database_selection(requirements: str) -> str:
            """Provide guidance on database selection and allocation"""
            database_profiles = {
                'sci_hub': {
                    'strengths': 'Full paper access, broad coverage, recent publications',
                    'best_for': 'Getting complete papers for detailed analysis',
                    'limitations': 'Variable availability, potential access issues'
                },
                'arxiv': {
                    'strengths': 'Latest preprints, computer science/physics focus, free access',
                    'best_for': 'Recent methods, preliminary results, emerging trends',
                    'limitations': 'Not peer-reviewed, limited domain coverage'
                },
                'pubmed': {
                    'strengths': 'Medical/biological focus, high quality curation, MeSH terms',
                    'best_for': 'Medical applications, clinical studies, biological research', 
                    'limitations': 'Domain-specific, mostly abstracts without full text'
                }
            }
            
            result = "Database Selection Guide:\n\n"
            for db, profile in database_profiles.items():
                result += f"{db.upper()}:\n"
                for key, value in profile.items():
                    result += f"  {key}: {value}\n"
                result += "\n"
            
            return result
        
        # Register tools with PromptChain
        self.chain.register_tool_function(search_optimization)
        self.chain.register_tool_function(keyword_generation)
        self.chain.register_tool_function(database_selection)
        
        # Add tool schemas
        self.chain.add_tools([
            {
                "type": "function",
                "function": {
                    "name": "search_optimization",
                    "description": "Get search optimization techniques and strategies",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "context": {"type": "string", "description": "Search context and requirements"}
                        },
                        "required": ["context"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "keyword_generation",
                    "description": "Generate effective keywords from research queries",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queries": {"type": "string", "description": "Research queries to analyze"}
                        },
                        "required": ["queries"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "database_selection",
                    "description": "Get guidance on database selection and resource allocation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "requirements": {"type": "string", "description": "Search requirements and constraints"}
                        },
                        "required": ["requirements"]
                    }
                }
            }
        ])
    
    async def generate_strategy(
        self,
        queries: List[str],
        existing_papers: List[str] = None,
        iteration: int = 0,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate optimal search strategy for given queries
        
        Args:
            queries: List of research queries to search for
            existing_papers: List of paper IDs already found
            iteration: Current research iteration number
            context: Additional context including findings, gaps, etc.
            
        Returns:
            JSON string containing search strategy
        """
        logger.info(f"Generating search strategy for {len(queries)} queries, iteration {iteration}")
        
        try:
            # Prepare input context
            input_data = {
                'queries': queries,
                'iteration': iteration,
                'existing_papers_count': len(existing_papers) if existing_papers else 0,
                'context': context or {}
            }
            
            # Add iteration-specific strategy guidance
            if iteration > 0:
                strategy_focus = self._get_iteration_strategy_focus(iteration, context)
                input_data['iteration_strategy'] = strategy_focus
            
            # Add gap-based targeting if context available
            if context and 'gaps' in context:
                input_data['gap_targeting'] = context['gaps']
            
            # Process through PromptChain
            result = await self.chain.process_prompt_async(
                json.dumps(input_data, indent=2)
            )
            
            # Validate and enhance strategy
            validated_strategy = self._validate_strategy_response(result)
            
            logger.info("Search strategy generated successfully")
            
            return validated_strategy
            
        except Exception as e:
            logger.error(f"Search strategy generation failed: {e}")
            raise
    
    def _get_iteration_strategy_focus(self, iteration: int, context: Dict[str, Any]) -> str:
        """Get strategy focus based on iteration number and context"""
        if iteration == 1:
            return "Focus on foundational papers, key methodologies, and core concepts"
        elif iteration == 2:
            return "Target gaps identified in first iteration, explore alternative approaches"
        elif iteration == 3:
            return "Deep dive into specific areas, look for recent advances and applications"
        elif iteration >= 4:
            return "Fill remaining gaps, focus on edge cases and specialized applications"
        else:
            return "Comprehensive coverage across all aspects of the research topic"
    
    def _validate_strategy_response(self, response: str) -> str:
        """Validate and enhance strategy response"""
        try:
            parsed = json.loads(response)
            
            # Ensure required structure
            if 'search_strategy' not in parsed:
                parsed['search_strategy'] = {}
            
            if 'database_allocation' not in parsed:
                parsed['database_allocation'] = self._get_default_database_allocation()
            
            if 'search_optimization' not in parsed:
                parsed['search_optimization'] = {}
            
            # Validate search strategy section
            search_strategy = parsed['search_strategy']
            if 'primary_keywords' not in search_strategy:
                search_strategy['primary_keywords'] = []
            if 'secondary_keywords' not in search_strategy:
                search_strategy['secondary_keywords'] = []
            if 'boolean_queries' not in search_strategy:
                search_strategy['boolean_queries'] = []
            
            # Validate database allocation
            db_allocation = parsed['database_allocation']
            required_databases = ['sci_hub', 'arxiv', 'pubmed']
            
            for db in required_databases:
                if db not in db_allocation:
                    db_allocation[db] = {
                        'priority': 0.5,
                        'max_papers': 20,
                        'search_terms': [],
                        'rationale': f'Default allocation for {db}'
                    }
            
            # Ensure paper limits don't exceed total budget
            total_papers = sum(
                db.get('max_papers', 0) 
                for db in db_allocation.values()
                if isinstance(db, dict)
            )
            
            if total_papers > 100:  # Max paper budget
                self._rebalance_paper_allocation(db_allocation, 100)
            
            return json.dumps(parsed, indent=2)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in strategy response: {e}")
            return self._fallback_strategy_generation(response)
        except Exception as e:
            logger.error(f"Strategy validation failed: {e}")
            return self._fallback_strategy_generation(response)
    
    def _get_default_database_allocation(self) -> Dict[str, Any]:
        """Get default database allocation"""
        return {
            'sci_hub': {
                'priority': 1.0,
                'max_papers': 40,
                'search_terms': ['primary research papers'],
                'rationale': 'Primary source for full paper access'
            },
            'arxiv': {
                'priority': 0.8,
                'max_papers': 30,
                'search_terms': ['recent preprints'],
                'rationale': 'Latest research and methodologies'
            },
            'pubmed': {
                'priority': 0.6,
                'max_papers': 20,
                'search_terms': ['medical applications'],
                'rationale': 'Domain-specific medical/biological focus'
            }
        }
    
    def _rebalance_paper_allocation(self, allocation: Dict[str, Any], max_total: int):
        """Rebalance paper allocation to fit within budget"""
        databases = ['sci_hub', 'arxiv', 'pubmed']
        current_total = sum(
            allocation[db].get('max_papers', 0)
            for db in databases
            if db in allocation
        )
        
        if current_total > max_total:
            # Proportional reduction
            reduction_factor = max_total / current_total
            
            for db in databases:
                if db in allocation and isinstance(allocation[db], dict):
                    current = allocation[db].get('max_papers', 0)
                    allocation[db]['max_papers'] = max(1, int(current * reduction_factor))
    
    def _fallback_strategy_generation(self, original_response: str) -> str:
        """Generate fallback strategy if parsing fails"""
        logger.warning("Using fallback strategy generation")
        
        fallback_strategy = {
            'search_strategy': {
                'primary_keywords': ['research', 'analysis', 'methodology'],
                'secondary_keywords': ['application', 'evaluation', 'performance'],
                'boolean_queries': [
                    {
                        'database': 'sci_hub',
                        'query': 'research AND methodology',
                        'priority': 1.0
                    }
                ],
                'filters': {
                    'publication_years': [2020, 2024],
                    'paper_types': ['research'],
                    'languages': ['en']
                }
            },
            'database_allocation': self._get_default_database_allocation(),
            'search_optimization': {
                'iteration_focus': 'Broad search for foundational papers',
                'gap_targeting': [],
                'expansion_areas': ['methodology', 'applications'],
                'exclusion_criteria': ['non-English papers']
            },
            'fallback_note': 'Generated using fallback strategy due to parsing error'
        }
        
        return json.dumps(fallback_strategy, indent=2)
    
    def extract_search_terms(self, strategy: str) -> Dict[str, List[str]]:
        """Extract search terms for each database from strategy"""
        try:
            parsed = json.loads(strategy)
            
            search_terms = {}
            db_allocation = parsed.get('database_allocation', {})
            
            for db, config in db_allocation.items():
                if isinstance(config, dict):
                    terms = config.get('search_terms', [])
                    
                    # Also include primary keywords
                    primary_keywords = parsed.get('search_strategy', {}).get('primary_keywords', [])
                    terms.extend(primary_keywords[:3])  # Top 3 keywords
                    
                    search_terms[db] = list(set(terms))  # Remove duplicates
            
            return search_terms
            
        except Exception as e:
            logger.error(f"Failed to extract search terms: {e}")
            return {
                'sci_hub': ['research'],
                'arxiv': ['research'],
                'pubmed': ['research']
            }
    
    def get_database_priorities(self, strategy: str) -> Dict[str, float]:
        """Extract database priorities from strategy"""
        try:
            parsed = json.loads(strategy)
            
            priorities = {}
            db_allocation = parsed.get('database_allocation', {})
            
            for db, config in db_allocation.items():
                if isinstance(config, dict):
                    priorities[db] = config.get('priority', 0.5)
            
            return priorities
            
        except Exception as e:
            logger.error(f"Failed to extract priorities: {e}")
            return {'sci_hub': 1.0, 'arxiv': 0.8, 'pubmed': 0.6}
    
    def get_paper_limits(self, strategy: str) -> Dict[str, int]:
        """Extract paper limits for each database"""
        try:
            parsed = json.loads(strategy)
            
            limits = {}
            db_allocation = parsed.get('database_allocation', {})
            
            for db, config in db_allocation.items():
                if isinstance(config, dict):
                    limits[db] = config.get('max_papers', 20)
            
            return limits
            
        except Exception as e:
            logger.error(f"Failed to extract paper limits: {e}")
            return {'sci_hub': 40, 'arxiv': 30, 'pubmed': 20}
    
    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration"""
        return self.config.copy()
    
    def update_config(self, config: Dict[str, Any]):
        """Update agent configuration"""
        self.config.update(config)
        logger.info("SearchStrategistAgent configuration updated")