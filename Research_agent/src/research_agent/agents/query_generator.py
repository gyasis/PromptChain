"""
Query Generation Agent

Transforms research topics into comprehensive sets of specific, targeted questions.
"""

import json
import logging
from typing import Dict, List, Any, Optional

from promptchain import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

logger = logging.getLogger(__name__)


class QueryGenerationAgent:
    """
    Agent that generates comprehensive research questions from research topics
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize PromptChain with agentic processing
        self.chain = PromptChain(
            models=[config.get('model', 'openai/gpt-4o')],
            instructions=[
                "You are a research query generation specialist. Analyze the research topic: {topic}",
                AgenticStepProcessor(
                    objective=config.get('processor', {}).get('objective', 
                               "Generate comprehensive research questions for multi-query processing"),
                    max_internal_steps=config.get('processor', {}).get('max_internal_steps', 5)
                ),
                self._format_query_output_instruction()
            ],
            verbose=True
        )
        
        # Register tools
        self._register_tools()
        
        logger.info("QueryGenerationAgent initialized")
    
    def _format_query_output_instruction(self) -> str:
        """Instruction for formatting query output"""
        return """
        Format your output as a JSON object with the following structure:
        {
            "primary_queries": [
                {"text": "What are the main techniques used in...", "priority": 1.0, "category": "techniques"},
                {"text": "What are the limitations of...", "priority": 0.9, "category": "limitations"}
            ],
            "secondary_queries": [
                {"text": "What datasets are commonly used for...", "priority": 0.7, "category": "datasets"},
                {"text": "Which companies/organizations are leading in...", "priority": 0.6, "category": "industry"}
            ],
            "exploratory_queries": [
                {"text": "What are emerging trends in...", "priority": 0.5, "category": "trends"},
                {"text": "What ethical considerations exist for...", "priority": 0.4, "category": "ethics"}
            ]
        }
        
        Guidelines:
        1. Generate 8-12 total queries covering different aspects
        2. Primary queries (priority 0.8-1.0): Core research questions
        3. Secondary queries (priority 0.5-0.7): Supporting questions  
        4. Exploratory queries (priority 0.3-0.5): Broader context questions
        5. Each query should be specific and answerable from academic literature
        6. Categories should be: techniques, limitations, datasets, applications, industry, trends, ethics, challenges, future_work, comparative
        """
    
    def _register_tools(self):
        """Register tools for query generation"""
        
        def topic_analysis(topic: str) -> str:
            """Analyze research topic to identify key components"""
            components = {
                'domain': 'Research domain/field',
                'methods': 'Methods/techniques mentioned',
                'applications': 'Applications/use cases',
                'scope': 'Scope and boundaries',
                'context': 'Contextual factors'
            }
            
            analysis_result = f"Topic Analysis for: {topic}\n\n"
            for component, description in components.items():
                analysis_result += f"{component.upper()}: {description}\n"
            
            return analysis_result
        
        def question_generation(context: str) -> str:
            """Generate specific research questions based on context"""
            question_types = [
                "What are the current state-of-the-art methods for...",
                "What are the main challenges and limitations in...",
                "How do different approaches compare in terms of...",
                "What datasets and benchmarks are commonly used for...",
                "What are the practical applications and use cases of...",
                "What are the ethical considerations and implications of...",
                "What are the emerging trends and future directions in...",
                "How has the field evolved over the past 5 years in...",
                "What are the key performance metrics used to evaluate...",
                "What are the computational and resource requirements for..."
            ]
            
            return f"Question generation templates:\n" + "\n".join(f"- {q}" for q in question_types)
        
        def priority_scoring(queries: str) -> str:
            """Provide guidance on priority scoring for queries"""
            scoring_guide = """
            Priority Scoring Guidelines:
            
            1.0 - CRITICAL: Core research questions that must be answered
            0.9 - HIGH: Important questions that provide key insights
            0.8 - HIGH: Significant questions for comprehensive understanding
            0.7 - MEDIUM-HIGH: Supporting questions that add valuable context
            0.6 - MEDIUM: Useful questions for broader perspective
            0.5 - MEDIUM: Exploratory questions for additional insights
            0.4 - LOW-MEDIUM: Background or contextual questions
            0.3 - LOW: Optional questions for completeness
            
            Consider:
            - Relevance to core research topic
            - Availability of literature/sources
            - Actionability of insights
            - Uniqueness of perspective
            """
            
            return scoring_guide
        
        # Register tools with PromptChain
        self.chain.register_tool_function(topic_analysis)
        self.chain.register_tool_function(question_generation)
        self.chain.register_tool_function(priority_scoring)
        
        # Add tool schemas
        self.chain.add_tools([
            {
                "type": "function",
                "function": {
                    "name": "topic_analysis",
                    "description": "Analyze research topic to identify key components and structure",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string", "description": "Research topic to analyze"}
                        },
                        "required": ["topic"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "question_generation",
                    "description": "Generate research question templates based on analysis context",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "context": {"type": "string", "description": "Analysis context for question generation"}
                        },
                        "required": ["context"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "priority_scoring", 
                    "description": "Get guidance on priority scoring for research queries",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queries": {"type": "string", "description": "List of queries to score"}
                        },
                        "required": ["queries"]
                    }
                }
            }
        ])
    
    async def generate_queries(
        self, 
        topic: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate comprehensive research queries for a given topic
        
        Args:
            topic: Research topic to analyze
            context: Additional context including max_queries, iteration, etc.
            
        Returns:
            JSON string containing structured queries
        """
        logger.info(f"Generating queries for topic: '{topic}'")
        
        try:
            # Prepare input context
            input_data = {
                'topic': topic,
                'max_queries': context.get('max_queries', 12) if context else 12,
                'iteration': context.get('iteration', 0) if context else 0,
                'previous_queries': context.get('previous_queries', []) if context else []
            }
            
            # Add iteration-specific guidance
            if input_data['iteration'] > 0:
                iteration_guidance = f"""
                This is iteration {input_data['iteration']} of research. 
                Previous queries have been processed. Focus on:
                - Filling identified gaps
                - Exploring under-researched areas
                - Adding complementary perspectives
                - Refining scope based on findings
                """
                input_data['iteration_guidance'] = iteration_guidance
            
            # Process through PromptChain
            result = await self.chain.process_prompt_async(
                json.dumps(input_data, indent=2)
            )
            
            # Validate and clean result
            validated_result = self._validate_query_response(result)
            
            logger.info(f"Generated queries successfully: {self._count_queries(validated_result)} total queries")
            
            return validated_result
            
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            raise
    
    def _validate_query_response(self, response: str) -> str:
        """Validate and clean query response"""
        try:
            # Try to parse as JSON
            parsed = json.loads(response)
            
            # Ensure required structure
            if not isinstance(parsed, dict):
                raise ValueError("Response must be a JSON object")
            
            required_sections = ['primary_queries', 'secondary_queries', 'exploratory_queries']
            for section in required_sections:
                if section not in parsed:
                    parsed[section] = []
            
            # Validate query structure
            total_queries = 0
            for section in required_sections:
                if isinstance(parsed[section], list):
                    for i, query in enumerate(parsed[section]):
                        if isinstance(query, dict):
                            # Ensure required fields
                            if 'text' not in query:
                                query['text'] = f"Query {i+1} in {section}"
                            if 'priority' not in query:
                                query['priority'] = 0.5
                            if 'category' not in query:
                                query['category'] = 'general'
                            
                            total_queries += 1
            
            # Limit total queries
            if total_queries > 15:
                logger.warning(f"Too many queries generated ({total_queries}), truncating to 15")
                parsed = self._truncate_queries(parsed, 15)
            
            return json.dumps(parsed, indent=2)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
            # Fallback to simple parsing
            return self._fallback_parsing(response)
        except Exception as e:
            logger.error(f"Query validation failed: {e}")
            return self._fallback_parsing(response)
    
    def _truncate_queries(self, parsed: dict, max_queries: int) -> dict:
        """Truncate queries to maximum limit while preserving balance"""
        total = 0
        sections = ['primary_queries', 'secondary_queries', 'exploratory_queries']
        limits = [6, 5, 4]  # Balanced distribution
        
        for section, limit in zip(sections, limits):
            if section in parsed and isinstance(parsed[section], list):
                parsed[section] = parsed[section][:limit]
                total += len(parsed[section])
                
                if total >= max_queries:
                    break
        
        return parsed
    
    def _fallback_parsing(self, response: str) -> str:
        """Fallback parsing for non-JSON responses"""
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        # Simple extraction of question-like lines
        queries = []
        for line in lines:
            if ('?' in line or 
                line.lower().startswith('what') or 
                line.lower().startswith('how') or 
                line.lower().startswith('why') or
                line.lower().startswith('which')):
                queries.append({
                    'text': line.rstrip('?') + '?',
                    'priority': 0.7,
                    'category': 'general'
                })
        
        # Distribute across categories
        result = {
            'primary_queries': queries[:4],
            'secondary_queries': queries[4:8],  
            'exploratory_queries': queries[8:12]
        }
        
        return json.dumps(result, indent=2)
    
    def _count_queries(self, response: str) -> int:
        """Count total queries in response"""
        try:
            parsed = json.loads(response)
            total = 0
            for section in ['primary_queries', 'secondary_queries', 'exploratory_queries']:
                if section in parsed and isinstance(parsed[section], list):
                    total += len(parsed[section])
            return total
        except:
            return 0
    
    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration"""
        return self.config.copy()
    
    def update_config(self, config: Dict[str, Any]):
        """Update agent configuration"""
        self.config.update(config)
        logger.info("QueryGenerationAgent configuration updated")