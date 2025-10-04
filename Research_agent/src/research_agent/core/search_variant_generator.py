#!/usr/bin/env python3
"""
Intelligent Search Variant Generation with PromptChain

This module implements a sophisticated 3-stage PromptChain pipeline that transforms
the Research Agent's query generation from simple rule-based to AI-powered semantic analysis.

Architecture:
1. TopicDecompositionChain: Analyzes research topics and extracts key components
2. QueryGenerationChain: Generates diverse queries based on decomposed components
3. QueryOptimizationChain: Optimizes queries for specific databases (ArXiv, PubMed, Sci-Hub)

Features:
- Adaptive spanning: 4-20 queries based on topic complexity
- Semantic understanding and component extraction
- Database-specific optimization
- Backward compatibility and fallback mechanisms
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from promptchain import PromptChain
import litellm

logger = logging.getLogger(__name__)


@dataclass
class TopicComponents:
    """Structured representation of decomposed research topic components"""
    primary_concept: str
    methodology: str
    application_area: str
    domain: str
    temporal_focus: str
    related_fields: List[str]
    technical_aspects: List[str]
    clinical_context: Optional[str] = None
    geographic_scope: Optional[str] = None


@dataclass
class ComplexityMetrics:
    """Metrics for calculating adaptive query count"""
    concept_count: int
    methodology_count: int
    application_count: int
    scope_breadth: int  # 1-10 scale
    cross_disciplinary_elements: int
    temporal_span: int  # 1-10 scale
    technical_depth: int  # 1-10 scale


@dataclass
class GeneratedQuery:
    """Container for generated search query with metadata"""
    query: str
    category: str  # methodological, temporal, application, cross-disciplinary, comparative
    database_preference: str  # arxiv, pubmed, sci_hub, general
    confidence_score: float
    reasoning: str


class TopicComplexityAnalyzer:
    """Analyzes topic complexity to determine adaptive query count"""
    
    def __init__(self, model: str = "openai/gpt-4"):
        self.model = model
    
    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON from GPT-4o response that might contain markdown or extra text"""
        import re
        
        # Remove markdown code blocks
        response = response.replace('```json', '').replace('```', '')
        
        # Try to find JSON object boundaries
        json_start = response.find('{')
        if json_start != -1:
            # Find the matching closing brace
            brace_count = 0
            json_end = json_start
            for i, char in enumerate(response[json_start:], json_start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if json_end > json_start:
                return response[json_start:json_end]
        
        # If no clear JSON boundaries, return cleaned version
        return response.strip()
    
    async def analyze_complexity(self, topic: str, components: TopicComponents) -> ComplexityMetrics:
        """Analyze topic complexity and return metrics for adaptive spanning"""
        
        analysis_prompt = f"""
        You are an expert research analyst. Analyze the provided research topic and its components 
        to calculate complexity metrics that will determine how many search queries to generate.
        
        Focus on:
        1. Count distinct concepts, methodologies, and applications
        2. Assess scope breadth and cross-disciplinary elements
        3. Evaluate temporal span and technical depth
        4. Consider integration complexity between different fields
        
        Research Topic: "{topic}"
        
        Decomposed Components:
        - Primary Concept: {components.primary_concept}
        - Methodology: {components.methodology}
        - Application Area: {components.application_area}
        - Domain: {components.domain}
        - Temporal Focus: {components.temporal_focus}
        - Related Fields: {', '.join(components.related_fields)}
        - Technical Aspects: {', '.join(components.technical_aspects)}
        - Clinical Context: {components.clinical_context or 'None'}
        
        Analyze complexity and return JSON with these exact fields:
        {{
            "concept_count": <number of distinct concepts>,
            "methodology_count": <number of methodologies/techniques>,
            "application_count": <number of application areas>,
            "scope_breadth": <breadth of research scope from 1-10>,
            "cross_disciplinary_elements": <number of related fields>,
            "temporal_span": <temporal research span from 1-10>,
            "technical_depth": <technical complexity from 1-10>
        }}
        
        Guidelines:
        - concept_count: Count unique research concepts (2-8 typical)
        - methodology_count: Count distinct methodologies/approaches (1-5 typical)
        - application_count: Count application domains (1-4 typical)
        - scope_breadth: 1=narrow, 5=moderate, 10=very broad interdisciplinary scope
        - cross_disciplinary_elements: Count related fields (0-6 typical)
        - temporal_span: 1=current only, 5=historical context, 10=full timeline
        - technical_depth: 1=basic, 5=moderate, 10=highly technical/specialized
        """
        
        try:
            # Use direct LiteLLM call for complete control
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert research analyst. Analyze the provided research topic and its components to calculate complexity metrics that will determine how many search queries to generate. Return detailed complexity analysis as structured JSON."
                },
                {
                    "role": "user", 
                    "content": analysis_prompt
                }
            ]
            
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                temperature=0.3
            )
            
            result = response.choices[0].message.content
            
            # Parse JSON response with improved handling for GPT-4o
            if isinstance(result, str):
                # Clean the response to extract JSON
                cleaned_result = self._extract_json_from_response(result)
                complexity_data = json.loads(cleaned_result)
            else:
                complexity_data = result
            
            return ComplexityMetrics(**complexity_data)
        
        except Exception as e:
            logger.warning(f"Error in complexity analysis: {e}. Using default metrics.")
            # Fallback to reasonable defaults
            return ComplexityMetrics(
                concept_count=2,
                methodology_count=1,
                application_count=1,
                scope_breadth=3,
                cross_disciplinary_elements=1,
                temporal_span=3,
                technical_depth=3
            )


class TopicDecompositionChain:
    """First stage: Analyzes research topics and extracts key components"""
    
    def __init__(self, model: str = "openai/gpt-4"):
        self.model = model
    
    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON from GPT-4o response that might contain markdown or extra text"""
        import re
        
        # Remove markdown code blocks
        response = response.replace('```json', '').replace('```', '')
        
        # Try to find JSON object boundaries
        json_start = response.find('{')
        if json_start != -1:
            # Find the matching closing brace
            brace_count = 0
            json_end = json_start
            for i, char in enumerate(response[json_start:], json_start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if json_end > json_start:
                return response[json_start:json_end]
        
        # If no clear JSON boundaries, return cleaned version
        return response.strip()
    
    async def decompose_topic(self, topic: str) -> TopicComponents:
        """Decompose research topic into structured components"""
        
        decomposition_prompt = f"""
        You are an expert research analyst specializing in academic topic decomposition.
        
        Analyze the provided research topic and extract its key semantic components.
        Break down the topic into structured elements that will guide intelligent query generation.
        
        Research Topic: "{topic}"
        
        Analyze this research topic and extract its key components. Return JSON with these exact fields:
        {{
            "primary_concept": "<main research focus or objective>",
            "methodology": "<primary research methods or techniques>",
            "application_area": "<main application domain>",
            "domain": "<academic/scientific domain>",
            "temporal_focus": "<temporal aspect: current, historical, future, or longitudinal>",
            "related_fields": ["<field1>", "<field2>", "<field3>"],
            "technical_aspects": ["<aspect1>", "<aspect2>", "<aspect3>"],
            "clinical_context": "<clinical context if applicable, null if not>",
            "geographic_scope": "<geographic scope if applicable, null if not>"
        }}
        
        Guidelines:
        - primary_concept: Core research objective (e.g., "early detection", "pattern analysis")
        - methodology: Key approaches (e.g., "machine learning", "clinical assessment")
        - application_area: Main domain (e.g., "medical diagnosis", "predictive modeling")
        - domain: Academic field (e.g., "neurology", "computer science", "biomedical engineering")
        - temporal_focus: Time dimension (e.g., "early detection", "longitudinal tracking")
        - related_fields: Connected disciplines (up to 5)
        - technical_aspects: Technical elements (up to 5)
        - clinical_context: Clinical setting if relevant
        - geographic_scope: Geographic context if specified
        
        Be specific and precise in your analysis.
        """
        
        try:
            # Use direct LiteLLM call for complete control
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert research analyst specializing in academic topic decomposition. Analyze the provided research topic and extract its key semantic components. Return structured JSON with the specified fields."
                },
                {
                    "role": "user", 
                    "content": decomposition_prompt
                }
            ]
            
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                temperature=0.3
            )
            
            result = response.choices[0].message.content
            
            # Parse JSON response with improved handling for GPT-4o
            if isinstance(result, str):
                # Clean the response to extract JSON
                cleaned_result = self._extract_json_from_response(result)
                components_data = json.loads(cleaned_result)
            else:
                components_data = result
            
            # Handle None values
            if components_data.get('clinical_context') == 'null':
                components_data['clinical_context'] = None
            if components_data.get('geographic_scope') == 'null':
                components_data['geographic_scope'] = None
            
            return TopicComponents(**components_data)
        
        except Exception as e:
            logger.warning(f"Error in topic decomposition: {e}. Using fallback analysis.")
            # Fallback decomposition
            return TopicComponents(
                primary_concept=topic.split()[0] if topic.split() else "research",
                methodology="analysis",
                application_area="general research",
                domain="interdisciplinary",
                temporal_focus="current",
                related_fields=["computer science"],
                technical_aspects=["data analysis"],
                clinical_context=None,
                geographic_scope=None
            )


class QueryGenerationChain:
    """Second stage: Generates diverse queries based on decomposed components"""
    
    def __init__(self, model: str = "openai/gpt-4"):
        self.model = model
    
    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON from GPT-4o response that might contain markdown or extra text"""
        import re
        
        # Remove markdown code blocks
        response = response.replace('```json', '').replace('```', '')
        
        # Try to find JSON object boundaries
        json_start = response.find('{') if '{' in response else response.find('[')
        if json_start != -1:
            # Handle both object and array responses
            start_char = response[json_start]
            end_char = '}' if start_char == '{' else ']'
            
            # Find the matching closing brace/bracket
            brace_count = 0
            json_end = json_start
            for i, char in enumerate(response[json_start:], json_start):
                if char == start_char:
                    brace_count += 1
                elif char == end_char:
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if json_end > json_start:
                return response[json_start:json_end]
        
        # If no clear JSON boundaries, return cleaned version
        return response.strip()
    
    async def generate_queries(
        self, 
        topic: str, 
        components: TopicComponents, 
        target_count: int
    ) -> List[GeneratedQuery]:
        """Generate diverse search queries with adaptive count"""
        
        generation_prompt = f"""
        You are an expert academic search strategist specializing in query diversification.
        
        Generate semantically diverse, high-quality search queries based on the provided 
        topic components and complexity analysis. Create queries that explore different 
        research angles and methodologies.
        
        Generate queries across these categories:
        1. Methodological: Focus on techniques and approaches
        2. Temporal: Recent advances, historical developments, trends
        3. Application: Clinical applications, real-world use cases
        4. Cross-disciplinary: Integration with related fields
        5. Comparative: Different approaches and their effectiveness
        
        Original Topic: "{topic}"
        Target Query Count: {target_count}
        
        Topic Components:
        - Primary Concept: {components.primary_concept}
        - Methodology: {components.methodology}
        - Application Area: {components.application_area}
        - Domain: {components.domain}
        - Temporal Focus: {components.temporal_focus}
        - Related Fields: {', '.join(components.related_fields)}
        - Technical Aspects: {', '.join(components.technical_aspects)}
        - Clinical Context: {components.clinical_context or 'None'}
        
        Generate exactly {target_count} diverse search queries. Return JSON array with this structure:
        [
            {{
                "query": "<search query text>",
                "category": "<methodological|temporal|application|cross_disciplinary|comparative>",
                "database_preference": "<arxiv|pubmed|sci_hub|general>",
                "confidence_score": <0.1 to 1.0>,
                "reasoning": "<brief explanation of query design>"
            }}
        ]
        
        Query Design Guidelines:
        1. **Methodological queries**: Focus on techniques, methods, algorithms
        2. **Temporal queries**: Recent advances, reviews, historical development
        3. **Application queries**: Clinical applications, real-world implementations
        4. **Cross-disciplinary queries**: Integration with related fields
        5. **Comparative queries**: Comparison studies, effectiveness analysis
        
        Database Preferences:
        - arxiv: Computer science, machine learning, AI, theoretical work
        - pubmed: Medical, clinical, biological, health-related research
        - sci_hub: Broad academic coverage, especially when specific papers needed
        - general: Suitable for multiple databases
        
        Ensure queries are:
        - Semantically diverse (avoid redundancy)
        - Academically precise (use proper terminology)
        - Searchable (appropriate length and structure)
        - Relevant (connected to original topic)
        
        Confidence scores should reflect query quality and expected relevance.
        """
        
        try:
            # Use direct LiteLLM call for complete control
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert academic search strategist specializing in query diversification. Generate semantically diverse, high-quality search queries based on the provided topic components. Return high-quality queries with confidence scores and rationale as structured JSON."
                },
                {
                    "role": "user", 
                    "content": generation_prompt
                }
            ]
            
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                temperature=0.7  # Higher creativity for query generation
            )
            
            result = response.choices[0].message.content
            
            # Parse JSON response with improved handling for GPT-4o
            if isinstance(result, str):
                # Clean the response to extract JSON
                cleaned_result = self._extract_json_from_response(result)
                queries_data = json.loads(cleaned_result)
            else:
                queries_data = result
            
            # Convert to GeneratedQuery objects
            generated_queries = []
            for query_data in queries_data:
                generated_queries.append(GeneratedQuery(**query_data))
            
            return generated_queries
        
        except Exception as e:
            logger.warning(f"Error in query generation: {e}. Using fallback queries.")
            # Fallback to basic queries
            fallback_queries = [
                GeneratedQuery(
                    query=topic,
                    category="methodological",
                    database_preference="general",
                    confidence_score=0.7,
                    reasoning="Original topic as fallback"
                ),
                GeneratedQuery(
                    query=f'"{topic}"',
                    category="methodological",
                    database_preference="general",
                    confidence_score=0.6,
                    reasoning="Exact phrase search as fallback"
                )
            ]
            return fallback_queries[:target_count]


class QueryOptimizationChain:
    """Third stage: Optimizes queries for specific databases"""
    
    def __init__(self, model: str = "openai/gpt-4"):
        self.model = model
    
    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON from GPT-4o response that might contain markdown or extra text"""
        import re
        
        # Remove markdown code blocks
        response = response.replace('```json', '').replace('```', '')
        
        # Try to find JSON object boundaries
        json_start = response.find('{')
        if json_start != -1:
            # Find the matching closing brace
            brace_count = 0
            json_end = json_start
            for i, char in enumerate(response[json_start:], json_start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if json_end > json_start:
                return response[json_start:json_end]
        
        # If no clear JSON boundaries, return cleaned version
        return response.strip()
    
    async def optimize_queries(self, queries: List[GeneratedQuery]) -> Dict[str, List[str]]:
        """Optimize queries for specific databases"""
        
        # Organize queries by database preference
        arxiv_queries = [q for q in queries if q.database_preference in ['arxiv', 'general']]
        pubmed_queries = [q for q in queries if q.database_preference in ['pubmed', 'general']]
        scihub_queries = [q for q in queries if q.database_preference in ['sci_hub', 'general']]
        
        optimization_prompt = f"""
        You are an expert in academic database search optimization.
        
        Optimize the provided search queries for specific academic databases (ArXiv, PubMed, Sci-Hub).
        Each database has different search syntax, field structures, and optimization strategies.
        
        Database Optimization Focus:
        1. ArXiv: CS/physics terminology, preprint format, author/category fields
        2. PubMed: Medical terminology, MeSH terms, clinical context
        3. Sci-Hub: DOI-based access, journal coverage, publication metadata
        
        Optimize these search queries for ArXiv, PubMed, and Sci-Hub databases.
        
        ArXiv Queries to Optimize:
        {json.dumps([{'query': q.query, 'category': q.category} for q in arxiv_queries], indent=2)}
        
        PubMed Queries to Optimize:
        {json.dumps([{'query': q.query, 'category': q.category} for q in pubmed_queries], indent=2)}
        
        Sci-Hub Queries to Optimize:
        {json.dumps([{'query': q.query, 'category': q.category} for q in scihub_queries], indent=2)}
        
        Return JSON with optimized queries for each database:
        {{
            "arxiv": ["<optimized query 1>", "<optimized query 2>", ...],
            "pubmed": ["<optimized query 1>", "<optimized query 2>", ...],
            "sci_hub": ["<optimized query 1>", "<optimized query 2>", ...]
        }}
        
        Optimization Guidelines:
        
        **ArXiv Optimization**:
        - Use CS/physics terminology where appropriate
        - Include category filters when relevant (cs.LG, cs.AI, etc.)
        - Optimize for preprint and theoretical work terminology
        - Use author and subject classification hints
        
        **PubMed Optimization**:
        - Use medical/clinical terminology
        - Include MeSH terms and medical subject headings
        - Add publication type filters ([Review], [Clinical Trial])
        - Use field tags ([Title/Abstract], [MeSH Terms])
        - Include date ranges for temporal queries
        
        **Sci-Hub Optimization**:
        - Focus on journal names and publication metadata
        - Use precise terminology for better DOI matching
        - Include author names when relevant
        - Optimize for specific publication types
        
        Maintain semantic intent while adapting syntax and terminology for each database.
        """
        
        try:
            # Use direct LiteLLM call for complete control
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert in academic database search optimization. Optimize the provided search queries for specific academic databases (ArXiv, PubMed, Sci-Hub). Each database has different search syntax, field structures, and optimization strategies. Ensure optimized queries maintain semantic intent while maximizing database-specific effectiveness."
                },
                {
                    "role": "user", 
                    "content": optimization_prompt
                }
            ]
            
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                temperature=0.3  # Lower creativity for optimization
            )
            
            result = response.choices[0].message.content
            
            # Parse JSON response with improved handling for GPT-4o
            if isinstance(result, str):
                # Clean the response to extract JSON
                cleaned_result = self._extract_json_from_response(result)
                optimized_data = json.loads(cleaned_result)
            else:
                optimized_data = result
            
            return optimized_data
        
        except Exception as e:
            logger.warning(f"Error in query optimization: {e}. Using unoptimized queries.")
            # Fallback to original queries
            return {
                'arxiv': [q.query for q in arxiv_queries],
                'pubmed': [q.query for q in pubmed_queries],
                'sci_hub': [q.query for q in scihub_queries]
            }


class SearchVariantGenerator:
    """
    Main orchestrator for intelligent search variant generation.
    
    Implements a 3-stage PromptChain pipeline that transforms simple topic strings
    into sophisticated, database-optimized search queries with adaptive spanning.
    """
    
    def __init__(
        self, 
        model: str = "openai/gpt-4",
        min_queries: int = 4,
        max_queries: int = 20,
        enable_adaptive_spanning: bool = True,
        enable_query_optimization: bool = True
    ):
        self.model = model
        self.min_queries = min_queries
        self.max_queries = max_queries
        self.enable_adaptive_spanning = enable_adaptive_spanning
        self.enable_query_optimization = enable_query_optimization
        
        # Initialize PromptChain components
        self.complexity_analyzer = TopicComplexityAnalyzer(model)
        self.decomposition_chain = TopicDecompositionChain(model)
        self.generation_chain = QueryGenerationChain(model)
        self.optimization_chain = QueryOptimizationChain(model)
        
        # Performance tracking
        self.generation_metrics = {
            'total_generations': 0,
            'successful_generations': 0,
            'average_query_count': 0,
            'average_generation_time': 0,
            'fallback_usage': 0
        }
    
    def _calculate_adaptive_count(self, complexity_metrics: ComplexityMetrics) -> int:
        """Calculate adaptive query count based on topic complexity"""
        if not self.enable_adaptive_spanning:
            return 8  # Default fixed count
        
        # Base count
        base_count = 4
        
        # Complexity-based adjustments
        complexity_bonus = complexity_metrics.concept_count * 1.5
        methodology_bonus = complexity_metrics.methodology_count * 1.0
        scope_bonus = complexity_metrics.scope_breadth * 0.8
        cross_disciplinary_bonus = complexity_metrics.cross_disciplinary_elements * 1.2
        technical_bonus = complexity_metrics.technical_depth * 0.5
        
        # Calculate total
        total_count = (
            base_count + 
            complexity_bonus + 
            methodology_bonus + 
            scope_bonus + 
            cross_disciplinary_bonus + 
            technical_bonus
        )
        
        # Apply bounds
        adaptive_count = max(self.min_queries, min(int(total_count), self.max_queries))
        
        logger.info(f"Adaptive query count calculation: {adaptive_count} queries")
        logger.debug(f"Complexity breakdown: base={base_count}, complexity={complexity_bonus}, "
                    f"methodology={methodology_bonus}, scope={scope_bonus}, "
                    f"cross_disciplinary={cross_disciplinary_bonus}, technical={technical_bonus}")
        
        return adaptive_count
    
    async def generate_search_variants(self, topic: str) -> Dict[str, Any]:
        """
        Generate intelligent search variants using the 3-stage PromptChain pipeline.
        
        Args:
            topic: Research topic string
            
        Returns:
            Dictionary containing generated queries, metadata, and performance metrics
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting intelligent search variant generation for: '{topic}'")
            
            # Stage 1: Topic Decomposition
            logger.info("Stage 1: Topic decomposition")
            topic_components = await self.decomposition_chain.decompose_topic(topic)
            logger.debug(f"Topic components: {asdict(topic_components)}")
            
            # Stage 1.5: Complexity Analysis (for adaptive spanning)
            logger.info("Stage 1.5: Complexity analysis")
            complexity_metrics = await self.complexity_analyzer.analyze_complexity(
                topic, topic_components
            )
            target_query_count = self._calculate_adaptive_count(complexity_metrics)
            logger.info(f"Target query count: {target_query_count}")
            
            # Stage 2: Query Generation
            logger.info("Stage 2: Query generation")
            generated_queries = await self.generation_chain.generate_queries(
                topic, topic_components, target_query_count
            )
            logger.info(f"Generated {len(generated_queries)} queries")
            
            # Stage 3: Query Optimization (optional)
            optimized_queries = {}
            if self.enable_query_optimization:
                logger.info("Stage 3: Query optimization")
                optimized_queries = await self.optimization_chain.optimize_queries(generated_queries)
                logger.info(f"Optimized queries for {len(optimized_queries)} databases")
            else:
                # Use unoptimized queries
                optimized_queries = {
                    'general': [q.query for q in generated_queries]
                }
            
            # Calculate performance metrics
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            
            # Update metrics
            self.generation_metrics['total_generations'] += 1
            self.generation_metrics['successful_generations'] += 1
            self.generation_metrics['average_query_count'] = (
                (self.generation_metrics['average_query_count'] * 
                 (self.generation_metrics['total_generations'] - 1) + len(generated_queries)) /
                self.generation_metrics['total_generations']
            )
            self.generation_metrics['average_generation_time'] = (
                (self.generation_metrics['average_generation_time'] * 
                 (self.generation_metrics['total_generations'] - 1) + generation_time) /
                self.generation_metrics['total_generations']
            )
            
            # Return comprehensive results
            return {
                'queries': optimized_queries,
                'metadata': {
                    'original_topic': topic,
                    'topic_components': asdict(topic_components),
                    'complexity_metrics': asdict(complexity_metrics),
                    'target_query_count': target_query_count,
                    'actual_query_count': len(generated_queries),
                    'generation_time': generation_time,
                    'model_used': self.model,
                    'adaptive_spanning_enabled': self.enable_adaptive_spanning,
                    'query_optimization_enabled': self.enable_query_optimization
                },
                'generated_queries_details': [asdict(q) for q in generated_queries],
                'performance_metrics': self.generation_metrics.copy()
            }
            
        except Exception as e:
            logger.error(f"Error in search variant generation: {e}")
            
            # Update failure metrics
            self.generation_metrics['total_generations'] += 1
            self.generation_metrics['fallback_usage'] += 1
            
            # Return fallback results
            return self._generate_fallback_variants(topic)
    
    def _generate_fallback_variants(self, topic: str) -> Dict[str, Any]:
        """Generate fallback search variants using simple rule-based approach"""
        logger.warning("Using fallback search variant generation")
        
        # Simple rule-based fallback (original approach)
        fallback_queries = [
            topic,
            f'"{topic}"',
            f'{topic} review',
            f'{topic} methods',
            f'{topic} applications',
            f'{topic} recent advances',
            f'{topic} clinical',
            f'{topic} analysis'
        ]
        
        return {
            'queries': {
                'general': fallback_queries
            },
            'metadata': {
                'original_topic': topic,
                'fallback_used': True,
                'generation_time': 0.1,
                'model_used': 'fallback',
                'adaptive_spanning_enabled': False,
                'query_optimization_enabled': False
            },
            'generated_queries_details': [],
            'performance_metrics': self.generation_metrics.copy()
        }
    
    def get_generation_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for search variant generation"""
        return self.generation_metrics.copy()
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.generation_metrics = {
            'total_generations': 0,
            'successful_generations': 0,
            'average_query_count': 0,
            'average_generation_time': 0,
            'fallback_usage': 0
        }