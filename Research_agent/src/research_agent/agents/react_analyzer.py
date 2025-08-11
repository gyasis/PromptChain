"""
ReAct Analysis Agent

Performs ReAct-style analysis to identify gaps in research and generate new queries.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from promptchain import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

logger = logging.getLogger(__name__)


class ReActAnalysisAgent:
    """
    Agent that performs ReAct-style analysis to identify research gaps and generate refinement queries
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize PromptChain with ReAct-style reasoning
        self.chain = PromptChain(
            models=[config.get('model', 'openai/gpt-4o')],
            instructions=[
                "You are a ReAct analysis specialist. Analyze research progress and identify gaps: {context}",
                AgenticStepProcessor(
                    objective=config.get('processor', {}).get('objective', 
                               "Analyze research completeness and identify strategic gaps for next iteration"),
                    max_internal_steps=config.get('processor', {}).get('max_internal_steps', 7)
                ),
                self._format_react_output_instruction()
            ],
            verbose=True
        )
        
        # Register ReAct analysis tools
        self._register_tools()
        
        logger.info("ReActAnalysisAgent initialized")
    
    def _format_react_output_instruction(self) -> str:
        """Instruction for formatting ReAct analysis output"""
        return """
        Format your analysis as a JSON object with the following structure:
        {
            "analysis_summary": {
                "current_state": "Brief description of research progress",
                "completion_score": 0.85,
                "iteration_effectiveness": "Assessment of current iteration",
                "key_findings": ["Finding 1", "Finding 2", "Finding 3"]
            },
            "gaps_identified": [
                {
                    "gap_type": "methodology",
                    "description": "Specific gap description",
                    "severity": "high",
                    "evidence": "Why this gap exists based on current results",
                    "impact": "How this gap affects research completeness"
                }
            ],
            "coverage_analysis": {
                "techniques_covered": ["technique1", "technique2"],
                "techniques_missing": ["missing1", "missing2"], 
                "applications_covered": ["app1", "app2"],
                "applications_missing": ["missing_app1"],
                "temporal_coverage": {
                    "recent_papers": 15,
                    "older_papers": 8,
                    "missing_periods": ["2021-2022"]
                }
            },
            "new_queries": [
                {
                    "text": "What are the recent advances in...",
                    "priority": 0.9,
                    "category": "methodology",
                    "rationale": "Addresses identified methodology gap",
                    "target_papers": 10
                }
            ],
            "iteration_recommendation": {
                "should_continue": true,
                "confidence": 0.8,
                "focus_areas": ["area1", "area2"],
                "search_strategy_adjustments": ["adjustment1", "adjustment2"],
                "expected_improvement": "What this iteration should achieve"
            }
        }
        
        Analysis Guidelines:
        1. Completion score: 0.0-1.0 based on comprehensive coverage
        2. Gap severity: low, medium, high, critical
        3. Gap types: methodology, applications, datasets, evaluation, comparative, temporal, domain-specific
        4. New queries should be specific and targeted (max 8 queries)
        5. Consider paper quality, recency, and diversity
        """
    
    def _register_tools(self):
        """Register ReAct analysis tools"""
        
        def coverage_analysis(research_context: str) -> str:
            """Analyze coverage across different research dimensions"""
            analysis_framework = """
            Research Coverage Analysis Framework:
            
            METHODOLOGICAL COVERAGE:
            - Core techniques and algorithms
            - Variations and improvements
            - Comparative evaluations
            - Novel approaches
            
            APPLICATION COVERAGE:
            - Primary use cases
            - Domain-specific applications
            - Cross-domain applications
            - Emerging applications
            
            TEMPORAL COVERAGE:
            - Historical development
            - Recent advances (2-3 years)
            - Cutting-edge research (last year)
            - Future directions
            
            EVALUATION COVERAGE:
            - Standard benchmarks
            - Evaluation metrics
            - Comparative studies
            - Reproducibility
            
            QUALITY INDICATORS:
            - Publication venues
            - Citation patterns
            - Author authority
            - Methodological rigor
            """
            return analysis_framework
        
        def gap_identification(analysis_data: str) -> str:
            """Identify specific gaps in research coverage"""
            gap_identification_guide = """
            Gap Identification Strategy:
            
            METHODOLOGY GAPS:
            - Missing algorithmic approaches
            - Unexplored variations
            - Insufficient comparative studies
            - Limited evaluation dimensions
            
            APPLICATION GAPS:
            - Underexplored domains
            - Scale/complexity variations
            - Real-world deployments
            - Cross-domain transfers
            
            TEMPORAL GAPS:
            - Outdated information
            - Missing recent developments
            - Insufficient trend analysis
            - Future direction gaps
            
            QUALITY GAPS:
            - Insufficient high-impact papers
            - Missing authoritative sources
            - Incomplete experimental validation
            - Limited reproducibility evidence
            
            GAP PRIORITIZATION:
            - Impact on research completeness
            - Availability of literature
            - Time/effort to address
            - Relevance to research goals
            """
            return gap_identification_guide
        
        def query_generation_strategy(gap_analysis: str) -> str:
            """Generate strategic queries to address identified gaps"""
            query_strategy = """
            Strategic Query Generation:
            
            GAP-TARGETED QUERIES:
            - Direct gap addressing: "What are the limitations of X in Y domain?"
            - Comparative analysis: "How do X and Y approaches compare for Z?"
            - Recent developments: "What are the recent advances in X since 2023?"
            - Cross-domain: "How has X been applied to Y domain?"
            
            QUERY OPTIMIZATION:
            - Specific and answerable
            - Literature-rich topics
            - Complementary to existing queries
            - Balanced priority distribution
            
            SEARCH EFFICIENCY:
            - Target high-yield databases
            - Use effective keywords
            - Consider paper accessibility
            - Avoid redundant searches
            
            ITERATIVE REFINEMENT:
            - Build on previous findings
            - Address revealed complexities
            - Explore unexpected directions
            - Fill systematic gaps
            """
            return query_strategy
        
        def iteration_planning(progress_data: str) -> str:
            """Plan next iteration based on current progress"""
            iteration_planning_guide = """
            Iteration Planning Framework:
            
            CONTINUATION CRITERIA:
            - Significant gaps remain (completion < 0.9)
            - High-value opportunities identified
            - Iteration budget available
            - Clear improvement potential
            
            TERMINATION CRITERIA:
            - High completion score (>0.9)
            - Diminishing returns evident
            - No significant gaps remain
            - Resource constraints
            
            FOCUS PRIORITIZATION:
            - Critical gaps first
            - High-impact areas
            - Resource-efficient targets
            - Complementary coverage
            
            STRATEGY ADJUSTMENTS:
            - Database rebalancing
            - Query refinement
            - Search term optimization
            - Quality threshold adjustment
            """
            return iteration_planning_guide
        
        # Register tools
        self.chain.register_tool_function(coverage_analysis)
        self.chain.register_tool_function(gap_identification)
        self.chain.register_tool_function(query_generation_strategy)
        self.chain.register_tool_function(iteration_planning)
        
        # Add tool schemas
        self.chain.add_tools([
            {
                "type": "function",
                "function": {
                    "name": "coverage_analysis",
                    "description": "Analyze research coverage across multiple dimensions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "research_context": {"type": "string", "description": "Current research context and findings"}
                        },
                        "required": ["research_context"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "gap_identification",
                    "description": "Identify specific gaps in research coverage",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "analysis_data": {"type": "string", "description": "Analysis data for gap identification"}
                        },
                        "required": ["analysis_data"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_generation_strategy",
                    "description": "Generate strategic queries to address identified gaps",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "gap_analysis": {"type": "string", "description": "Gap analysis results"}
                        },
                        "required": ["gap_analysis"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "iteration_planning",
                    "description": "Plan next iteration based on progress analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "progress_data": {"type": "string", "description": "Current progress data"}
                        },
                        "required": ["progress_data"]
                    }
                }
            }
        ])
    
    async def analyze_research_progress(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze current research progress and identify gaps for next iteration
        
        Args:
            context: Research context including session data, results, and progress
            
        Returns:
            Dictionary containing analysis results and recommendations
        """
        logger.info(f"Analyzing research progress for session {context.get('session_id')}")
        
        try:
            # Prepare analysis input
            analysis_input = {
                'session_info': {
                    'session_id': context.get('session_id'),
                    'topic': context.get('topic'),
                    'iteration': context.get('iteration', 0),
                    'timestamp': datetime.now().isoformat()
                },
                'current_progress': {
                    'completed_queries': context.get('completed_queries', 0),
                    'total_papers': context.get('total_papers', 0),
                    'completion_score': context.get('completion_score', 0.0)
                },
                'recent_results': context.get('processing_results', []),
                'iteration_history': context.get('iteration_summaries', [])
            }
            
            # Process through PromptChain with ReAct reasoning
            result = await self.chain.process_prompt_async(
                json.dumps(analysis_input, indent=2)
            )
            
            # Validate and enhance analysis result
            validated_analysis = self._validate_analysis_response(result)
            
            # Add analysis metadata
            analysis_data = json.loads(validated_analysis)
            analysis_data['analysis_metadata'] = {
                'agent_version': self.config.get('version', '1.0'),
                'analysis_timestamp': datetime.now().isoformat(),
                'session_id': context.get('session_id'),
                'iteration_analyzed': context.get('iteration', 0)
            }
            
            logger.info("ReAct analysis completed successfully")
            
            return analysis_data
            
        except Exception as e:
            logger.error(f"ReAct analysis failed: {e}")
            return self._fallback_analysis(context)
    
    def _validate_analysis_response(self, response: str) -> str:
        """Validate and enhance analysis response"""
        try:
            parsed = json.loads(response)
            
            # Ensure required structure
            required_sections = [
                'analysis_summary', 'gaps_identified', 'coverage_analysis',
                'new_queries', 'iteration_recommendation'
            ]
            
            for section in required_sections:
                if section not in parsed:
                    parsed[section] = self._get_default_section(section)
            
            # Validate analysis summary
            analysis_summary = parsed['analysis_summary']
            if 'completion_score' not in analysis_summary:
                analysis_summary['completion_score'] = 0.5
            elif not (0.0 <= analysis_summary['completion_score'] <= 1.0):
                analysis_summary['completion_score'] = max(0.0, min(1.0, analysis_summary['completion_score']))
            
            # Validate gaps
            gaps = parsed['gaps_identified']
            if not isinstance(gaps, list):
                parsed['gaps_identified'] = []
            else:
                for gap in gaps:
                    if not isinstance(gap, dict):
                        continue
                    # Ensure required gap fields
                    gap.setdefault('gap_type', 'general')
                    gap.setdefault('severity', 'medium')
                    gap.setdefault('description', 'Identified research gap')
            
            # Validate new queries
            new_queries = parsed['new_queries']
            if not isinstance(new_queries, list):
                parsed['new_queries'] = []
            else:
                # Limit to max 8 queries
                parsed['new_queries'] = new_queries[:8]
                
                for query in parsed['new_queries']:
                    if not isinstance(query, dict):
                        continue
                    query.setdefault('text', 'Research query')
                    query.setdefault('priority', 0.5)
                    query.setdefault('category', 'general')
                    
                    # Validate priority range
                    if not (0.0 <= query.get('priority', 0.5) <= 1.0):
                        query['priority'] = 0.5
            
            # Validate iteration recommendation
            iteration_rec = parsed['iteration_recommendation']
            if 'should_continue' not in iteration_rec:
                iteration_rec['should_continue'] = True
            if 'confidence' not in iteration_rec:
                iteration_rec['confidence'] = 0.7
            elif not (0.0 <= iteration_rec['confidence'] <= 1.0):
                iteration_rec['confidence'] = 0.7
            
            return json.dumps(parsed, indent=2)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in analysis response: {e}")
            return self._fallback_analysis_json()
        except Exception as e:
            logger.error(f"Analysis validation failed: {e}")
            return self._fallback_analysis_json()
    
    def _get_default_section(self, section_name: str) -> Dict[str, Any]:
        """Get default structure for missing sections"""
        defaults = {
            'analysis_summary': {
                'current_state': 'Research in progress',
                'completion_score': 0.5,
                'iteration_effectiveness': 'Moderate progress',
                'key_findings': []
            },
            'gaps_identified': [],
            'coverage_analysis': {
                'techniques_covered': [],
                'techniques_missing': [],
                'applications_covered': [],
                'applications_missing': [],
                'temporal_coverage': {
                    'recent_papers': 0,
                    'older_papers': 0,
                    'missing_periods': []
                }
            },
            'new_queries': [],
            'iteration_recommendation': {
                'should_continue': True,
                'confidence': 0.7,
                'focus_areas': [],
                'search_strategy_adjustments': [],
                'expected_improvement': 'Continue research for better coverage'
            }
        }
        return defaults.get(section_name, {})
    
    def _fallback_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback analysis if main analysis fails"""
        logger.warning("Using fallback ReAct analysis")
        
        completion_score = context.get('completion_score', 0.0)
        iteration = context.get('iteration', 0)
        
        return {
            'analysis_summary': {
                'current_state': f'Research in progress (iteration {iteration})',
                'completion_score': completion_score,
                'iteration_effectiveness': 'Analysis failed, using fallback',
                'key_findings': ['Analysis system encountered error']
            },
            'gaps_identified': [
                {
                    'gap_type': 'analysis',
                    'description': 'Unable to perform detailed gap analysis',
                    'severity': 'medium',
                    'evidence': 'ReAct analysis system failure',
                    'impact': 'May miss optimization opportunities'
                }
            ],
            'coverage_analysis': {
                'techniques_covered': [],
                'techniques_missing': ['unknown due to analysis failure'],
                'applications_covered': [],
                'applications_missing': ['unknown due to analysis failure'],
                'temporal_coverage': {
                    'recent_papers': context.get('total_papers', 0),
                    'older_papers': 0,
                    'missing_periods': []
                }
            },
            'new_queries': self._generate_fallback_queries(context),
            'iteration_recommendation': {
                'should_continue': iteration < 4 and completion_score < 0.8,
                'confidence': 0.5,
                'focus_areas': ['broad search'],
                'search_strategy_adjustments': ['increase search diversity'],
                'expected_improvement': 'Basic continuation without detailed analysis'
            },
            'analysis_metadata': {
                'fallback_used': True,
                'analysis_timestamp': datetime.now().isoformat(),
                'session_id': context.get('session_id'),
                'iteration_analyzed': iteration
            }
        }
    
    def _fallback_analysis_json(self) -> str:
        """Generate fallback analysis as JSON string"""
        fallback_data = {
            'analysis_summary': {
                'current_state': 'Research in progress',
                'completion_score': 0.5,
                'iteration_effectiveness': 'Unable to assess',
                'key_findings': []
            },
            'gaps_identified': [],
            'coverage_analysis': {
                'techniques_covered': [],
                'techniques_missing': [],
                'applications_covered': [],
                'applications_missing': [],
                'temporal_coverage': {
                    'recent_papers': 0,
                    'older_papers': 0,
                    'missing_periods': []
                }
            },
            'new_queries': [],
            'iteration_recommendation': {
                'should_continue': True,
                'confidence': 0.5,
                'focus_areas': [],
                'search_strategy_adjustments': [],
                'expected_improvement': 'Continue with basic strategy'
            }
        }
        return json.dumps(fallback_data, indent=2)
    
    def _generate_fallback_queries(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate basic queries when detailed analysis fails"""
        topic = context.get('topic', 'research topic')
        iteration = context.get('iteration', 0)
        
        # Basic query templates based on iteration
        if iteration == 1:
            templates = [
                f"What are the recent advances in {topic}?",
                f"What are the main challenges in {topic}?",
                f"What datasets are used for {topic}?"
            ]
        elif iteration == 2:
            templates = [
                f"What are alternative approaches to {topic}?",
                f"How is {topic} evaluated?",
                f"What are the applications of {topic}?"
            ]
        else:
            templates = [
                f"What are the limitations of current {topic} methods?",
                f"What are emerging trends in {topic}?",
                f"What are the future directions for {topic}?"
            ]
        
        queries = []
        for i, template in enumerate(templates):
            queries.append({
                'text': template,
                'priority': 0.7 - (i * 0.1),
                'category': 'fallback',
                'rationale': 'Generated due to analysis failure',
                'target_papers': 10
            })
        
        return queries
    
    def extract_gaps(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Extract gap descriptions from analysis result"""
        try:
            gaps = analysis_result.get('gaps_identified', [])
            return [gap.get('description', '') for gap in gaps if isinstance(gap, dict)]
        except Exception as e:
            logger.error(f"Failed to extract gaps: {e}")
            return []
    
    def extract_new_queries(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract new queries from analysis result"""
        try:
            queries = analysis_result.get('new_queries', [])
            return [q for q in queries if isinstance(q, dict) and q.get('text')]
        except Exception as e:
            logger.error(f"Failed to extract new queries: {e}")
            return []
    
    def should_continue_iteration(self, analysis_result: Dict[str, Any]) -> bool:
        """Determine if iteration should continue based on analysis"""
        try:
            iteration_rec = analysis_result.get('iteration_recommendation', {})
            return iteration_rec.get('should_continue', False)
        except Exception as e:
            logger.error(f"Failed to determine continuation: {e}")
            return False
    
    def get_completion_score(self, analysis_result: Dict[str, Any]) -> float:
        """Extract completion score from analysis"""
        try:
            analysis_summary = analysis_result.get('analysis_summary', {})
            return analysis_summary.get('completion_score', 0.5)
        except Exception as e:
            logger.error(f"Failed to get completion score: {e}")
            return 0.5
    
    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration"""
        return self.config.copy()
    
    def update_config(self, config: Dict[str, Any]):
        """Update agent configuration"""
        self.config.update(config)
        logger.info("ReActAnalysisAgent configuration updated")