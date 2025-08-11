"""
Synthesis Agent

Synthesizes comprehensive literature reviews from multi-query processing results.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict, Counter

from promptchain import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

logger = logging.getLogger(__name__)


class SynthesisAgent:
    """
    Agent that synthesizes comprehensive literature reviews from research results
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize PromptChain for synthesis
        self.chain = PromptChain(
            models=[config.get('model', 'openai/gpt-4o')],
            instructions=[
                "You are a literature review synthesis specialist. Create comprehensive reviews from: {research_context}",
                AgenticStepProcessor(
                    objective=config.get('processor', {}).get('objective', 
                               "Synthesize comprehensive literature review with statistics and insights"),
                    max_internal_steps=config.get('processor', {}).get('max_internal_steps', 8)
                ),
                self._format_synthesis_output_instruction()
            ],
            verbose=True
        )
        
        # Register synthesis tools
        self._register_tools()
        
        logger.info("SynthesisAgent initialized")
    
    def _format_synthesis_output_instruction(self) -> str:
        """Instruction for formatting synthesis output"""
        return """
        Format your synthesis as a JSON object with the following structure:
        {
            "literature_review": {
                "executive_summary": {
                    "overview": "Brief overview of the research area",
                    "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
                    "research_gaps": ["Gap 1", "Gap 2"],
                    "future_directions": ["Direction 1", "Direction 2"]
                },
                "sections": {
                    "introduction": {
                        "content": "Comprehensive introduction to the field...",
                        "citations": [{"paper_id": "id1", "context": "Supporting statement"}]
                    },
                    "methodology_review": {
                        "content": "Analysis of methodological approaches...",
                        "subsections": {
                            "traditional_approaches": "Content...",
                            "modern_techniques": "Content...",
                            "comparative_analysis": "Content..."
                        },
                        "citations": [{"paper_id": "id2", "context": "Methodology reference"}]
                    },
                    "applications_and_use_cases": {
                        "content": "Review of applications across domains...",
                        "subsections": {
                            "domain_specific_applications": "Content...",
                            "cross_domain_applications": "Content...",
                            "emerging_applications": "Content..."
                        }
                    },
                    "evaluation_and_benchmarks": {
                        "content": "Analysis of evaluation methodologies...",
                        "benchmarks_table": [
                            {"dataset": "Dataset1", "metrics": ["metric1", "metric2"], "papers_count": 5}
                        ]
                    },
                    "challenges_and_limitations": {
                        "content": "Discussion of current challenges...",
                        "limitation_categories": [
                            {"category": "Technical", "limitations": ["Limitation 1", "Limitation 2"]},
                            {"category": "Practical", "limitations": ["Limitation 3"]}
                        ]
                    },
                    "future_research_directions": {
                        "content": "Identification of future opportunities...",
                        "research_priorities": [
                            {"priority": "High", "area": "Research Area 1", "rationale": "Why important"},
                            {"priority": "Medium", "area": "Research Area 2", "rationale": "Justification"}
                        ]
                    }
                }
            },
            "statistics": {
                "paper_statistics": {
                    "total_papers": 50,
                    "papers_by_year": {"2020": 5, "2021": 10, "2022": 15, "2023": 15, "2024": 5},
                    "papers_by_source": {"arxiv": 20, "journals": 25, "conferences": 5},
                    "top_authors": [{"author": "Author Name", "paper_count": 3}],
                    "top_venues": [{"venue": "Venue Name", "paper_count": 5}]
                },
                "research_coverage": {
                    "techniques_covered": 15,
                    "applications_covered": 8,
                    "datasets_mentioned": 12,
                    "evaluation_metrics": 10
                },
                "query_statistics": {
                    "total_queries_processed": 12,
                    "successful_queries": 10,
                    "queries_by_category": {"methodology": 4, "applications": 3, "evaluation": 3}
                },
                "synthesis_metrics": {
                    "total_processing_time": 45.5,
                    "papers_per_section": {"methodology": 25, "applications": 30, "evaluation": 20},
                    "cross_references": 15,
                    "unique_insights": 8
                }
            },
            "visualizations": {
                "charts_data": [
                    {
                        "type": "timeline",
                        "title": "Research Evolution Over Time",
                        "data": [{"year": 2020, "papers": 5, "key_developments": ["Development 1"]}]
                    },
                    {
                        "type": "network",
                        "title": "Author Collaboration Network",
                        "nodes": [{"id": "author1", "papers": 3}],
                        "edges": [{"source": "author1", "target": "author2", "collaborations": 2}]
                    },
                    {
                        "type": "heatmap",
                        "title": "Research Focus Areas",
                        "data": [{"technique": "Method1", "application": "Domain1", "papers": 5}]
                    }
                ]
            },
            "citations": {
                "bibliography": [
                    {
                        "paper_id": "id1",
                        "title": "Paper Title",
                        "authors": ["Author 1", "Author 2"],
                        "venue": "Conference/Journal",
                        "year": 2023,
                        "doi": "10.1000/example",
                        "key_contributions": ["Contribution 1", "Contribution 2"]
                    }
                ],
                "citation_network": {
                    "most_cited": [{"paper_id": "id1", "citation_count": 15}],
                    "citation_clusters": [{"cluster": "Methodology", "papers": ["id1", "id2"]}]
                }
            },
            "recommendations": {
                "for_researchers": [
                    "Focus on addressing Gap 1 using Method X",
                    "Explore applications in Domain Y"
                ],
                "for_practitioners": [
                    "Consider implementing Technique Z for Problem A",
                    "Evaluate using Benchmark B for your use case"
                ],
                "research_priorities": [
                    {
                        "priority": 1,
                        "area": "Scalability improvements",
                        "expected_impact": "High",
                        "timeline": "2-3 years"
                    }
                ]
            }
        }
        
        Guidelines:
        1. Each section should be comprehensive (300-500 words)
        2. Include proper academic writing style and transitions
        3. Cite papers appropriately throughout
        4. Provide quantitative insights where possible
        5. Identify genuine research gaps and opportunities
        6. Create actionable recommendations
        7. Ensure logical flow between sections
        """
    
    def _register_tools(self):
        """Register synthesis tools"""
        
        def literature_structuring(research_context: str) -> str:
            """Guide structuring of literature review content"""
            structuring_guide = """
            Literature Review Structuring Framework:
            
            INTRODUCTION SECTION:
            - Define the research area and scope
            - Establish importance and motivation
            - Preview key themes and findings
            - Outline review methodology
            
            METHODOLOGY REVIEW:
            - Categorize approaches chronologically
            - Compare and contrast techniques
            - Analyze evolution of methods
            - Identify methodological trends
            
            APPLICATIONS SECTION:
            - Group by application domains
            - Highlight successful implementations
            - Analyze domain-specific adaptations
            - Identify cross-domain opportunities
            
            EVALUATION ANALYSIS:
            - Standard benchmarks and datasets
            - Evaluation metrics and protocols
            - Comparative performance analysis
            - Reproducibility considerations
            
            CHALLENGES SECTION:
            - Technical limitations
            - Practical implementation issues
            - Scalability concerns
            - Ethical considerations
            
            FUTURE DIRECTIONS:
            - Emerging trends and opportunities
            - Technological enablers
            - Research gaps to address
            - Long-term vision
            """
            return structuring_guide
        
        def citation_management(papers_data: str) -> str:
            """Manage citations and references throughout review"""
            citation_guide = """
            Citation Management Strategy:
            
            CITATION PRINCIPLES:
            - Support all claims with appropriate references
            - Use recent papers for current state
            - Include foundational works for background
            - Distribute citations evenly across sections
            
            CITATION FORMATS:
            - Primary claims: (Author et al., Year)
            - Multiple supporting: (Author1 et al., Year1; Author2 et al., Year2)
            - Direct quotes: "Quote text" (Author et al., Year, p. X)
            - Comparative: While Author1 found X, Author2 demonstrated Y
            
            CITATION STRATEGY:
            - 2-3 citations per major claim
            - Mix of journal and conference papers
            - Balance between recent and foundational
            - Include diverse research groups
            
            REFERENCE QUALITY:
            - Prioritize peer-reviewed sources
            - Check citation counts and impact
            - Verify methodological rigor
            - Ensure source diversity
            """
            return citation_guide
        
        def statistical_analysis(processing_results: str) -> str:
            """Generate statistical insights from research data"""
            analysis_guide = """
            Statistical Analysis Framework:
            
            TEMPORAL ANALYSIS:
            - Papers published per year
            - Research trend identification
            - Growth rate calculations
            - Future projections
            
            SOURCE ANALYSIS:
            - Distribution across venues
            - Journal vs conference breakdown
            - Preprint vs published ratios
            - Geographic distribution
            
            AUTHORSHIP ANALYSIS:
            - Most prolific authors
            - Collaboration patterns
            - Institution rankings
            - Research group identification
            
            CONTENT ANALYSIS:
            - Technique popularity trends
            - Application domain coverage
            - Evaluation metric usage
            - Dataset adoption rates
            
            IMPACT METRICS:
            - Citation count distributions
            - H-index calculations
            - Influence measurements
            - Network centrality metrics
            """
            return analysis_guide
        
        def visualization_planning(synthesis_data: str) -> str:
            """Plan effective visualizations for literature review"""
            visualization_guide = """
            Visualization Planning Framework:
            
            TIMELINE VISUALIZATIONS:
            - Research evolution over time
            - Technique development history
            - Publication trends by venue
            - Breakthrough moments identification
            
            NETWORK VISUALIZATIONS:
            - Author collaboration networks
            - Citation networks
            - Concept relationship maps
            - Institution partnerships
            
            DISTRIBUTION VISUALIZATIONS:
            - Performance metric distributions
            - Dataset usage frequencies
            - Technique adoption rates
            - Geographic research spread
            
            COMPARATIVE VISUALIZATIONS:
            - Method performance comparisons
            - Benchmark result tables
            - Evaluation metric heatmaps
            - Cross-domain application matrices
            
            TREND VISUALIZATIONS:
            - Research focus shifts
            - Emerging topic identification
            - Declining area analysis
            - Future direction indicators
            """
            return visualization_guide
        
        def insight_generation(comprehensive_data: str) -> str:
            """Generate novel insights from literature analysis"""
            insight_guide = """
            Insight Generation Framework:
            
            PATTERN IDENTIFICATION:
            - Recurring themes across papers
            - Common methodological choices
            - Consistent evaluation approaches
            - Shared limitations and challenges
            
            GAP ANALYSIS:
            - Underexplored research areas
            - Missing comparative studies
            - Insufficient evaluation dimensions
            - Limited real-world applications
            
            TREND ANALYSIS:
            - Emerging research directions
            - Shifting paradigms
            - Technology adoption patterns
            - Community consensus evolution
            
            SYNTHESIS INSIGHTS:
            - Cross-paper connections
            - Implicit assumptions
            - Methodological biases
            - Evaluation blindspots
            
            FUTURE PREDICTIONS:
            - Logical next steps
            - Technology enablers
            - Market drivers
            - Research catalysts
            """
            return insight_guide
        
        # Register tools
        self.chain.register_tool_function(literature_structuring)
        self.chain.register_tool_function(citation_management)
        self.chain.register_tool_function(statistical_analysis)
        self.chain.register_tool_function(visualization_planning)
        self.chain.register_tool_function(insight_generation)
        
        # Add tool schemas
        self.chain.add_tools([
            {
                "type": "function",
                "function": {
                    "name": "literature_structuring",
                    "description": "Guide structuring of comprehensive literature review",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "research_context": {"type": "string", "description": "Research context for structuring"}
                        },
                        "required": ["research_context"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "citation_management",
                    "description": "Manage citations and references throughout review",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "papers_data": {"type": "string", "description": "Paper data for citation management"}
                        },
                        "required": ["papers_data"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "statistical_analysis",
                    "description": "Generate statistical insights from research data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "processing_results": {"type": "string", "description": "Processing results for analysis"}
                        },
                        "required": ["processing_results"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "visualization_planning",
                    "description": "Plan effective visualizations for literature review",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "synthesis_data": {"type": "string", "description": "Data for visualization planning"}
                        },
                        "required": ["synthesis_data"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "insight_generation",
                    "description": "Generate novel insights from literature analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "comprehensive_data": {"type": "string", "description": "Comprehensive data for insight generation"}
                        },
                        "required": ["comprehensive_data"]
                    }
                }
            }
        ])
    
    async def synthesize_literature_review(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesize comprehensive literature review from research context
        
        Args:
            context: Research context including papers, queries, results, and statistics
            
        Returns:
            Comprehensive literature review dictionary
        """
        logger.info(f"Synthesizing literature review for session {context.get('session_id')}")
        
        try:
            # Prepare synthesis input
            synthesis_input = {
                'session_info': {
                    'session_id': context.get('session_id'),
                    'topic': context.get('topic'),
                    'synthesis_timestamp': datetime.now().isoformat()
                },
                'research_data': {
                    'queries': context.get('queries', []),
                    'papers': context.get('papers', []),
                    'processing_results': context.get('processing_results', []),
                    'iteration_summaries': context.get('iterations', [])
                },
                'statistics': context.get('statistics', {}),
                'synthesis_requirements': {
                    'target_length': self.config.get('target_length', 'comprehensive'),
                    'academic_style': self.config.get('academic_style', True),
                    'include_visualizations': self.config.get('include_visualizations', True),
                    'citation_style': self.config.get('citation_style', 'academic')
                }
            }
            
            # Process through PromptChain
            result = await self.chain.process_prompt_async(
                json.dumps(synthesis_input, indent=2)
            )
            
            # Validate and enhance synthesis
            validated_synthesis = self._validate_synthesis_response(result, context)
            
            # Add synthesis metadata
            synthesis_data = json.loads(validated_synthesis)
            synthesis_data['synthesis_metadata'] = {
                'agent_version': self.config.get('version', '1.0'),
                'synthesis_timestamp': datetime.now().isoformat(),
                'session_id': context.get('session_id'),
                'total_papers_analyzed': len(context.get('papers', [])),
                'total_queries_processed': len(context.get('queries', [])),
                'processing_time': 0.0  # Would be calculated
            }
            
            logger.info("Literature review synthesis completed successfully")
            
            return synthesis_data
            
        except Exception as e:
            logger.error(f"Literature review synthesis failed: {e}")
            return self._fallback_synthesis(context)
    
    def _validate_synthesis_response(self, response: str, context: Dict[str, Any]) -> str:
        """Validate and enhance synthesis response"""
        try:
            parsed = json.loads(response)
            
            # Ensure required structure
            required_sections = [
                'literature_review', 'statistics', 'visualizations', 
                'citations', 'recommendations'
            ]
            
            for section in required_sections:
                if section not in parsed:
                    parsed[section] = self._get_default_synthesis_section(section, context)
            
            # Validate literature review structure
            lit_review = parsed['literature_review']
            if 'executive_summary' not in lit_review:
                lit_review['executive_summary'] = self._generate_executive_summary(context)
            
            if 'sections' not in lit_review:
                lit_review['sections'] = self._generate_default_sections(context)
            
            # Validate statistics
            if 'paper_statistics' not in parsed['statistics']:
                parsed['statistics']['paper_statistics'] = self._calculate_paper_statistics(context)
            
            # Validate citations
            if 'bibliography' not in parsed['citations']:
                parsed['citations']['bibliography'] = self._generate_bibliography(context)
            
            return json.dumps(parsed, indent=2)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in synthesis response: {e}")
            return self._fallback_synthesis_json(context)
        except Exception as e:
            logger.error(f"Synthesis validation failed: {e}")
            return self._fallback_synthesis_json(context)
    
    def _get_default_synthesis_section(self, section_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get default structure for missing synthesis sections"""
        topic = context.get('topic', 'research topic')
        papers = context.get('papers', [])
        
        defaults = {
            'literature_review': {
                'executive_summary': {
                    'overview': f"This literature review examines {len(papers)} papers on {topic}.",
                    'key_findings': [f"Key finding about {topic}", "Important research trend identified"],
                    'research_gaps': ["Research gap identified through analysis"],
                    'future_directions': ["Promising future research direction"]
                },
                'sections': self._generate_default_sections(context)
            },
            'statistics': {
                'paper_statistics': self._calculate_paper_statistics(context),
                'research_coverage': {
                    'techniques_covered': 0,
                    'applications_covered': 0,
                    'datasets_mentioned': 0
                }
            },
            'visualizations': {
                'charts_data': []
            },
            'citations': {
                'bibliography': self._generate_bibliography(context)
            },
            'recommendations': {
                'for_researchers': [f"Continue investigating {topic}"],
                'for_practitioners': [f"Consider implementing {topic} solutions"]
            }
        }
        
        return defaults.get(section_name, {})
    
    def _generate_executive_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary from context"""
        topic = context.get('topic', 'research topic')
        papers = context.get('papers', [])
        
        return {
            'overview': f"This comprehensive literature review analyzes {len(papers)} papers in the field of {topic}, providing insights into current methodologies, applications, and future research directions.",
            'key_findings': [
                f"Significant progress has been made in {topic} research",
                "Multiple methodological approaches show promise",
                "Applications span diverse domains with varying success rates"
            ],
            'research_gaps': [
                "Limited comparative evaluation across different approaches",
                "Insufficient real-world deployment studies",
                "Need for standardized evaluation benchmarks"
            ],
            'future_directions': [
                f"Integration of emerging technologies with {topic}",
                "Development of more robust evaluation frameworks",
                "Expansion to underexplored application domains"
            ]
        }
    
    def _generate_default_sections(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate default literature review sections"""
        topic = context.get('topic', 'research topic')
        
        return {
            'introduction': {
                'content': f"The field of {topic} has experienced significant growth in recent years, driven by advances in technology and increasing practical applications. This review synthesizes current research to provide a comprehensive overview of the state-of-the-art, identify key trends, and highlight promising future directions.",
                'citations': []
            },
            'methodology_review': {
                'content': f"Current approaches to {topic} encompass a diverse range of methodologies, each with distinct advantages and limitations. This section analyzes the evolution of techniques and compares their effectiveness across different scenarios.",
                'subsections': {
                    'traditional_approaches': f"Traditional methods in {topic} have established foundational principles...",
                    'modern_techniques': f"Recent advances have introduced sophisticated approaches...",
                    'comparative_analysis': "Comparative studies reveal varying performance characteristics..."
                },
                'citations': []
            },
            'applications_and_use_cases': {
                'content': f"The practical applications of {topic} span multiple domains, demonstrating versatility and real-world impact. This section examines successful implementations and emerging use cases.",
                'subsections': {
                    'domain_specific_applications': f"Applications within specific domains show {topic}'s targeted effectiveness...",
                    'cross_domain_applications': "Cross-domain implementations reveal broader applicability...",
                    'emerging_applications': "Emerging applications indicate future potential..."
                },
                'citations': []
            },
            'evaluation_and_benchmarks': {
                'content': f"Evaluation methodologies for {topic} vary significantly across studies, making comparative analysis challenging. This section reviews common evaluation approaches and identifies standardization opportunities.",
                'benchmarks_table': [],
                'citations': []
            },
            'challenges_and_limitations': {
                'content': f"Despite significant progress, {topic} faces several persistent challenges that limit broader adoption and effectiveness. This section catalogues current limitations and ongoing research efforts to address them.",
                'limitation_categories': [
                    {'category': 'Technical', 'limitations': ['Technical challenge 1', 'Technical challenge 2']},
                    {'category': 'Practical', 'limitations': ['Practical limitation 1']}
                ],
                'citations': []
            },
            'future_research_directions': {
                'content': f"The future of {topic} research holds significant promise, with several emerging trends and technological enablers creating new opportunities. This section identifies high-priority research areas and expected developments.",
                'research_priorities': [
                    {'priority': 'High', 'area': f'Advanced {topic} techniques', 'rationale': 'Critical for field advancement'},
                    {'priority': 'Medium', 'area': 'Evaluation standardization', 'rationale': 'Important for comparative research'}
                ],
                'citations': []
            }
        }
    
    def _calculate_paper_statistics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics from papers data"""
        papers = context.get('papers', [])
        
        if not papers:
            return {
                'total_papers': 0,
                'papers_by_year': {},
                'papers_by_source': {},
                'top_authors': [],
                'top_venues': []
            }
        
        # Calculate year distribution
        years = [p.get('publication_year') for p in papers if p.get('publication_year')]
        year_counts = Counter(years)
        
        # Calculate source distribution
        sources = [p.get('source', 'unknown') for p in papers]
        source_counts = Counter(sources)
        
        # Calculate author statistics
        all_authors = []
        for paper in papers:
            authors = paper.get('authors', [])
            if isinstance(authors, list):
                all_authors.extend(authors)
        author_counts = Counter(all_authors)
        
        # Calculate venue statistics
        venues = [p.get('journal', p.get('venue', 'Unknown')) for p in papers]
        venue_counts = Counter(venues)
        
        return {
            'total_papers': len(papers),
            'papers_by_year': dict(year_counts),
            'papers_by_source': dict(source_counts),
            'top_authors': [{'author': author, 'paper_count': count} 
                           for author, count in author_counts.most_common(10)],
            'top_venues': [{'venue': venue, 'paper_count': count} 
                          for venue, count in venue_counts.most_common(10)]
        }
    
    def _generate_bibliography(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate bibliography from papers"""
        papers = context.get('papers', [])
        
        bibliography = []
        for paper in papers:
            bib_entry = {
                'paper_id': paper.get('id', 'unknown'),
                'title': paper.get('title', 'Untitled'),
                'authors': paper.get('authors', []),
                'venue': paper.get('journal', paper.get('venue', 'Unknown Venue')),
                'year': paper.get('publication_year', 'Unknown'),
                'doi': paper.get('doi'),
                'key_contributions': [
                    'Key contribution identified from paper analysis',
                    'Additional contribution noted'
                ]
            }
            bibliography.append(bib_entry)
        
        return bibliography
    
    def _fallback_synthesis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback synthesis if main synthesis fails"""
        logger.warning("Using fallback literature review synthesis")
        
        topic = context.get('topic', 'research topic')
        papers = context.get('papers', [])
        
        return {
            'literature_review': {
                'executive_summary': {
                    'overview': f"Fallback synthesis for {len(papers)} papers on {topic}",
                    'key_findings': ["Analysis system encountered error"],
                    'research_gaps': ["Unable to perform detailed gap analysis"],
                    'future_directions': ["Manual review recommended"]
                },
                'sections': {
                    'introduction': {
                        'content': f"This is a basic synthesis of research on {topic} generated due to system limitations.",
                        'citations': []
                    }
                }
            },
            'statistics': {
                'paper_statistics': self._calculate_paper_statistics(context),
                'synthesis_metrics': {
                    'fallback_used': True,
                    'total_papers': len(papers)
                }
            },
            'visualizations': {'charts_data': []},
            'citations': {'bibliography': self._generate_bibliography(context)},
            'recommendations': {
                'for_researchers': ["Manual literature review recommended due to synthesis error"],
                'for_practitioners': ["Consult individual papers directly"]
            }
        }
    
    def _fallback_synthesis_json(self, context: Dict[str, Any]) -> str:
        """Generate fallback synthesis as JSON string"""
        return json.dumps(self._fallback_synthesis(context), indent=2)
    
    def generate_summary_report(self, synthesis_data: Dict[str, Any]) -> str:
        """Generate a concise summary report from synthesis data"""
        try:
            lit_review = synthesis_data.get('literature_review', {})
            stats = synthesis_data.get('statistics', {})
            
            # Extract key information
            exec_summary = lit_review.get('executive_summary', {})
            paper_stats = stats.get('paper_statistics', {})
            
            summary = f"""
# Literature Review Summary

## Overview
{exec_summary.get('overview', 'No overview available')}

## Key Statistics
- Total Papers Analyzed: {paper_stats.get('total_papers', 0)}
- Publication Years: {', '.join(map(str, paper_stats.get('papers_by_year', {}).keys()))}
- Sources: {', '.join(paper_stats.get('papers_by_source', {}).keys())}

## Key Findings
{chr(10).join(f'- {finding}' for finding in exec_summary.get('key_findings', []))}

## Research Gaps Identified
{chr(10).join(f'- {gap}' for gap in exec_summary.get('research_gaps', []))}

## Future Research Directions
{chr(10).join(f'- {direction}' for direction in exec_summary.get('future_directions', []))}
            """.strip()
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")
            return f"Summary generation failed: {e}"
    
    def extract_key_insights(self, synthesis_data: Dict[str, Any]) -> List[str]:
        """Extract key insights from synthesis data"""
        try:
            insights = []
            
            # Extract from executive summary
            exec_summary = synthesis_data.get('literature_review', {}).get('executive_summary', {})
            insights.extend(exec_summary.get('key_findings', []))
            
            # Extract from statistics
            stats = synthesis_data.get('statistics', {})
            paper_stats = stats.get('paper_statistics', {})
            
            if paper_stats.get('total_papers', 0) > 0:
                insights.append(f"Analysis covers {paper_stats['total_papers']} papers")
            
            # Extract from recommendations
            recommendations = synthesis_data.get('recommendations', {})
            insights.extend(recommendations.get('for_researchers', [])[:2])  # Top 2 recommendations
            
            return insights[:10]  # Return top 10 insights
            
        except Exception as e:
            logger.error(f"Failed to extract key insights: {e}")
            return [f"Insight extraction failed: {e}"]
    
    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration"""
        return self.config.copy()
    
    def update_config(self, config: Dict[str, Any]):
        """Update agent configuration"""
        self.config.update(config)
        logger.info("SynthesisAgent configuration updated")