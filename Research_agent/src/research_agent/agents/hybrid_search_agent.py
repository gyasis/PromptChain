#!/usr/bin/env python3
"""
Intelligent Hybrid Search Agent
Combines LightRAG corpus knowledge with web search using ReACT-style reasoning
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

# PromptChain imports
from promptchain import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

# LightRAG imports
from lightrag import LightRAG, QueryParam

# Research Agent tools
from research_agent.tools import web_search_tool, WEB_SEARCH_AVAILABLE

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Structured search result with source attribution"""
    content: str
    source_type: str  # "corpus", "web", "hybrid"
    confidence: float  # 0.0 to 1.0
    citations: List[str]
    metadata: Dict[str, Any]
    
@dataclass
class CorpusAnalysis:
    """Analysis of LightRAG corpus results"""
    completeness_score: float  # 0.0 to 1.0
    knowledge_gaps: List[str]
    temporal_coverage: str  # "current", "outdated", "mixed"
    recommendation: str  # "sufficient", "needs_web_search", "inconclusive"
    reasoning: str

@dataclass
class WebSearchDecision:
    """Decision about whether and how to search the web"""
    should_search: bool
    search_queries: List[str]
    reasoning: str
    confidence: float

class CorpusAnalyzer:
    """Analyzes LightRAG corpus results for completeness and quality"""
    
    def __init__(self, model_name: str = "openai/gpt-4o-mini"):
        self.model_name = model_name
        
        # Analysis chain for evaluating corpus results
        self.analysis_chain = PromptChain(
            models=[model_name],
            instructions=[
                """Analyze the following research question and corpus results to determine completeness and quality.

Question: {question}
Corpus Results: {corpus_results}

Evaluate:
1. COMPLETENESS (0.0-1.0): How well do the results answer the question?
2. KNOWLEDGE GAPS: What specific information is missing or unclear?
3. TEMPORAL COVERAGE: Are the results current, outdated, or mixed?
4. RECOMMENDATION: sufficient | needs_web_search | inconclusive

Provide analysis in this JSON format:
{{
    "completeness_score": 0.0-1.0,
    "knowledge_gaps": ["gap1", "gap2"],
    "temporal_coverage": "current|outdated|mixed",
    "recommendation": "sufficient|needs_web_search|inconclusive",
    "reasoning": "detailed explanation of analysis"
}}"""
            ],
            verbose=False
        )
    
    def analyze_corpus_results(self, question: str, corpus_results: str) -> CorpusAnalysis:
        """Analyze corpus results for completeness and quality"""
        try:
            # Get analysis from the chain
            analysis_json = self.analysis_chain.process_prompt(
                question=question,
                corpus_results=corpus_results
            )
            
            # Parse JSON response
            analysis_data = json.loads(analysis_json)
            
            return CorpusAnalysis(
                completeness_score=analysis_data.get("completeness_score", 0.0),
                knowledge_gaps=analysis_data.get("knowledge_gaps", []),
                temporal_coverage=analysis_data.get("temporal_coverage", "mixed"),
                recommendation=analysis_data.get("recommendation", "inconclusive"),
                reasoning=analysis_data.get("reasoning", "Analysis failed")
            )
            
        except Exception as e:
            logger.error(f"Corpus analysis failed: {e}")
            return CorpusAnalysis(
                completeness_score=0.0,
                knowledge_gaps=["Analysis error occurred"],
                temporal_coverage="unknown",
                recommendation="inconclusive",
                reasoning=f"Analysis failed: {str(e)}"
            )

class WebSearchDecisionMaker:
    """Makes intelligent decisions about web search necessity using ReACT reasoning"""
    
    def __init__(self, model_name: str = "openai/gpt-4o-mini"):
        self.model_name = model_name
        
        # ReACT-style decision processor
        self.decision_processor = AgenticStepProcessor(
            objective="""
            Based on the research question, corpus analysis, and search context, decide whether web search is needed.
            
            Consider:
            1. Completeness of corpus results
            2. Temporal relevance (recent events, current technology)
            3. Knowledge gaps identified
            4. Question context and keywords
            
            Generate targeted web search queries if needed.
            """,
            max_internal_steps=3,
            model_name=model_name
        )
    
    async def decide_web_search(self, question: str, corpus_analysis: CorpusAnalysis) -> WebSearchDecision:
        """Make intelligent decision about web search necessity"""
        
        # Prepare context for decision making
        context = f"""
        RESEARCH QUESTION: {question}
        
        CORPUS ANALYSIS:
        - Completeness Score: {corpus_analysis.completeness_score}
        - Knowledge Gaps: {', '.join(corpus_analysis.knowledge_gaps)}
        - Temporal Coverage: {corpus_analysis.temporal_coverage}
        - Recommendation: {corpus_analysis.recommendation}
        - Reasoning: {corpus_analysis.reasoning}
        
        WEB SEARCH AVAILABLE: {WEB_SEARCH_AVAILABLE}
        
        DECISION CRITERIA:
        - If completeness_score < 0.7, consider web search
        - If temporal keywords present (recent, latest, 2024, current), lean toward web search
        - If knowledge gaps include recent developments, prioritize web search
        - If corpus recommendation is "needs_web_search", strongly consider it
        
        Provide decision in JSON format:
        {{
            "should_search": true/false,
            "search_queries": ["query1", "query2"],
            "reasoning": "detailed explanation",
            "confidence": 0.0-1.0
        }}
        """
        
        try:
            # Use AgenticStepProcessor for reasoning
            decision_result = await self.decision_processor.process_async(context)
            
            # Extract JSON from the result
            import re
            json_match = re.search(r'\{.*\}', decision_result, re.DOTALL)
            if json_match:
                decision_data = json.loads(json_match.group(0))
                
                return WebSearchDecision(
                    should_search=decision_data.get("should_search", False),
                    search_queries=decision_data.get("search_queries", []),
                    reasoning=decision_data.get("reasoning", "Decision reasoning not provided"),
                    confidence=decision_data.get("confidence", 0.0)
                )
            else:
                # Fallback decision based on corpus analysis
                should_search = (
                    corpus_analysis.completeness_score < 0.7 or
                    corpus_analysis.recommendation == "needs_web_search" or
                    any(keyword in question.lower() for keyword in ["recent", "latest", "current", "2024", "new"])
                )
                
                return WebSearchDecision(
                    should_search=should_search,
                    search_queries=[question] if should_search else [],
                    reasoning="Fallback decision based on corpus analysis",
                    confidence=0.5
                )
                
        except Exception as e:
            logger.error(f"Web search decision failed: {e}")
            return WebSearchDecision(
                should_search=False,
                search_queries=[],
                reasoning=f"Decision failed: {str(e)}",
                confidence=0.0
            )

class QueryGenerator:
    """Generates targeted web search queries to fill knowledge gaps"""
    
    def __init__(self, model_name: str = "openai/gpt-4o-mini"):
        self.model_name = model_name
        
        self.query_chain = PromptChain(
            models=[model_name],
            instructions=[
                """Generate targeted web search queries to fill the identified knowledge gaps.

Original Question: {question}
Knowledge Gaps: {knowledge_gaps}
Corpus Analysis: {analysis_reasoning}

Create 2-4 specific web search queries that will find the missing information.
Focus on:
- Recent developments and current information
- Specific technologies, companies, or products mentioned
- Concrete data or examples not in the corpus

Provide queries in JSON format:
{{
    "queries": [
        "specific search query 1",
        "specific search query 2",
        "specific search query 3"
    ],
    "reasoning": "explanation of query selection strategy"
}}"""
            ],
            verbose=False
        )
    
    def generate_queries(self, question: str, corpus_analysis: CorpusAnalysis) -> List[str]:
        """Generate targeted web search queries"""
        try:
            # Generate queries using the chain
            query_result = self.query_chain.process_prompt(
                question=question,
                knowledge_gaps=", ".join(corpus_analysis.knowledge_gaps),
                analysis_reasoning=corpus_analysis.reasoning
            )
            
            # Parse JSON response
            query_data = json.loads(query_result)
            return query_data.get("queries", [question])  # Fallback to original question
            
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            # Fallback: generate basic queries
            base_queries = [question]
            
            # Add queries for knowledge gaps
            for gap in corpus_analysis.knowledge_gaps[:2]:  # Limit to 2 gaps
                base_queries.append(f"{gap} recent developments")
            
            return base_queries

class SourceSynthesizer:
    """Combines and attributes results from corpus and web sources"""
    
    def __init__(self, model_name: str = "openai/gpt-4o-mini"):
        self.model_name = model_name
        
        self.synthesis_chain = PromptChain(
            models=[model_name],
            instructions=[
                """Synthesize information from research corpus and web search results into a comprehensive answer.

ORIGINAL QUESTION: {question}

CORPUS RESULTS (Research Database):
{corpus_results}

WEB SEARCH RESULTS (Current Information):
{web_results}

SYNTHESIS REQUIREMENTS:
1. Provide a comprehensive answer that combines both sources
2. Clearly attribute information to sources ("Based on research corpus..." vs "Current web information shows...")
3. Note any limitations: "The current research corpus covers [X], while recent web search reveals [Y]"
4. Highlight where corpus and web information complement or contradict each other
5. Maintain academic rigor while incorporating current developments

ANSWER FORMAT:
- Start with a comprehensive response
- Include clear source attribution throughout
- End with a "Sources Summary" section noting corpus vs. web contributions
- Mention any limitations or areas where more research is needed"""
            ],
            verbose=False
        )
    
    def synthesize_results(self, question: str, corpus_results: str, web_results: str, corpus_analysis: CorpusAnalysis) -> SearchResult:
        """Synthesize corpus and web results into comprehensive answer"""
        try:
            # Synthesize results
            synthesis = self.synthesis_chain.process_prompt(
                question=question,
                corpus_results=corpus_results,
                web_results=web_results
            )
            
            # Create result with metadata
            return SearchResult(
                content=synthesis,
                source_type="hybrid",
                confidence=min(0.9, corpus_analysis.completeness_score + 0.3),  # Boost confidence with web data
                citations=["Research Corpus", "Web Search"],
                metadata={
                    "corpus_completeness": corpus_analysis.completeness_score,
                    "knowledge_gaps_filled": corpus_analysis.knowledge_gaps,
                    "synthesis_timestamp": datetime.now().isoformat(),
                    "temporal_coverage": corpus_analysis.temporal_coverage
                }
            )
            
        except Exception as e:
            logger.error(f"Result synthesis failed: {e}")
            # Return corpus results as fallback
            return SearchResult(
                content=corpus_results,
                source_type="corpus",
                confidence=corpus_analysis.completeness_score,
                citations=["Research Corpus"],
                metadata={"synthesis_error": str(e)}
            )

class HybridSearchAgent:
    """Main orchestrator for intelligent hybrid search using ReACT-style reasoning"""
    
    def __init__(self, lightrag_instance: LightRAG, model_name: str = "openai/gpt-4o-mini"):
        self.lightrag = lightrag_instance
        self.model_name = model_name
        
        # Initialize components
        self.corpus_analyzer = CorpusAnalyzer(model_name)
        self.decision_maker = WebSearchDecisionMaker(model_name)
        self.query_generator = QueryGenerator(model_name)
        self.synthesizer = SourceSynthesizer(model_name)
        
        logger.info("HybridSearchAgent initialized with intelligent ReACT-style reasoning")
    
    async def search(self, question: str, mode: str = "hybrid") -> SearchResult:
        """Execute intelligent hybrid search with ReACT-style reasoning"""
        
        logger.info(f"🔍 Starting hybrid search for: {question}")
        
        # Phase 1: Query corpus and analyze results
        print("📚 Phase 1: Analyzing research corpus...")
        corpus_results = await self._query_corpus(question, mode)
        corpus_analysis = self.corpus_analyzer.analyze_corpus_results(question, corpus_results)
        
        print(f"   Corpus completeness: {corpus_analysis.completeness_score:.2f}")
        print(f"   Recommendation: {corpus_analysis.recommendation}")
        
        # Phase 2: Intelligent decision making about web search
        print("🤔 Phase 2: Making intelligent search decision...")
        web_decision = await self.decision_maker.decide_web_search(question, corpus_analysis)
        
        print(f"   Web search needed: {web_decision.should_search}")
        print(f"   Decision confidence: {web_decision.confidence:.2f}")
        
        # Phase 3: Execute web search if needed
        web_results = ""
        if web_decision.should_search and WEB_SEARCH_AVAILABLE:
            print("🌐 Phase 3: Executing targeted web search...")
            
            # Generate targeted queries if decision maker didn't provide them
            if not web_decision.search_queries:
                web_decision.search_queries = self.query_generator.generate_queries(question, corpus_analysis)
            
            web_results = await self._execute_web_search(web_decision.search_queries)
            print(f"   Web search completed: {len(web_results)} characters retrieved")
            
        elif web_decision.should_search and not WEB_SEARCH_AVAILABLE:
            print("⚠️  Web search recommended but not available (SERPER_API_KEY not configured)")
            web_results = "Web search recommended but not available due to missing API configuration."
        
        # Phase 4: Synthesize results
        print("🔄 Phase 4: Synthesizing comprehensive answer...")
        
        if web_results:
            # Hybrid synthesis
            result = self.synthesizer.synthesize_results(question, corpus_results, web_results, corpus_analysis)
            print("✅ Hybrid analysis complete (corpus + web)")
        else:
            # Corpus-only result
            result = SearchResult(
                content=corpus_results,
                source_type="corpus",
                confidence=corpus_analysis.completeness_score,
                citations=["Research Corpus"],
                metadata={
                    "analysis": corpus_analysis.__dict__,
                    "web_decision": web_decision.__dict__
                }
            )
            print("✅ Corpus-only analysis complete")
        
        return result
    
    async def _query_corpus(self, question: str, mode: str) -> str:
        """Query LightRAG corpus"""
        try:
            # Use async LightRAG query
            query_param = QueryParam(mode=mode)
            result = await self.lightrag.aquery(question, param=query_param)
            return result
        except Exception as e:
            logger.error(f"Corpus query failed: {e}")
            return f"Corpus query failed: {str(e)}"
    
    async def _execute_web_search(self, queries: List[str]) -> str:
        """Execute web search with multiple queries"""
        if not WEB_SEARCH_AVAILABLE:
            return "Web search not available"
        
        try:
            # Execute searches for all queries
            all_results = []
            
            for i, query in enumerate(queries[:3], 1):  # Limit to 3 queries
                print(f"   🔍 Query {i}: {query}")
                search_results = web_search_tool.search_web(query, num_results=2)
                
                for result in search_results:
                    result_text = f"**{result['title']}** ({result['url']})\n{result['text']}\n"
                    all_results.append(result_text)
            
            return "\n---\n".join(all_results)
            
        except Exception as e:
            logger.error(f"Web search execution failed: {e}")
            return f"Web search failed: {str(e)}"
    
    def format_result(self, result: SearchResult) -> str:
        """Format search result for display"""
        output = f"""
🎯 **Comprehensive Research Analysis**

{result.content}

📊 **Analysis Metadata:**
- Source Type: {result.source_type.title()}
- Confidence: {result.confidence:.2f}
- Citations: {', '.join(result.citations)}

"""
        
        if result.metadata:
            output += "🔍 **Research Details:**\n"
            for key, value in result.metadata.items():
                if isinstance(value, (int, float, str, bool)):
                    output += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        return output