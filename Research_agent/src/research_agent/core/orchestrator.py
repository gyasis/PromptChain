"""
Advanced Research Orchestrator

Manages the complete research workflow with multi-query processing and iterative refinement.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from promptchain import PromptChain, AgentChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain.utils.execution_history_manager import ExecutionHistoryManager
from promptchain.utils.logging_utils import RunLogger

from .config import ResearchConfig
from .session import ResearchSession, SessionStatus, Query, Paper, ProcessingResult
from ..agents.query_generator import QueryGenerationAgent
from ..agents.search_strategist import SearchStrategistAgent  
from ..agents.literature_searcher import LiteratureSearchAgent
from ..agents.react_analyzer import ReActAnalysisAgent
from ..agents.synthesis_agent import SynthesisAgent
from ..integrations.multi_query_coordinator import MultiQueryCoordinator
from ..utils.chat_interface import InteractiveChatInterface


logger = logging.getLogger(__name__)


class AdvancedResearchOrchestrator:
    """
    Orchestrates the complete research workflow with iterative multi-query processing
    """
    
    def __init__(self, config: Optional[ResearchConfig] = None):
        self.config = config or ResearchConfig()
        self.run_logger = RunLogger(log_dir=self.config.get('logging.file.directory', './logs'))
        
        # Initialize core components
        self._initialize_agents()
        self._initialize_coordinators()
        
        # State tracking
        self.current_session: Optional[ResearchSession] = None
        self.chat_interface: Optional[InteractiveChatInterface] = None
        
        logger.info("AdvancedResearchOrchestrator initialized")
    
    def _initialize_agents(self):
        """Initialize all research agents"""
        try:
            # Query Generation Agent
            self.query_generator = QueryGenerationAgent(
                config=self.config.get_agent_config('query_generator')
            )
            
            # Search Strategy Agent  
            self.search_strategist = SearchStrategistAgent(
                config=self.config.get_agent_config('search_strategist')
            )
            
            # Literature Search Agent
            self.literature_searcher = LiteratureSearchAgent(
                config=self.config.get_agent_config('literature_searcher')
            )
            
            # ReAct Analysis Agent
            self.react_analyzer = ReActAnalysisAgent(
                config=self.config.get_agent_config('react_analyzer')
            )
            
            # Synthesis Agent
            self.synthesis_agent = SynthesisAgent(
                config=self.config.get_agent_config('synthesis_agent')
            )
            
            logger.info("All research agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            raise
    
    def _initialize_coordinators(self):
        """Initialize coordination components"""
        try:
            # Multi-Query Coordinator
            self.multi_query_coordinator = MultiQueryCoordinator(
                config=self.config
            )
            
            logger.info("Coordinators initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize coordinators: {e}")
            raise
    
    async def conduct_research_session(
        self, 
        research_topic: str,
        session_id: Optional[str] = None,
        callbacks: Optional[List[Callable]] = None
    ) -> ResearchSession:
        """
        Conduct a complete research session with iterative processing
        """
        logger.info(f"Starting research session for topic: '{research_topic}'")
        
        # Initialize session
        session = ResearchSession(
            topic=research_topic,
            session_id=session_id,
            config=self.config.research_session.__dict__
        )
        self.current_session = session
        
        # Log session start
        self.run_logger.log_run({
            'event': 'research_session_started',
            'session_id': session.session_id,
            'topic': research_topic,
            'timestamp': datetime.now().isoformat()
        })
        
        try:
            # Phase 1: Query Generation
            session.set_status(SessionStatus.QUERY_GENERATION)
            await self._phase1_query_generation(session, callbacks)
            
            # Phase 2: Iterative Research Processing
            await self._phase2_iterative_processing(session, callbacks)
            
            # Phase 3: Final Synthesis
            session.set_status(SessionStatus.SYNTHESIS)
            await self._phase3_synthesis(session, callbacks)
            
            # Phase 4: Interactive Chat Setup
            session.set_status(SessionStatus.INTERACTIVE)
            self.chat_interface = await self._phase4_interactive_setup(session)
            
            session.set_status(SessionStatus.COMPLETED)
            logger.info(f"Research session completed: {session.session_id}")
            
            # Log completion
            self.run_logger.log_run({
                'event': 'research_session_completed',
                'session_id': session.session_id,
                'statistics': session.get_session_statistics()
            })
            
            return session
            
        except Exception as e:
            logger.error(f"Research session failed: {e}")
            session.set_status(SessionStatus.ERROR)
            
            self.run_logger.log_run({
                'event': 'research_session_error',
                'session_id': session.session_id,
                'error': str(e)
            })
            
            raise
    
    async def _phase1_query_generation(
        self, 
        session: ResearchSession, 
        callbacks: Optional[List[Callable]] = None
    ):
        """Phase 1: Generate initial research queries"""
        logger.info("Phase 1: Query Generation")
        
        if callbacks:
            for callback in callbacks:
                callback("🔍 Analyzing research topic and generating questions...")
        
        try:
            # Generate comprehensive query set
            query_response = await self.query_generator.generate_queries(
                topic=session.topic,
                context={
                    'max_queries': session.max_queries_per_iteration,
                    'iteration': 0
                }
            )
            
            # Parse and add queries to session
            queries = self._parse_query_response(query_response)
            query_ids = session.add_queries(queries)
            
            logger.info(f"Generated {len(query_ids)} initial queries")
            
            # Log query generation
            self.run_logger.log_run({
                'event': 'queries_generated',
                'session_id': session.session_id,
                'phase': 'initial',
                'query_count': len(query_ids),
                'queries': [q['text'] for q in queries]
            })
            
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            raise
    
    async def _phase2_iterative_processing(
        self, 
        session: ResearchSession, 
        callbacks: Optional[List[Callable]] = None
    ):
        """Phase 2: Iterative research processing with ReAct-style refinement"""
        logger.info("Phase 2: Iterative Processing")
        
        iteration = 0
        while session.should_continue_iteration() and iteration < session.max_iterations:
            iteration = session.start_new_iteration()
            
            logger.info(f"Starting iteration {iteration}")
            
            if callbacks:
                for callback in callbacks:
                    callback(f"📊 Research Iteration {iteration}/5")
            
            try:
                # Step 2.1: Literature Search
                await self._step_literature_search(session, callbacks)
                
                # Step 2.2: Multi-Query Processing
                await self._step_multi_query_processing(session, callbacks)
                
                # Step 2.3: ReAct Analysis
                gaps, new_queries = await self._step_react_analysis(session, callbacks)
                
                # Complete iteration
                session.complete_iteration(gaps=gaps, new_queries=len(new_queries))
                
                # Add new queries if gaps found
                if new_queries and iteration < session.max_iterations:
                    query_ids = session.add_queries(new_queries)
                    logger.info(f"Added {len(query_ids)} new queries for next iteration")
                
                logger.info(f"Completed iteration {iteration}")
                
            except Exception as e:
                logger.error(f"Iteration {iteration} failed: {e}")
                # Continue to next iteration or exit gracefully
                continue
    
    async def _step_literature_search(
        self, 
        session: ResearchSession, 
        callbacks: Optional[List[Callable]] = None
    ):
        """Execute literature search for current queries"""
        active_queries = session.get_active_queries()
        
        if not active_queries:
            return
        
        if callbacks:
            for callback in callbacks:
                callback(f"🔎 Searching literature for {len(active_queries)} queries...")
        
        try:
            # Generate search strategy
            search_strategy = await self.search_strategist.generate_strategy(
                queries=[q.text for q in active_queries],
                existing_papers=list(session.papers.keys()),
                iteration=session.current_iteration
            )
            
            # Execute literature search
            papers = await self.literature_searcher.search_papers(
                strategy=search_strategy,
                max_papers=session.config.get('max_papers_total', 100)
            )
            
            # Add papers to session
            paper_ids = session.add_papers(papers)
            
            logger.info(f"Found {len(paper_ids)} papers in literature search")
            
            # Log search results
            self.run_logger.log_run({
                'event': 'literature_search_completed',
                'session_id': session.session_id,
                'iteration': session.current_iteration,
                'papers_found': len(paper_ids),
                'total_papers': len(session.papers)
            })
            
        except Exception as e:
            logger.error(f"Literature search failed: {e}")
            raise
    
    async def _step_multi_query_processing(
        self, 
        session: ResearchSession, 
        callbacks: Optional[List[Callable]] = None
    ):
        """Process papers across all tiers with multiple queries"""
        active_queries = session.get_active_queries()
        available_papers = list(session.papers.values())
        
        if not active_queries or not available_papers:
            return
        
        if callbacks:
            for callback in callbacks:
                callback(f"⚙️ Processing {len(available_papers)} papers across 3 tiers...")
        
        try:
            # Process papers with all queries across all tiers
            processing_results = await self.multi_query_coordinator.process_papers_with_queries(
                papers=available_papers,
                queries=active_queries,
                session=session
            )
            
            # Store results in session
            for result in processing_results:
                session.add_processing_result(result)
                
                # Mark query as completed if successful
                if result.result_data.get('success', False):
                    session.mark_query_completed(
                        result.query_id, 
                        result.result_data
                    )
                else:
                    session.mark_query_failed(
                        result.query_id,
                        result.result_data.get('error', 'Processing failed')
                    )
            
            logger.info(f"Multi-query processing completed: {len(processing_results)} results")
            
            # Log processing
            self.run_logger.log_run({
                'event': 'multi_query_processing_completed',
                'session_id': session.session_id,
                'iteration': session.current_iteration,
                'results_count': len(processing_results),
                'queries_completed': len(session.completed_queries)
            })
            
        except Exception as e:
            logger.error(f"Multi-query processing failed: {e}")
            raise
    
    async def _step_react_analysis(
        self, 
        session: ResearchSession, 
        callbacks: Optional[List[Callable]] = None
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """Perform ReAct-style analysis to identify gaps and generate new queries"""
        if callbacks:
            for callback in callbacks:
                callback("🧠 Analyzing results and identifying research gaps...")
        
        try:
            # Gather all results for analysis
            analysis_context = {
                'session_id': session.session_id,
                'topic': session.topic,
                'iteration': session.current_iteration,
                'completed_queries': len(session.completed_queries),
                'total_papers': len(session.papers),
                'processing_results': [r.result_data for r in session.processing_results[-10:]],  # Last 10 results
                'completion_score': session.calculate_completion_score()
            }
            
            # Perform ReAct analysis
            analysis_result = await self.react_analyzer.analyze_research_progress(
                context=analysis_context
            )
            
            gaps = analysis_result.get('gaps_identified', [])
            new_queries = analysis_result.get('new_queries', [])
            should_continue = analysis_result.get('should_continue', False)
            
            logger.info(f"ReAct analysis: {len(gaps)} gaps, {len(new_queries)} new queries, continue={should_continue}")
            
            # Log analysis
            self.run_logger.log_run({
                'event': 'react_analysis_completed',
                'session_id': session.session_id,
                'iteration': session.current_iteration,
                'gaps_count': len(gaps),
                'new_queries_count': len(new_queries),
                'should_continue': should_continue,
                'gaps': gaps
            })
            
            return gaps, new_queries
            
        except Exception as e:
            logger.error(f"ReAct analysis failed: {e}")
            return [], []
    
    async def _phase3_synthesis(
        self, 
        session: ResearchSession, 
        callbacks: Optional[List[Callable]] = None
    ):
        """Phase 3: Synthesize comprehensive literature review"""
        logger.info("Phase 3: Literature Review Synthesis")
        
        if callbacks:
            for callback in callbacks:
                callback("📝 Synthesizing comprehensive literature review...")
        
        try:
            # Prepare synthesis context
            synthesis_context = {
                'session_id': session.session_id,
                'topic': session.topic,
                'queries': [q.text for q in session.queries.values()],
                'papers': [{
                    'title': p.title,
                    'authors': p.authors,
                    'abstract': p.abstract,
                    'source': p.source,
                    'year': p.publication_year
                } for p in session.papers.values()],
                'processing_results': [r.result_data for r in session.processing_results],
                'iterations': session.iteration_summaries,
                'statistics': session.get_session_statistics()
            }
            
            # Generate literature review
            literature_review = await self.synthesis_agent.synthesize_literature_review(
                context=synthesis_context
            )
            
            # Store in session
            session.literature_review = literature_review
            
            logger.info("Literature review synthesis completed")
            
            # Log synthesis
            self.run_logger.log_run({
                'event': 'literature_review_synthesized',
                'session_id': session.session_id,
                'sections': list(literature_review.keys()) if literature_review else [],
                'word_count': len(str(literature_review)) if literature_review else 0
            })
            
        except Exception as e:
            logger.error(f"Literature review synthesis failed: {e}")
            raise
    
    async def _phase4_interactive_setup(
        self, 
        session: ResearchSession
    ) -> InteractiveChatInterface:
        """Phase 4: Setup interactive chat interface"""
        logger.info("Phase 4: Interactive Chat Setup")
        
        try:
            # Initialize chat interface
            chat_interface = InteractiveChatInterface(
                session=session,
                config=self.config,
                orchestrator=self
            )
            
            # Add welcome message
            session.add_chat_message(
                role='assistant',
                message=f"Research complete! I've analyzed {len(session.papers)} papers and generated a comprehensive literature review on '{session.topic}'. Ask me anything about the findings or request specific analyses!",
                metadata={
                    'phase': 'interactive_start',
                    'statistics': session.get_session_statistics()
                }
            )
            
            logger.info("Interactive chat interface ready")
            
            return chat_interface
            
        except Exception as e:
            logger.error(f"Interactive setup failed: {e}")
            raise
    
    def _parse_query_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse query generation response into structured format"""
        try:
            # This would be enhanced with actual parsing logic
            # For now, simple placeholder implementation
            import json
            
            # Try to parse as JSON first
            try:
                parsed = json.loads(response)
                if isinstance(parsed, dict) and 'queries' in parsed:
                    return parsed['queries']
                elif isinstance(parsed, list):
                    return [{'text': q, 'priority': 1.0} for q in parsed]
            except:
                pass
            
            # Fallback: split by lines and treat as individual queries
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            return [{'text': line, 'priority': 1.0} for line in lines[:15]]  # Max 15 queries
            
        except Exception as e:
            logger.error(f"Failed to parse query response: {e}")
            return []
    
    async def continue_session(self, session_id: str) -> Optional[ResearchSession]:
        """Continue an existing research session"""
        # This would load from persistent storage
        # For now, return current session if it matches
        if self.current_session and self.current_session.session_id == session_id:
            return self.current_session
        return None
    
    async def get_chat_interface(self, session_id: str) -> Optional[InteractiveChatInterface]:
        """Get chat interface for a session"""
        if self.current_session and self.current_session.session_id == session_id:
            return self.chat_interface
        return None
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get current session statistics"""
        if self.current_session:
            return self.current_session.get_session_statistics()
        return {}
    
    async def shutdown(self):
        """Shutdown orchestrator and cleanup resources"""
        logger.info("Shutting down AdvancedResearchOrchestrator")
        
        # Close any open resources
        if hasattr(self.multi_query_coordinator, 'cleanup'):
            await self.multi_query_coordinator.cleanup()
        
        # Save current session if exists
        if self.current_session:
            try:
                # This would save to persistent storage
                pass
            except Exception as e:
                logger.error(f"Failed to save session on shutdown: {e}")
        
        logger.info("Orchestrator shutdown complete")