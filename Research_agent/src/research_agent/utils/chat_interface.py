"""
Interactive Chat Interface

Provides interactive chat capabilities for research sessions.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum

from promptchain import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

logger = logging.getLogger(__name__)


class ChatMode(Enum):
    """Chat interaction modes"""
    QUESTION_ANSWERING = "qa"
    DEEP_DIVE = "deep_dive"
    COMPARATIVE_ANALYSIS = "compare"
    SUMMARY_GENERATION = "summary"
    CUSTOM_ANALYSIS = "custom"


class InteractiveChatInterface:
    """
    Interactive chat interface for research sessions
    """
    
    def __init__(
        self,
        session: Any,
        config: Dict[str, Any],
        orchestrator: Optional[Any] = None
    ):
        self.session = session
        self.config = config
        self.orchestrator = orchestrator
        self.chat_history: List[Dict[str, Any]] = []
        self.current_mode = ChatMode.QUESTION_ANSWERING
        
        # Initialize chat chain
        self.chat_chain = PromptChain(
            models=[config.get('chat', {}).get('model', 'openai/gpt-4o')],
            instructions=[
                self._get_system_prompt(),
                AgenticStepProcessor(
                    objective="Answer user questions based on comprehensive research analysis",
                    max_internal_steps=5
                ),
                "Provide detailed, evidence-based responses with proper citations"
            ],
            verbose=config.get('verbose', False)
        )
        
        # Register chat tools
        self._register_tools()
        
        logger.info(f"InteractiveChatInterface initialized for session {session.session_id}")
    
    def _get_system_prompt(self) -> str:
        """Generate system prompt based on session context"""
        papers_count = len(self.session.papers)
        queries_count = len(self.session.queries)
        
        return f"""You are an expert research assistant with access to analysis of {papers_count} papers on '{self.session.topic}'.
        
        Available Information:
        - {queries_count} research questions analyzed
        - {len(self.session.processing_results)} processing results from 3-tier RAG system
        - Comprehensive literature review with statistics and insights
        - Multi-iteration research with gap analysis
        
        Your role:
        1. Answer questions based on the research findings
        2. Provide evidence-based responses with citations
        3. Identify patterns and insights across papers
        4. Suggest areas for further exploration
        5. Maintain academic rigor and objectivity
        
        Current context: {{user_input}}"""
    
    def _register_tools(self):
        """Register tools for chat interactions"""
        
        def search_research_findings(query: str) -> str:
            """Search through research findings for specific information"""
            # Search through session data
            relevant_results = []
            
            # Search in processing results
            for result in self.session.processing_results[-20:]:  # Last 20 results
                if result.result_data and query.lower() in str(result.result_data).lower():
                    relevant_results.append({
                        'type': 'processing_result',
                        'tier': result.tier,
                        'data': str(result.result_data)[:500]  # Truncate for context
                    })
            
            # Search in paper abstracts
            for paper in list(self.session.papers.values())[:10]:  # Sample papers
                if query.lower() in paper.abstract.lower():
                    relevant_results.append({
                        'type': 'paper',
                        'title': paper.title,
                        'abstract_excerpt': paper.abstract[:300]
                    })
            
            if relevant_results:
                return f"Found {len(relevant_results)} relevant items:\n" + str(relevant_results)
            else:
                return f"No specific findings for '{query}' in current research data"
        
        def get_paper_details(paper_identifier: str) -> str:
            """Get detailed information about a specific paper"""
            # Search for paper by ID or title
            for paper_id, paper in self.session.papers.items():
                if (paper_identifier.lower() in paper_id.lower() or 
                    paper_identifier.lower() in paper.title.lower()):
                    return f"""Paper Details:
                    Title: {paper.title}
                    Authors: {', '.join(paper.authors[:3])}
                    Year: {paper.publication_year}
                    Source: {paper.source}
                    Abstract: {paper.abstract[:500]}...
                    """
            return f"Paper '{paper_identifier}' not found in research session"
        
        def compare_findings(aspect: str) -> str:
            """Compare findings across different papers or methods"""
            comparison_data = {
                'aspect': aspect,
                'papers_analyzed': len(self.session.papers),
                'comparison_note': f"Comparing {aspect} across research findings",
                'methods_found': [],
                'consensus_points': [],
                'disagreements': []
            }
            
            # Analyze for comparisons (simplified)
            if 'method' in aspect.lower():
                comparison_data['methods_found'] = ['Method A', 'Method B', 'Method C']
                comparison_data['consensus_points'] = ['Common evaluation metrics used']
                comparison_data['disagreements'] = ['Performance claims vary']
            
            return f"Comparison Analysis:\n{comparison_data}"
        
        def get_statistics_summary() -> str:
            """Get statistical summary of research session"""
            stats = self.session.get_session_statistics()
            
            summary = f"""Research Statistics:
            - Total Papers: {stats.get('total_papers', 0)}
            - Queries Processed: {stats.get('queries_completed', 0)}/{stats.get('total_queries', 0)}
            - Processing Success Rate: {stats.get('success_rate', 0):.1%}
            - Iterations Completed: {stats.get('iterations_completed', 0)}
            - Unique Authors: {stats.get('unique_authors', 0)}
            - Year Range: {stats.get('year_range', 'N/A')}
            - Top Sources: {stats.get('top_sources', [])}
            """
            return summary
        
        # Register tools
        self.chat_chain.register_tool_function(search_research_findings)
        self.chat_chain.register_tool_function(get_paper_details)
        self.chat_chain.register_tool_function(compare_findings)
        self.chat_chain.register_tool_function(get_statistics_summary)
        
        # Add tool schemas
        self.chat_chain.add_tools([
            {
                "type": "function",
                "function": {
                    "name": "search_research_findings",
                    "description": "Search through research findings for specific information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_paper_details",
                    "description": "Get detailed information about a specific paper",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "paper_identifier": {"type": "string", "description": "Paper ID or title"}
                        },
                        "required": ["paper_identifier"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "compare_findings",
                    "description": "Compare findings across papers or methods",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "aspect": {"type": "string", "description": "Aspect to compare"}
                        },
                        "required": ["aspect"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_statistics_summary",
                    "description": "Get statistical summary of research session",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ])
    
    async def process_message(
        self,
        user_message: str,
        mode: Optional[ChatMode] = None
    ) -> str:
        """
        Process user message and generate response
        
        Args:
            user_message: User's input message
            mode: Optional chat mode override
            
        Returns:
            Assistant's response
        """
        if mode:
            self.current_mode = mode
        
        # Add to chat history
        self.chat_history.append({
            'role': 'user',
            'message': user_message,
            'timestamp': datetime.now().isoformat(),
            'mode': self.current_mode.value
        })
        
        try:
            # Prepare context based on mode
            context = self._prepare_context_for_mode(user_message)
            
            # Process through chat chain
            response = await self.chat_chain.process_prompt_async(context)
            
            # Add response to history
            self.chat_history.append({
                'role': 'assistant',
                'message': response,
                'timestamp': datetime.now().isoformat(),
                'mode': self.current_mode.value
            })
            
            # Store in session
            self.session.add_chat_message('user', user_message)
            self.session.add_chat_message('assistant', response)
            
            return response
            
        except Exception as e:
            logger.error(f"Chat processing failed: {e}")
            error_response = f"I encountered an error processing your request: {e}"
            
            self.chat_history.append({
                'role': 'assistant',
                'message': error_response,
                'timestamp': datetime.now().isoformat(),
                'error': True
            })
            
            return error_response
    
    def _prepare_context_for_mode(self, user_message: str) -> str:
        """Prepare context based on chat mode"""
        base_context = f"User Question: {user_message}\n\n"
        
        if self.current_mode == ChatMode.QUESTION_ANSWERING:
            base_context += "Mode: Question Answering - Provide direct, evidence-based answers\n"
            
        elif self.current_mode == ChatMode.DEEP_DIVE:
            base_context += "Mode: Deep Dive - Provide comprehensive analysis with multiple perspectives\n"
            base_context += f"Include: methodology details, evaluation results, limitations\n"
            
        elif self.current_mode == ChatMode.COMPARATIVE_ANALYSIS:
            base_context += "Mode: Comparative Analysis - Compare and contrast different approaches\n"
            base_context += "Structure: similarities, differences, trade-offs, recommendations\n"
            
        elif self.current_mode == ChatMode.SUMMARY_GENERATION:
            base_context += "Mode: Summary Generation - Create concise summaries\n"
            base_context += "Focus: key points, main findings, actionable insights\n"
            
        elif self.current_mode == ChatMode.CUSTOM_ANALYSIS:
            base_context += "Mode: Custom Analysis - Flexible analysis based on user needs\n"
        
        # Add recent context
        if len(self.chat_history) > 0:
            recent_context = self.chat_history[-3:]  # Last 3 exchanges
            base_context += "\nRecent conversation:\n"
            for entry in recent_context:
                base_context += f"{entry['role']}: {entry['message'][:200]}...\n"
        
        # Add session summary
        base_context += f"\nResearch Context:\n"
        base_context += f"- Topic: {self.session.topic}\n"
        base_context += f"- Papers: {len(self.session.papers)}\n"
        base_context += f"- Completion: {self.session.calculate_completion_score():.1%}\n"
        
        return base_context
    
    def set_mode(self, mode: ChatMode):
        """Set chat interaction mode"""
        self.current_mode = mode
        logger.info(f"Chat mode set to: {mode.value}")
    
    def get_available_modes(self) -> List[Dict[str, str]]:
        """Get available chat modes with descriptions"""
        return [
            {
                'mode': ChatMode.QUESTION_ANSWERING.value,
                'name': 'Question Answering',
                'description': 'Direct answers to specific questions about the research'
            },
            {
                'mode': ChatMode.DEEP_DIVE.value,
                'name': 'Deep Dive Analysis',
                'description': 'Comprehensive exploration of specific topics'
            },
            {
                'mode': ChatMode.COMPARATIVE_ANALYSIS.value,
                'name': 'Comparative Analysis',
                'description': 'Compare different methods, papers, or findings'
            },
            {
                'mode': ChatMode.SUMMARY_GENERATION.value,
                'name': 'Summary Generation',
                'description': 'Generate concise summaries of research findings'
            },
            {
                'mode': ChatMode.CUSTOM_ANALYSIS.value,
                'name': 'Custom Analysis',
                'description': 'Flexible analysis based on specific needs'
            }
        ]
    
    def get_suggested_questions(self) -> List[str]:
        """Get suggested questions based on research content"""
        suggestions = [
            f"What are the main methodologies used in {self.session.topic} research?",
            f"What are the key challenges identified in the literature?",
            f"Which papers provide the strongest evidence for practical applications?",
            f"What are the most promising future research directions?",
            f"How do different approaches compare in terms of performance?",
            f"What datasets are commonly used for evaluation?",
            f"What are the limitations of current {self.session.topic} methods?",
            f"Which research groups are leading in this field?",
            f"What are the recent breakthroughs since 2023?",
            f"How has the field evolved over the past 5 years?"
        ]
        
        # Add topic-specific suggestions
        if 'machine learning' in self.session.topic.lower():
            suggestions.extend([
                "What are the computational requirements of different methods?",
                "How do these methods handle data scarcity?",
                "What are the interpretability trade-offs?"
            ])
        
        return suggestions[:10]  # Return top 10 suggestions
    
    def get_chat_history(
        self,
        limit: Optional[int] = None,
        mode_filter: Optional[ChatMode] = None
    ) -> List[Dict[str, Any]]:
        """Get chat history with optional filters"""
        history = self.chat_history
        
        if mode_filter:
            history = [h for h in history if h.get('mode') == mode_filter.value]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def export_chat_session(self) -> Dict[str, Any]:
        """Export chat session for saving or analysis"""
        return {
            'session_id': self.session.session_id,
            'topic': self.session.topic,
            'chat_history': self.chat_history,
            'mode_usage': self._calculate_mode_usage(),
            'total_messages': len(self.chat_history),
            'session_start': self.chat_history[0]['timestamp'] if self.chat_history else None,
            'session_end': self.chat_history[-1]['timestamp'] if self.chat_history else None
        }
    
    def _calculate_mode_usage(self) -> Dict[str, int]:
        """Calculate usage statistics for different modes"""
        from collections import Counter
        
        modes = [h.get('mode', 'unknown') for h in self.chat_history if h.get('role') == 'user']
        return dict(Counter(modes))
    
    async def generate_final_summary(self) -> str:
        """Generate a final summary of the chat session"""
        if not self.chat_history:
            return "No chat history to summarize."
        
        # Prepare summary context
        summary_context = f"""Generate a concise summary of this research chat session:
        
        Topic: {self.session.topic}
        Total Exchanges: {len(self.chat_history) // 2}
        Modes Used: {', '.join(set(h.get('mode', 'qa') for h in self.chat_history if h.get('role') == 'user'))}
        
        Key Questions Asked:
        {self._extract_key_questions()}
        
        Main Insights Shared:
        {self._extract_key_insights()}
        
        Create a 2-3 paragraph summary highlighting:
        1. Main topics discussed
        2. Key findings or insights shared
        3. Any action items or recommendations made
        """
        
        try:
            summary = await self.chat_chain.process_prompt_async(summary_context)
            return summary
        except Exception as e:
            logger.error(f"Failed to generate chat summary: {e}")
            return f"Summary generation failed: {e}"
    
    def _extract_key_questions(self) -> str:
        """Extract key questions from chat history"""
        questions = [
            h['message'][:100] 
            for h in self.chat_history 
            if h.get('role') == 'user'
        ][:5]  # Top 5 questions
        
        return '\n'.join(f"- {q}" for q in questions)
    
    def _extract_key_insights(self) -> str:
        """Extract key insights from assistant responses"""
        # This is simplified - could use NLP for better extraction
        insights = []
        
        for h in self.chat_history:
            if h.get('role') == 'assistant':
                message = h['message']
                # Look for insight indicators
                if any(indicator in message.lower() for indicator in ['found that', 'shows that', 'indicates', 'suggests']):
                    # Extract first sentence with indicator
                    sentences = message.split('.')
                    for sent in sentences[:3]:  # Check first 3 sentences
                        if any(ind in sent.lower() for ind in ['found', 'shows', 'indicates', 'suggests']):
                            insights.append(sent.strip())
                            break
        
        return '\n'.join(f"- {insight}" for insight in insights[:5])  # Top 5 insights