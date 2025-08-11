"""
Research Session Management

Handles research sessions, query management, paper tracking, and iterative processing state.
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import uuid


class SessionStatus(Enum):
    """Research session status"""
    INITIALIZING = "initializing"
    QUERY_GENERATION = "query_generation"
    LITERATURE_SEARCH = "literature_search"
    PROCESSING = "processing"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    INTERACTIVE = "interactive"
    COMPLETED = "completed"
    ERROR = "error"


class QueryStatus(Enum):
    """Query processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Query:
    """Research query with metadata"""
    id: str
    text: str
    priority: float
    iteration: int
    status: QueryStatus = QueryStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    results: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Query':
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['status'] = QueryStatus(data['status'])
        return cls(**data)


@dataclass
class Paper:
    """Research paper metadata"""
    id: str
    title: str
    authors: List[str]
    abstract: str
    source: str  # sci_hub, arxiv, pubmed
    url: str
    pdf_path: Optional[str] = None
    doi: Optional[str] = None
    citation_count: Optional[int] = None
    publication_year: Optional[int] = None
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed_by_tiers: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data['processed_by_tiers'] = list(self.processed_by_tiers)
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Paper':
        data = data.copy()
        data['processed_by_tiers'] = set(data.get('processed_by_tiers', []))
        return cls(**data)


@dataclass
class ProcessingResult:
    """Results from tier processing"""
    tier: str
    query_id: str
    paper_ids: List[str]
    result_data: Dict[str, Any]
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ProcessingResult':
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class IterationSummary:
    """Summary of a research iteration"""
    iteration: int
    queries_processed: int
    papers_found: int
    gaps_identified: List[str]
    new_queries_generated: int
    completion_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class ResearchSession:
    """
    Manages a complete research session with iterative query processing
    """
    
    def __init__(
        self,
        topic: str,
        session_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.topic = topic
        self.config = config or {}
        
        # Session metadata
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.status = SessionStatus.INITIALIZING
        
        # Research data
        self.queries: Dict[str, Query] = {}
        self.papers: Dict[str, Paper] = {}
        self.processing_results: List[ProcessingResult] = []
        self.iteration_summaries: List[IterationSummary] = []
        
        # State tracking
        self.current_iteration = 0
        self.active_queries: List[str] = []
        self.completed_queries: Set[str] = set()
        self.failed_queries: Set[str] = set()
        
        # Literature review data
        self.literature_review: Optional[Dict[str, Any]] = None
        self.chat_history: List[Dict[str, Any]] = []
        
        # Configuration from session config
        self.max_iterations = self.config.get('max_iterations', 5)
        self.max_queries_per_iteration = self.config.get('max_queries_per_iteration', 15)
        self.completeness_threshold = self.config.get('completeness_threshold', 0.85)
        self.gap_detection_threshold = self.config.get('gap_detection_threshold', 0.7)
    
    def add_queries(self, queries: List[Dict[str, Any]]) -> List[str]:
        """Add new queries to the session"""
        query_ids = []
        
        for query_data in queries:
            query_id = str(uuid.uuid4())
            query = Query(
                id=query_id,
                text=query_data['text'],
                priority=query_data.get('priority', 1.0),
                iteration=self.current_iteration
            )
            self.queries[query_id] = query
            query_ids.append(query_id)
        
        self.active_queries.extend(query_ids)
        self._update_timestamp()
        return query_ids
    
    def add_papers(self, papers: List[Dict[str, Any]]) -> List[str]:
        """Add new papers to the session"""
        paper_ids = []
        
        for paper_data in papers:
            paper_id = paper_data.get('id') or str(uuid.uuid4())
            paper = Paper(**paper_data)
            paper.id = paper_id
            
            # Avoid duplicates
            if paper_id not in self.papers:
                self.papers[paper_id] = paper
                paper_ids.append(paper_id)
        
        self._update_timestamp()
        return paper_ids
    
    def add_processing_result(self, result: ProcessingResult):
        """Add processing result"""
        self.processing_results.append(result)
        
        # Mark papers as processed by this tier
        for paper_id in result.paper_ids:
            if paper_id in self.papers:
                self.papers[paper_id].processed_by_tiers.add(result.tier)
        
        self._update_timestamp()
    
    def mark_query_completed(self, query_id: str, results: Dict[str, Any] = None):
        """Mark query as completed"""
        if query_id in self.queries:
            self.queries[query_id].status = QueryStatus.COMPLETED
            if results:
                self.queries[query_id].results = results
            
            self.completed_queries.add(query_id)
            if query_id in self.active_queries:
                self.active_queries.remove(query_id)
        
        self._update_timestamp()
    
    def mark_query_failed(self, query_id: str, error: str = None):
        """Mark query as failed"""
        if query_id in self.queries:
            self.queries[query_id].status = QueryStatus.FAILED
            if error:
                self.queries[query_id].results = {'error': error}
            
            self.failed_queries.add(query_id)
            if query_id in self.active_queries:
                self.active_queries.remove(query_id)
        
        self._update_timestamp()
    
    def get_active_queries(self) -> List[Query]:
        """Get active queries for processing"""
        return [self.queries[qid] for qid in self.active_queries if qid in self.queries]
    
    def get_pending_queries(self) -> List[Query]:
        """Get queries that haven't been processed yet"""
        return [
            query for query in self.queries.values()
            if query.status == QueryStatus.PENDING
        ]
    
    def get_papers_for_queries(self, query_ids: List[str] = None) -> List[Paper]:
        """Get papers relevant to specific queries or all papers"""
        if query_ids is None:
            return list(self.papers.values())
        
        # For now, return all papers - could be enhanced with relevance scoring
        return list(self.papers.values())
    
    def calculate_completion_score(self) -> float:
        """Calculate research completion score"""
        if not self.queries:
            return 0.0
        
        completed_queries = len(self.completed_queries)
        total_queries = len(self.queries)
        
        # Base completion from query processing
        query_completion = completed_queries / total_queries
        
        # Bonus for paper processing across tiers
        if self.papers:
            tier_coverage = 0.0
            for paper in self.papers.values():
                tier_coverage += len(paper.processed_by_tiers) / 3.0  # 3 tiers
            tier_coverage /= len(self.papers)
            
            # Weighted average
            return (query_completion * 0.7) + (tier_coverage * 0.3)
        
        return query_completion
    
    def identify_gaps(self) -> List[str]:
        """Identify research gaps based on current findings"""
        gaps = []
        
        # Analyze query completion rates by priority
        high_priority_incomplete = [
            query for query in self.queries.values()
            if query.priority > 0.8 and query.status != QueryStatus.COMPLETED
        ]
        
        if high_priority_incomplete:
            gaps.append("High-priority queries remain incomplete")
        
        # Check for coverage gaps
        if self.papers:
            unprocessed_papers = [
                paper for paper in self.papers.values()
                if len(paper.processed_by_tiers) < 3
            ]
            
            if len(unprocessed_papers) > len(self.papers) * 0.3:
                gaps.append("Significant number of papers not fully processed")
        
        # Domain-specific gap analysis could be added here
        
        return gaps
    
    def is_complete(self) -> bool:
        """Check if research session is complete"""
        completion_score = self.calculate_completion_score()
        gaps = self.identify_gaps()
        
        return (
            completion_score >= self.completeness_threshold and
            len(gaps) <= 2 and  # Allow some minor gaps
            self.current_iteration > 0  # At least one iteration completed
        )
    
    def should_continue_iteration(self) -> bool:
        """Check if iteration should continue"""
        return (
            self.current_iteration < self.max_iterations and
            not self.is_complete() and
            len(self.active_queries) > 0
        )
    
    def start_new_iteration(self) -> int:
        """Start a new research iteration"""
        self.current_iteration += 1
        self.status = SessionStatus.PROCESSING
        self._update_timestamp()
        return self.current_iteration
    
    def complete_iteration(self, gaps: List[str] = None, new_queries: int = 0):
        """Complete current iteration"""
        summary = IterationSummary(
            iteration=self.current_iteration,
            queries_processed=len(self.completed_queries),
            papers_found=len(self.papers),
            gaps_identified=gaps or [],
            new_queries_generated=new_queries,
            completion_score=self.calculate_completion_score()
        )
        
        self.iteration_summaries.append(summary)
        self._update_timestamp()
    
    def set_status(self, status: SessionStatus):
        """Set session status"""
        self.status = status
        self._update_timestamp()
    
    def add_chat_message(self, role: str, message: str, metadata: Dict[str, Any] = None):
        """Add message to chat history"""
        self.chat_history.append({
            'role': role,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        })
        self._update_timestamp()
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get session statistics"""
        return {
            'session_id': self.session_id,
            'topic': self.topic,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'iterations_completed': self.current_iteration,
            'total_queries': len(self.queries),
            'completed_queries': len(self.completed_queries),
            'failed_queries': len(self.failed_queries),
            'active_queries': len(self.active_queries),
            'total_papers': len(self.papers),
            'processing_results': len(self.processing_results),
            'completion_score': self.calculate_completion_score(),
            'gaps_identified': len(self.identify_gaps()),
            'chat_messages': len(self.chat_history),
            'has_literature_review': self.literature_review is not None
        }
    
    def _update_timestamp(self):
        """Update the last modified timestamp"""
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization"""
        return {
            'session_id': self.session_id,
            'topic': self.topic,
            'config': self.config,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'status': self.status.value,
            'queries': {qid: query.to_dict() for qid, query in self.queries.items()},
            'papers': {pid: paper.to_dict() for pid, paper in self.papers.items()},
            'processing_results': [result.to_dict() for result in self.processing_results],
            'iteration_summaries': [summary.to_dict() for summary in self.iteration_summaries],
            'current_iteration': self.current_iteration,
            'active_queries': self.active_queries,
            'completed_queries': list(self.completed_queries),
            'failed_queries': list(self.failed_queries),
            'literature_review': self.literature_review,
            'chat_history': self.chat_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResearchSession':
        """Create session from dictionary"""
        session = cls(
            topic=data['topic'],
            session_id=data['session_id'],
            config=data.get('config', {})
        )
        
        session.created_at = datetime.fromisoformat(data['created_at'])
        session.updated_at = datetime.fromisoformat(data['updated_at'])
        session.status = SessionStatus(data['status'])
        
        # Restore queries
        session.queries = {
            qid: Query.from_dict(qdata) 
            for qid, qdata in data.get('queries', {}).items()
        }
        
        # Restore papers
        session.papers = {
            pid: Paper.from_dict(pdata)
            for pid, pdata in data.get('papers', {}).items()
        }
        
        # Restore processing results
        session.processing_results = [
            ProcessingResult.from_dict(rdata)
            for rdata in data.get('processing_results', [])
        ]
        
        # Restore other data
        session.current_iteration = data.get('current_iteration', 0)
        session.active_queries = data.get('active_queries', [])
        session.completed_queries = set(data.get('completed_queries', []))
        session.failed_queries = set(data.get('failed_queries', []))
        session.literature_review = data.get('literature_review')
        session.chat_history = data.get('chat_history', [])
        
        return session
    
    def save(self, filepath: str):
        """Save session to file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'ResearchSession':
        """Load session from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    def __str__(self) -> str:
        return f"ResearchSession(id={self.session_id}, topic='{self.topic}', status={self.status.value})"
    
    def __repr__(self) -> str:
        return self.__str__()