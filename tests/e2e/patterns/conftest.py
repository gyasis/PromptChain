"""E2E test fixtures with complete infrastructure and realistic mock data.

Provides comprehensive mocking for multi-pattern workflow testing.
"""

import asyncio
import pytest
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch


# ===========================
# Mock Data Models
# ===========================


@dataclass
class MockDocument:
    """Realistic document mock."""
    doc_id: str
    title: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]

    def similarity_to(self, other: "MockDocument") -> float:
        """Calculate mock similarity based on content overlap."""
        words_self = set(self.content.lower().split())
        words_other = set(other.content.lower().split())
        if not words_self or not words_other:
            return 0.0
        intersection = len(words_self & words_other)
        union = len(words_self | words_other)
        return intersection / union if union > 0 else 0.0


@dataclass
class MockSearchResult:
    """Realistic search result mock."""
    content: str
    score: float
    doc_id: str
    source: str
    metadata: Dict[str, Any]


@dataclass
class MockEntity:
    """Knowledge graph entity mock."""
    name: str
    entity_type: str
    properties: Dict[str, Any]


@dataclass
class MockRelationship:
    """Knowledge graph relationship mock."""
    source: str
    relation_type: str
    target: str
    weight: float


# ===========================
# Mock Knowledge Graph
# ===========================


class MockKnowledgeGraph:
    """Realistic knowledge graph mock with entities and relationships."""

    def __init__(self):
        self.entities = {
            "machine_learning": MockEntity(
                name="Machine Learning",
                entity_type="concept",
                properties={"description": "AI technique for pattern recognition"}
            ),
            "deep_learning": MockEntity(
                name="Deep Learning",
                entity_type="concept",
                properties={"description": "ML using neural networks"}
            ),
            "neural_network": MockEntity(
                name="Neural Network",
                entity_type="concept",
                properties={"description": "Computational model inspired by brain"}
            ),
            "transformer": MockEntity(
                name="Transformer",
                entity_type="architecture",
                properties={"description": "Attention-based neural architecture"}
            ),
            "rnn": MockEntity(
                name="RNN",
                entity_type="architecture",
                properties={"description": "Recurrent neural network"}
            ),
        }

        self.relationships = [
            MockRelationship("deep_learning", "is_a", "machine_learning", 1.0),
            MockRelationship("transformer", "uses", "neural_network", 0.9),
            MockRelationship("rnn", "uses", "neural_network", 0.9),
            MockRelationship("transformer", "replaces", "rnn", 0.7),
        ]

    def get_entity(self, name: str) -> Optional[MockEntity]:
        """Get entity by name."""
        return self.entities.get(name.lower().replace(" ", "_"))

    def get_related_entities(self, entity_name: str, max_depth: int = 2) -> List[MockEntity]:
        """Get entities related to the given entity."""
        related = []
        entity_key = entity_name.lower().replace(" ", "_")

        for rel in self.relationships:
            if rel.source == entity_key:
                target = self.entities.get(rel.target)
                if target:
                    related.append(target)
            elif rel.target == entity_key:
                source = self.entities.get(rel.source)
                if source:
                    related.append(source)

        return related


# ===========================
# Mock Document Store
# ===========================


class MockDocumentStore:
    """Realistic document store with vector search."""

    def __init__(self):
        self.documents = [
            MockDocument(
                doc_id="doc1",
                title="Introduction to Machine Learning",
                content="Machine learning is a subset of artificial intelligence that enables systems to learn from data. It includes supervised, unsupervised, and reinforcement learning techniques.",
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
                metadata={"author": "Dr. Smith", "year": 2023}
            ),
            MockDocument(
                doc_id="doc2",
                title="Deep Learning Fundamentals",
                content="Deep learning uses neural networks with multiple layers to learn hierarchical representations. Popular architectures include CNNs, RNNs, and Transformers.",
                embedding=[0.2, 0.3, 0.4, 0.5, 0.6],
                metadata={"author": "Dr. Johnson", "year": 2023}
            ),
            MockDocument(
                doc_id="doc3",
                title="Transformer Architecture Explained",
                content="Transformers revolutionized NLP through self-attention mechanisms. They process sequences in parallel and capture long-range dependencies better than RNNs.",
                embedding=[0.3, 0.4, 0.5, 0.6, 0.7],
                metadata={"author": "Dr. Lee", "year": 2024}
            ),
            MockDocument(
                doc_id="doc4",
                title="RNN and LSTM Networks",
                content="Recurrent Neural Networks process sequential data but suffer from vanishing gradients. LSTMs address this with gating mechanisms.",
                embedding=[0.15, 0.25, 0.35, 0.45, 0.55],
                metadata={"author": "Dr. Chen", "year": 2022}
            ),
            MockDocument(
                doc_id="doc5",
                title="Comparing Transformers vs RNNs",
                content="Transformers excel at parallelization and long-range dependencies, while RNNs are memory-efficient for sequential tasks. Each has distinct advantages.",
                embedding=[0.25, 0.35, 0.45, 0.55, 0.65],
                metadata={"author": "Dr. Williams", "year": 2024}
            ),
        ]
        self.search_history = []

    def search(self, query: str, top_k: int = 3, mode: str = "hybrid") -> List[MockSearchResult]:
        """Simulate vector + keyword search."""
        self.search_history.append({"query": query, "mode": mode, "top_k": top_k})

        # Simple keyword matching for mock
        query_lower = query.lower()
        scored_docs = []

        for doc in self.documents:
            # Score based on keyword overlap
            content_words = set(doc.content.lower().split())
            query_words = set(query_lower.split())
            overlap = len(content_words & query_words)
            score = overlap / len(query_words) if query_words else 0.0

            if score > 0:
                scored_docs.append((doc, score))

        # Sort by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top-k results
        results = []
        for doc, score in scored_docs[:top_k]:
            results.append(MockSearchResult(
                content=doc.content,
                score=score,
                doc_id=doc.doc_id,
                source=doc.title,
                metadata=doc.metadata
            ))

        return results


# ===========================
# Mock LightRAG Integration
# ===========================


class MockLightRAGIntegration:
    """Comprehensive mock of LightRAGIntegration."""

    def __init__(self):
        self.kg = MockKnowledgeGraph()
        self.doc_store = MockDocumentStore()
        self._call_history = []

    async def extract_context(self, query: str, **kwargs) -> Dict[str, Any]:
        """Mock context extraction from knowledge graph."""
        self._call_history.append({"method": "extract_context", "query": query})

        # Extract entities from query
        entities = []
        relationships = []

        query_lower = query.lower()
        for entity_name, entity in self.kg.entities.items():
            if entity_name.replace("_", " ") in query_lower:
                entities.append({
                    "name": entity.name,
                    "type": entity.entity_type,
                    **entity.properties
                })

                # Add related entities
                related = self.kg.get_related_entities(entity_name)
                for rel_entity in related:
                    entities.append({
                        "name": rel_entity.name,
                        "type": rel_entity.entity_type,
                    })

        # Add relationships
        for rel in self.kg.relationships:
            if any(e["name"].lower().replace(" ", "_") in [rel.source, rel.target] for e in entities):
                relationships.append({
                    "source": rel.source.replace("_", " ").title(),
                    "type": rel.relation_type,
                    "target": rel.target.replace("_", " ").title(),
                    "weight": rel.weight
                })

        return {
            "entities": entities,
            "relationships": relationships,
            "query": query
        }

    async def local_query(self, query: str, top_k: int = 10, **kwargs) -> List[MockSearchResult]:
        """Mock local (entity-specific) query."""
        self._call_history.append({"method": "local_query", "query": query, "top_k": top_k})
        return self.doc_store.search(query, top_k, mode="local")

    async def global_query(self, query: str, top_k: int = 10, **kwargs) -> List[MockSearchResult]:
        """Mock global (conceptual) query."""
        self._call_history.append({"method": "global_query", "query": query, "top_k": top_k})
        return self.doc_store.search(query, top_k, mode="global")

    async def hybrid_query(self, query: str, top_k: int = 10, **kwargs) -> List[MockSearchResult]:
        """Mock hybrid query."""
        self._call_history.append({"method": "hybrid_query", "query": query, "top_k": top_k})
        return self.doc_store.search(query, top_k, mode="hybrid")

    def get_call_history(self) -> List[Dict[str, Any]]:
        """Get history of all calls."""
        return self._call_history


# ===========================
# Mock SearchInterface
# ===========================


class MockSearchInterface:
    """Mock SearchInterface with multi-query and agentic search."""

    def __init__(self, lightrag_integration: MockLightRAGIntegration):
        self.integration = lightrag_integration
        self.search_calls = []

    async def multi_query_search(
        self, queries: List[str], mode: str = "hybrid", **kwargs
    ) -> List[MockSearchResult]:
        """Mock multi-query parallel search."""
        self.search_calls.append({"type": "multi_query", "queries": queries, "mode": mode})

        # Execute each query and aggregate
        all_results = []
        for query in queries:
            if mode == "local":
                results = await self.integration.local_query(query, **kwargs)
            elif mode == "global":
                results = await self.integration.global_query(query, **kwargs)
            else:
                results = await self.integration.hybrid_query(query, **kwargs)
            all_results.extend(results)

        return all_results

    async def agentic_search(
        self, query: str, objective: str, max_steps: int = 5, **kwargs
    ) -> Dict[str, Any]:
        """Mock agentic multi-hop search."""
        self.search_calls.append({
            "type": "agentic",
            "query": query,
            "objective": objective,
            "max_steps": max_steps
        })

        # Simulate multi-hop reasoning
        hops = []
        current_query = query

        for step in range(min(max_steps, 3)):  # Simulate 3 hops
            results = await self.integration.hybrid_query(current_query, top_k=2)
            hops.append({
                "step": step + 1,
                "query": current_query,
                "results": results
            })

            # Generate follow-up query based on results
            if results and step < 2:
                current_query = f"Follow-up to {current_query}"

        # Synthesize answer
        all_content = " ".join([
            r.content for hop in hops for r in hop["results"]
        ])

        return {
            "answer": f"Comprehensive answer based on {len(hops)} hops: {all_content[:200]}...",
            "steps_taken": len(hops),
            "hops": hops
        }

    async def hybrid_search(self, query: str, **kwargs) -> List[MockSearchResult]:
        """Mock hybrid search."""
        return await self.integration.hybrid_query(query, **kwargs)


# ===========================
# Mock MessageBus
# ===========================


class MockMessageBus:
    """Mock MessageBus for event capture."""

    def __init__(self):
        self.events = []
        self.subscribers = {}

    def publish(self, event_type: str, data: Dict[str, Any]) -> None:
        """Capture published event."""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.events.append(event)

        # Notify subscribers
        for pattern, callback in self.subscribers.items():
            if self._match_pattern(event_type, pattern):
                callback(event_type, data)

    def subscribe(self, pattern: str, callback) -> None:
        """Subscribe to event pattern."""
        self.subscribers[pattern] = callback

    def _match_pattern(self, event_type: str, pattern: str) -> bool:
        """Simple wildcard pattern matching."""
        if pattern == "*" or pattern == "pattern.*":
            return True
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return event_type.startswith(prefix)
        return event_type == pattern

    def get_events_by_type(self, event_type_pattern: str) -> List[Dict[str, Any]]:
        """Get events matching a pattern."""
        return [
            e for e in self.events
            if self._match_pattern(e["type"], event_type_pattern)
        ]

    def get_event_sequence(self) -> List[str]:
        """Get chronological sequence of event types."""
        return [e["type"] for e in self.events]

    def clear(self) -> None:
        """Clear all events."""
        self.events.clear()


# ===========================
# Pytest Fixtures
# ===========================


@pytest.fixture
def mock_knowledge_graph():
    """Provide mock knowledge graph."""
    return MockKnowledgeGraph()


@pytest.fixture
def mock_document_store():
    """Provide mock document store."""
    return MockDocumentStore()


@pytest.fixture
def mock_lightrag_integration():
    """Provide mock LightRAG integration."""
    return MockLightRAGIntegration()


@pytest.fixture
def mock_search_interface(mock_lightrag_integration):
    """Provide mock SearchInterface."""
    return MockSearchInterface(mock_lightrag_integration)


@pytest.fixture
def mock_message_bus():
    """Provide mock MessageBus."""
    return MockMessageBus()


@pytest.fixture
def event_capture(mock_message_bus):
    """Fixture for capturing and analyzing events."""
    def _capture():
        return mock_message_bus.events.copy()
    return _capture


@pytest.fixture
async def e2e_test_context(
    mock_lightrag_integration,
    mock_search_interface,
    mock_message_bus
):
    """Complete E2E test context with all infrastructure."""
    context = {
        "lightrag": mock_lightrag_integration,
        "search": mock_search_interface,
        "message_bus": mock_message_bus,
        "kg": mock_lightrag_integration.kg,
        "doc_store": mock_lightrag_integration.doc_store,
    }

    yield context

    # Cleanup
    mock_message_bus.clear()


# ===========================
# Test Data Generators
# ===========================


def generate_research_queries() -> List[str]:
    """Generate realistic research queries."""
    return [
        "What is machine learning?",
        "How do transformers work?",
        "Compare transformers vs RNNs",
        "What are the advantages of deep learning?",
        "Explain neural network architectures",
    ]


def generate_conversation_history() -> List[Dict[str, str]]:
    """Generate realistic conversation history."""
    return [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI..."},
        {"role": "user", "content": "How does deep learning differ?"},
        {"role": "assistant", "content": "Deep learning uses neural networks..."},
        {"role": "user", "content": "Tell me about transformers"},
    ]


@pytest.fixture
def research_queries():
    """Provide research queries."""
    return generate_research_queries()


@pytest.fixture
def conversation_history():
    """Provide conversation history."""
    return generate_conversation_history()
