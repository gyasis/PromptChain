"""Complete Research Workflow Using Multiple LightRAG Patterns

This example demonstrates a comprehensive research workflow that combines
multiple patterns for sophisticated information retrieval and analysis:

1. Query Expansion - Broaden search scope
2. Sharded Retrieval - Query multiple knowledge bases
3. Multi-Hop Retrieval - Complex multi-step reasoning
4. Hybrid Search Fusion - Combine search techniques
5. Branching Thoughts - Evaluate hypotheses

Prerequisites:
    pip install git+https://github.com/gyasis/hybridrag.git
    pip install litellm

Environment:
    Set OPENAI_API_KEY in .env file
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import patterns
from promptchain.integrations.lightrag import (
    LightRAGIntegration,
    LightRAGQueryExpander,
    LightRAGShardRegistry,
    LightRAGShardedRetriever,
    LightRAGMultiHop,
    LightRAGHybridSearcher,
    LightRAGBranchingThoughts,
    QueryExpansionConfig,
    ExpansionStrategy,
    ShardConfig,
    ShardType,
    ShardedRetrievalConfig,
    MultiHopConfig,
    HybridSearchConfig,
    SearchTechnique,
    FusionAlgorithm,
    BranchingConfig,
)

# Import infrastructure for multi-agent coordination
from promptchain.cli.models import MessageBus, Blackboard


class ResearchWorkflow:
    """Complete research workflow using LightRAG patterns."""

    def __init__(self, working_dir: Path, use_multi_agent: bool = True):
        """Initialize research workflow.

        Args:
            working_dir: Base directory for LightRAG data
            use_multi_agent: Enable MessageBus and Blackboard integration
        """
        self.working_dir = working_dir
        self.use_multi_agent = use_multi_agent

        # Multi-agent infrastructure
        self.message_bus = MessageBus() if use_multi_agent else None
        self.blackboard = Blackboard() if use_multi_agent else None

        # Pattern instances (initialized in setup)
        self.integration: LightRAGIntegration = None
        self.expander: LightRAGQueryExpander = None
        self.shard_retriever: LightRAGShardedRetriever = None
        self.multi_hop: LightRAGMultiHop = None
        self.hybrid_searcher: LightRAGHybridSearcher = None
        self.branching: LightRAGBranchingThoughts = None

    async def setup(self):
        """Setup LightRAG integration and patterns."""
        print("Setting up research workflow...\n")

        # 1. Create LightRAG integration
        print("1. Initializing LightRAG integration...")
        self.integration = LightRAGIntegration(working_dir=str(self.working_dir))

        # 2. Index sample research documents
        print("2. Indexing research documents...")
        await self._index_research_documents()

        # 3. Initialize patterns
        print("3. Initializing patterns...")

        # Query Expansion
        self.expander = LightRAGQueryExpander(
            lightrag_integration=self.integration,
            config=QueryExpansionConfig(
                strategies=[
                    ExpansionStrategy.SEMANTIC,
                    ExpansionStrategy.REFORMULATION,
                ],
                max_expansions_per_strategy=2,
                emit_events=self.use_multi_agent,
                use_blackboard=self.use_multi_agent,
            ),
        )

        # Sharded Retrieval (if multiple shards available)
        registry = LightRAGShardRegistry()
        registry.register_shard(
            ShardConfig(
                shard_id="main_shard",
                shard_type=ShardType.LIGHTRAG,
                working_dir=str(self.working_dir),
                priority=1,
            )
        )
        self.shard_retriever = LightRAGShardedRetriever(
            registry=registry,
            config=ShardedRetrievalConfig(
                parallel=True,
                emit_events=self.use_multi_agent,
            ),
        )

        # Multi-Hop Retrieval
        self.multi_hop = LightRAGMultiHop(
            search_interface=self.integration.search,
            config=MultiHopConfig(
                max_hops=5,
                decompose_first=True,
                emit_events=self.use_multi_agent,
                use_blackboard=self.use_multi_agent,
            ),
        )

        # Hybrid Search Fusion
        self.hybrid_searcher = LightRAGHybridSearcher(
            lightrag_integration=self.integration,
            config=HybridSearchConfig(
                techniques=[SearchTechnique.LOCAL, SearchTechnique.GLOBAL],
                fusion_algorithm=FusionAlgorithm.RRF,
                emit_events=self.use_multi_agent,
            ),
        )

        # Branching Thoughts
        self.branching = LightRAGBranchingThoughts(
            lightrag_core=self.integration,
            config=BranchingConfig(
                hypothesis_count=3,
                judge_model="openai/gpt-4o",
                emit_events=self.use_multi_agent,
                use_blackboard=self.use_multi_agent,
            ),
        )

        # Connect to multi-agent infrastructure
        if self.use_multi_agent:
            for pattern in [
                self.expander,
                self.shard_retriever,
                self.multi_hop,
                self.hybrid_searcher,
                self.branching,
            ]:
                pattern.connect_messagebus(self.message_bus)
                pattern.connect_blackboard(self.blackboard)

            # Subscribe to events
            self.message_bus.subscribe("pattern.*", self._event_logger)

        print("✓ Setup complete\n")

    async def _index_research_documents(self):
        """Index sample research documents."""
        docs = [
            """
            Machine Learning and Deep Learning Overview:
            Machine learning is a subset of AI that enables systems to learn from data.
            Deep learning uses neural networks with multiple layers for complex pattern recognition.
            Key applications include computer vision, natural language processing, and speech recognition.
            Recent advances include transformer architectures and attention mechanisms.
            """,
            """
            Climate Change and Environmental Impact:
            Climate change is driven by greenhouse gas emissions from fossil fuels.
            Rising global temperatures lead to ice melt, sea level rise, and extreme weather.
            Renewable energy sources (solar, wind, hydro) offer sustainable alternatives.
            Carbon capture technologies and reforestation help mitigate emissions.
            """,
            """
            Renewable Energy Technologies:
            Solar photovoltaic systems convert sunlight directly to electricity.
            Wind turbines harness kinetic energy from air movement.
            Hydroelectric dams use water flow for power generation.
            Battery storage systems enable grid-scale renewable integration.
            Costs have decreased significantly making renewables competitive with fossil fuels.
            """,
            """
            Transformer Architecture in NLP:
            Transformers use self-attention mechanisms to process sequences in parallel.
            BERT and GPT models are based on transformer architecture.
            Attention allows models to focus on relevant parts of input.
            Pre-training on large corpora followed by fine-tuning is effective.
            """,
        ]

        await self.integration.insert_documents(docs)
        print(f"   Indexed {len(docs)} documents")

    def _event_logger(self, event_type: str, data: Dict[str, Any]):
        """Log pattern events for monitoring."""
        # Simple event logging
        if "started" in event_type:
            print(f"   → {event_type}")
        elif "completed" in event_type:
            print(f"   ✓ {event_type}")

    async def execute_research(
        self, research_question: str, use_all_patterns: bool = True
    ) -> Dict[str, Any]:
        """Execute complete research workflow.

        Args:
            research_question: The research question to investigate
            use_all_patterns: Whether to use all patterns or basic workflow

        Returns:
            Dictionary with results from each pattern
        """
        print("\n" + "=" * 70)
        print(f"Research Question: {research_question}")
        print("=" * 70 + "\n")

        results = {
            "question": research_question,
            "patterns_used": [],
        }

        if use_all_patterns:
            # Complete workflow using all patterns

            # Phase 1: Query Expansion
            print("Phase 1: Query Expansion")
            print("-" * 70)
            expansion_result = await self.expander.execute(query=research_question)
            results["expansion"] = {
                "expanded_queries": [eq.expanded_query for eq in expansion_result.expanded_queries],
                "unique_results": expansion_result.unique_results_found,
            }
            results["patterns_used"].append("Query Expansion")
            print(f"✓ Generated {len(expansion_result.expanded_queries)} query variations")
            print(f"✓ Found {expansion_result.unique_results_found} unique results\n")

            # Phase 2: Sharded Retrieval
            print("Phase 2: Sharded Retrieval")
            print("-" * 70)
            shard_result = await self.shard_retriever.execute(query=research_question)
            results["sharded"] = {
                "shards_queried": shard_result.shards_queried,
                "results_count": len(shard_result.aggregated_results),
            }
            results["patterns_used"].append("Sharded Retrieval")
            print(f"✓ Queried {shard_result.shards_queried} shards")
            print(f"✓ Retrieved {len(shard_result.aggregated_results)} aggregated results\n")

            # Phase 3: Multi-Hop Retrieval
            print("Phase 3: Multi-Hop Retrieval")
            print("-" * 70)
            multi_hop_result = await self.multi_hop.execute(question=research_question)
            results["multi_hop"] = {
                "hops_executed": multi_hop_result.hops_executed,
                "sub_questions": len(multi_hop_result.sub_questions),
                "unified_answer": multi_hop_result.unified_answer,
            }
            results["patterns_used"].append("Multi-Hop Retrieval")
            print(f"✓ Executed {multi_hop_result.hops_executed} reasoning hops")
            print(f"✓ Decomposed into {len(multi_hop_result.sub_questions)} sub-questions")
            print(f"✓ Synthesized unified answer\n")

            # Phase 4: Hybrid Search Fusion
            print("Phase 4: Hybrid Search Fusion")
            print("-" * 70)
            hybrid_result = await self.hybrid_searcher.execute(query=research_question)
            results["hybrid_search"] = {
                "techniques": [tr.technique.value for tr in hybrid_result.technique_results],
                "fused_results": len(hybrid_result.fused_results),
                "contributions": hybrid_result.technique_contributions,
            }
            results["patterns_used"].append("Hybrid Search Fusion")
            print(f"✓ Combined {len(hybrid_result.technique_results)} search techniques")
            print(f"✓ Fused to {len(hybrid_result.fused_results)} top results")
            print(f"✓ Contributions: {hybrid_result.technique_contributions}\n")

            # Phase 5: Branching Thoughts (Final Hypothesis Evaluation)
            print("Phase 5: Branching Thoughts")
            print("-" * 70)
            branching_result = await self.branching.execute(problem=research_question)
            results["branching"] = {
                "hypotheses_generated": len(branching_result.hypotheses),
                "selected_mode": branching_result.selected_hypothesis.mode if branching_result.selected_hypothesis else None,
                "final_answer": branching_result.selected_hypothesis.reasoning if branching_result.selected_hypothesis else None,
            }
            results["patterns_used"].append("Branching Thoughts")
            print(f"✓ Generated {len(branching_result.hypotheses)} hypotheses")
            if branching_result.selected_hypothesis:
                print(f"✓ Selected best hypothesis from {branching_result.selected_hypothesis.mode} mode")
                print(f"\nFinal Answer:\n{branching_result.selected_hypothesis.reasoning}\n")
            else:
                print("✗ No hypothesis selected")

        else:
            # Basic workflow - just hybrid search and branching
            print("Basic Workflow: Hybrid Search + Branching")
            print("-" * 70)

            hybrid_result = await self.hybrid_searcher.execute(query=research_question)
            results["hybrid_search"] = {
                "fused_results": len(hybrid_result.fused_results)
            }

            branching_result = await self.branching.execute(problem=research_question)
            results["branching"] = {
                "final_answer": branching_result.selected_hypothesis.reasoning if branching_result.selected_hypothesis else None,
            }
            results["patterns_used"] = ["Hybrid Search", "Branching Thoughts"]

        return results

    async def print_statistics(self):
        """Print execution statistics for all patterns."""
        print("\n" + "=" * 70)
        print("Pattern Execution Statistics")
        print("=" * 70 + "\n")

        patterns = {
            "Query Expansion": self.expander,
            "Sharded Retrieval": self.shard_retriever,
            "Multi-Hop Retrieval": self.multi_hop,
            "Hybrid Search Fusion": self.hybrid_searcher,
            "Branching Thoughts": self.branching,
        }

        for name, pattern in patterns.items():
            stats = pattern.get_stats()
            print(f"{name}:")
            print(f"  Executions: {stats['execution_count']}")
            print(f"  Avg Time: {stats['average_execution_time_ms']:.2f}ms")
            print()


async def main():
    """Run complete research workflow example."""
    # Setup
    working_dir = Path("./lightrag_research_data")
    working_dir.mkdir(exist_ok=True)

    workflow = ResearchWorkflow(working_dir=working_dir, use_multi_agent=True)
    await workflow.setup()

    # Example 1: Complete workflow
    await workflow.execute_research(
        research_question="How do transformers and deep learning contribute to modern NLP?",
        use_all_patterns=True,
    )

    # Example 2: Basic workflow
    await workflow.execute_research(
        research_question="What are the main renewable energy technologies?",
        use_all_patterns=False,
    )

    # Print statistics
    await workflow.print_statistics()

    print("\n" + "=" * 70)
    print("✓ Research Workflow Complete")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
