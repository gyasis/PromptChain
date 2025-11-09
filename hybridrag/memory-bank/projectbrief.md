# HybridRAG Project Brief

## Project Name
**HybridRAG** - LightRAG + DeepLake Hybrid RAG System for Medical Database Schema Querying

## Project Scope
A proof-of-concept system that creates a knowledge graph-based retrieval system for medical database schema exploration, combining DeepLake's vector database capabilities with LightRAG's knowledge graph construction and multi-mode querying.

## Core Objectives
1. **Data Integration**: Extract and transform 15,149 medical table descriptions from DeepLake's athena_descriptions_v4 database
2. **Knowledge Graph Creation**: Build an intelligent graph representation of medical database relationships
3. **Multi-Mode Querying**: Provide local, global, hybrid, naive, and mix query modes for different exploration needs
4. **Interactive Interface**: Deliver a command-line demo for real-time schema exploration

## Primary Deliverables
- **deeplake_to_lightrag.py**: Batch ingestion pipeline with progress tracking and rate limiting
- **lightrag_query_demo.py**: Interactive CLI with colored output and multiple query modes
- **CustomDeepLake v4**: Enhanced recency-based search functionality for temporal relevance
- **Comprehensive Documentation**: Setup guides, architecture docs, and usage examples

## Value Proposition
Transforms static medical database schema information into an intelligent, queryable knowledge graph that enables natural language exploration of complex table relationships and medical data structures. Demonstrates the potential for knowledge graph-based database discovery and documentation.

## Success Criteria
- Successfully ingest all 15,149 medical table records from DeepLake
- Create functional knowledge graph supporting multi-hop reasoning queries
- Provide sub-5 second query response times across different query modes
- Enable complex relationship discovery (e.g., "What tables relate to patient appointments?")
- Maintain backward compatibility with existing DeepLake functionality

## Technical Foundation
- **Python 3.9+** with UV package management for isolated environments
- **LightRAG-HKU** for knowledge graph construction and querying
- **DeepLake v4** for vector storage and embedding management
- **OpenAI API** for embeddings and language model operations
- **Async/Sync Architecture** for flexible execution patterns

## Project Status
:white_check_mark: **Core Implementation Complete** - Full ingestion and query pipeline functional
:white_check_mark: **Documentation Mature** - Comprehensive setup and usage guides
:white_check_mark: **Enhanced Features** - Recency-based search with detailed commenting
:hourglass_flowing_sand: **Current Focus** - Memory bank initialization for session continuity

## Key Innovation
Successfully demonstrates hybrid RAG architecture where vector similarity search from DeepLake feeds into LightRAG's knowledge graph construction, creating a two-tier retrieval system that maintains both semantic similarity and graph-based reasoning capabilities.