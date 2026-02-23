# PRD: Intelligent Search Variant Generation with PromptChain

## Executive Summary

**Problem**: The current search variant generation in the Research Agent uses a simplistic rule-based approach that merely appends words like "review", "methods", "applications" to the base topic. This approach is generic, lacks semantic understanding, and doesn't leverage the agentic capabilities of the system.

**Solution**: Implement a PromptChain-based pipeline that intelligently decomposes research topics and generates semantically diverse, contextually relevant search queries that explore different aspects of the research domain.

**Impact**: Transform search from generic keyword variations to intelligent, multi-dimensional query exploration that significantly improves research discovery and coverage.

## Problem Statement

### Current Limitations

1. **Generic Approach**: Current implementation in `LiteratureSearchAgent.generate_search_queries()` simply appends predefined words:
   ```python
   base_queries.extend([
       clean_topic,  # Basic topic
       f'"{clean_topic}"',  # Exact phrase
       f'{clean_topic} review',  # Review papers
       f'{clean_topic} methods',  # Methodological papers
       # ... etc
   ])
   ```

2. **Lack of Semantic Understanding**: No analysis of the research topic's components, domains, or relationships.

3. **Poor Coverage**: Misses important research angles and related fields.

4. **Not Agentic**: Doesn't leverage AI capabilities for intelligent query generation.

### Example: Current vs Desired Output

**Input**: "early detection of neurological diseases with gait analysis"

**Current Output**:
1. early detection of neurological diseases with gait analysis
2. "early detection of neurological diseases with gait analysis"
3. early detection of neurological diseases with gait analysis review
4. early detection of neurological diseases with gait analysis methods
5. early detection of neurological diseases with gait analysis applications

**Desired Output**:
1. "neurological disease detection methods"
2. "gait analysis techniques for medical diagnosis"
3. "early detection systems in neurology"
4. "gait analysis applications in disease detection"
5. "recent advances in neurological disease screening"
6. "machine learning in gait analysis for medical diagnosis"
7. "biomarkers for early neurological disease detection"
8. "gait pattern analysis in clinical neurology"

## Solution Overview

### PromptChain Pipeline Architecture

Implement a three-stage PromptChain pipeline:

1. **Topic Decomposition Chain**: Analyzes research topic and extracts key components
2. **Query Generation Chain**: Generates diverse queries based on decomposed components
3. **Query Optimization Chain**: Optimizes queries for specific databases

### Adaptive Query Spanning Approach

The system will use an **adaptive spanning approach** where the number of generated queries is determined by the topic's complexity and scope, not a fixed number. This ensures comprehensive coverage for complex topics while avoiding unnecessary queries for simple topics.

**Spanning Factors**:
- **Topic Complexity**: Number of distinct concepts and relationships
- **Research Scope**: Breadth of applications and methodologies
- **Cross-disciplinary Elements**: Integration with related fields
- **Temporal Aspects**: Historical, current, and future research directions
- **Geographic/Clinical Context**: Specific applications and settings

**Example Spanning Behavior**:
- **Simple Topic**: "machine learning" → 4-6 queries
- **Moderate Topic**: "early detection of neurological diseases" → 6-8 queries  
- **Complex Topic**: "early detection of neurological diseases with gait analysis using machine learning and biomarkers in clinical settings" → 10-12 queries
- **Multi-domain Topic**: "AI applications in healthcare for disease prediction using multimodal data from wearables, imaging, and genomics" → 12-15 queries

### Core Components

#### 1. TopicDecompositionChain
- **Input**: Research topic string
- **Output**: Structured topic components
- **Purpose**: Break down topic into research dimensions

**Example Output**:
```json
{
  "primary_concept": "early detection",
  "disease_domain": "neurological diseases",
  "methodology": "gait analysis",
  "application_area": "medical diagnosis",
  "temporal_focus": "early stage",
  "related_fields": ["biomarkers", "machine learning", "clinical neurology"]
}
```

#### 2. QueryGenerationChain
- **Input**: Decomposed topic components
- **Output**: Multiple semantically diverse queries (adaptive count)
- **Purpose**: Generate queries exploring different research angles

**Query Categories**:
- **Methodological**: Focus on techniques and approaches
- **Temporal**: Recent advances, historical developments
- **Application**: Clinical applications, real-world use cases
- **Cross-disciplinary**: Integration with related fields
- **Comparative**: Different approaches and their effectiveness

**Adaptive Spanning Logic**:
- **Complexity Analysis**: Count distinct concepts, methodologies, and applications
- **Scope Assessment**: Evaluate breadth of research domain and applications
- **Query Count Formula**: `base_count + (complexity_score * 2) + (scope_breadth * 1.5)`
- **Dynamic Adjustment**: Add queries for cross-disciplinary elements and temporal aspects

#### 3. QueryOptimizationChain
- **Input**: Generated queries
- **Output**: Database-optimized queries
- **Purpose**: Adapt queries for ArXiv, PubMed, Sci-Hub

## Technical Requirements

### 1. New PromptChain Classes

#### TopicComplexityAnalyzer
```python
class TopicComplexityAnalyzer(PromptChain):
    def __init__(self, model="openai/gpt-4"):
        super().__init__(
            instructions=[
                "Analyze topic complexity and scope for adaptive query spanning",
                "Count distinct concepts, methodologies, and applications",
                "Assess cross-disciplinary elements and temporal aspects",
                "Return structured complexity metrics for query count calculation"
            ],
            model=model
        )
    
    async def analyze_complexity(self, topic: str, components: dict) -> dict:
        """
        Analyze topic complexity and return metrics for adaptive spanning
        """
        analysis_prompt = f"""
        Topic: {topic}
        Components: {components}
        
        Analyze complexity and return JSON with:
        - concept_count: Number of distinct concepts
        - methodology_count: Number of methodologies/techniques
        - application_count: Number of application areas
        - scope_breadth: Breadth of research scope (1-10)
        - cross_disciplinary_elements: Number of related fields
        - temporal_span: Historical to future research span (1-10)
        """
        return await self.process_prompt(analysis_prompt)
```

#### TopicDecompositionChain
```python
class TopicDecompositionChain(PromptChain):
    def __init__(self, model="openai/gpt-4"):
        super().__init__(
            instructions=[
                "Analyze the research topic and extract key components",
                "Identify primary concepts, methodologies, and applications",
                "Map relationships between different aspects of the topic"
            ],
            model=model
        )
```

#### QueryGenerationChain
```python
class QueryGenerationChain(PromptChain):
    def __init__(self, model="openai/gpt-4"):
        super().__init__(
            instructions=[
                "Generate diverse search queries based on topic components",
                "Explore different research angles and methodologies",
                "Consider temporal, methodological, and application dimensions"
            ],
            model=model
        )
```

#### QueryOptimizationChain
```python
class QueryOptimizationChain(PromptChain):
    def __init__(self, model="openai/gpt-4"):
        super().__init__(
            instructions=[
                "Optimize queries for specific academic databases",
                "Adapt syntax and structure for ArXiv, PubMed, Sci-Hub",
                "Ensure maximum relevance and coverage"
            ],
            model=model
        )
```

### 2. Integration with LiteratureSearchAgent

#### Modified generate_search_queries Method
```python
async def generate_search_queries(self, topic: str) -> List[str]:
    """
    Generate intelligent search queries using PromptChain pipeline with adaptive spanning
    """
    try:
        # Stage 1: Topic Decomposition
        decomposition_chain = TopicDecompositionChain(self.model)
        topic_components = await decomposition_chain.process_prompt(topic)
        
        # Stage 1.5: Complexity Analysis for Spanning
        complexity_analyzer = TopicComplexityAnalyzer(self.model)
        complexity_score = await complexity_analyzer.analyze_complexity(topic, topic_components)
        
        # Stage 2: Query Generation with Adaptive Count
        generation_chain = QueryGenerationChain(self.model)
        target_query_count = self._calculate_adaptive_count(complexity_score)
        base_queries = await generation_chain.process_prompt(
            f"Topic: {topic}\nComponents: {topic_components}\nTarget Queries: {target_query_count}"
        )
        
        # Stage 3: Query Optimization
        optimization_chain = QueryOptimizationChain(self.model)
        optimized_queries = await optimization_chain.process_prompt(
            f"Queries: {base_queries}\nDatabases: ArXiv, PubMed, Sci-Hub"
        )
        
        return optimized_queries  # No fixed limit - adaptive count
        
    except Exception as e:
        logger.error(f"Error in intelligent query generation: {e}")
        return self._fallback_queries(topic)

def _calculate_adaptive_count(self, complexity_score: dict) -> int:
    """
    Calculate adaptive query count based on topic complexity
    """
    base_count = 4
    complexity_bonus = complexity_score.get('concept_count', 0) * 2
    scope_bonus = complexity_score.get('scope_breadth', 0) * 1.5
    cross_disciplinary_bonus = complexity_score.get('cross_disciplinary_elements', 0) * 1
    
    total_count = base_count + complexity_bonus + scope_bonus + cross_disciplinary_bonus
    return min(max(total_count, 4), 20)  # Between 4 and 20 queries
```

### 3. Configuration Updates

#### research_config.yaml Additions
```yaml
search_variants:
  enabled: true
  model: "openai/gpt-4"
  adaptive_spanning: true
  min_queries: 4
  max_queries: 20
  base_complexity_weight: 2.0
  scope_breadth_weight: 1.5
  cross_disciplinary_weight: 1.0
  diversity_threshold: 0.7
  include_temporal: true
  include_methodological: true
  include_applications: true
  include_cross_disciplinary: true
  complexity_analysis:
    concept_count_weight: 2.0
    methodology_count_weight: 1.5
    application_count_weight: 1.0
    temporal_span_weight: 1.2
```

## Implementation Plan

### Phase 1: Core PromptChain Development (Week 1)
1. Create `TopicDecompositionChain` class
2. Create `QueryGenerationChain` class
3. Create `QueryOptimizationChain` class
4. Add comprehensive tests for each chain

### Phase 2: Integration (Week 2)
1. Modify `LiteratureSearchAgent.generate_search_queries()`
2. Add configuration options
3. Implement fallback mechanisms
4. Add logging and monitoring

### Phase 3: Testing and Optimization (Week 3)
1. Test with diverse research topics
2. Optimize prompt engineering
3. Performance tuning
4. User acceptance testing

### Phase 4: Deployment and Monitoring (Week 4)
1. Deploy to development environment
2. Monitor query quality and diversity
3. Gather user feedback
4. Iterate based on results

## Success Metrics

### Quantitative Metrics
1. **Query Diversity**: Measure semantic diversity using embeddings
2. **Coverage Improvement**: Compare paper discovery rates
3. **Relevance Score**: User ratings of query relevance
4. **Performance**: Query generation time and API costs

### Qualitative Metrics
1. **User Satisfaction**: Feedback on search results quality
2. **Research Discovery**: Ability to find relevant papers missed by current approach
3. **Cross-disciplinary Coverage**: Discovery of papers from related fields

### Baseline Comparison
- **Current**: 8 generic queries (topic + word variations)
- **Target**: 4-20 adaptive queries based on topic complexity
- **Success Criteria**: 40% improvement in paper discovery rate
- **Spanning Success**: 90% of complex topics generate 10+ queries, simple topics generate 4-6 queries

## Risks and Considerations

### Technical Risks
1. **API Costs**: LLM calls for query generation may increase costs
2. **Latency**: PromptChain processing adds time to search execution
3. **Model Dependencies**: Reliance on external LLM APIs

### Mitigation Strategies
1. **Caching**: Cache query generation results for similar topics
2. **Fallback**: Maintain current system as backup
3. **Cost Optimization**: Use efficient models and prompt engineering
4. **Async Processing**: Generate queries asynchronously

### Quality Assurance
1. **Human Review**: Sample and review generated queries
2. **A/B Testing**: Compare old vs new system performance
3. **Continuous Monitoring**: Track query quality metrics
4. **User Feedback Loop**: Regular feedback collection and iteration

## Future Enhancements

### Phase 2 Features
1. **Personalization**: Adapt queries based on user research history
2. **Collaborative Filtering**: Learn from successful queries across users
3. **Domain Specialization**: Specialized chains for different research domains
4. **Multi-language Support**: Generate queries in multiple languages

### Advanced Capabilities
1. **Query Evolution**: Learn and improve from search result feedback
2. **Research Trend Integration**: Incorporate current research trends
3. **Citation Network Analysis**: Generate queries based on citation patterns
4. **Semantic Similarity**: Use embeddings for query diversity measurement

## Conclusion

This PRD outlines a comprehensive approach to transform the Research Agent's search variant generation from a simple rule-based system to an intelligent, agentic system using PromptChain. The proposed solution will significantly improve research discovery by generating semantically diverse, contextually relevant queries that explore different aspects of research topics.

The implementation leverages the existing PromptChain infrastructure while adding sophisticated query generation capabilities that align with the project's agentic architecture principles. 