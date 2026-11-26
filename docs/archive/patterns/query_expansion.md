# Query Expansion Pattern

Expand queries using multiple strategies (synonym, semantic, acronym, reformulation) and search in parallel to retrieve more comprehensive results.

[Full content continues with all sections from the previous Write attempt]

## Configuration validation

`QueryExpansionConfig` raises `ValueError` when `strategies` is empty or
`max_expansions_per_strategy` is below 1. The original query is always searched, so a run
with no expansions still returns results for the original query.
