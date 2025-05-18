---
noteId: "80ea65d0332f11f0a51a37b27c0392c3"
tags: []

---

# State Agent

## Overview

The State Agent is a specialized agent within the PromptChain framework that manages conversation session state and history. It provides a powerful interface for users to search, navigate, summarize, and manipulate conversation histories across multiple sessions.

Unlike traditional conversational agents, the State Agent acts as a "meta-agent" that can operate on the history itself, allowing users to leverage past conversations and build contextual continuity across different interactions.

## Key Features

### Session Management
- **Session Listing**: View recent or all conversation sessions
- **Session Loading**: Replace current conversation history with a previous session
- **Session Appending**: Add a previous session to the current conversation 
- **Cross-Session Search**: Find specific content across all stored sessions
- **Session Summarization**: Generate concise or detailed summaries of sessions
- **Inter-Session Comparison**: Analyze relationships between multiple sessions

### Memory Management
- **Internal Memory**: Maintains its own memory of session interactions
- **Session Tracking**: Remembers which sessions have been discovered
- **Cached Summaries**: Stores generated summaries for efficiency
- **Search History**: Tracks recent search queries and results

## Commands

| Command Pattern | Description | Example |
|----------------|-------------|---------|
| `@state: list recent sessions` | Show recently used sessions | `@state: list recent sessions` |
| `@state: list all sessions` | Show all available sessions | `@state: list all sessions` |
| `@state: list sessions about [topic]` | Show sessions related to a topic | `@state: list sessions about python` |
| `@state: find conversations about [topic]` | Search across sessions for content | `@state: find conversations about neural networks` |
| `@state: summarize session [UUID]` | Generate a detailed summary of a session | `@state: summarize session 550e8400-e29b-41d4-a716-446655440000` |
| `@state: load session [UUID]` | Replace current history with a session | `@state: load session 550e8400-e29b-41d4-a716-446655440000` |
| `@state: add session [UUID] to current conversation` | Append a session to current history | `@state: add session 550e8400-e29b-41d4-a716-446655440000` |
| `@state: compare sessions [UUID] [UUID]` | Compare and analyze relationships between sessions | `@state: compare sessions 550e8400-e29b-41d4-a716-446655440000 61c8809a-1dea-11eb-adc1-0242ac120002` |

## Practical Examples

### Finding and Continuing a Previous Conversation

```
User: @state: find conversations about machine learning
Agent: Found 3 matches for 'machine learning' across 2 sessions:

## Session 8a7b4c2d... (2 matches)
**Session Summary**: Discussion about machine learning frameworks and TensorFlow implementation details.
1. **user** (2023-04-12): Do you have examples of implementing CNN with TensorFlow?
...

## Session 9f4e7d8c... (1 match)
**Session Summary**: Debugging a PyTorch model training issue with CUDA.
1. **assistant (engineer)** (2023-04-15): To fix your CUDA memory issue in PyTorch, try...

Options:
- Load a session: 'load session [UUID]'
- Summarize a session: 'summarize session [UUID]'
- Continue searching: 'search for [new terms]'

User: @state: load session 8a7b4c2d
Agent: Successfully loaded session 8a7b4c2d. Loaded 12 conversation entries into the current context. You can now continue the conversation from this loaded session.

User: Can you show me how to implement a transformer architecture instead?
```

### Comparing Related Sessions

```
User: @state: list sessions about database
Agent: Found 3 sessions related to 'database':

1. **Session a1b2c3d4...** (Created: 2023-05-01)
   - Messages: 14
   - **Summary**: Discussion about SQL query optimization and indexing strategies.

2. **Session e5f6g7h8...** (Created: 2023-05-15)
   - Messages: 8
   - **Summary**: Troubleshooting MongoDB connection issues and configuration.

3. **Session i9j0k1l2...** (Created: 2023-06-02)
   - Messages: 21
   - **Summary**: Comparing different database systems for a web application.

User: @state: compare sessions a1b2c3d4 e5f6g7h8
Agent: # Comparison of 2 Sessions

## Session Summaries:
- Session a1b2c3d4...: Discussion about SQL query optimization and indexing strategies.
- Session e5f6g7h8...: Troubleshooting MongoDB connection issues and configuration.

## Relationship Analysis:
These sessions represent two different database paradigms - SQL and NoSQL. The first session focuses on optimization techniques within a relational database context, including indexing strategies, query planning, and normalization. The second session deals with operational aspects of MongoDB, focusing on connection issues, authentication, and server configuration. Together, they highlight different aspects of database management: performance optimization in SQL systems versus operational troubleshooting in NoSQL environments.

## Common Themes:
Common topics across sessions: database, performance, query, configuration, optimization

## Chronological Order:
1. Session a1b2c3d4... (Created: 2023-05-01)
2. Session e5f6g7h8... (Created: 2023-05-15)
```

## Implementation Details

The State Agent is implemented in `promptchain/utils/strategies/state_agent.py` as a specialized agent that integrates with the AgentChain and SQLite caching system. The agent maintains its own internal memory to track sessions and searches while manipulating the shared conversation history.

### Core Components

1. **StateAgent Class**: Main implementation with methods for handling different commands
2. **SQLite Integration**: Uses SQLite database for persistent storage of sessions
3. **Automatic Summarization**: Generates mini-summaries and full summaries of sessions
4. **Cross-Session Analysis**: Identifies relationships and common themes between sessions

### Technical Implementation

The State Agent operates by:
1. Parsing commands to identify the intended operation
2. Querying the SQLite database for relevant session data
3. Processing raw conversation data into structured information
4. Generating summaries or comparisons using a dedicated LLM instance
5. Manipulating the AgentChain's conversation history as needed
6. Maintaining internal memory for context across interactions

### Memory Structure

The agent's internal memory tracks:

```python
self.internal_memory = {
    "known_sessions": [],  # Sessions we've discovered
    "recent_searches": [],  # Search queries and results
    "session_summaries": {},  # Cached full summaries
    "mini_summaries": {},  # Cached mini-summaries
    "last_mentioned_session": None,  # Most recently referenced session
}
```

## Integration with Agent Orchestrator

The State Agent is integrated with the main Agent Orchestrator through:

```python
# Create the state agent
state_agent = create_state_agent(agent_orchestrator, verbose=True)

# Add the state agent to the orchestrator
agent_orchestrator.add_agent("state", state_agent)

# Add a description for the state agent
agent_orchestrator.agent_descriptions["state"] = "Manages conversation state and history..."
```

When users interact with the State Agent through the `@state:` command prefix, the agent manipulates the shared conversation history in the AgentChain instance, which affects all other agents' access to conversation context.

## Global vs. Internal Memory

It's important to understand that the State Agent maintains two types of memory:

1. **Global Conversation History**: Stored in `agent_chain._conversation_history` and shared with all agents
2. **Internal Agent Memory**: Stored in `self.internal_memory` and private to the State Agent

When loading or appending sessions, the State Agent modifies the global conversation history, affecting all agents. This allows seamless transition from managing sessions to continuing conversations with other specialized agents.

## Notes and Limitations

- Session UUIDs are shortened in UI displays but must be complete for operations
- Very large sessions may impact performance when summarizing or loading
- Token limits may truncate very long conversation histories
- Session search is currently keyword-based; future versions may add vector search
- The agent remembers sessions discovered in the current run but not across restarts

## Future Enhancements

Planned enhancements for the State Agent include:
- Vector embedding-based semantic search across sessions
- Multi-session merging and selective entry inclusion
- Enhanced relationship analysis between sessions
- Session tagging and categorization
- Session compression for very large conversation histories
- Collaborative session sharing between users 