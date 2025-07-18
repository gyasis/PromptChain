import os
import json
import re
import logging
import sqlite3
import uuid
import tiktoken
from typing import List, Dict, Any, Optional, Union, Callable
from datetime import datetime
import asyncio

from promptchain.utils.promptchaining import PromptChain
from promptchain.utils.agent_chain import AgentChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

logger = logging.getLogger(__name__)

# Initialize tiktoken encoder
try:
    GPT4_ENCODER = tiktoken.encoding_for_model("gpt-4")
except Exception as e:
    logger.warning(f"Failed to initialize tiktoken encoder: {e}. Will use character-based estimation.")
    GPT4_ENCODER = None

def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text string using the GPT-4 tokenizer.
    Falls back to character-based estimation if tiktoken is not available.
    
    Args:
        text: The text to count tokens for
        
    Returns:
        The number of tokens in the text
    """
    if GPT4_ENCODER is not None:
        try:
            return len(GPT4_ENCODER.encode(text))
        except Exception as e:
            logger.warning(f"Error counting tokens with tiktoken: {e}. Falling back to character estimation.")
    
    # Fallback to character-based estimation (roughly 4 chars per token)
    return len(text) // 4

# Constants
STATE_AGENT_PROMPT = """
You are the State Agent, responsible for managing conversation sessions and history.
Your job is to help users locate, search, summarize, and manipulate previous conversations.

Key capabilities:
1. Search through conversation history to find relevant content
2. Identify and list sessions based on topics or keywords
3. Load previous sessions into the current conversation
4. Generate summaries of previous conversations
5. Remember and track session UUIDs you've interacted with

You have access to the following tools:
- list_sessions: List available sessions with optional filtering
- list_sessions_table: List available conversation sessions in a markdown table format with summaries
- search_conversations: Search for content across conversation sessions
- load_session: Load a specific session by UUID
- append_session: Add a session to the current conversation
- summarize_session: Generate a summary of a session
- compare_sessions: Analyze the relationship between multiple sessions
- delete_session: Delete a specific session by UUID, removing all conversation entries and session data

When interacting:
1. Understand the user's intent even if they use natural language
2. Maintain context about searches and sessions the user has performed
3. Take a step-by-step approach to complex tasks
4. Always confirm before making changes that affect the conversation state

Your internal memory allows you to reference sessions discovered in previous interactions.
For example, if a user asks to "summarize that session we talked about earlier", you should 
use your memory to identify which session they mean.

FORMAT YOUR RESPONSES:
- Use clear, concise language
- Organize search results in tables or bullet points
- For session summaries, organize by topic
- Include actionable options after each response (e.g., "Would you like to load this session?")

Remember: You are handling the user's conversation history, which may contain important information.
Be precise, helpful, and maintain context across interactions.
"""

class StateAgent(AgenticStepProcessor):
    """
    A specialized agent for managing conversation session state and history.
    This agent provides capabilities to search, load, and manipulate conversation histories
    across different sessions while maintaining its own memory of interactions.
    """
    
    # Tool schemas for AgenticStepProcessor
    TOOL_SCHEMAS = [
        {
            "type": "function",
            "function": {
                "name": "list_sessions",
                "description": "List available conversation sessions with optional filtering",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filter_terms": {
                            "type": "string",
                            "description": "Optional keywords to filter sessions by topic"
                        },
                        "all_sessions": {
                            "type": "boolean",
                            "description": "Whether to include all sessions or just those we've interacted with"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of sessions to return"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_sessions_table",
                "description": "List available conversation sessions in a markdown table format with summaries",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filter_terms": {
                            "type": "string",
                            "description": "Optional keywords to filter sessions by topic"
                        },
                        "all_sessions": {
                            "type": "boolean",
                            "description": "Whether to include all sessions or just those we've interacted with"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of sessions to return"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_conversations",
                "description": "Search for content across conversation sessions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_terms": {
                            "type": "string",
                            "description": "Terms to search for in conversation content"
                        },
                        "search_all_instances": {
                            "type": "boolean",
                            "description": "Whether to search across all instances or just the current one"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return"
                        }
                    },
                    "required": ["search_terms"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "load_session",
                "description": "Load a previous conversation session by its UUID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_uuid": {
                            "type": "string",
                            "description": "UUID of the session to load"
                        },
                        "mode": {
                            "type": "string",
                            "description": "How to handle the loaded session",
                            "enum": ["replace_current", "append", "search_only"]
                        }
                    },
                    "required": ["session_uuid"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "summarize_session",
                "description": "Generate a summary of a specific conversation session",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_uuid": {
                            "type": "string",
                            "description": "UUID of the session to summarize"
                        },
                        "force_regeneration": {
                            "type": "boolean",
                            "description": "Force regeneration of the summary even if a cached one exists",
                            "default": False
                        }
                    },
                    "required": ["session_uuid"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "compare_sessions",
                "description": "Compare multiple sessions to identify relationships and common themes",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_uuids": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of session UUIDs to compare"
                        }
                    },
                    "required": ["session_uuids"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "add_search_to_global_history",
                "description": "Add a search query and its results to the global conversation history",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_term": {
                            "type": "string",
                            "description": "Optional specific search term to add. If not provided, adds the most recent search."
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "summary_crawler",
                "description": "Crawls through all sessions in the database and generates summaries for those that don't have one or need to be regenerated",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "regenerate_all": {
                            "type": "boolean",
                            "description": "Whether to regenerate all summaries regardless of existing values",
                            "default": False
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "regenerate_session_summary",
                "description": "Regenerates the summary for a specific session and updates it in the database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_uuid": {
                            "type": "string",
                            "description": "UUID of the session to regenerate the summary for"
                        }
                    },
                    "required": ["session_uuid"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "delete_session",
                "description": "Delete a specific session by UUID, removing all conversation entries and session data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_uuid": {
                            "type": "string",
                            "description": "UUID of the session to delete"
                        },
                        "confirm": {
                            "type": "boolean",
                            "description": "Confirmation flag to prevent accidental deletion",
                            "default": False
                        }
                    },
                    "required": ["session_uuid", "confirm"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "cleanup_tiny_sessions",
                "description": "Clean up sessions with 2 or fewer messages to reduce database clutter",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "confirm": {
                            "type": "boolean",
                            "description": "Confirmation flag to prevent accidental deletion",
                            "default": False
                        },
                        "max_messages": {
                            "type": "integer",
                            "description": "Maximum number of messages for a session to be considered 'tiny'",
                            "default": 2
                        }
                    },
                    "required": ["confirm"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "delete_small_sessions",
                "description": "Delete all sessions with a message count less than or equal to a specified threshold.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "max_messages": {
                            "type": "integer",
                            "description": "Maximum number of messages for a session to be deleted (inclusive)",
                            "default": 2
                        },
                        "confirm": {
                            "type": "boolean",
                            "description": "Confirmation flag to prevent accidental deletion",
                            "default": False
                        }
                    },
                    "required": ["max_messages", "confirm"]
                }
            }
        }
    ]
    
    def __init__(self, agent_chain: AgentChain, verbose: bool = False):
        """
        Initialize the StateAgent with a reference to the AgentChain it will manipulate.
        
        Args:
            agent_chain: The AgentChain instance this agent will work with
            verbose: Whether to enable verbose logging
        """
        # Initialize the AgenticStepProcessor parent class
        super().__init__(
            objective=STATE_AGENT_PROMPT,
            max_internal_steps=7,
            model_name="openai/gpt-4o-mini",
            model_params={"temperature": 0.2, "tool_choice": "auto"}
        )
        
        self.agent_chain = agent_chain
        self.verbose = verbose
        self.internal_memory = {
            "known_sessions": [],  # Sessions we've discovered or interacted with
            "recent_searches": [],  # Recent search queries and results
            "session_summaries": {},  # Cached summaries of sessions
            "mini_summaries": {},  # Cached mini-summaries of sessions
            "last_mentioned_session": None,  # Most recently discussed session
        }
        
        # Initialize the preprompter for loading external prompt files
        try:
            from promptchain.utils.preprompt import PrePrompt
            self.preprompter = PrePrompt()
            logger.info("Initialized PrePrompt for loading external prompt files")
        except ImportError as e:
            logger.warning(f"Failed to import PrePrompt: {e}. Will use inline prompts.")
            # Define a simple stub class that returns the input as fallback
            class FallbackPrePrompt:
                def load(self, prompt_id: str) -> str:
                    logger.warning(f"Using fallback for prompt ID '{prompt_id}', PrePrompt not available")
                    return f"FALLBACK_PROMPT_{prompt_id}" # This will trigger the except blocks in the methods
            self.preprompter = FallbackPrePrompt()
        
        # Ensure the agent chain has caching enabled
        if not self.agent_chain.enable_cache:
            logger.warning("StateAgent initialized with AgentChain that doesn't have caching enabled")
        
        # Initialize the summary database
        asyncio.create_task(self._initialize_summary_database())
        
        logger.info("StateAgent initialized successfully")
    
    async def _initialize_summary_database(self):
        """
        Initialize the summary database by ensuring the session table has the summary column
        and populating any missing summaries. This runs at startup.
        """
        if not self.agent_chain.enable_cache or not self.agent_chain.db_connection:
            logger.warning("Cannot initialize summary database - no cache or connection")
            return
            
        try:
            # Run the summary crawler to populate summaries for sessions that need them
            # We don't want to regenerate everything, just sessions with missing or generic summaries
            result = await self.summary_crawler(regenerate_all=False)
            
            if result.get("success", False):
                logger.info(f"Summary database initialization complete: {result.get('message')}")
            else:
                logger.error(f"Summary database initialization failed: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"Error initializing summary database: {e}", exc_info=True)
    
    async def process_command(self, command: str) -> str:
        """
        Process a state agent command and return a response.
        
        Args:
            command: The command string from the user
            
        Returns:
            Response message after processing the command
        """
        # Log the incoming command
        if self.verbose:
            print(f"StateAgent processing command: {command}")
        logger.info(f"StateAgent processing command: {command}")
        
        # Check for specific command types using more flexible pattern matching
        
        # List sessions commands - expanded pattern matching
        if re.search(r"list.*sessions|show.*sessions|what.*sessions|list_recent_sessions|sessions.*list|recent.*sessions", command.lower()):
            return await self._handle_list_sessions(command)
            
        # Search commands
        elif re.search(r"find|search|look for|where did|about|conversations.*about", command.lower()):
            return await self._handle_search(command)
            
        # Load session commands
        elif re.search(r"load session|use session|switch to session|session.*load|^load\s+\w+", command.lower()):
            return await self._handle_load_session(command)
            
        # Append session commands
        elif re.search(r"add session|append session|include session|merge session", command.lower()):
            return await self._handle_append_session(command)
            
        # Delete session commands - new section
        elif re.search(r"delete session|remove session|erase session|drop session|purge session", command.lower()):
            return await self._handle_delete_session(command)
            
        # Cleanup tiny sessions command
        elif re.search(r"cleanup tiny|cleanup small|cleanup empty|cleanup sessions|remove tiny|remove small|remove empty|delete tiny|delete small|delete empty|purge tiny|purge small|purge empty", command.lower()):
            return await self._handle_cleanup_tiny_sessions(command)
            
        # Add search to history command
        elif re.search(r"add search|save search|store search|add results|record search", command.lower()):
            # Extract the search term if specified
            search_term_match = re.search(r"for\s+['\"](.*?)['\"]|for\s+([\w\s]+)", command.lower())
            search_term = None
            if search_term_match:
                search_term = search_term_match.group(1) or search_term_match.group(2)
                search_term = search_term.strip()
            
            return await self.add_search_to_global_history(search_term)
            
        # Summarize commands
        elif re.search(r"summarize|summary|recap|sum up|brief|overview", command.lower()):
            # Check if it's a compare or inter-session summary request
            if re.search(r"relation|connect|related|between|across|compare", command.lower()):
                return await self._handle_compare_sessions(command)
            else:
                return await self._handle_summarize(command)
        
        # General/help command
        elif re.search(r"help|commands|what can you do|capabilities|features", command.lower()):
            help_text = """
I'm the State Agent, and I can help you manage your conversation history. Here are my main commands:

- **List Sessions**: `@state: list recent sessions` or `@state: list all sessions`
- **Search Content**: `@state: find conversations about [topic]` (search results are stored in my memory)
- **Add Search to History**: `@state: add search to history` (adds the most recent search to global history)
- **Summarize Sessions**: `@state: summarize session [ID]`
- **Load Sessions**: `@state: load session [ID]`
- **Delete Sessions**: `@state: delete session [ID] confirm` (irreversibly removes the session)
- **Cleanup Tiny Sessions**: `@state: cleanup tiny sessions confirm` (removes all sessions with 2 or fewer messages)
- **Compare Sessions**: `@state: compare sessions [ID1] and [ID2]`

You can use natural language for any of these commands. Just make sure to start with `@state:` to direct your message to me.

When you search for content, the results are stored in my internal memory but not automatically added to the conversation history. If you want to add search results to the global conversation history, use the "add search to history" command.
"""
            return help_text
            
        # General command processing using the LLM
        else:
            return await self._process_with_llm(command)
    
    async def _handle_search(self, command: str) -> str:
        """Handle search commands by finding relevant content in sessions."""
        # Extract search terms from command
        search_terms = re.sub(r"find|search|look for|where did|conversations about|we talk about", "", command.lower()).strip()
        
        if not search_terms:
            return "Please specify what you'd like to search for in the conversation history."
        
        # Search across all sessions
        search_results = self._search_sessions(search_terms, search_all_instances=True)
        
        if not search_results.get("success", False):
            return f"Error searching sessions: {search_results.get('error', 'Unknown error')}"
        
        # Track search in history (this updates both internal memory and global history)
        await self.track_search_in_history(search_terms, search_results)
        
        # If no matches were found
        if len(search_results.get("matches", [])) == 0:
            return f"I couldn't find any conversations about '{search_terms}'. Would you like to try a different search term?"
        
        # Format the search results
        matches = search_results.get("matches", [])
        
        # Group matches by session
        sessions = {}
        for match in matches:
            session_uuid = match.get("session_instance_uuid")
            if session_uuid not in sessions:
                sessions[session_uuid] = []
            sessions[session_uuid].append(match)
        
        # Format the response
        response_parts = [f"Found {len(matches)} matches for '{search_terms}' across {len(sessions)} sessions:"]
        
        for session_uuid, session_matches in sessions.items():
            # Store this session in known_sessions if not already there
            if session_uuid not in self.internal_memory["known_sessions"]:
                self.internal_memory["known_sessions"].append(session_uuid)
            
            # Set as last mentioned session
            self.internal_memory["last_mentioned_session"] = session_uuid
                
            response_parts.append(f"\n## Session ID: {session_uuid} ({len(session_matches)} matches)")
            
            # Get or generate a mini-summary for this session
            mini_summary = await self._get_or_generate_mini_summary(session_uuid)
            if mini_summary:
                response_parts.append(f"**Session Summary**: {mini_summary}")
            
            # Add a few sample matches from this session
            for i, match in enumerate(session_matches[:3]):  # Limit to 3 matches per session
                role = match.get("role", "")
                # Format role to show agent name if available
                if ":" in role and role.startswith("assistant:"):
                    role_parts = role.split(":", 1)
                    display_role = f"{role_parts[0]} ({role_parts[1]})"
                else:
                    display_role = role
                    
                timestamp = match.get("timestamp", "").split("T")[0]
                snippet = match.get("snippet", match.get("content", "")[:100] + "...")
                response_parts.append(f"{i+1}. **{display_role}** ({timestamp}): {snippet}")
            
            if len(session_matches) > 3:
                response_parts.append(f"...and {len(session_matches) - 3} more matches in this session.")
        
        response_parts.append("\nOptions:")
        response_parts.append("- Load a session: 'load session [Session ID]'")
        response_parts.append("- Summarize a session: 'summarize session [Session ID]'")
        response_parts.append("- Continue searching: 'search for [new terms]'")
        
        return "\n".join(response_parts)
    
    async def _handle_load_session(self, command: str) -> str:
        """Handle commands to load a specific session."""
        # Extract the session UUID from the command
        session_uuid_match = re.search(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", command)
        
        if not session_uuid_match:
            # Check if they're referring to "the session we just found" or similar
            if re.search(r"that session|the session|previous session|last session", command.lower()):
                # Use the last mentioned session from memory
                if self.internal_memory["last_mentioned_session"]:
                    session_uuid = self.internal_memory["last_mentioned_session"]
                else:
                    return "I don't have a specific session in mind. Please specify a session UUID or search for a session first."
            else:
                return "Please specify a valid session UUID to load, or search for sessions first."
        else:
            session_uuid = session_uuid_match.group(1)
        
        # Validate the session exists
        if not self._validate_session_exists(session_uuid):
            return f"Session with UUID {session_uuid} not found. Please check the UUID or search for available sessions."
        
        # Load the session using AgentChain's load_session method
        result = self._load_session(session_uuid, mode="replace_current")
        
        if not result.get("success", False):
            return f"Error loading session: {result.get('error', 'Unknown error')}"
        
        # Update internal memory
        if session_uuid not in self.internal_memory["known_sessions"]:
            self.internal_memory["known_sessions"].append(session_uuid)
        self.internal_memory["last_mentioned_session"] = session_uuid
        
        return f"Successfully loaded session {session_uuid}. Loaded {result.get('entries_loaded', 0)} conversation entries into the current context. You can now continue the conversation from this loaded session."
    
    async def _handle_append_session(self, command: str) -> str:
        """Handle commands to append a session to the current history."""
        # Extract the session UUID from the command
        session_uuid_match = re.search(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", command)
        
        if not session_uuid_match:
            # Check if they're referring to "the session we just found" or similar
            if re.search(r"that session|the session|previous session|last session", command.lower()):
                # Use the last mentioned session from memory
                if self.internal_memory["last_mentioned_session"]:
                    session_uuid = self.internal_memory["last_mentioned_session"]
                else:
                    return "I don't have a specific session in mind. Please specify a session UUID or search for a session first."
            else:
                return "Please specify a valid session UUID to append, or search for sessions first."
        else:
            session_uuid = session_uuid_match.group(1)
        
        # Validate the session exists
        if not self._validate_session_exists(session_uuid):
            return f"Session with UUID {session_uuid} not found. Please check the UUID or search for available sessions."
        
        # Append the session using AgentChain's load_session method
        result = self._load_session(session_uuid, mode="append")
        
        if not result.get("success", False):
            return f"Error appending session: {result.get('error', 'Unknown error')}"
        
        # Update internal memory
        if session_uuid not in self.internal_memory["known_sessions"]:
            self.internal_memory["known_sessions"].append(session_uuid)
        self.internal_memory["last_mentioned_session"] = session_uuid
        
        return f"Successfully appended session {session_uuid}. Added {result.get('entries_loaded', 0)} conversation entries to the current context. The current conversation now includes content from both sessions."
    
    async def _handle_list_sessions(self, command: str) -> str:
        """Handle list sessions commands by returning a formatted list of available sessions."""
        # Extract filter terms if present
        filter_terms = re.sub(r"list sessions|show sessions|what sessions|list_recent_sessions|list recent sessions", "", command.lower()).strip()
        
        # Determine if we want all sessions
        all_sessions = "all" in command.lower()
        
        # Check if we're requesting a simple format (non-table)
        simple_format = re.search(r"simple|no table|plain|text only", command.lower()) is not None
        
        # Check if we want to force regeneration of summaries
        regenerate_summaries = any(term in command.lower() for term in ["regenerate", "refresh", "update", "recrawl", "new summaries"])
        
        # Default limit
        limit = 20  # Increased default limit
        if re.search(r"show (\d+)|list (\d+)", command.lower()):
            limit_match = re.search(r"show (\d+)|list (\d+)", command.lower())
            if limit_match:
                limit = int(limit_match.group(1) or limit_match.group(2))
        
        # If we need to regenerate summaries, run the crawler first
        if regenerate_summaries:
            try:
                logger.info(f"Running summary crawler with regenerate_all={regenerate_summaries}")
                result = await self.summary_crawler(regenerate_all=regenerate_summaries)
                logger.info(f"Summary crawler completed: {result.get('message', 'No message')}")
            except Exception as e:
                logger.error(f"Error running summary crawler: {e}")
        
        # Use table format by default (it's more structured and readable)
        if simple_format:
            return await self.list_sessions(
                filter_terms=filter_terms if filter_terms else None,
                all_sessions=all_sessions,
                limit=limit
            )
        else:
            # Use the markdown table format for better readability
            return await self.get_sessions_markdown_table(
                filter_terms=filter_terms if filter_terms else None,
                all_sessions=all_sessions,
                limit=limit
            )
    
    async def _handle_summarize(self, command: str) -> str:
        """Handle commands to summarize a session."""
        # Extract the session UUID from the command
        session_uuid_match = re.search(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", command)
        
        # Check if this is a request to force regeneration
        force_regeneration = False
        if any(term in command.lower() for term in [
            "force", "regenerate", "refresh", "update", "complete", "detailed", "full", 
            "comprehensive", "new summary", "what we talked about", "talked about so far"
        ]):
            force_regeneration = True
            logger.info(f"Forcing summary regeneration based on command: '{command}'")
        
        if not session_uuid_match:
            # Check if they're referring to "the session we just found" or similar
            if re.search(r"that session|the session|previous session|last session", command.lower()):
                # Use the last mentioned session from memory
                if self.internal_memory["last_mentioned_session"]:
                    session_uuid = self.internal_memory["last_mentioned_session"]
                else:
                    return "I don't have a specific session in mind. Please specify a session UUID or search for a session first."
            else:
                # See if this is a request to summarize the current session
                if re.search(r"current|this session|what we talked about|so far", command.lower()):
                    # Assume they want to summarize the current session
                    # Use the session_id from the agent_chain
                    if not self.agent_chain.db_connection:
                        return "No active session found. Please start a conversation first."
                    
                    try:
                        # Get the current session instance UUID
                        current_session_id = self.agent_chain.cache_config.get("name", "default")
                        cursor = self.agent_chain.db_connection.cursor()
                        
                        cursor.execute(
                            """
                            SELECT session_instance_uuid 
                            FROM sessions 
                            WHERE session_id = ? 
                            ORDER BY created_at DESC LIMIT 1
                            """,
                            (current_session_id,)
                        )
                        
                        result = cursor.fetchone()
                        if result and result[0]:
                            session_uuid = result[0]
                            logger.info(f"Using current session {session_uuid} for summarization")
                        else:
                            return "No active session found. Please start a conversation first."
                    except Exception as e:
                        logger.error(f"Error fetching current session: {e}")
                        return "Could not determine the current session for summarization."
                else:
                    return "Please specify a valid session UUID to summarize, or search for sessions first."
        else:
            session_uuid = session_uuid_match.group(1)
        
        # Validate the session exists
        if not self._validate_session_exists(session_uuid):
            return f"Session with UUID {session_uuid} not found. Please check the UUID or search for available sessions."
        
        # Call the summarize_session method with the force_regeneration flag
        return await self.summarize_session(session_uuid, force_regeneration=force_regeneration)
    
    async def _handle_compare_sessions(self, command: str) -> str:
        """Handle commands to compare sessions and summarize relationships between them."""
        # Extract session UUIDs from the command
        # Look for multiple UUIDs in the command
        uuid_matches = re.findall(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", command)
        
        # If we don't have at least 2 UUIDs, check if we're referring to recent sessions
        if len(uuid_matches) < 2:
            # Check if they want to compare recently mentioned sessions
            if re.search(r"recent|latest|previous", command.lower()):
                # Use the most recent sessions from internal memory
                known_sessions = self.internal_memory["known_sessions"]
                if len(known_sessions) >= 2:
                    uuid_matches = known_sessions[-2:]  # Use the 2 most recent
                else:
                    return "I need at least 2 sessions to compare. Please specify session UUIDs or search for sessions first."
        
        # Validate all sessions exist
        for session_uuid in uuid_matches:
            if not self._validate_session_exists(session_uuid):
                return f"Session with UUID {session_uuid} not found. Please check the UUID or search for available sessions."
        
        # Compare the sessions
        comparison = self._compare_sessions(uuid_matches)
        
        if not comparison.get("success", False):
            return f"Error comparing sessions: {comparison.get('error', 'Unknown error')}"
        
        # Update internal memory to include all sessions being compared
        for session_uuid in uuid_matches:
            if session_uuid not in self.internal_memory["known_sessions"]:
                self.internal_memory["known_sessions"].append(session_uuid)
        
        # Generate a comprehensive inter-session summary using the LLM
        session_summaries = []
        
        for session_uuid in uuid_matches:
            # Get a summary for each session
            mini_summary = await self._get_or_generate_mini_summary(session_uuid)
            uuid_short = session_uuid[:8]
            session_summaries.append(f"- Session {uuid_short}...: {mini_summary}")
        
        relationships = comparison.get("relationships", [])
        common_terms = comparison.get("common_terms", [])
        
        inter_session_summarization_prompt = f"""
Please analyze the relationship between these conversation sessions and provide a concise summary of how they connect.

SESSION SUMMARIES:
{chr(10).join(session_summaries)}

IDENTIFIED RELATIONSHIPS:
{chr(10).join([rel.get("description", "") for rel in relationships])}

COMMON THEMES:
{', '.join(common_terms[:10]) if common_terms else 'No common themes identified'}

Provide a brief (3-5 sentences) analysis of how these sessions relate to each other and what insights might be gained by considering them together. Include mentions of progression, thematic connections, or contextual relationships.

INTER-SESSION ANALYSIS:
"""
        
        # Use a separate PromptChain for the inter-session analysis
        analyzer = PromptChain(
            models=[{"name": "openai/gpt-4o-mini", "params": {"temperature": 0.2}}],
            instructions=[inter_session_summarization_prompt],
            verbose=self.verbose
        )
        
        inter_session_summary = await analyzer.process_prompt_async("")
        
        # Format the response
        response_parts = [f"# Comparison of {len(uuid_matches)} Sessions"]
        
        # Add the mini-summaries
        response_parts.append("\n## Session Summaries:")
        for summary in session_summaries:
            response_parts.append(summary)
        
        # Add the inter-session analysis
        response_parts.append("\n## Relationship Analysis:")
        response_parts.append(inter_session_summary)
        
        # Add common themes if any
        if common_terms:
            response_parts.append("\n## Common Themes:")
            response_parts.append(f"Common topics across sessions: {', '.join(common_terms[:10])}")
        
        # Add chronological relationship if available
        for relationship in relationships:
            if relationship.get("type") == "chronological" and "order" in relationship:
                response_parts.append("\n## Chronological Order:")
                chronology = relationship.get("order", [])
                for i, entry in enumerate(chronology):
                    uuid = entry.get("uuid", "")
                    created_at = entry.get("created_at", "").split("T")[0]
                    response_parts.append(f"{i+1}. Session {uuid[:8]}... (Created: {created_at})")
        
        response_parts.append("\nOptions:")
        response_parts.append("- Load a session: 'load session [UUID]'")
        response_parts.append("- Search within these sessions: 'search for [terms]'")
        
        return "\n".join(response_parts)
    
    async def _process_with_llm(self, command: str) -> str:
        """Process commands that don't match specific patterns using the LLM."""
        # Check for table formatting requests - handle lots of variations
        table_pattern = r"table|markdown|md|tabular|structur|as a table"
        list_pattern = r"list.*sessions|show.*sessions|sessions.*list|recent.*sessions|what.*sessions|sessions.*can you"
        
        if re.search(table_pattern, command.lower()) and re.search(list_pattern, command.lower()):
            return await self.get_sessions_markdown_table()
            
        # Check for list command patterns first since they're common
        if re.search(r"list.*sessions|show.*sessions|sessions.*list|recent.*sessions|available.*sessions|list_recent_sessions", command.lower()):
            return await self._handle_list_sessions(command)
            
        # For general commands, update internal memory representation
        internal_memory_str = json.dumps(self.internal_memory, indent=2)
        
        prompt = f"""
Command from user: "{command}"

Your internal memory state:
{internal_memory_str}

Analyze this command in the context of your role as a State Agent. 
If it's related to session management, history search, or conversation state,
respond appropriately.

If you need to perform a specific operation like searching or loading sessions,
explain what you would do and what the expected outcome would be.

If this doesn't seem like a command for the State Agent, explain that you're
specialized in managing conversation history and sessions, and suggest some
relevant commands the user might want to try.
"""
        
        try:
            response = await self.prompt_chain.process_prompt_async(prompt)
            
            # Add some action suggestions if not already present
            if "you can" not in response.lower() and "available commands" not in response.lower():
                response += "\n\nHere are some commands you can use:\n"
                response += "- Search for content: `@state: find conversations about X`\n"
                response += "- List sessions: `@state: list recent sessions`\n"
                response += "- Summarize a session: `@state: summarize session [ID]`\n"
                response += "- Load a session: `@state: load session [ID]`"
                
            return response
        except Exception as e:
            logger.error(f"Error processing command with LLM: {e}")
            return f"I encountered an error processing your command: {e}. Please try again with a more specific command like 'list recent sessions' or 'find conversations about [topic]'."
    
    def _search_sessions(self, query: str, search_all_instances: bool = True, max_results: int = 20) -> Dict[str, Any]:
        """
        Search across conversation sessions for content matching the query.
        
        Args:
            query: Search terms to look for in conversation content
            search_all_instances: Whether to search across all sessions or just the current one
            max_results: Maximum number of matching entries to return
            
        Returns:
            Dict with search results and metadata
        """
        if not self.agent_chain.enable_cache:
            return {"success": False, "error": "Cache is not enabled", "matches": []}
            
        if not self.agent_chain.db_connection:
            return {"success": False, "error": "Database connection not established", "matches": []}
            
        try:
            cursor = self.agent_chain.db_connection.cursor()
            
            # Get the database name
            db_name = self.agent_chain.cache_config.get("name", "default")
            
            # Prepare query condition based on search_all_instances flag
            instance_condition = ""
            params = [f"%{query}%", f"%{query}%", db_name]
            
            if not search_all_instances:
                instance_condition = "AND session_instance_uuid = ?"
                params.append(self.agent_chain.session_instance_uuid)
            
            # Search for matching content
            cursor.execute(
                f"""
                SELECT session_instance_uuid, role, content, timestamp 
                FROM conversation_entries 
                WHERE (content LIKE ? OR role LIKE ?) 
                AND session_id = ? 
                {instance_condition}
                ORDER BY id DESC
                LIMIT ?
                """,
                params + [max_results]
            )
            
            matches = []
            for session_uuid, role, content, timestamp in cursor.fetchall():
                matches.append({
                    "session_instance_uuid": session_uuid,
                    "role": role,
                    "content": content,
                    "timestamp": timestamp,
                    # Include a snippet around the matching part
                    "snippet": self._extract_snippet(content, query, context_chars=100)
                })
                    
            return {
                "success": True,
                "query": query,
                "search_all_instances": search_all_instances,
                "matches_found": len(matches),
                "matches": matches
            }
            
        except Exception as e:
            logger.error(f"Error searching sessions: {e}")
            return {"success": False, "error": str(e), "matches": []}
    
    def _extract_snippet(self, text: str, query: str, context_chars: int = 100) -> str:
        """Extract a snippet of text around the query match with context."""
        if not query or not text:
            return text[:200] + "..." if len(text) > 200 else text
            
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Find position of the query in the text
        pos = text_lower.find(query_lower)
        if pos == -1:
            # If exact match not found, return start of text
            return text[:200] + "..." if len(text) > 200 else text
        
        # Calculate start and end positions for snippet
        start = max(0, pos - context_chars)
        end = min(len(text), pos + len(query) + context_chars)
        
        # Add ellipsis if the snippet doesn't start or end at the text boundaries
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        
        return prefix + text[start:end] + suffix
    
    def _load_session(self, session_uuid: str, mode: str = "replace_current", max_entries: int = None) -> Dict[str, Any]:
        """
        Load a previous conversation session by its UUID.
        Wrapper around the agent_chain's similar functionality.
        
        Args:
            session_uuid: The UUID of the session to load
            mode: How to handle the loaded session ('replace_current', 'append', 'search_only')
            max_entries: Maximum number of entries to load (newest first if limited)
            
        Returns:
            Dict with operation results including success status and loaded entries
        """
        if not self.agent_chain.enable_cache:
            return {"success": False, "error": "Cache is not enabled", "entries_loaded": 0}
            
        if not self.agent_chain.db_connection:
            return {"success": False, "error": "Database connection not established", "entries_loaded": 0}
            
        try:
            cursor = self.agent_chain.db_connection.cursor()
            
            # Get the database name
            db_name = self.agent_chain.cache_config.get("name", "default")
            
            # Verify the session exists
            cursor.execute(
                "SELECT session_id FROM sessions WHERE session_instance_uuid = ?",
                (session_uuid,)
            )
            session_result = cursor.fetchone()
            
            if not session_result:
                return {"success": False, "error": f"Session with ID {session_uuid} not found", "entries_loaded": 0}
            
            # Get conversation entries for the specified session
            query = """
            SELECT role, content, timestamp 
            FROM conversation_entries 
            WHERE session_id = ? AND session_instance_uuid = ? 
            ORDER BY id
            """
            params = (db_name, session_uuid)
            
            if max_entries:
                # Add LIMIT to the query if max_entries is specified
                query += f" LIMIT {int(max_entries)}"
                
            cursor.execute(query, params)
            entries = []
            
            for role, content, timestamp in cursor.fetchall():
                entries.append({"role": role, "content": content, "timestamp": timestamp})
            
            # Handle the loaded entries based on mode
            if mode == "replace_current":
                # Clear current history and replace with loaded session
                self.agent_chain._conversation_history = entries
                if self.verbose:
                    print(f"Replaced current history with {len(entries)} entries from session {session_uuid}")
                    
            elif mode == "append":
                # Append loaded entries to current history
                self.agent_chain._conversation_history.extend(entries)
                if self.verbose:
                    print(f"Appended {len(entries)} entries from session {session_uuid} to current history")
                    
            # For search_only mode, we don't modify the history
            
            return {
                "success": True, 
                "entries_loaded": len(entries),
                "session_id": db_name,
                "session_uuid": session_uuid,
                "mode": mode,
                "entries": entries if mode == "search_only" else None
            }
            
        except Exception as e:
            logger.error(f"Error loading session {session_uuid}: {e}")
            return {"success": False, "error": str(e), "entries_loaded": 0}
    
    async def load_session(self, session_uuid: str, mode: str = "replace_current") -> str:
        """
        Load a previous conversation session by UUID for the user.
        This is a public wrapper around the internal _load_session method.
        
        Args:
            session_uuid: UUID of the session to load
            mode: How to handle the loaded session ('replace_current', 'append', 'search_only')
            
        Returns:
            Formatted response message about the loading operation
        """
        if not session_uuid:
            return "Please provide a valid session UUID to load."
            
        # Validate the session exists
        if not self._validate_session_exists(session_uuid):
            return f"Session with UUID {session_uuid} not found. Please check the UUID or search for available sessions."
            
        # Load the session
        result = self._load_session(session_uuid, mode)
        
        if not result.get("success", False):
            return f"Error loading session: {result.get('error', 'Unknown error')}"
            
        # Update internal memory
        if session_uuid not in self.internal_memory["known_sessions"]:
            self.internal_memory["known_sessions"].append(session_uuid)
        self.internal_memory["last_mentioned_session"] = session_uuid
        
        return f"Successfully loaded session {session_uuid}. Loaded {result.get('entries_loaded', 0)} conversation entries into the current context. You can now continue the conversation from this loaded session."

    async def delete_session(self, session_uuid: str, confirm: bool = False) -> Dict[str, Any]:
        """
        Delete a session and all its conversation entries from the database.
        
        Args:
            session_uuid: UUID of the session to delete
            confirm: Confirmation flag to prevent accidental deletion
            
        Returns:
            Dict with operation results
        """
        if not confirm:
            return {"success": False, "error": "Deletion requires confirmation", "message": "Please set confirm=True to proceed with deletion"}
            
        if not self.agent_chain.enable_cache:
            return {"success": False, "error": "Cache is not enabled", "message": "Cannot delete session when caching is disabled"}
            
        if not self.agent_chain.db_connection:
            return {"success": False, "error": "Database connection not established", "message": "No database connection available"}
            
        # Validate the session exists
        if not self._validate_session_exists(session_uuid):
            return {"success": False, "error": f"Session with UUID {session_uuid} not found", "message": "Session not found in database"}
            
        try:
            cursor = self.agent_chain.db_connection.cursor()
            db_name = self.agent_chain.cache_config.get("name", "default")
            
            # Count messages before deletion for reporting
            cursor.execute(
                """
                SELECT COUNT(*) 
                FROM conversation_entries 
                WHERE session_id = ? AND session_instance_uuid = ?
                """,
                (db_name, session_uuid)
            )
            
            count_result = cursor.fetchone()
            message_count = count_result[0] if count_result else 0
            
            # Start a transaction
            self.agent_chain.db_connection.execute("BEGIN TRANSACTION")
            
            # Delete conversation entries
            cursor.execute(
                """
                DELETE FROM conversation_entries 
                WHERE session_id = ? AND session_instance_uuid = ?
                """,
                (db_name, session_uuid)
            )
            
            entries_deleted = cursor.rowcount
            
            # Delete the session record
            cursor.execute(
                """
                DELETE FROM sessions 
                WHERE session_id = ? AND session_instance_uuid = ?
                """,
                (db_name, session_uuid)
            )
            
            session_deleted = cursor.rowcount
            
            # Commit the transaction
            self.agent_chain.db_connection.commit()
            
            logger.info(f"Deleted session {session_uuid} with {entries_deleted} entries")
            
            return {
                "success": True,
                "session_uuid": session_uuid,
                "entries_deleted": entries_deleted,
                "session_deleted": session_deleted,
                "message": f"Deleted {entries_deleted} conversation entries and session record"
            }
            
        except Exception as e:
            # Rollback on error
            try:
                self.agent_chain.db_connection.rollback()
            except:
                pass
                
            logger.error(f"Error deleting session {session_uuid}: {e}", exc_info=True)
            return {"success": False, "error": str(e), "message": "Database error during deletion"}

    async def _handle_cleanup_tiny_sessions(self, command: str) -> str:
        """Handle commands to clean up sessions with 2 or fewer messages."""
        # Check for automatic confirmation in the command
        confirm = any(word in command.lower() for word in 
                     ["confirm", "yes", "definitely", "sure", "please", "go ahead", "do it"])
        
        # Extract max_messages if specified
        max_messages = 2  # Default
        max_msg_match = re.search(r"(\d+)\s*(or fewer|or less|messages|msg)", command.lower())
        if max_msg_match:
            try:
                max_messages = int(max_msg_match.group(1))
                # Cap at reasonable value to prevent accidental deletion of large sessions
                max_messages = min(max_messages, 5)
            except ValueError:
                pass
        
        # If not confirmed, ask for confirmation
        if not confirm:
            return f"Are you sure you want to delete all sessions with {max_messages} or fewer messages? This action cannot be undone. To confirm, say '@state: cleanup tiny sessions confirm'."
        
        # Call the cleanup function
        result = await self.cleanup_tiny_sessions(confirm=True, max_messages=max_messages)
        
        if result.get("success", False):
            # Update internal memory to remove deleted sessions
            for session_uuid in result.get("deleted_sessions", []):
                if session_uuid in self.internal_memory["known_sessions"]:
                    self.internal_memory["known_sessions"].remove(session_uuid)
                
                if self.internal_memory["last_mentioned_session"] == session_uuid:
                    self.internal_memory["last_mentioned_session"] = None
                    
                if session_uuid in self.internal_memory["mini_summaries"]:
                    del self.internal_memory["mini_summaries"][session_uuid]
            
            return f"Successfully cleaned up tiny sessions. {result.get('message', '')}"
        else:
            return f"Error cleaning up tiny sessions: {result.get('error', 'Unknown error')}"

    async def cleanup_tiny_sessions(self, confirm: bool = False, max_messages: int = 2) -> Dict[str, Any]:
        """
        Clean up sessions with few messages to reduce database clutter.
        
        Args:
            confirm: Confirmation flag to prevent accidental deletion
            max_messages: Maximum number of messages for a session to be considered 'tiny'
            
        Returns:
            Dict with operation results
        """
        if not confirm:
            return {"success": False, "error": "Deletion requires confirmation", "message": "Please set confirm=True to proceed with deletion"}
            
        if not self.agent_chain.enable_cache:
            return {"success": False, "error": "Cache is not enabled", "message": "Cannot delete sessions when caching is disabled"}
            
        if not self.agent_chain.db_connection:
            return {"success": False, "error": "Database connection not established", "message": "No database connection available"}
        
        try:
            cursor = self.agent_chain.db_connection.cursor()
            db_name = self.agent_chain.cache_config.get("name", "default")
            
            # First, find all sessions with message count <= max_messages
            cursor.execute(
                """
                SELECT s.session_instance_uuid, COUNT(c.id) as msg_count
                FROM sessions s
                LEFT JOIN conversation_entries c ON s.session_id = c.session_id AND s.session_instance_uuid = c.session_instance_uuid
                WHERE s.session_id = ?
                GROUP BY s.session_instance_uuid
                HAVING COUNT(c.id) <= ?
                """,
                (db_name, max_messages)
            )
            
            tiny_sessions = cursor.fetchall()
            
            if not tiny_sessions:
                return {"success": True, "message": f"No sessions found with {max_messages} or fewer messages", "deleted_count": 0}
            
            # Start a transaction for the batch deletion
            self.agent_chain.db_connection.execute("BEGIN TRANSACTION")
            
            total_entries_deleted = 0
            deleted_sessions = []
            
            # Process each tiny session
            for session_uuid, msg_count in tiny_sessions:
                # Delete conversation entries
                cursor.execute(
                    """
                    DELETE FROM conversation_entries 
                    WHERE session_id = ? AND session_instance_uuid = ?
                    """,
                    (db_name, session_uuid)
                )
                
                entries_deleted = cursor.rowcount
                total_entries_deleted += entries_deleted
                
                # Delete the session record
                cursor.execute(
                    """
                    DELETE FROM sessions 
                    WHERE session_id = ? AND session_instance_uuid = ?
                    """,
                    (db_name, session_uuid)
                )
                
                deleted_sessions.append(session_uuid)
                
                logger.info(f"Deleted tiny session {session_uuid} with {msg_count} messages")
            
            # Commit the transaction
            self.agent_chain.db_connection.commit()
            
            return {
                "success": True,
                "deleted_count": len(deleted_sessions),
                "deleted_sessions": deleted_sessions,
                "total_entries_deleted": total_entries_deleted,
                "message": f"Deleted {len(deleted_sessions)} sessions with {max_messages} or fewer messages ({total_entries_deleted} total entries removed)"
            }
            
        except Exception as e:
            # Rollback on error
            try:
                self.agent_chain.db_connection.rollback()
            except:
                pass
                
            logger.error(f"Error cleaning up tiny sessions: {e}", exc_info=True)
            return {"success": False, "error": str(e), "message": "Database error during cleanup"}

    async def search_conversations(self, search_terms: str, search_all_instances: bool = True, max_results: int = 20) -> str:
        """
        Search across conversation sessions for specific content.
        Always returns a user-facing, final answer to prevent tool call loops.
        """
        if not search_terms:
            return "Please specify search terms to find in conversation history."
        search_results = self._search_sessions(search_terms, search_all_instances, max_results)
        if not search_results.get("success", False):
            return f"Error searching conversations: {search_results.get('error', 'Unknown error')}"
        await self.track_search_in_history(search_terms, search_results)
        matches = search_results.get("matches", [])
        if not matches:
            return f"No conversations found matching '{search_terms}'. Try a different search term or check if there are any conversations stored."
        sessions = {}
        for match in matches:
            session_uuid = match.get("session_instance_uuid")
            if session_uuid not in sessions:
                sessions[session_uuid] = []
            sessions[session_uuid].append(match)
            if session_uuid not in self.internal_memory["known_sessions"]:
                self.internal_memory["known_sessions"].append(session_uuid)
            self.internal_memory["last_mentioned_session"] = session_uuid
        response_parts = [f"# Search Results for '{search_terms}'"]
        response_parts.append(f"Found {len(matches)} matches across {len(sessions)} sessions.")
        for session_uuid, session_matches in sessions.items():
            response_parts.append(f"\n## Session {session_uuid}")
            try:
                mini_summary = await self._get_or_generate_mini_summary(session_uuid)
                if mini_summary:
                    response_parts.append(f"**Summary**: {mini_summary}")
            except Exception as e:
                logger.error(f"Error getting summary during search: {e}")
            response_parts.append(f"\n**Matching Messages ({len(session_matches)}):**")
            for i, match in enumerate(session_matches[:5]):
                role = match.get("role", "unknown")
                timestamp = match.get("timestamp", "").split("T")[0] if match.get("timestamp") else ""
                if ":" in role and role.startswith("assistant:"):
                    role_parts = role.split(":", 1)
                    display_role = f"{role_parts[0]} ({role_parts[1]})"
                else:
                    display_role = role.capitalize()
                snippet = match.get("snippet", "")
                if not snippet and match.get("content"):
                    content = match.get("content", "")
                    snippet = content[:200] + "..." if len(content) > 200 else content
                response_parts.append(f"{i+1}. **{display_role}** ({timestamp}): {snippet}")
            if len(session_matches) > 5:
                response_parts.append(f"...and {len(session_matches) - 5} more matches in this session.")
        response_parts.append("\n## Next Actions")
        response_parts.append("- **Load a session**: `@state: load session [UUID]`")
        response_parts.append("- **Get full summary**: `@state: summarize session [UUID]`")
        response_parts.append("- **Refine search**: `@state: search for [refined terms]`")
        response_parts.append("- **Add to history**: `@state: add search to history`")
        # Return a user-facing, final message
        return "\n".join(response_parts)

    async def summarize_session(self, session_uuid: str, force_regeneration: bool = False) -> str:
        """
        Generate a summary of a specific conversation session.
        
        Args:
            session_uuid: UUID of the session to summarize
            force_regeneration: Whether to force regeneration of the summary
            
        Returns:
            A detailed summary of the session
        """
        if not session_uuid:
            return "Please provide a valid session UUID to summarize."
            
        # Validate the session exists
        if not self._validate_session_exists(session_uuid):
            return f"Session with UUID {session_uuid} not found. Please check the UUID or search for available sessions."
            
        try:
            # First, get basic session info
            cursor = self.agent_chain.db_connection.cursor()
            db_name = self.agent_chain.cache_config.get("name", "default")
            
            # Get creation time and message count
            cursor.execute(
                """
                SELECT s.created_at, COUNT(c.id) as message_count
                FROM sessions s
                LEFT JOIN conversation_entries c ON s.session_id = c.session_id AND s.session_instance_uuid = c.session_instance_uuid
                WHERE s.session_id = ? AND s.session_instance_uuid = ?
                GROUP BY s.session_instance_uuid
                """,
                (db_name, session_uuid)
            )
            
            result = cursor.fetchone()
            if not result:
                return f"Session with UUID {session_uuid} found, but no metadata is available."
                
            created_at, message_count = result
            
            # Get a mini-summary first
            mini_summary = await self._get_or_generate_mini_summary(session_uuid, force_regeneration)
            
            # For very small sessions, the mini summary is sufficient
            if message_count <= 2:
                return f"""
# Session Summary: {session_uuid}

**Created:** {created_at.split('T')[0] if 'T' in created_at else created_at}
**Messages:** {message_count}

{mini_summary}

*This session contains only {message_count} messages, so a brief summary is provided.*
"""
            
            # For larger sessions, generate a more detailed summary
            # Get a representative sample of messages
            cursor.execute(
                """
                SELECT role, content 
                FROM conversation_entries 
                WHERE session_id = ? AND session_instance_uuid = ? 
                ORDER BY id
                LIMIT 20
                """,
                (db_name, session_uuid)
            )
            
            messages = cursor.fetchall()
            
            # Format conversation for summarization
            conversation = "\n\n".join([f"{role}: {content[:200] + '...' if len(content) > 200 else content}" for role, content in messages])
            
            # Prepare the summarization prompt
            summary_prompt = f"""
# Conversation Session Summary Generator

## Task
Create a comprehensive summary of a conversation session, capturing key topics, questions, and conclusions.

## Input
Conversation session from {created_at.split('T')[0] if 'T' in created_at else created_at} with {message_count} total messages.
Sample of conversation (first 20 messages or fewer):

{conversation}

## Instructions
1. Summarize the main topics discussed in the conversation
2. Identify any key questions asked and answers provided
3. Note any decisions made or conclusions reached
4. Highlight any action items or next steps mentioned
5. Format your summary using Markdown with clear sections

## Output Format
Provide a comprehensive yet concise summary organized into these sections:
- Main Topics
- Key Points
- Conclusions (if any)
"""
            
            # Use a temporary PromptChain for summarization
            summary_chain = PromptChain(
                models=[{"name": "openai/gpt-4o-mini", "params": {"temperature": 0.2}}],
                instructions=[summary_prompt],
                verbose=self.verbose
            )
            
            detailed_summary = await summary_chain.process_prompt_async("")
            
            # Format the final response
            response = f"""
# Session Summary: {session_uuid}

**Created:** {created_at.split('T')[0] if 'T' in created_at else created_at}
**Messages:** {message_count}
**Brief Description:** {mini_summary}

## Detailed Summary:
{detailed_summary}

*This summary was generated from an analysis of the conversation. For the full conversation, use the load session command.*
"""
            
            # Add this session to known sessions
            if session_uuid not in self.internal_memory["known_sessions"]:
                self.internal_memory["known_sessions"].append(session_uuid)
                
            self.internal_memory["last_mentioned_session"] = session_uuid
            
            return response
            
        except Exception as e:
            logger.error(f"Error summarizing session {session_uuid}: {e}", exc_info=True)
            return f"Error generating summary: {str(e)}"
    
    async def compare_sessions(self, session_uuids: List[str]) -> str:
        """
        Compare multiple sessions to identify relationships and common themes.
        
        Args:
            session_uuids: List of session UUIDs to compare
            
        Returns:
            Formatted comparison report
        """
        if not session_uuids or len(session_uuids) < 2:
            return "Please provide at least two session UUIDs to compare."
            
        # Validate all sessions exist
        invalid_uuids = []
        for uuid in session_uuids:
            if not self._validate_session_exists(uuid):
                invalid_uuids.append(uuid)
                
        if invalid_uuids:
            return f"The following sessions were not found: {', '.join(invalid_uuids)}. Please verify the UUIDs and try again."
            
        # Call the internal comparison method
        comparison_result = self._compare_sessions(session_uuids)
        
        if not comparison_result.get("success", False):
            return f"Error comparing sessions: {comparison_result.get('error', 'Unknown error')}"
            
        # Build a formatted comparison report
        session_summaries = []
        
        # Get summaries for each session
        for uuid in session_uuids:
            try:
                summary = await self._get_or_generate_mini_summary(uuid)
                
                # Get creation time and message count
                cursor = self.agent_chain.db_connection.cursor()
                db_name = self.agent_chain.cache_config.get("name", "default")
                
                cursor.execute(
                    """
                    SELECT s.created_at, COUNT(c.id) as message_count
                    FROM sessions s
                    LEFT JOIN conversation_entries c ON s.session_id = c.session_id AND s.session_instance_uuid = c.session_instance_uuid
                    WHERE s.session_id = ? AND s.session_instance_uuid = ?
                    GROUP BY s.session_instance_uuid
                    """,
                    (db_name, uuid)
                )
                
                session_info = cursor.fetchone()
                created_at = session_info[0] if session_info else "Unknown date"
                message_count = session_info[1] if session_info else 0
                
                session_summaries.append({
                    "uuid": uuid,
                    "summary": summary,
                    "created_at": created_at.split('T')[0] if isinstance(created_at, str) and 'T' in created_at else created_at,
                    "message_count": message_count
                })
                
                # Add to known sessions
                if uuid not in self.internal_memory["known_sessions"]:
                    self.internal_memory["known_sessions"].append(uuid)
                    
            except Exception as e:
                logger.error(f"Error getting summary for session {uuid}: {e}")
                session_summaries.append({
                    "uuid": uuid,
                    "summary": "Summary unavailable",
                    "created_at": "Unknown",
                    "message_count": "Unknown"
                })
        
        # Get relationship information
        relationships = comparison_result.get("relationships", [])
        common_terms = comparison_result.get("common_terms", [])
        
        # Format the response
        response_parts = [f"# Comparison of {len(session_uuids)} Sessions"]
        
        # Add session summaries
        response_parts.append("\n## Session Summaries")
        for summary in session_summaries:
            response_parts.append(f"### Session {summary['uuid'][:8]}...")
            response_parts.append(f"- **Created:** {summary['created_at']}")
            response_parts.append(f"- **Messages:** {summary['message_count']}")
            response_parts.append(f"- **Summary:** {summary['summary']}")
            response_parts.append("")
        
        # Add common themes
        if common_terms:
            response_parts.append("## Common Themes")
            response_parts.append(f"These terms appear across all compared sessions: {', '.join(common_terms[:10])}")
            response_parts.append("")
        
        # Add relationships
        if relationships:
            response_parts.append("## Relationships")
            for relationship in relationships:
                response_parts.append(f"- {relationship.get('description', 'Unknown relationship')}")
                
            # Add chronological relationship if available
            chronological_rel = next((r for r in relationships if r.get("type") == "chronological"), None)
            if chronological_rel and "order" in chronological_rel:
                response_parts.append("\n### Chronological Order")
                chronology = chronological_rel.get("order", [])
                for i, entry in enumerate(chronology, 1):
                    uuid = entry.get("uuid", "")
                    created_at = entry.get("created_at", "").split("T")[0] if isinstance(entry.get("created_at", ""), str) else entry.get("created_at", "")
                    response_parts.append(f"{i}. Session {uuid[:8]}... (Created: {created_at})")
        
        # Generate an inter-session analysis using the LLM
        session_summary_texts = [f"Session {s['uuid'][:8]}...: {s['summary']}" for s in session_summaries]
        
        analysis_prompt = f"""
Analyze the relationship between these conversation sessions and provide a concise summary of how they connect.

SESSION SUMMARIES:
{chr(10).join(session_summary_texts)}

COMMON THEMES:
{', '.join(common_terms[:10]) if common_terms else 'No common themes identified'}

Provide a brief (3-5 sentences) analysis of how these sessions relate to each other and what insights might be gained by considering them together. Include mentions of progression, thematic connections, or contextual relationships.
"""
        
        try:
            # Use a temporary PromptChain for analysis
            analysis_chain = PromptChain(
                models=[{"name": "openai/gpt-4o-mini", "params": {"temperature": 0.2}}],
                instructions=[analysis_prompt],
                verbose=self.verbose
            )
            
            inter_session_analysis = await analysis_chain.process_prompt_async("")
            
            response_parts.append("\n## Analysis")
            response_parts.append(inter_session_analysis)
        except Exception as e:
            logger.error(f"Error generating inter-session analysis: {e}")
            response_parts.append("\n## Analysis")
            response_parts.append("Error generating inter-session analysis.")
        
        # Add next steps
        response_parts.append("\n## Next Steps")
        response_parts.append("- **Load a session**: `@state: load session [UUID]`")
        response_parts.append("- **Get detailed summary**: `@state: summarize session [UUID]`")
        response_parts.append("- **Search within these sessions**: `@state: search for [topic]`")
        
        return "\n".join(response_parts)

    async def track_search_in_history(self, search_term: str, search_results: Dict[str, Any]) -> None:
        """
        Track search queries in the agent's internal memory only.
        
        Args:
            search_term: The search term used for the query
            search_results: The results dictionary from _search_sessions
            
        Returns:
            None (modifies internal memory only)
        """
        # Add to internal memory for agent's reference
        self.internal_memory["recent_searches"].append({
            "timestamp": datetime.now().isoformat(),
            "query": search_term,
            "results": search_results,
            "result_count": len(search_results.get("matches", []))
        })
        
        return None
        
    async def add_search_to_global_history(self, search_term: str = None) -> str:
        """
        Add the most recent search (or a specific search term) to the global conversation history.
        Ensures a user-facing, final answer is always returned to break tool call loops.
        """
        # Find the relevant search in internal memory
        if search_term:
            matching_searches = [s for s in self.internal_memory["recent_searches"] 
                               if s.get("query", "").lower() == search_term.lower()]
            if not matching_searches:
                return f"No search for '{search_term}' found in memory. Please search for this term first."
            target_search = matching_searches[-1]
        else:
            if not self.internal_memory["recent_searches"]:
                return "No recent searches found in memory. Please perform a search first."
            target_search = self.internal_memory["recent_searches"][-1]
        search_term = target_search.get("query", "unknown")
        search_results = target_search.get("results", {})
        match_count = len(search_results.get("matches", []))
        session_count = len({match.get("session_instance_uuid") for match in search_results.get("matches", [])})
        summary_text = f"Search query: '{search_term}'\n"
        summary_text += f"Found {match_count} matches across {session_count} sessions.\n"
        if match_count > 0:
            summary_text += "\nSample matches:\n"
            for i, match in enumerate(search_results.get("matches", [])[:3]):
                snippet = match.get("snippet", "")[:100] + "..." if len(match.get("snippet", "")) > 100 else match.get("snippet", "")
                summary_text += f"{i+1}. {snippet}\n"
        search_entry = {
            "role": "assistant:state_agent",
            "content": summary_text,
            "timestamp": datetime.now().isoformat(),
            "metadata": {"search_term": search_term, "is_search_result": True}
        }
        self.agent_chain._conversation_history.append(search_entry)
        # Return a user-facing, final message to break tool call loops
        return f"✅ Search results for '{search_term}' have been added to your global conversation history.\n\nWould you like to view them or perform another action?"

    async def list_sessions(self, filter_terms: str = None, all_sessions: bool = True, limit: int = 10) -> str:
        """
        List available sessions, optionally filtered by topic.
        
        Args:
            filter_terms: Optional keywords to filter sessions by topic
            all_sessions: Whether to include all sessions or just those we've interacted with
            limit: Maximum number of sessions to return
            
        Returns:
            Formatted string with list of available sessions
        """
        if not self.agent_chain.enable_cache:
            return "Error: Cache is not enabled"
            
        if not self.agent_chain.db_connection:
            return "Error: Database connection not established"
            
        try:
            sessions_result = self._list_sessions(
                topic_filter=filter_terms, 
                all_sessions=all_sessions,
                limit=limit
            )
            
            if not sessions_result.get("success", False):
                return f"Error retrieving sessions: {sessions_result.get('error', 'Unknown error')}"
                
            sessions = sessions_result.get("sessions", [])
            
            if not sessions:
                if filter_terms:
                    return f"No sessions found matching '{filter_terms}'. Try a different search term or use 'list sessions' to see all available sessions."
                else:
                    return "No conversation sessions found."
            
            # Format the output
            output_lines = [f"## Available Sessions ({len(sessions)})"]
            
            for session in sessions:
                uuid = session.get("uuid", "")
                created_at = session.get("created_at", "")
                if isinstance(created_at, str) and 'T' in created_at:
                    created_at = created_at.split('T')[0]  # Just the date part
                    
                # Get a mini summary if available
                summary = session.get("summary", "No summary available")
                
                # Get message count
                message_count = session.get("message_count", 0)
                
                # Format the session info
                entry = f"### Session: {uuid}\n"
                entry += f"- **Created:** {created_at}\n"
                entry += f"- **Messages:** {message_count}\n"
                entry += f"- **Summary:** {summary}\n"
                
                output_lines.append(entry)
                
                # Add this session to the known sessions in memory
                if uuid not in self.internal_memory["known_sessions"]:
                    self.internal_memory["known_sessions"].append(uuid)
            
            # Add usage instructions
            output_lines.append("\n## Commands")
            output_lines.append("- **Load a session:** `@state: load session [UUID]`")
            output_lines.append("- **Summarize a session:** `@state: summarize session [UUID]`")
            output_lines.append("- **Compare sessions:** `@state: compare sessions [UUID1] [UUID2]`")
            output_lines.append("- **Delete a session:** `@state: delete session [UUID]`")
            output_lines.append("- **Search within sessions:** `@state: search for [terms]`")
            
            return "\n".join(output_lines)
            
        except Exception as e:
            logger.error(f"Error listing sessions: {e}", exc_info=True)
            return f"Error listing sessions: {str(e)}"
    
    async def summary_crawler(self, regenerate_all: bool = False) -> Dict[str, Any]:
        """
        Crawls the database for sessions without summaries and generates mini-summaries for them.
        
        Args:
            regenerate_all: Whether to regenerate summaries for all sessions or just those without summaries
            
        Returns:
            Dictionary with results of the operation
        """
        if not self.agent_chain.enable_cache:
            return {"success": False, "error": "Cache is not enabled", "sessions_updated": 0}
            
        if not self.agent_chain.db_connection:
            return {"success": False, "error": "Database connection not established", "sessions_updated": 0}
            
        try:
            cursor = self.agent_chain.db_connection.cursor()
            db_name = self.agent_chain.cache_config.get("name", "default")
            
            # Check if the summary column exists in the sessions table
            cursor.execute("PRAGMA table_info(sessions)")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            if "summary" not in column_names:
                # Add the summary column if it doesn't exist
                cursor.execute("ALTER TABLE sessions ADD COLUMN summary TEXT")
                self.agent_chain.db_connection.commit()
                logger.info("Added summary column to the sessions table")
            
            # Get sessions without summaries or all sessions if regenerate_all is True
            if regenerate_all:
                cursor.execute(
                    """
                    SELECT session_instance_uuid
                    FROM sessions
                    WHERE session_id = ?
                    """,
                    (db_name,)
                )
            else:
                cursor.execute(
                    """
                    SELECT session_instance_uuid
                    FROM sessions
                    WHERE session_id = ? AND (summary IS NULL OR summary = '' OR summary = 'No summary available')
                    """,
                    (db_name,)
                )
            
            sessions_needing_summaries = cursor.fetchall()
            logger.info(f"Found {len(sessions_needing_summaries)} sessions needing summaries")
            
            updated_sessions = 0
            skipped_sessions = 0
            
            # Process each session
            for (session_uuid,) in sessions_needing_summaries:
                # Check message count first to avoid unnecessary processing
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM conversation_entries
                    WHERE session_id = ? AND session_instance_uuid = ?
                    """,
                    (db_name, session_uuid)
                )
                
                message_count = cursor.fetchone()[0]
                
                if message_count <= 0:
                    logger.info(f"Session {session_uuid} has only {message_count} messages, skipping summary generation")
                    skipped_sessions += 1
                    continue
                
                # For sessions with actual content, generate a summary
                try:
                    mini_summary = await self._get_or_generate_mini_summary(session_uuid, force_regeneration=regenerate_all)
                    
                    # Update the database with the new summary
                    cursor.execute(
                        """
                        UPDATE sessions
                        SET summary = ?
                        WHERE session_id = ? AND session_instance_uuid = ?
                        """,
                        (mini_summary, db_name, session_uuid)
                    )
                    
                    self.agent_chain.db_connection.commit()
                    updated_sessions += 1
                    
                    logger.info(f"Updated summary for session {session_uuid}: {mini_summary[:50]}...")
                    
                except Exception as summary_error:
                    logger.error(f"Error generating summary for session {session_uuid}: {summary_error}")
                    skipped_sessions += 1
            
            logger.info(f"Summary database initialization complete: Updated {updated_sessions} session summaries, skipped {skipped_sessions}")
            
            return {
                "success": True,
                "sessions_updated": updated_sessions,
                "sessions_skipped": skipped_sessions,
                "message": f"Updated {updated_sessions} session summaries, skipped {skipped_sessions}"
            }
            
        except Exception as e:
            logger.error(f"Error initializing summary database: {e}", exc_info=True)
            return {"success": False, "error": str(e), "sessions_updated": 0}

    async def list_sessions_table(self, filter_terms: str = None, all_sessions: bool = True, limit: int = 10) -> str:
        """
        List available sessions in a markdown table format, optionally filtered by topic.
        
        Args:
            filter_terms: Optional keywords to filter sessions by topic
            all_sessions: Whether to include all sessions or just those we've interacted with
            limit: Maximum number of sessions to return
            
        Returns:
            Markdown-formatted table of available sessions
        """
        if not self.agent_chain.enable_cache:
            return "Error: Cache is not enabled"
            
        if not self.agent_chain.db_connection:
            return "Error: Database connection not established"
            
        try:
            return await self.get_sessions_markdown_table(filter_terms, all_sessions, limit)
        except Exception as e:
            logger.error(f"Error listing sessions in table format: {e}", exc_info=True)
            return f"Error listing sessions in table format: {str(e)}"
            
    async def get_sessions_markdown_table(self, filter_terms: str = None, all_sessions: bool = True, limit: int = 20) -> str:
        """
        Get a markdown table of available sessions, optionally filtered by topic.
        
        Args:
            filter_terms: Optional keywords to filter sessions by topic
            all_sessions: Whether to include all sessions or just those we've interacted with
            limit: Maximum number of sessions to return
            
        Returns:
            Markdown-formatted table of available sessions
        """
        if not self.agent_chain.enable_cache:
            return "Error: Cache is not enabled"
            
        if not self.agent_chain.db_connection:
            return "Error: Database connection not established"
            
        try:
            sessions_result = self._list_sessions(
                topic_filter=filter_terms, 
                all_sessions=all_sessions,
                limit=limit
            )
            
            if not sessions_result.get("success", False):
                return f"Error retrieving sessions: {sessions_result.get('error', 'Unknown error')}"
                
            sessions = sessions_result.get("sessions", [])
            
            if not sessions:
                if filter_terms:
                    return f"No sessions found matching '{filter_terms}'. Try a different search term or use 'list sessions' to see all available sessions."
                else:
                    return "No conversation sessions with 3+ messages found. You can use '@state cleanup tiny sessions confirm' to remove sessions with fewer messages."
            
            # Format the output as a markdown table
            table_lines = ["# Available Sessions\n"]
            if filter_terms:
                table_lines[0] = f"# Sessions Matching '{filter_terms}'\n"
                
            table_lines.append("| UUID | Created | Messages | Summary |")
            table_lines.append("|------|---------|----------|---------|")
            
            for session in sessions:
                uuid = session.get("uuid", "")
                created_at = session.get("created_at", "")
                if isinstance(created_at, str) and 'T' in created_at:
                    created_at = created_at.split('T')[0]  # Just the date part
                    
                # Get a mini summary if available
                summary = session.get("summary", "No summary available")
                # Truncate summary if too long for table
                if len(summary) > 50:
                    summary = summary[:47] + "..."
                # Escape pipe characters in summary to prevent table formatting issues
                summary = summary.replace("|", "\\|")
                
                # Get message count
                message_count = session.get("message_count", 0)
                
                # Add to table
                table_lines.append(f"| `{uuid}` | {created_at} | {message_count} | {summary} |")
                
                # Add this session to the known sessions in memory
                if uuid not in self.internal_memory["known_sessions"]:
                    self.internal_memory["known_sessions"].append(uuid)
            
            # Add usage instructions
            table_lines.append("\n## Available Commands")
            table_lines.append("- **Load a session**: `@state load session [UUID]`")
            table_lines.append("- **Summarize a session**: `@state summarize session [UUID]`")
            table_lines.append("- **Search conversations**: `@state find conversations about [topic]`")
            table_lines.append("- **Clean up small sessions**: `@state cleanup tiny sessions confirm`")
            
            return "\n".join(table_lines)
            
        except Exception as e:
            logger.error(f"Error generating sessions markdown table: {e}", exc_info=True)
            return f"Error generating sessions markdown table: {str(e)}"

    async def regenerate_session_summary(self, session_uuid: str) -> str:
        """
        Force regeneration of a session summary.
        
        Args:
            session_uuid: UUID of the session to regenerate summary for
            
        Returns:
            The newly generated summary or an error message
        """
        if not session_uuid:
            return "Please provide a valid session UUID to regenerate summary."
            
        # Validate the session exists
        if not self._validate_session_exists(session_uuid):
            return f"Session with UUID {session_uuid} not found. Please check the UUID or search for available sessions."
            
        try:
            # Generate a new summary
            summary = await self._get_or_generate_mini_summary(session_uuid, force_regeneration=True)
            
            # Store the new summary in the database
            success = self._store_session_summary(session_uuid, summary)
            
            if success:
                return f"Summary regenerated for session {session_uuid}:\n\n{summary}"
            else:
                return f"Summary was regenerated but could not be stored in the database: {summary}"
                
        except Exception as e:
            error_msg = f"Error regenerating summary for session {session_uuid}: {str(e)}"
            if self.verbose:
                print(error_msg)
            return error_msg

    def _list_sessions(self, topic_filter: str = None, all_sessions: bool = True, limit: int = 20, min_messages: int = 3) -> Dict[str, Any]:
        """
        Internal method to retrieve sessions from the database, optionally filtered by topic.
        
        Args:
            topic_filter: Optional keywords to filter sessions by topic
            all_sessions: Whether to include all sessions or just those we've interacted with
            limit: Maximum number of sessions to return
            min_messages: Minimum number of messages a session must have to be included
            
        Returns:
            Dict with session data and operation status
        """
        if not self.agent_chain.enable_cache:
            return {"success": False, "error": "Cache is not enabled", "sessions": []}
            
        if not self.agent_chain.db_connection:
            return {"success": False, "error": "Database connection not established", "sessions": []}
            
        try:
            cursor = self.agent_chain.db_connection.cursor()
            db_name = self.agent_chain.cache_config.get("name", "default")
            
            # Build the query based on parameters
            query = """
            SELECT s.session_instance_uuid, s.created_at, s.summary, COUNT(c.id) as message_count
            FROM sessions s
            LEFT JOIN conversation_entries c ON s.session_id = c.session_id AND s.session_instance_uuid = c.session_instance_uuid
            WHERE s.session_id = ?
            """
            
            params = [db_name]
            
            # Add filter for sessions we know about if not showing all
            if not all_sessions:
                # We'll use a placeholder for array binding
                if self.internal_memory["known_sessions"]:
                    placeholder = ','.join(['?'] * len(self.internal_memory["known_sessions"]))
                    query += f" AND s.session_instance_uuid IN ({placeholder})"
                    params.extend(self.internal_memory["known_sessions"])
                else:
                    # If we don't know any sessions and only want known ones, return empty
                    return {"success": True, "sessions": [], "filter_applied": False}
            
            # Add topic filter if specified
            if topic_filter:
                # Look for the filter term in either summaries or content
                query += """ 
                AND (
                    s.summary LIKE ? 
                    OR s.session_instance_uuid IN (
                        SELECT DISTINCT session_instance_uuid 
                        FROM conversation_entries 
                        WHERE session_id = ? AND content LIKE ?
                    )
                )
                """
                params.extend([f"%{topic_filter}%", db_name, f"%{topic_filter}%"])
            
            # Group by session
            query += " GROUP BY s.session_instance_uuid"
            
            # Filter out sessions with too few messages
            if min_messages > 0 and not topic_filter:  # Don't filter by message count when specifically searching
                query += f" HAVING COUNT(c.id) >= {min_messages}"
            
            # Order by message count first (prioritize sessions with content), then by creation date
            query += " ORDER BY message_count DESC, s.created_at DESC"
            
            # Add limit
            query += f" LIMIT {int(limit)}"
            
            # Execute the query
            cursor.execute(query, params)
            
            # Format the results
            sessions = []
            for session_uuid, created_at, summary, message_count in cursor.fetchall():
                # Format session data
                session_data = {
                    "uuid": session_uuid,
                    "created_at": created_at,
                    "message_count": message_count,
                    "summary": summary if summary else "No summary available"
                }
                
                # Try to get sample content for sessions without summaries
                if (not summary or summary == "No summary available") and message_count > 0:
                    try:
                        # Get a sample message to use as a mini-summary
                        cursor.execute(
                            """
                            SELECT content FROM conversation_entries
                            WHERE session_id = ? AND session_instance_uuid = ? AND role = 'user'
                            ORDER BY id ASC
                            LIMIT 1
                            """,
                            (db_name, session_uuid)
                        )
                        sample_content = cursor.fetchone()
                        if sample_content and sample_content[0]:
                            content = sample_content[0]
                            # Truncate and clean up for display
                            if len(content) > 80:
                                content = content[:77] + "..."
                            session_data["summary"] = f"Session contains {message_count} messages with substantive content."
                    except Exception as e:
                        logger.error(f"Error getting sample content for session {session_uuid}: {e}")
                
                sessions.append(session_data)
            
            # Store these sessions in the known_sessions list for future reference
            for session in sessions:
                if session["uuid"] not in self.internal_memory["known_sessions"]:
                    self.internal_memory["known_sessions"].append(session["uuid"])
            
            return {"success": True, "sessions": sessions, "filter_applied": topic_filter is not None}
            
        except Exception as e:
            logger.error(f"Error listing sessions: {e}", exc_info=True)
            return {"success": False, "error": str(e), "sessions": []}

    async def _get_or_generate_mini_summary(self, session_uuid: str, force_regeneration: bool = False) -> str:
        """
        Get a cached mini-summary or generate a new one for the specified session.
        Always returns a string and never triggers further tool calls.
        """
        # Use stored summary if available and not forcing regeneration
        if not force_regeneration and session_uuid in self.internal_memory["mini_summaries"]:
            return self.internal_memory["mini_summaries"][session_uuid]
        # Try to get from DB
        try:
            cursor = self.agent_chain.db_connection.cursor()
            db_name = self.agent_chain.cache_config.get("name", "default")
            cursor.execute(
                "SELECT summary FROM sessions WHERE session_id = ? AND session_instance_uuid = ?",
                (db_name, session_uuid)
            )
            result = cursor.fetchone()
            if result and result[0]:
                mini_summary = result[0]
                self.internal_memory["mini_summaries"][session_uuid] = mini_summary
                return mini_summary
        except Exception as e:
            logger.error(f"Error fetching mini-summary from DB: {e}")
        # If not found, generate a new one
        try:
            cursor = self.agent_chain.db_connection.cursor()
            db_name = self.agent_chain.cache_config.get("name", "default")
            cursor.execute(
                "SELECT content FROM conversation_entries WHERE session_id = ? AND session_instance_uuid = ? ORDER BY id LIMIT 10",
                (db_name, session_uuid)
            )
            messages = [row[0] for row in cursor.fetchall()]
            if not messages:
                mini_summary = "No messages found in this session."
            else:
                # Use a simple heuristic for a mini-summary
                mini_summary = f"Conversation with {len(messages)} messages: " + ", ".join([m[:30] + ("..." if len(m) > 30 else "") for m in messages[:3]])
            self.internal_memory["mini_summaries"][session_uuid] = mini_summary
            return mini_summary
        except Exception as e:
            logger.error(f"Error generating mini-summary: {e}")
            return "Summary unavailable."

    async def delete_small_sessions(self, max_messages: int = 2, confirm: bool = False) -> dict:
        """
        Delete all sessions with a message count less than or equal to max_messages.
        Args:
            max_messages: Maximum number of messages for a session to be deleted (inclusive)
            confirm: Confirmation flag to prevent accidental deletion
        Returns:
            Dict with operation results
        """
        if not confirm:
            return {"success": False, "error": "Deletion requires confirmation", "message": "Please set confirm=True to proceed with deletion"}
        if not self.agent_chain.enable_cache:
            return {"success": False, "error": "Cache is not enabled", "message": "Cannot delete sessions when caching is disabled"}
        if not self.agent_chain.db_connection:
            return {"success": False, "error": "Database connection not established", "message": "No database connection available"}
        try:
            cursor = self.agent_chain.db_connection.cursor()
            db_name = self.agent_chain.cache_config.get("name", "default")
            # Find all sessions with message count <= max_messages
            cursor.execute(
                """
                SELECT s.session_instance_uuid, COUNT(c.id) as msg_count
                FROM sessions s
                LEFT JOIN conversation_entries c ON s.session_id = c.session_id AND s.session_instance_uuid = c.session_instance_uuid
                WHERE s.session_id = ?
                GROUP BY s.session_instance_uuid
                HAVING COUNT(c.id) <= ?
                """,
                (db_name, max_messages)
            )
            small_sessions = cursor.fetchall()
            if not small_sessions:
                return {"success": True, "message": f"No sessions found with {max_messages} or fewer messages", "deleted_count": 0}
            self.agent_chain.db_connection.execute("BEGIN TRANSACTION")
            total_entries_deleted = 0
            deleted_sessions = []
            for session_uuid, msg_count in small_sessions:
                cursor.execute(
                    """
                    DELETE FROM conversation_entries 
                    WHERE session_id = ? AND session_instance_uuid = ?
                    """,
                    (db_name, session_uuid)
                )
                entries_deleted = cursor.rowcount
                total_entries_deleted += entries_deleted
                cursor.execute(
                    """
                    DELETE FROM sessions 
                    WHERE session_id = ? AND session_instance_uuid = ?
                    """,
                    (db_name, session_uuid)
                )
                deleted_sessions.append(session_uuid)
            self.agent_chain.db_connection.commit()
            # Clean up internal memory
            for session_uuid in deleted_sessions:
                if session_uuid in self.internal_memory["known_sessions"]:
                    self.internal_memory["known_sessions"].remove(session_uuid)
                if self.internal_memory["last_mentioned_session"] == session_uuid:
                    self.internal_memory["last_mentioned_session"] = None
                if session_uuid in self.internal_memory["mini_summaries"]:
                    del self.internal_memory["mini_summaries"][session_uuid]
            return {
                "success": True,
                "deleted_count": len(deleted_sessions),
                "deleted_sessions": deleted_sessions,
                "total_entries_deleted": total_entries_deleted,
                "message": f"Deleted {len(deleted_sessions)} sessions with {max_messages} or fewer messages ({total_entries_deleted} total entries removed)"
            }
        except Exception as e:
            try:
                self.agent_chain.db_connection.rollback()
            except:
                pass
            logger.error(f"Error deleting small sessions: {e}", exc_info=True)
            return {"success": False, "error": str(e), "message": "Database error during deletion"}

# Function to create a state agent PromptChain that can be added to an AgentChain
def create_state_agent(agent_chain: AgentChain, verbose: bool = False) -> PromptChain:
    """
    Create a PromptChain for the state agent that can be added to an AgentChain.
    
    Args:
        agent_chain: The AgentChain this agent will work with
        verbose: Whether to enable verbose logging
        
    Returns:
        A configured PromptChain for the state agent
    """
    # Create the StateAgent instance
    state_agent = StateAgent(agent_chain, verbose)
    
    # Create a PromptChain that wraps the AgenticStepProcessor
    state_agent_chain = PromptChain(
        models=[],  # No models needed since AgenticStepProcessor handles this internally
        instructions=[state_agent],  # Use the AgenticStepProcessor directly as an instruction
        verbose=verbose
    )
    
    # Register the tools on the chain
    state_agent_chain.add_tools(state_agent.TOOL_SCHEMAS)
    
    # Register all tool functions - make sure to add ALL tools we define in TOOL_SCHEMAS
    tool_functions = {
        "list_sessions": state_agent.list_sessions,
        "list_sessions_table": state_agent.list_sessions_table,
        "search_conversations": state_agent.search_conversations, 
        "load_session": state_agent.load_session,
        "summarize_session": state_agent.summarize_session,
        "compare_sessions": state_agent.compare_sessions,
        "add_search_to_global_history": state_agent.add_search_to_global_history,
        "summary_crawler": state_agent.summary_crawler,
        "regenerate_session_summary": state_agent.regenerate_session_summary,
        "delete_session": state_agent.delete_session,
        "cleanup_tiny_sessions": state_agent.cleanup_tiny_sessions,
        "delete_small_sessions": state_agent.delete_small_sessions
    }
    
    # Register each function, with error handling for missing methods
    for tool_name, tool_func in tool_functions.items():
        try:
            state_agent_chain.register_tool_function(tool_func)
            logger.info(f"Successfully registered tool function: {tool_name}")
        except Exception as e:
            logger.error(f"Error registering tool function {tool_name}: {e}")
            # Continue with other tools rather than failing completely
    
    return state_agent_chain