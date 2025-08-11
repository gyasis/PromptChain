
# Product Requirements Document (PRD): RAG CLI Chat with PromptChain

## 1. Overview

This document outlines the requirements for a new feature: a command-line interface (CLI) chat application that integrates with an existing Retrieval-Augmented Generation (RAG) database. The application will leverage the `promptchain` library to manage conversational flow, maintain session history, and interact with the user. The primary goal is to create a persistent chat session where a user can ask questions, receive answers augmented by the RAG database, and continue the conversation until they choose to exit.

## 2. Project Goals

*   **Primary Goal:** To create a functional, CLI-based chat application that uses a RAG database to provide contextually relevant answers.
*   **Secondary Goal:** To utilize the `promptchain` library, specifically `AgentChain` and `ExecutionHistoryManager`, to handle the chat logic and conversation history.
*   **Target Audience:** This tool is for developers and users who need to interact with the RAG database in a conversational manner through a terminal.
*   **Success Metric:** A user can successfully run the script, ask multiple questions, receive coherent and context-aware responses from the RAG-powered agent, and end the session gracefully.

## 3. Key Features

### 3.1. CLI Chat Interface
- The application must run entirely within a command-line terminal.
- It will present a simple `You:` prompt for user input and `AI:` for the agent's response.
- The chat session will be persistent, maintaining context from the conversation until explicitly terminated.

### 3.2. RAG Integration
- The application must connect to a pre-existing RAG database.
- Before generating a response, the application will first query the RAG database with the user's input to retrieve relevant context.
- This retrieved context will be injected into the prompt sent to the Language Model (LLM).

### 3.3. Conversation Management with `promptchain`
- **Agent:** A `PromptChain` instance will be configured to act as the conversational agent. Its prompt will be templated to accept both the RAG-retrieved context and the conversation history.
- **Orchestrator:** An `AgentChain` will be used to manage the agent. For this initial version, a simple `pipeline` execution mode is sufficient.
- **History:** The `ExecutionHistoryManager` will be used to track the conversation history (user inputs and agent outputs). The history will be truncated to a specific token limit to manage context window size.

### 3.4. Advanced Session Management
- The application will support multiple, named chat sessions within a single run.
- Each session will have its own independent conversation history.
- Users can create, switch between, and list sessions using interactive commands.
- The application starts with a "default" session.
- The user can end the entire application using the `/exit` or `/quit` command.

### 3.5. Interactive Commands
- The application will support slash commands to perform actions outside of the normal chat flow.
- A simple command processor will be implemented to handle these commands.
- Initial commands will include:
  - `/help`: Show available commands.
  - `/exit`, `/quit`: Exit the entire application.
  - `/new <name>`: Create and switch to a new, empty session.
  - `/switch <name>`: Switch to an existing session.
  - `/sessions`: List all available session names.

### 3.6. Configuration Management
- The application will load its settings from a `config.json` file.
- This allows users to configure the default model, RAG settings, and other parameters without modifying the source code.
- The script will look for `config.json` in the current directory.

## 4. Technical Implementation Plan (for an LLM Developer)

This section provides a step-by-step guide to implement the RAG CLI Chat application.

### Step 1: Project Setup
1.  Create a new directory for the project (e.g., `rag_cli_chat`).
2.  Inside this directory, create the main application file (e.g., `main.py`).
3.  Create a `requirements.txt` file listing the necessary dependencies, including `promptchain`, `litellm`, and any libraries needed to connect to the RAG database.

### Step 2: Create Configuration File
1.  Create a `config.json` file to store application settings. This separates configuration from code, making the tool more flexible.
2.  Define the structure of the JSON file. It should include settings for the model, RAG, and logging.
    ```json
    {
      "model": {
        "name": "ollama/mistral",
        "params": {
          "temperature": 0.7
        }
      },
      "rag": {
        "retriever_module": "rag_utils",
        "retriever_function": "retrieve_rag_context"
      },
      "logging": {
        "log_dir": "rag_chat_logs"
      }
    }
    ```
3.  In the main script, add logic to read this `config.json` file at startup and use its values to configure the application, such as setting up the `PromptChain` model and the `RunLogger` directory.

### Step 3: Implement the RAG Retrieval Function
1.  In a separate utility file (e.g., `rag_utils.py`) or directly in `main.py`, create a function to handle RAG retrieval.
2.  This function, let's call it `retrieve_rag_context(query: str) -> str`, will:
    *   Accept the user's query as a string.
    *   Connect to the RAG database (implementation details depend on the specific database).
    *   Perform a similarity search or other retrieval mechanism to find relevant documents/chunks.
    *   Format the retrieved content into a single string.
    *   Return the formatted context string.
    *   **Note:** For initial development, this can be a placeholder function that returns a fixed string.

### Step 3: Configure the `promptchain` Agent
1.  In `main.py`, import the necessary classes: `PromptChain`, `AgentChain`, `ExecutionHistoryManager`.
2.  Define the prompt template for the conversational agent. This template is crucial and must include placeholders for the history and the RAG context.
    ```python
    PROMPT_TEMPLATE = """
    You are a helpful assistant. Answer the user's question based on the provided context and conversation history.

    --- CONTEXT ---
    {rag_context}
    --- END CONTEXT ---

    --- HISTORY ---
    {history}
    --- END HISTORY ---

    User: {input}
    Assistant:
    """
    ```
3.  Instantiate the `PromptChain` to create the agent. Configure it with the desired LLM (e.g., a local Ollama model or a cloud-based one via LiteLLM).
    ```python
    rag_agent = PromptChain(
        models=[{"name": "ollama/mistral"}], # Or any other model
        instructions=[PROMPT_TEMPLATE],
        verbose=False
    )
    ```

### Step 4: Set Up the `AgentChain` Orchestrator
1.  Instantiate `AgentChain` to manage the `rag_agent`.
2.  Use the `pipeline` execution mode for this simple, single-agent setup.
    ```python
    agent_orchestrator = AgentChain(
        agents={"rag_agent": rag_agent},
        agent_descriptions={"rag_agent": "Answers questions using RAG context."},
        execution_mode="pipeline"
    )
    ```

### Step 5: Implement the Main Chat Loop
1.  Create an `async def main():` function to house the chat logic.
2.  **Initialize Session Management:** Create a dictionary to hold the `ExecutionHistoryManager` for each session. Initialize it with a default session.
    ```python
    sessions = {"default": ExecutionHistoryManager(max_tokens=4096)}
    active_session_name = "default"
    ```
3.  Start a `while True:` loop to handle the conversation.
4.  Inside the loop:
    a.  Prompt the user for input: `user_input = input("You: ")`.
    b.  **Check for and process commands:** If `user_input` starts with a `/`, parse and execute the command. This will involve logic to manage the `sessions` dictionary (e.g., creating new history managers, changing `active_session_name`).
    c.  **If not a command, process as a chat message:**
        i.  Get the active history manager: `history_manager = sessions[active_session_name]`.
        ii. Call the RAG retrieval function: `context = retrieve_rag_context(user_input)`.
        iii. Get the formatted history: `formatted_history = history_manager.get_formatted_history()`.
        iv. Add the user's input to history: `history_manager.add_entry("user", user_input)`.
        v.  Use `agent_orchestrator.process_input()` to get the agent's response.
        vi. Print the agent's response: `print(f"AI: {response}")`.
        vii. Add the agent's response to history: `history_manager.add_entry("assistant", response)`.
5.  After the loop, print a closing message.
6.  Run the `main` function using `asyncio.run(main())`.

## 5. Logging and Debugging

For effective development and monitoring, the application will use the `RunLogger` class from `promptchain.utils.logging_utils`.

### 5.1. Implementation
1.  **Import necessary modules:**
    ```python
    from promptchain.utils.logging_utils import RunLogger
    from datetime import datetime
    import os
    ```
2.  **Initialize the Logger:** At the beginning of the `main` function, create a single instance of `RunLogger`. Define a `log_dir` and create a unique `session_filename` for the entire chat session. This ensures all logs for one conversation go into the same file.
    ```python
    # --- Inside main() ---
    log_dir = "rag_chat_logs"
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_filename = os.path.join(log_dir, f"chat_session_{session_timestamp}.jsonl")

    # Initialize the logger for the session
    logger = RunLogger(log_dir=log_dir, session_filename=session_filename)

    logger.log_run({
        "event": "session_start",
        "session_id": session_timestamp
    })
    ```
3.  **Log Events in the Chat Loop:** Inside the `while` loop, use the `logger.log_run()` method to record key events. This provides a structured, detailed record of each turn in the conversation.
    ```python
    # --- Inside the while loop ---

    # Log the user's input
    logger.log_run({
        "event": "user_input",
        "content": user_input
    })

    # Log the retrieved RAG context
    context = retrieve_rag_context(user_input)
    logger.log_run({
        "event": "rag_retrieval",
        "query": user_input,
        "retrieved_context": context
    })

    # ... (rest of the logic to get agent response) ...

    # Log the agent's final response
    logger.log_run({
        "event": "agent_response",
        "content": response
    })
    ```
4.  **Log Session End:** After the loop terminates, log a final event to mark the end of the session.
    ```python
    # --- After the while loop ---
    logger.log_run({
        "event": "session_end",
        "session_id": session_timestamp
    })
    print("Session ended. Logs saved to:", session_filename)
    ```

## 7. Future Enhancements (Out of Scope for V1)

*   **Advanced Agent with Tool-Use:** Transition from the `pipeline` execution mode to the `router` mode in `AgentChain`. This will enable the agent to decide *if* and *when* to call the RAG retrieval function (which would be defined as a "tool"). This allows for more flexible and intelligent responses, as the agent won't be forced to retrieve context for every single query.
*   **Streaming Responses:** Implement streaming to show the agent's response token-by-token for a better user experience.
*   **Advanced Session Management:** Allow users to save, load, or reset chat sessions, potentially by using the session-based logging files.
