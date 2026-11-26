# PromptChain Changelog

## 2024-05-10: Rich Text Output for Assistant Responses

- The chat interface (router_chat_advanced.py, via AgentChain) now uses the `rich` Python library to render the assistant's output as rich text in the terminal.
- This means assistant responses are now colorized and formatted for better readability, making it easier to distinguish between user input and assistant output.
- Only the assistant's output is affected; user prompts and other system messages remain unchanged.
- This change improves the overall user experience for interactive CLI chat sessions.

## 2024-06: Advanced Router Web Chat UI/UX and Routing Improvements

- Direct agent addressing via @agent_name in user input (router prioritizes direct calls)
- UI agent hints: clickable agent names below input box
- User messages rendered as plain text; assistant messages as Markdown
- All assistant links open in new tab (target="_blank")
- Fullscreen toggle button in chat header
- Fullscreen mode uses 95% of viewport, centered, with border radius and shadow
- Header layout uses Flexbox for correct positioning
- Code blocks wrap lines (no horizontal scroll)
- Next: Implement websearch agent
