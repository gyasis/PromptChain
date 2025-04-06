# CogniUI

This project provides a set of reusable front-end templates built with **HTML**, **Tailwind CSS**, and **Vanilla JavaScript**. It replicates various UI layouts suitable for AI-centric applications, inspired by the designs presented in the "Designing for AI Engineers" article.

The primary goal is to offer a flexible "base system" that can be easily adapted and integrated with backend systems, particularly an **agentic LLM script**, to create functional developer tools and product interfaces.

## Technology Stack

*   **HTML5:** Semantic markup for structure.
*   **Tailwind CSS:** Utility-first CSS framework for rapid styling. Included via CDN.
*   **Vanilla JavaScript:** For basic interactivity, primarily template switching and demonstrating potential hooks for backend integration.

## Framework Objectives & Design Reasoning

*   **Clone and Template Creation:** Replicates layouts for common AI application interfaces (Chat, Model Repo, Playground, Console, Docs, Recipes). Templates are structurally complete but contain placeholder content, ready for dynamic data.
*   **Multi-page Support:** The current implementation uses JS to show/hide templates within a single HTML file (SPA-like feel). However, each template is self-contained within its main `div` (`#template-xxx`) and can be easily extracted into separate HTML files for a traditional multi-page application (MPA) architecture. Navigation would then use standard `<a>` tags instead of JS-driven buttons.
*   **Modular and Maintainable Structure:**
    *   Each product layout resides in its own clearly marked `div` (e.g., `<div id="template-chat" class="template-content">...</div>`).
    *   Within each template, major sections have unique IDs (e.g., `#chat-message-area`, `#model-grid-area`, `#docs-sidebar-nav`) serving as **connection points** for JavaScript to inject or read data.
    *   Repeatable elements use classes (e.g., `.model-card`, `.step`, `.widget`).
    *   Tailwind's utility classes keep styling co-located with the HTML structure, promoting component encapsulation.
*   **Framework Adaptability:** The core idea is to provide pre-styled structures. An LLM agent or backend system can interact with this framework by:
    1.  **Selecting a Template:** Choose the layout that best fits the current need (e.g., Documentation layout for showing help, Console layout for showing stats).
    2.  **Targeting Connection Points:** Use JavaScript to select elements by their IDs (e.g., `document.getElementById('agent-status')`) and update their `textContent`, `innerHTML`, or attributes based on the agent's state or data.
    3.  **Populating Components:** Generate lists or grids (like model cards or recipe steps) by creating HTML strings or DOM elements in JavaScript and appending them to container elements (e.g., `#model-grid-area`, `#recipe-steps-container`).
    4.  **Handling User Input:** Attach event listeners (like the example in `app.js` for `#agent-action-button`) to capture user actions and send commands back to the agent/backend.

## Available Templates & Connection Points

*(This section details each template, its purpose, key IDs, and reasoning)*

1.  **Product 1: Chat LLM (`#template-chat`)**
    *   **Purpose:** Basic conversational interface.
    *   **Key IDs:** `#chat-message-area` (for displaying messages), `#chat-input` (for user input), `#chat-send-button` (to trigger sending).
    *   **Reasoning:** Simple, focused layout. Clear separation of concerns. Includes a dark mode console variant concept.

2.  **Product 2: Model Repository (`#template-model-repo`)**
    *   **Purpose:** Browsing and filtering a collection of items (models).
    *   **Key IDs:** `#model-filter-options` (container for filter controls), `#model-grid-area` (container for model cards), `#model-detail-area` (placeholder for detail view). Model cards use the class `.model-card`.
    *   **Reasoning:** Common pattern using sidebar/filters and a main grid. Cards enhance scannability. Easily adaptable for lists instead of grids.

3.  **Product 3: Playground (`#template-playground`)**
    *   **Purpose:** Interactive environment for experimenting with AI models/parameters.
    *   **Key IDs:** `#playground-config-panel` (for parameter sliders, dropdowns), `#playground-input-area` (for user prompt/code), `#playground-output-area` (to display results).
    *   **Reasoning:** Multi-column layout logically separates configuration, input, and output.

4.  **Product 4: Developer Console (`#template-dev-console`)**
    *   **Purpose:** Dashboard for monitoring, management, and quick actions.
    *   **Key IDs:** `#dev-console-sidebar` (for navigation), `#dev-console-main-content` (main container for widgets), various widget IDs (e.g., `#api-call-stat`, `#usage-chart-widget`). Widgets often use the class `.widget`.
    *   **Reasoning:** Standard dashboard pattern. Sidebar nav is common. Grid layout for flexible widget placement.

5.  **Product 5: Documentation (`#template-documentation`)**
    *   **Purpose:** Displaying technical documentation with navigation and search.
    *   **Key IDs:** `#docs-sidebar-nav` (for navigation tree), `#docs-search-input` (for search functionality), `#docs-main-content` (where article content is loaded).
    *   **Reasoning:** Classic, effective documentation layout (like Docusaurus, GitBook). Prioritizes readability and findability.

6.  **Product 6: Recipes & Code (`#template-recipes`)**
    *   **Purpose:** Presenting step-by-step guides or tutorials heavy on code examples.
    *   **Key IDs:** `#recipe-steps-container` (holds all steps), steps use the class `.step`, code blocks use `pre > code` and should have language classes (e.g., `language-python`) for syntax highlighting.
    *   **Reasoning:** Focus on clarity for instructions and code. Assumes integration with a syntax highlighting library (like Prism.js or highlight.js).

7.  **Agentic LLM Hook Example (`#template-agentic-llm-hook`)**
    *   **Purpose:** Explicitly demonstrates how a backend agent script could interface with the UI.
    *   **Key IDs:** `#agent-status`, `#agent-task`, `#agent-output`, `#agent-instruction`, `#agent-action-button`.
    *   **Reasoning:** Provides a concrete example of the framework's intended use case – connecting UI elements to an agent's state and actions.

## How to Use

1.  **Clone/Download:** Get the `index.html`, `assets/` folder, and `README.md`.
2.  **Open `index.html`:** Open the file in your web browser.
3.  **Navigate Templates:** Use the buttons in the header to switch between the different layout templates.
4.  **Inspect Structure:** Use your browser's developer tools (F12) to inspect the HTML structure and see the IDs and Tailwind classes used.
5.  **Integrate with Backend:**
    *   Choose the template(s) you need.
    *   Modify `assets/js/app.js` or create new JS files.
    *   Use JavaScript to:
        *   Fetch data from your LLM agent/backend API.
        *   Target the specific `#id` elements within the chosen template.
        *   Update the `innerHTML` or `textContent` of these elements with your data.
        *   Dynamically create and append elements (e.g., list items, cards) to container elements.
        *   Add event listeners to buttons/inputs to send user actions back to your backend.

## Future Development

*   **Syntax Highlighting:** Integrate a library like Prism.js or highlight.js for code blocks in Recipes and Documentation.
*   **Charting:** Integrate a library like Chart.js for the Developer Console template.
*   **Component Extraction:** For larger applications, break down templates into smaller, reusable web components (using Vanilla JS or a framework like Vue/React/Svelte).
*   **Build Process:** Implement a build step (e.g., using Vite or Parcel) to bundle assets and enable features like purging unused Tailwind classes.
*   **Accessibility:** Perform thorough accessibility checks and improvements.
*   **Responsiveness:** Further refine responsiveness for various screen sizes.
