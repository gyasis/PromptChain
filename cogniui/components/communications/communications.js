/**
 * @fileoverview Communications Component Frame
 * @description A modular component for handling LLM (Language Learning Model) communications.
 * Manages message display, history, and API formatting with configurable input/output paths.
 * 
 * @example Basic Usage:
 * ```javascript
 * import Communications from './communications.js';
 * 
 * // Initialize with default liteLLM config
 * const comms = new Communications();
 * comms.init('output-div', 'input-div');
 * 
 * // Display messages
 * comms.displayInput('User message');
 * comms.displayOutput({
 *   choices: [{
 *     message: { content: 'AI response' }
 *   }]
 * });
 * ```
 */

// Default configuration for different LLM services
const DEFAULT_CONFIGS = {
    liteLLM: {
        outputKey: 'choices[0].message.content',
        inputKey: 'messages'
    }
};

/**
 * Communications class for managing LLM interactions
 * @class
 * @description Handles all aspects of LLM communication including:
 * - Message display and formatting
 * - Conversation history management
 * - Input/Output parsing
 * - UI updates and scrolling
 * - Error handling
 */
class Communications {
    /**
     * Create a Communications instance
     * @param {Object} config - Configuration for input/output parsing
     * @param {string} config.outputKey - Path to message content in API response
     * @param {string} config.inputKey - Key for sending messages to API
     */
    constructor(config = DEFAULT_CONFIGS.liteLLM) {
        this.config = config;
        this.messageHistory = [];
        this.outputElement = null;
        this.inputElement = null;
    }

    /**
     * Initialize UI elements
     * @param {string} outputElementId - ID of the output container element
     * @param {string} inputElementId - ID of the input field element
     * @throws {Error} If elements are not found in the DOM
     */
    init(outputElementId, inputElementId) {
        this.outputElement = document.getElementById(outputElementId);
        this.inputElement = document.getElementById(inputElementId);
        
        if (!this.outputElement || !this.inputElement) {
            throw new Error('Output or input element not found');
        }
    }

    /**
     * Parse nested object paths
     * @param {Object} obj - Object to parse
     * @param {string} path - Dot notation path (e.g., 'choices[0].message.content')
     * @returns {*} Value at the specified path
     */
    parseObjectPath(obj, path) {
        return path.split(/[\.\[\]]+/).filter(Boolean).reduce((current, key) => {
            return current?.[key];
        }, obj);
    }

    /**
     * Display user/system input in the UI
     * @param {string} input - Message to display
     * @param {string} role - Message role ('user', 'system', etc.)
     */
    displayInput(input, role = 'user') {
        const inputDiv = document.createElement('div');
        inputDiv.className = `message ${role}-message`;
        inputDiv.innerHTML = `
            <div class="message-content">
                <span class="message-role">${role}:</span>
                <span class="message-text">${input}</span>
            </div>
        `;
        this.outputElement.appendChild(inputDiv);
        this.messageHistory.push({ role, content: input });
        this.scrollToBottom();
    }

    /**
     * Display LLM output in the UI
     * @param {Object} rawOutput - Raw API response
     * @param {string} role - Message role (default: 'assistant')
     * @returns {string|null} Parsed message content or null if error
     */
    displayOutput(rawOutput, role = 'assistant') {
        try {
            // Parse the output based on configuration
            const message = this.parseObjectPath(rawOutput, this.config.outputKey);
            
            const outputDiv = document.createElement('div');
            outputDiv.className = `message ${role}-message`;
            outputDiv.innerHTML = `
                <div class="message-content">
                    <span class="message-role">${role}:</span>
                    <span class="message-text">${message}</span>
                </div>
            `;
            this.outputElement.appendChild(outputDiv);
            this.messageHistory.push({ role, content: message });
            this.scrollToBottom();
            
            return message;
        } catch (error) {
            console.error('Error parsing output:', error);
            this.displayError('Error processing response');
            return null;
        }
    }

    /**
     * Display error messages
     * @param {string} error - Error message to display
     */
    displayError(error) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'message error-message';
        errorDiv.innerHTML = `
            <div class="message-content">
                <span class="message-role">error:</span>
                <span class="message-text">${error}</span>
            </div>
        `;
        this.outputElement.appendChild(errorDiv);
        this.scrollToBottom();
    }

    /**
     * Clear the output display and message history
     */
    clearOutput() {
        if (this.outputElement) {
            this.outputElement.innerHTML = '';
        }
        this.messageHistory = [];
    }

    /**
     * Get the current message history
     * @returns {Array} Array of message objects with role and content
     */
    getMessageHistory() {
        return this.messageHistory;
    }

    /**
     * Format input for LLM API
     * @param {string} input - User input message
     * @param {string} role - Message role
     * @returns {Object} Formatted input for API request
     */
    formatInput(input, role = 'user') {
        return {
            [this.config.inputKey]: [
                ...this.messageHistory,
                { role, content: input }
            ]
        };
    }

    /**
     * Scroll the output container to the bottom
     * @private
     */
    scrollToBottom() {
        if (this.outputElement) {
            this.outputElement.scrollTop = this.outputElement.scrollHeight;
        }
    }
}

// Export the Communications class
export default Communications; 