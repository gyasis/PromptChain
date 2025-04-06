# Communications Component

A modular component for handling LLM (Language Learning Model) communications in web applications. This component provides a complete solution for managing chat-like interactions between users and AI models.

## Features

- 🔄 Bi-directional message handling
- 📝 Message history management
- 🎨 Styled message bubbles with dark mode support
- 🔧 Configurable API input/output parsing
- ⚠️ Error handling and display
- 📱 Responsive design
- 🌗 Dark mode support

## Installation

1. Copy the component files to your project:
   ```
   communications/
   ├── communications.js
   ├── communications.css
   └── README.md
   ```

2. Import the component and styles:
   ```javascript
   import Communications from './communications/communications.js';
   import './communications/communications.css';
   ```

## Basic Usage

```javascript
// Initialize with default liteLLM configuration
const comms = new Communications();

// Initialize with HTML elements
comms.init('output-container', 'input-field');

// Display user input
comms.displayInput('Hello, how are you?');

// Display AI response
const response = {
    choices: [{
        message: {
            content: "I'm doing well, thank you!"
        }
    }]
};
comms.displayOutput(response);
```

## Configuration

The component can be configured for different LLM APIs:

```javascript
// Custom configuration example
const customConfig = {
    outputKey: 'response.text',
    inputKey: 'prompt'
};

const comms = new Communications(customConfig);
```

### Default Configuration (liteLLM)
- Output path: `choices[0].message.content`
- Input key: `messages`

## API Reference

### Constructor
```javascript
const comms = new Communications(config);
```

### Methods

#### `init(outputElementId, inputElementId)`
Initialize the component with DOM elements.

#### `displayInput(input, role = 'user')`
Display a user or system message.

#### `displayOutput(rawOutput, role = 'assistant')`
Display an AI response message.

#### `displayError(error)`
Display an error message.

#### `clearOutput()`
Clear all messages and history.

#### `getMessageHistory()`
Get the current conversation history.

#### `formatInput(input, role = 'user')`
Format input for API requests.

## Styling

The component includes a comprehensive CSS file with:
- Message bubble styling
- Role-based colors
- Dark mode support
- Code block formatting
- Responsive design

### CSS Classes

- `.message` - Base message container
- `.user-message` - User message styling
- `.assistant-message` - AI response styling
- `.error-message` - Error message styling
- `.message-content` - Message content wrapper
- `.message-role` - Role indicator
- `.message-text` - Message text content

## Best Practices

1. **Initialization**
   - Always initialize with both output and input elements
   - Handle initialization errors appropriately

2. **Message Display**
   - Use appropriate roles for different message types
   - Handle long messages and code blocks properly

3. **Error Handling**
   - Display errors clearly to users
   - Log errors for debugging

4. **Configuration**
   - Use custom configs for non-standard APIs
   - Verify API response structure matches config

## Example Implementation

```javascript
// Initialize component
const comms = new Communications();
comms.init('chat-output', 'user-input');

// Handle user input
document.getElementById('user-input').addEventListener('submit', async (e) => {
    e.preventDefault();
    const input = e.target.elements.message.value;
    
    // Display user message
    comms.displayInput(input);
    
    try {
        // Format input for API
        const apiInput = comms.formatInput(input);
        
        // Make API call (example)
        const response = await fetch('/api/chat', {
            method: 'POST',
            body: JSON.stringify(apiInput)
        });
        
        const data = await response.json();
        
        // Display AI response
        comms.displayOutput(data);
    } catch (error) {
        comms.displayError('Failed to get response');
        console.error(error);
    }
});
```

## Contributing

Feel free to submit issues and enhancement requests! 