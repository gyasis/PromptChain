// Component registry
const components = new Map();

// Load a component module
async function loadComponent(name) {
    try {
        const module = await import(`../../components/${name}/index.js`);
        components.set(name, module.default);
        return module.default;
    } catch (error) {
        console.error(`Error loading component ${name}:`, error);
        return null;
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', async () => {
    // Get all template navigation buttons
    const navButtons = document.querySelectorAll('.template-nav-button');
    const templateContents = document.querySelectorAll('.template-content');
    const defaultTemplate = 'chat'; // Show chat by default

    async function setActiveTemplate(templateId) {
        // Hide all templates
        templateContents.forEach(content => {
            content.classList.remove('active');
            content.style.display = 'none';
        });

        // Deactivate all buttons
        navButtons.forEach(button => {
            button.classList.remove('ring-2', 'ring-offset-2', 'ring-blue-400');
        });

        // Show and initialize the selected template
        const activeTemplate = document.getElementById(`template-${templateId}`);
        if (activeTemplate) {
            activeTemplate.classList.add('active');
            activeTemplate.style.display = 'block';

            // Load the component if not already loaded
            if (!components.has(templateId)) {
                const component = await loadComponent(templateId);
                if (component) {
                    component.render(activeTemplate);
                }
            }

            // Activate the corresponding button
            const activeButton = document.querySelector(`.template-nav-button[data-template="${templateId}"]`);
            if (activeButton) {
                activeButton.classList.add('ring-2', 'ring-offset-2', 'ring-blue-400');
            }
        }
    }

    // Add click listeners to navigation buttons
    navButtons.forEach(button => {
        button.addEventListener('click', () => {
            const templateId = button.getAttribute('data-template');
            setActiveTemplate(templateId);
        });
    });

    // Initialize with default template
    await setActiveTemplate(defaultTemplate);

    // --- Placeholder for Agentic LLM Interaction Logic ---
    // This section demonstrates where you might hook up your LLM agent script.
    // This would typically involve WebSockets or regular API polling/fetching.

    const agentActionButton = document.getElementById('agent-action-button');
    const agentInstructionInput = document.getElementById('agent-instruction');
    const agentStatusEl = document.getElementById('agent-status');
    const agentTaskEl = document.getElementById('agent-task');
    const agentOutputEl = document.getElementById('agent-output');

    if (agentActionButton) {
        agentActionButton.addEventListener('click', () => {
            const instruction = agentInstructionInput.value;
            if (!instruction) {
                alert('Please enter an instruction for the agent.');
                return;
            }

            // 1. Update UI immediately (optional, for responsiveness)
            if(agentStatusEl) agentStatusEl.textContent = 'Processing...';
            if(agentStatusEl) agentStatusEl.className = 'text-lg font-medium text-yellow-600'; // Change color
            if(agentTaskEl) agentTaskEl.textContent = instruction;
             if(agentOutputEl) agentOutputEl.textContent += `\n> Sending task: ${instruction}`;


            // 2. Send instruction to backend/LLM script (Simulated here)
            console.log(`Sending instruction to backend: ${instruction}`);
            // Example: Replace with actual API call or WebSocket message
            // fetch('/api/agent/instruct', { method: 'POST', body: JSON.stringify({ instruction }) })
            //   .then(response => response.json())
            //   .then(data => updateAgentUI(data)); // Backend should respond with status updates

            // Simulate backend processing and response after a delay
            setTimeout(() => {
                 updateAgentUI({
                     status: 'Completed',
                     task: instruction,
                     output: `Task "${instruction}" executed successfully. Result: [Some Result Data]`,
                     statusColor: 'text-green-600'
                 });
                 // Clear input after simulated success
                if(agentInstructionInput) agentInstructionInput.value = '';
             }, 2500); // Simulate 2.5 second processing time
        });
    }

    // Function to update the Agentic LLM UI elements based on data from backend
    function updateAgentUI(data) {
         console.log("Updating Agent UI with data: ", data);
        if (agentStatusEl && data.status) {
            agentStatusEl.textContent = data.status;
            agentStatusEl.className = `text-lg font-medium ${data.statusColor || 'text-gray-600'}`;
        }
        if (agentTaskEl && data.task) agentTaskEl.textContent = data.task;
        if (agentOutputEl && data.output) {
            // Append new output, keep some history maybe
            agentOutputEl.textContent += `\n[${new Date().toLocaleTimeString()}] ${data.output}`;
            agentOutputEl.scrollTop = agentOutputEl.scrollHeight; // Scroll to bottom
        }
    }

     // Example: Simulate receiving an update from the backend (e.g., via WebSocket)
     /*
     setInterval(() => {
         updateAgentUI({
             status: 'Monitoring',
             output: 'System nominal.',
             statusColor: 'text-blue-600'
         });
     }, 10000); // Update every 10 seconds
     */

});