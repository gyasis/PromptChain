import { template } from './template.js';
import { initModelRepo } from './handlers.js';

export default {
    name: 'model-repo',
    render(container) {
        // Insert the template
        container.innerHTML = template;
        
        // Initialize the component
        initModelRepo();
        
        return {
            // Return any methods that should be accessible from outside
            update() {
                initModelRepo();
            }
        };
    }
}; 