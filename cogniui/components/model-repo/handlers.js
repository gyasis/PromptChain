// Sample model data (in real app, this would come from an API)
const sampleModels = [
    {
        id: 1,
        name: 'Awesome Model Alpha',
        description: 'Brief description of the model and its capabilities.',
        category: 'Text Generation',
        compatibility: ['max'],
        thumbnail: 'https://via.placeholder.com/150/771796'
    },
    {
        id: 2,
        name: 'Vision Model Beta',
        description: 'Advanced image analysis and processing capabilities.',
        category: 'Image Analysis',
        compatibility: ['pytorch'],
        thumbnail: 'https://via.placeholder.com/150/24f355'
    }
];

// Create a model card element
function createModelCard(model) {
    return `
        <div class="bg-white p-4 rounded shadow border hover:shadow-lg transition-shadow">
            <img src="${model.thumbnail}" alt="${model.name} Thumbnail" class="w-full h-32 object-cover rounded mb-2">
            <h4 class="font-bold">${model.name}</h4>
            <p class="text-sm text-gray-600 mb-2">${model.description}</p>
            <span class="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full mr-1">${model.category}</span>
            ${model.compatibility.map(c => 
                `<span class="text-xs bg-green-100 text-green-800 px-2 py-1 rounded-full">${c}</span>`
            ).join(' ')}
            <button class="mt-3 w-full text-sm bg-green-500 hover:bg-green-700 text-white py-1 px-3 rounded" 
                    onclick="showModelDetails(${model.id})">
                View Details
            </button>
        </div>
    `;
}

// Filter models based on selected criteria
function filterModels(models, filters) {
    return models.filter(model => {
        if (filters.category && filters.category !== 'All' && model.category !== filters.category) {
            return false;
        }
        if (filters.compatibility.length > 0) {
            return filters.compatibility.some(c => model.compatibility.includes(c));
        }
        return true;
    });
}

// Update the grid with filtered models
function updateModelGrid(filters = { category: 'All', compatibility: [] }) {
    const filteredModels = filterModels(sampleModels, filters);
    const gridArea = document.getElementById('model-grid-area');
    gridArea.innerHTML = filteredModels.map(createModelCard).join('');
}

// Show model details
function showModelDetails(modelId) {
    const model = sampleModels.find(m => m.id === modelId);
    if (!model) return;

    const detailArea = document.getElementById('model-detail-area');
    const detailContent = document.getElementById('model-detail-content');
    
    detailContent.innerHTML = `
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
                <img src="${model.thumbnail}" alt="${model.name}" class="w-full rounded">
                <h4 class="text-xl font-bold mt-4">${model.name}</h4>
                <p class="text-gray-600">${model.description}</p>
            </div>
            <div>
                <h5 class="font-semibold mb-2">Implementation Example:</h5>
                <pre class="bg-gray-900 text-white p-3 rounded mt-3 text-sm overflow-x-auto"><code>
import model_loader

model = model_loader.load("${model.name}")
result = model.predict("Input text...")
                </code></pre>
            </div>
        </div>
    `;
    
    detailArea.classList.remove('hidden');
}

// Initialize event listeners
export function initModelRepo() {
    // Category filter
    const categorySelect = document.querySelector('#model-filter-options select');
    categorySelect.addEventListener('change', (e) => {
        const filters = {
            category: e.target.value,
            compatibility: Array.from(document.querySelectorAll('[data-filter]:checked')).map(cb => cb.dataset.filter)
        };
        updateModelGrid(filters);
    });

    // Compatibility checkboxes
    const compatibilityCheckboxes = document.querySelectorAll('[data-filter]');
    compatibilityCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', () => {
            const filters = {
                category: categorySelect.value,
                compatibility: Array.from(document.querySelectorAll('[data-filter]:checked')).map(cb => cb.dataset.filter)
            };
            updateModelGrid(filters);
        });
    });

    // Initial load
    updateModelGrid();
}

// Export functions that need to be accessible
export { showModelDetails, updateModelGrid }; 