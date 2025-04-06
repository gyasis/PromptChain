export const template = `
<div class="container mx-auto">
    <h2 class="text-xl font-semibold mb-4">Model Repository</h2>
    <div class="flex flex-col md:flex-row gap-6">
        <!-- Filters -->
        <aside class="md:w-1/4 bg-white p-4 rounded shadow">
            <h3 class="font-semibold mb-3 border-b pb-2">Filters</h3>
            <div id="model-filter-options">
                <div class="mb-2">
                    <label class="block text-sm font-medium text-gray-700">Category</label>
                    <select class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50">
                        <option>All</option>
                        <option>Text Generation</option>
                        <option>Image Analysis</option>
                    </select>
                </div>
                <div class="mb-2">
                    <label class="block text-sm font-medium text-gray-700">Compatibility</label>
                    <div><input type="checkbox" class="rounded" data-filter="max"> MAX Engine</div>
                    <div><input type="checkbox" class="rounded" data-filter="pytorch"> PyTorch</div>
                </div>
            </div>
        </aside>

        <!-- Model Grid/List -->
        <section id="model-grid-area" class="flex-1 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <!-- Model cards will be dynamically inserted here -->
        </section>
    </div>

    <!-- Model Detail View -->
    <div id="model-detail-area" class="mt-8 bg-white p-6 rounded shadow border hidden">
        <h3 class="text-lg font-bold mb-3">Model Detail View</h3>
        <div id="model-detail-content"></div>
    </div>
</div>
`; 