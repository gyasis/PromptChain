<script lang="ts">
  import { createEventDispatcher, onMount, onDestroy } from 'svelte';
  import { ZoomIn, ZoomOut, RotateCw, Download, Maximize2, Minimize2, ChevronLeft, ChevronRight, X } from 'lucide-svelte';

  interface Paper {
    id: string;
    title: string;
    authors: string[];
    pdfUrl?: string;
  }

  export let paper: Paper | null = null;
  export let isOpen = false;
  export let isFullscreen = false;

  const dispatch = createEventDispatcher<{
    close: void;
    toggleFullscreen: void;
    download: { paper: Paper };
  }>();

  let currentPage = 1;
  let totalPages = 0;
  let zoomLevel = 1.0;
  let rotation = 0;
  let isLoading = true;
  let error: string | null = null;
  
  let pdfContainer: HTMLDivElement;
  let pdfIframe: HTMLIFrameElement;

  // PDF viewer state
  let pdfViewerUrl = '';

  $: if (paper?.pdfUrl && isOpen) {
    loadPdf();
  }

  function loadPdf() {
    if (!paper?.pdfUrl) return;
    
    isLoading = true;
    error = null;
    
    // For demo purposes, simulate loading and then show demo content
    setTimeout(() => {
      isLoading = false;
      totalPages = 15; // Demo page count
      // In production, this would:
      // 1. Use PDF.js viewer embedded in iframe for better control
      // 2. Be served from the backend with proper CORS headers
      // 3. Handle actual PDF loading and parsing
      // const viewerUrl = '/pdf-viewer/viewer.html';
      // pdfViewerUrl = `${viewerUrl}?file=${encodeURIComponent(paper.pdfUrl)}`;
    }, 1000); // Simulate 1 second loading time
  }

  function handleClose() {
    dispatch('close');
  }

  function handleToggleFullscreen() {
    dispatch('toggleFullscreen');
  }

  function handleDownload() {
    if (paper) {
      dispatch('download', { paper });
    }
  }

  function zoomIn() {
    zoomLevel = Math.min(zoomLevel * 1.2, 3.0);
    updatePdfViewer();
  }

  function zoomOut() {
    zoomLevel = Math.max(zoomLevel / 1.2, 0.5);
    updatePdfViewer();
  }

  function rotate() {
    rotation = (rotation + 90) % 360;
    updatePdfViewer();
  }

  function previousPage() {
    if (currentPage > 1) {
      currentPage--;
      updatePdfViewer();
    }
  }

  function nextPage() {
    if (currentPage < totalPages) {
      currentPage++;
      updatePdfViewer();
    }
  }

  function goToPage(page: number) {
    if (page >= 1 && page <= totalPages) {
      currentPage = page;
      updatePdfViewer();
    }
  }

  function updatePdfViewer() {
    // Send commands to PDF.js viewer via postMessage
    if (pdfIframe?.contentWindow) {
      pdfIframe.contentWindow.postMessage({
        type: 'zoom',
        scale: zoomLevel
      }, '*');
      
      pdfIframe.contentWindow.postMessage({
        type: 'page',
        page: currentPage
      }, '*');
      
      pdfIframe.contentWindow.postMessage({
        type: 'rotate',
        rotation: rotation
      }, '*');
    }
  }

  function handleKeydown(event: KeyboardEvent) {
    if (!isOpen) return;
    
    switch (event.key) {
      case 'Escape':
        handleClose();
        break;
      case 'ArrowLeft':
        previousPage();
        break;
      case 'ArrowRight':
        nextPage();
        break;
      case '+':
      case '=':
        zoomIn();
        break;
      case '-':
        zoomOut();
        break;
      case 'r':
        rotate();
        break;
      case 'f':
        handleToggleFullscreen();
        break;
    }
  }

  onMount(() => {
    document.addEventListener('keydown', handleKeydown);
    
    // Listen for messages from PDF.js viewer
    function handleMessage(event: MessageEvent) {
      if (event.data.type === 'pdfLoaded') {
        totalPages = event.data.totalPages;
        isLoading = false;
      } else if (event.data.type === 'pageChanged') {
        currentPage = event.data.page;
      } else if (event.data.type === 'error') {
        error = event.data.message;
        isLoading = false;
      }
    }
    
    window.addEventListener('message', handleMessage);
    
    return () => {
      window.removeEventListener('message', handleMessage);
    };
  });

  onDestroy(() => {
    document.removeEventListener('keydown', handleKeydown);
  });

  // Simple fallback PDF viewer for demo purposes
  function createFallbackViewer(url: string): string {
    return `
      <div style="width: 100%; height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; background: #f5f5f5;">
        <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center; max-width: 400px;">
          <h3 style="margin: 0 0 10px 0; color: #333;">PDF Viewer</h3>
          <p style="margin: 0 0 15px 0; color: #666; font-size: 14px;">
            In production, this would show the PDF with full navigation controls.
          </p>
          <p style="margin: 0 0 15px 0; color: #888; font-size: 12px; word-break: break-all;">
            PDF URL: ${url}
          </p>
          <div style="display: flex; gap: 10px; justify-content: center;">
            <button onclick="window.open('${url}', '_blank')" style="padding: 8px 16px; background: #ff7733; color: white; border: none; border-radius: 4px; cursor: pointer;">
              Open in New Tab
            </button>
          </div>
        </div>
      </div>
    `;
  }
</script>

{#if isOpen && paper}
  <div class="fixed inset-0 z-50 bg-black bg-opacity-75 flex {isFullscreen ? 'p-0' : 'p-4'}" 
       role="dialog" 
       aria-modal="true">
    
    <!-- PDF Viewer Container -->
    <div class="bg-white rounded-lg overflow-hidden flex flex-col w-full max-w-7xl mx-auto {isFullscreen ? 'rounded-none max-w-none h-screen' : 'h-[90vh]'}">
      
      <!-- Header -->
      <div class="bg-neutral-100 border-b border-neutral-200 p-3 flex items-center justify-between flex-shrink-0">
        <!-- Paper Info -->
        <div class="flex-1 min-w-0">
          <h3 class="font-semibold text-neutral-900 truncate text-sm">
            {paper.title}
          </h3>
          <p class="text-xs text-neutral-600 truncate">
            {paper.authors.length > 0 ? paper.authors.slice(0, 3).join(', ') : 'Unknown authors'}
          </p>
        </div>

        <!-- Controls -->
        <div class="flex items-center gap-2 ml-4">
          <!-- Page Navigation -->
          <div class="flex items-center gap-1 text-sm">
            <button 
              type="button"
              class="p-1 hover:bg-neutral-200 rounded"
              disabled={currentPage <= 1}
              on:click={previousPage}
              title="Previous page (←)"
            >
              <ChevronLeft class="h-4 w-4" />
            </button>
            
            <div class="flex items-center gap-1 px-2 py-1 bg-white rounded border text-xs">
              <input 
                type="number" 
                bind:value={currentPage}
                on:change={() => goToPage(currentPage)}
                class="w-8 text-center border-none bg-transparent outline-none"
                min="1"
                max={totalPages}
              />
              <span>of {totalPages}</span>
            </div>
            
            <button 
              type="button"
              class="p-1 hover:bg-neutral-200 rounded"
              disabled={currentPage >= totalPages}
              on:click={nextPage}
              title="Next page (→)"
            >
              <ChevronRight class="h-4 w-4" />
            </button>
          </div>

          <!-- Zoom Controls -->
          <div class="flex items-center gap-1">
            <button 
              type="button"
              class="p-1.5 hover:bg-neutral-200 rounded"
              on:click={zoomOut}
              title="Zoom out (-)"
            >
              <ZoomOut class="h-4 w-4" />
            </button>
            
            <span class="text-xs px-2 py-1 bg-white rounded border min-w-[3rem] text-center">
              {Math.round(zoomLevel * 100)}%
            </span>
            
            <button 
              type="button"
              class="p-1.5 hover:bg-neutral-200 rounded"
              on:click={zoomIn}
              title="Zoom in (+)"
            >
              <ZoomIn class="h-4 w-4" />
            </button>
          </div>

          <!-- Other Controls -->
          <button 
            type="button"
            class="p-1.5 hover:bg-neutral-200 rounded"
            on:click={rotate}
            title="Rotate (R)"
          >
            <RotateCw class="h-4 w-4" />
          </button>

          <button 
            type="button"
            class="p-1.5 hover:bg-neutral-200 rounded"
            on:click={handleDownload}
            title="Download PDF"
          >
            <Download class="h-4 w-4" />
          </button>

          <button 
            type="button"
            class="p-1.5 hover:bg-neutral-200 rounded"
            on:click={handleToggleFullscreen}
            title="Toggle fullscreen (F)"
          >
            {#if isFullscreen}
              <Minimize2 class="h-4 w-4" />
            {:else}
              <Maximize2 class="h-4 w-4" />
            {/if}
          </button>

          <button 
            type="button"
            class="p-1.5 hover:bg-neutral-200 rounded text-neutral-600"
            on:click={handleClose}
            title="Close (Esc)"
          >
            <X class="h-4 w-4" />
          </button>
        </div>
      </div>

      <!-- PDF Content -->
      <div class="flex-1 relative bg-neutral-50 overflow-hidden" bind:this={pdfContainer}>
        {#if isLoading}
          <!-- Loading State -->
          <div class="absolute inset-0 flex items-center justify-center bg-white">
            <div class="text-center">
              <div class="spinner mb-4"></div>
              <p class="text-neutral-600">Loading PDF...</p>
            </div>
          </div>
        {:else if error}
          <!-- Error State -->
          <div class="absolute inset-0 flex items-center justify-center bg-white">
            <div class="text-center max-w-md mx-auto p-6">
              <div class="text-red-500 text-4xl mb-4">📄</div>
              <h3 class="text-lg font-medium text-neutral-900 mb-2">Failed to load PDF</h3>
              <p class="text-sm text-neutral-600 mb-4">{error}</p>
              <div class="flex gap-2 justify-center">
                <button class="btn-secondary" on:click={() => loadPdf()}>
                  Try Again
                </button>
                {#if paper.pdfUrl}
                  <a href={paper.pdfUrl} target="_blank" class="btn-primary">
                    Open in New Tab
                  </a>
                {/if}
              </div>
            </div>
          </div>
        {:else}
          <!-- PDF Viewer Demo -->
          <!-- For demo purposes, we'll show a mockup. In production, integrate PDF.js -->
          <div class="w-full h-full bg-neutral-50 flex flex-col items-center justify-center p-8">
            <div class="bg-white rounded-lg shadow-lg p-8 max-w-md text-center border border-neutral-200">
              <div class="text-6xl mb-4">📄</div>
              <h3 class="text-lg font-semibold text-neutral-900 mb-3">PDF Viewer Demo</h3>
              <p class="text-sm text-neutral-600 mb-4">
                In production, this would display the actual PDF content with full navigation, zoom, and annotation capabilities.
              </p>
              <div class="bg-neutral-50 rounded-md p-3 mb-4">
                <p class="text-xs text-neutral-500 font-medium mb-1">Paper:</p>
                <p class="text-xs text-neutral-800 font-mono break-all">{paper?.title?.substring(0, 50)}...</p>
              </div>
              <div class="grid grid-cols-2 gap-3 text-xs">
                <div class="bg-green-50 border border-green-200 rounded-md p-2">
                  <div class="text-green-700 font-medium">✓ PDF.js Integration</div>
                  <div class="text-green-600">Full PDF rendering</div>
                </div>
                <div class="bg-blue-50 border border-blue-200 rounded-md p-2">
                  <div class="text-blue-700 font-medium">⚡ Navigation</div>
                  <div class="text-blue-600">Page controls, zoom</div>
                </div>
                <div class="bg-purple-50 border border-purple-200 rounded-md p-2">
                  <div class="text-purple-700 font-medium">🔍 Search</div>
                  <div class="text-purple-600">Text search & highlight</div>
                </div>
                <div class="bg-orange-50 border border-orange-200 rounded-md p-2">
                  <div class="text-orange-700 font-medium">📝 Annotations</div>
                  <div class="text-orange-600">Notes & highlighting</div>
                </div>
              </div>
              {#if paper?.url}
                <div class="mt-4 pt-4 border-t border-neutral-200">
                  <a 
                    href={paper.url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    class="inline-flex items-center gap-2 text-sm font-medium text-primary-600 hover:text-primary-700"
                  >
                    <span>View Original Source</span>
                    <svg class="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                    </svg>
                  </a>
                </div>
              {/if}
            </div>
          </div>
          
          <!-- Overlay for interaction (when using iframe) -->
          <div class="absolute inset-0 pointer-events-none">
            <!-- Custom overlay content could go here -->
          </div>
        {/if}
      </div>

      <!-- Footer Status -->
      <div class="bg-neutral-100 border-t border-neutral-200 px-3 py-2 text-xs text-neutral-600 flex items-center justify-between flex-shrink-0">
        <div class="flex items-center gap-4">
          <span>Zoom: {Math.round(zoomLevel * 100)}%</span>
          <span>Page: {currentPage} of {totalPages}</span>
          {#if rotation > 0}
            <span>Rotated: {rotation}°</span>
          {/if}
        </div>
        
        <div class="text-xs">
          Press <kbd class="px-1 py-0.5 bg-white rounded border">Esc</kbd> to close • 
          <kbd class="px-1 py-0.5 bg-white rounded border">← →</kbd> navigate • 
          <kbd class="px-1 py-0.5 bg-white rounded border">+ -</kbd> zoom
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  kbd {
    font-family: ui-monospace, monospace;
    font-size: 0.75rem;
  }
  
  input[type="number"]::-webkit-outer-spin-button,
  input[type="number"]::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
  }
  
  input[type="number"] {
    -moz-appearance: textfield;
  }
</style>