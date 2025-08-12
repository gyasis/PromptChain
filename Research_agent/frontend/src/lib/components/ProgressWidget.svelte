<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  
  export let sessionId: string | null = null;
  export let topic = '';
  export let progress = 0;
  export let status: 'running' | 'completed' | 'error' = 'running';
  export let currentStep = '';
  export let isVisible = false;

  const dispatch = createEventDispatcher<{
    expand: void;
    close: void;
  }>();

  function handleExpand() {
    dispatch('expand');
  }

  function handleClose() {
    dispatch('close');
  }

  function getStatusColor(): string {
    switch (status) {
      case 'completed': return 'border-green-500 bg-green-50';
      case 'error': return 'border-red-500 bg-red-50';
      case 'running': return 'border-primary-500 bg-primary-50';
      default: return 'border-neutral-300 bg-white';
    }
  }

  function getStatusIcon(): string {
    switch (status) {
      case 'completed': return '✅';
      case 'error': return '❌';
      case 'running': return '🔄';
      default: return '⏳';
    }
  }
</script>

{#if isVisible}
  <div class="fixed bottom-6 right-6 z-40 animate-in">
    <!-- Progress Widget -->
    <div class="bg-white rounded-2xl shadow-large border-2 {getStatusColor()} p-4 min-w-[280px] max-w-[320px]">
      <!-- Widget Header -->
      <div class="flex items-center justify-between mb-3">
        <div class="flex items-center gap-2">
          <span class="text-lg">{getStatusIcon()}</span>
          <h3 class="font-semibold text-neutral-900 text-sm">Research Progress</h3>
        </div>
        
        <div class="flex items-center gap-1">
          <!-- Expand Button -->
          <button
            type="button"
            class="p-1 hover:bg-neutral-100 rounded text-neutral-600 hover:text-neutral-900"
            on:click={handleExpand}
            title="Expand progress view"
          >
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                    d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"/>
            </svg>
          </button>
          
          <!-- Close Button -->
          <button
            type="button"
            class="p-1 hover:bg-neutral-100 rounded text-neutral-600 hover:text-neutral-900"
            on:click={handleClose}
            title="Close progress tracker"
          >
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                    d="M6 18L18 6M6 6l12 12"/>
            </svg>
          </button>
        </div>
      </div>
      
      <!-- Topic -->
      {#if topic}
        <p class="text-sm text-neutral-700 mb-2 truncate" title={topic}>{topic}</p>
      {/if}
      
      <!-- Current Step -->
      {#if currentStep}
        <p class="text-xs text-neutral-600 mb-3">{currentStep}</p>
      {/if}
      
      <!-- Progress Bar -->
      <div class="mb-2">
        <div class="flex justify-between items-center mb-1">
          <span class="text-xs font-medium text-neutral-700">Progress</span>
          <span class="text-xs text-neutral-600">{Math.round(progress)}%</span>
        </div>
        <div class="h-2 bg-neutral-200 rounded-full overflow-hidden">
          <div 
            class="h-full transition-all duration-300 ease-out {status === 'completed' ? 'bg-green-500' : 
                  status === 'error' ? 'bg-red-500' : 'bg-primary-400'}"
            style="width: {progress}%"
          ></div>
        </div>
      </div>
      
      <!-- Status Message -->
      <div class="flex items-center justify-between">
        {#if status === 'running'}
          <div class="flex items-center gap-2">
            <div class="w-3 h-3 border-2 border-primary-400 border-t-transparent rounded-full animate-spin"></div>
            <span class="text-xs text-neutral-600">Processing...</span>
          </div>
        {:else if status === 'completed'}
          <span class="text-xs text-green-600 font-medium">Completed successfully</span>
        {:else if status === 'error'}
          <span class="text-xs text-red-600 font-medium">Error occurred</span>
        {/if}
        
        <!-- Session ID (for debugging) -->
        {#if sessionId}
          <span class="text-xs text-neutral-400 font-mono">#{sessionId.slice(-6)}</span>
        {/if}
      </div>
      
      <!-- Clickable Expand Area -->
      <button
        type="button"
        class="absolute inset-0 w-full h-full opacity-0"
        on:click={handleExpand}
        aria-label="Expand progress view"
      ></button>
    </div>
  </div>
{/if}

<style>
  .animate-in {
    animation: slideInUp 0.3s ease-out;
  }

  @keyframes slideInUp {
    from {
      opacity: 0;
      transform: translateY(20px) scale(0.95);
    }
    to {
      opacity: 1;
      transform: translateY(0) scale(1);
    }
  }
</style>