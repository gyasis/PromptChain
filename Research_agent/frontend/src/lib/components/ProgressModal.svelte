<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import ProgressTracker from './ProgressTracker.svelte';

  export let isOpen = false;
  export let sessionId: string | null = null;
  export let topic = '';

  const dispatch = createEventDispatcher<{
    close: void;
    minimize: void;
  }>();

  function handleClose() {
    dispatch('close');
  }

  function handleMinimize() {
    dispatch('minimize');
  }

  function handleBackdropClick(event: MouseEvent) {
    if (event.target === event.currentTarget) {
      handleClose();
    }
  }

  function handleKeydown(event: KeyboardEvent) {
    if (event.key === 'Escape') {
      handleClose();
    }
  }
</script>

{#if isOpen}
  <!-- Modal Backdrop -->
  <div 
    class="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
    on:click={handleBackdropClick}
    on:keydown={handleKeydown}
    role="dialog"
    aria-modal="true"
    aria-labelledby="progress-modal-title"
    tabindex="-1"
  >
    <!-- Modal Content -->
    <div class="bg-white rounded-2xl shadow-large max-w-4xl w-full max-h-[90vh] overflow-hidden animate-in">
      <!-- Modal Header -->
      <div class="flex items-center justify-between p-6 border-b border-neutral-200">
        <div>
          <h2 id="progress-modal-title" class="text-xl font-semibold text-neutral-900">
            Research Progress
          </h2>
          {#if topic}
            <p class="text-sm text-neutral-600 mt-1">{topic}</p>
          {/if}
        </div>
        
        <div class="flex items-center gap-2">
          <!-- Minimize Button -->
          <button
            type="button"
            class="btn-ghost p-2 rounded-lg"
            on:click={handleMinimize}
            title="Minimize to corner"
          >
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                    d="M20 12H4"/>
            </svg>
          </button>
          
          <!-- Close Button -->
          <button
            type="button"
            class="btn-ghost p-2 rounded-lg"
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
      
      <!-- Modal Body -->
      <div class="p-6 overflow-y-auto max-h-[calc(90vh-120px)] scrollbar-thin">
        <ProgressTracker {sessionId} isVisible={true} />
      </div>
    </div>
  </div>
{/if}

<style>
  .animate-in {
    animation: modalIn 0.2s ease-out;
  }

  @keyframes modalIn {
    from {
      opacity: 0;
      transform: scale(0.95) translateY(20px);
    }
    to {
      opacity: 1;
      transform: scale(1) translateY(0);
    }
  }
</style>