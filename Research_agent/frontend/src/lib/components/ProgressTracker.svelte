<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  
  interface ProgressStep {
    id: string;
    name: string;
    status: 'pending' | 'running' | 'completed' | 'error';
    progress?: number;
    message?: string;
    startTime?: Date;
    endTime?: Date;
    substeps?: ProgressStep[];
  }

  interface ProgressSession {
    sessionId: string;
    topic: string;
    status: 'initializing' | 'searching' | 'processing' | 'synthesizing' | 'completed' | 'error';
    overallProgress: number;
    currentStep: string;
    steps: ProgressStep[];
    startTime: Date;
    estimatedCompletion?: Date;
    papersFound: number;
    queriesProcessed: number;
  }

  export let sessionId: string | null = null;
  export let isVisible = false;

  let session: ProgressSession | null = null;
  let websocket: WebSocket | null = null;
  let connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error' = 'disconnected';
  let retryCount = 0;
  let maxRetries = 5;

  onMount(() => {
    if (sessionId) {
      connectWebSocket();
    }
  });

  onDestroy(() => {
    if (websocket) {
      websocket.close();
    }
  });

  function connectWebSocket() {
    if (!sessionId) return;

    connectionStatus = 'connecting';
    const wsUrl = `ws://localhost:8078/ws/progress/${sessionId}`;
    
    websocket = new WebSocket(wsUrl);

    websocket.onopen = () => {
      connectionStatus = 'connected';
      retryCount = 0;
      console.log('Progress WebSocket connected');
    };

    websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'progress_update') {
          session = data.session;
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    websocket.onclose = () => {
      connectionStatus = 'disconnected';
      console.log('Progress WebSocket disconnected');
      
      // Auto-reconnect with exponential backoff
      if (retryCount < maxRetries && sessionId) {
        retryCount++;
        const delay = Math.pow(2, retryCount) * 1000;
        setTimeout(connectWebSocket, delay);
      }
    };

    websocket.onerror = (error) => {
      connectionStatus = 'error';
      console.error('WebSocket error:', error);
    };
  }

  function getStepIcon(step: ProgressStep): string {
    switch (step.status) {
      case 'completed': return '✅';
      case 'running': return '🔄';
      case 'error': return '❌';
      case 'pending': return '⏳';
      default: return '⏳';
    }
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'completed': return 'text-green-600';
      case 'running': return 'text-primary-500';
      case 'error': return 'text-red-600';
      case 'pending': return 'text-neutral-500';
      default: return 'text-neutral-500';
    }
  }

  function formatDuration(startTime: Date, endTime?: Date): string {
    const end = endTime || new Date();
    const diff = end.getTime() - startTime.getTime();
    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);
    
    if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    }
    return `${seconds}s`;
  }

  function formatETA(eta?: Date): string {
    if (!eta) return 'Calculating...';
    const now = new Date();
    const diff = eta.getTime() - now.getTime();
    const minutes = Math.ceil(diff / (1000 * 60));
    return minutes > 0 ? `~${minutes}m remaining` : 'Almost done';
  }

  $: isActive = session?.status && !['completed', 'error'].includes(session.status);
</script>

{#if isVisible && session}
  <div class="card p-6 animate-in">
    <!-- Header -->
    <div class="section-header">
      <div>
        <h3 class="section-title">Research Progress</h3>
        <p class="section-subtitle">{session.topic}</p>
      </div>
      
      <div class="flex items-center gap-3">
        <!-- Connection Status -->
        <div class="flex items-center gap-2 text-sm">
          <div class="status-dot {connectionStatus === 'connected' ? 'bg-green-500' : connectionStatus === 'connecting' ? 'bg-amber-500 animate-pulse' : 'bg-red-500'}"></div>
          <span class="text-neutral-600">{connectionStatus}</span>
        </div>
        
        <!-- Overall Progress -->
        <div class="flex items-center gap-2">
          <span class="text-2xl font-bold text-primary-600">{Math.round(session.overallProgress)}%</span>
        </div>
      </div>
    </div>

    <!-- Overall Progress Bar -->
    <div class="mb-6">
      <div class="flex justify-between items-center mb-2">
        <span class="text-sm font-medium text-neutral-700">Overall Progress</span>
        <span class="text-sm text-neutral-500">{formatETA(session.estimatedCompletion)}</span>
      </div>
      <div class="h-3 bg-neutral-200 rounded-full overflow-hidden">
        <div 
          class="h-full bg-gradient-to-r from-primary-400 to-primary-500 transition-all duration-300 ease-out"
          style="width: {session.overallProgress}%"
        ></div>
      </div>
    </div>

    <!-- Current Status -->
    <div class="mb-6 p-4 bg-neutral-50 rounded-xl">
      <div class="flex items-center gap-3">
        {#if isActive}
          <div class="spinner"></div>
        {:else if session.status === 'completed'}
          <div class="text-green-500 text-xl">✅</div>
        {:else if session.status === 'error'}
          <div class="text-red-500 text-xl">❌</div>
        {/if}
        
        <div>
          <div class="font-medium text-neutral-900">{session.currentStep}</div>
          <div class="text-sm text-neutral-600">
            {session.papersFound} papers found • {session.queriesProcessed} queries processed
          </div>
        </div>
      </div>
    </div>

    <!-- Detailed Steps -->
    <div class="space-y-4">
      <h4 class="font-semibold text-neutral-900 mb-3">Research Steps</h4>
      
      {#each session.steps as step, index}
        <div class="flex gap-4">
          <!-- Step Icon -->
          <div class="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm
                      {step.status === 'completed' ? 'bg-green-100 text-green-600' :
                       step.status === 'running' ? 'bg-primary-100 text-primary-600' :
                       step.status === 'error' ? 'bg-red-100 text-red-600' :
                       'bg-neutral-100 text-neutral-500'}">
            {getStepIcon(step)}
          </div>
          
          <!-- Step Content -->
          <div class="flex-1 min-w-0">
            <div class="flex items-center justify-between">
              <h5 class="font-medium text-neutral-900">{step.name}</h5>
              <div class="text-sm text-neutral-500">
                {#if step.startTime}
                  {formatDuration(step.startTime, step.endTime)}
                {/if}
              </div>
            </div>
            
            {#if step.message}
              <p class="text-sm text-neutral-600 mt-1">{step.message}</p>
            {/if}
            
            <!-- Step Progress Bar -->
            {#if step.progress !== undefined && step.status === 'running'}
              <div class="mt-2">
                <div class="h-2 bg-neutral-200 rounded-full overflow-hidden">
                  <div 
                    class="h-full bg-primary-400 transition-all duration-300"
                    style="width: {step.progress}%"
                  ></div>
                </div>
              </div>
            {/if}
            
            <!-- Substeps -->
            {#if step.substeps && step.substeps.length > 0}
              <div class="mt-3 ml-4 space-y-2">
                {#each step.substeps as substep}
                  <div class="flex items-center gap-2 text-sm">
                    <span class={getStatusColor(substep.status)}>{getStepIcon(substep)}</span>
                    <span class="text-neutral-700">{substep.name}</span>
                    {#if substep.progress !== undefined}
                      <span class="text-neutral-500">({substep.progress}%)</span>
                    {/if}
                  </div>
                {/each}
              </div>
            {/if}
          </div>
        </div>
        
        {#if index < session.steps.length - 1}
          <div class="ml-4 h-4 w-px bg-neutral-200"></div>
        {/if}
      {/each}
    </div>

    <!-- Summary Stats -->
    {#if session.status === 'completed'}
      <div class="mt-6 pt-4 border-t border-neutral-200">
        <div class="grid grid-cols-3 gap-4 text-center">
          <div>
            <div class="text-2xl font-bold text-neutral-900">{session.papersFound}</div>
            <div class="text-sm text-neutral-600">Papers Found</div>
          </div>
          <div>
            <div class="text-2xl font-bold text-neutral-900">{session.queriesProcessed}</div>
            <div class="text-sm text-neutral-600">Queries Processed</div>
          </div>
          <div>
            <div class="text-2xl font-bold text-neutral-900">{formatDuration(session.startTime)}</div>
            <div class="text-sm text-neutral-600">Total Time</div>
          </div>
        </div>
      </div>
    {/if}
  </div>
{:else if isVisible && !session}
  <!-- Loading State -->
  <div class="card p-6">
    <div class="flex items-center justify-center py-8">
      <div class="spinner mr-3"></div>
      <span class="text-neutral-600">Connecting to research session...</span>
    </div>
  </div>
{/if}

<style>
  .animate-in {
    animation: slideIn 0.3s ease-out;
  }

  @keyframes slideIn {
    from {
      opacity: 0;
      transform: translateY(20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
</style>