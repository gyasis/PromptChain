<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Search, BookOpen, Brain, Database, BarChart3, Settings, Plus, Play, History, MessageCircle, TrendingUp, FileText } from 'lucide-svelte';
  import { Button } from "$lib/components/ui/button/index.js";
  import * as Card from "$lib/components/ui/card/index.js";
  import { Input } from "$lib/components/ui/input/index.js";
  import * as Table from "$lib/components/ui/table/index.js";
  import { Badge } from "$lib/components/ui/badge/index.js";
  import { Skeleton } from "$lib/components/ui/skeleton/index.js";
  import ProgressModal from "$lib/components/ProgressModal.svelte";
  import ProgressWidget from "$lib/components/ProgressWidget.svelte";
  import PaperDiscoveryFeed from "$lib/components/PaperDiscoveryFeed.svelte";
  import PdfViewer from "$lib/components/PdfViewer.svelte";
  import ChatInterface from "$lib/components/ChatInterface.svelte";
  import ResearchVisualization from "$lib/components/ResearchVisualization.svelte";
  import ReportGenerator from "$lib/components/ReportGenerator.svelte";
  import { progressStore, activeSession, progressWebSocketManager, type Paper } from "$lib/stores/progress";
  import { startProgressDemo } from "$lib/utils/progressDemo";
  import { errorHandler, handleApiCall } from "$lib/utils/errorHandler";
  import { logger, LogCategory, logApiCall } from "$lib/utils/logger";
  
  // API base URL
  const API_BASE = 'http://localhost:8078/api';
  
  // Reactive state
  let currentView = 'dashboard';
  let researchSessions: any[] = [];
  let systemHealth = { status: 'loading', timestamp: '', active_sessions: 0, websocket_connections: 0 };
  let newResearchTopic = '';
  let isLoading = false;
  
  // Progress tracking state
  let progressModalOpen = false;
  let progressWidgetVisible = false;
  
  // PDF viewer state
  let selectedPaper: Paper | null = null;
  let isPdfViewerOpen = false;
  let isPdfViewerFullscreen = false;
  
  // Mock data for demonstration
  let mockMetrics = {
    totalPapers: 127,
    totalSessions: 8,
    avgProcessingTime: '3.2m',
    successRate: '94%'
  };
  
  let mockRecentSessions = [
    {
      id: 'session-1',
      title: 'Machine Learning in Healthcare',
      status: 'completed',
      created_at: '2025-01-12T10:30:00Z',
      papers_found: 45,
      queries_processed: 12
    },
    {
      id: 'session-2', 
      title: 'Climate Change Impact on Agriculture',
      status: 'in_progress',
      created_at: '2025-01-12T14:15:00Z',
      papers_found: 23,
      queries_processed: 7
    },
    {
      id: 'session-3',
      title: 'Quantum Computing Algorithms',
      status: 'completed',
      created_at: '2025-01-11T16:45:00Z',
      papers_found: 38,
      queries_processed: 15
    }
  ];
  
  onMount(async () => {
    try {
      logger.info('Application initializing', undefined, LogCategory.SYSTEM);
      await checkSystemHealth();
      await loadResearchSessions();
      logger.info('Application initialized successfully', undefined, LogCategory.SYSTEM);
    } catch (error) {
      logger.error('Application initialization failed', error, LogCategory.SYSTEM);
      errorHandler.handleError({
        type: 'unknown',
        severity: 'high',
        message: 'Failed to initialize application'
      });
    }
  });
  
  async function checkSystemHealth() {
    const apiLogger = logApiCall('GET', `${API_BASE}/health`);
    
    try {
      logger.info('Checking system health', undefined, LogCategory.API);
      const response = await fetch(`${API_BASE}/health`);
      
      if (response.ok) {
        systemHealth = await response.json();
        apiLogger.success(response.status, systemHealth);
        logger.info('System health check successful', systemHealth, LogCategory.SYSTEM);
      } else {
        apiLogger.error(response.status);
        errorHandler.handleApiError(response, { operation: 'health_check' });
        systemHealth.status = 'error';
      }
    } catch (error) {
      apiLogger.error(0, error);
      logger.error('System health check failed', error, LogCategory.API);
      errorHandler.handleError({
        type: 'network',
        severity: 'medium',
        message: 'Failed to check system health',
        details: error
      });
      systemHealth.status = 'error';
    }
  }
  
  async function loadResearchSessions() {
    try {
      logger.info('Loading research sessions', undefined, LogCategory.SESSION);
      // For now using mock data, would connect to actual API
      researchSessions = mockRecentSessions;
      logger.info('Research sessions loaded', { count: researchSessions.length }, LogCategory.SESSION);
    } catch (error) {
      logger.error('Failed to load research sessions', error, LogCategory.SESSION);
      errorHandler.handleError({
        type: 'session',
        severity: 'medium',
        message: 'Failed to load research sessions',
        details: error
      });
    }
  }
  
  async function startNewResearch() {
    if (!newResearchTopic.trim()) {
      logger.warn('Research started with empty topic', undefined, LogCategory.USER);
      return;
    }
    
    const sessionId = `session-${Date.now()}`;
    const topic = newResearchTopic;
    
    logger.logUserAction('start_research', 'research_input', { topic, sessionId });
    logger.startPerformanceTracking(`research_session_${sessionId}`);
    
    isLoading = true;
    try {
      logger.info('Starting new research session', { topic, sessionId }, LogCategory.RESEARCH);
      
      // Start progress tracking
      progressStore.startSession(sessionId, topic);
      logger.logSessionEvent('session_created', sessionId, { topic });
      
      // For demo purposes, use WebSocket simulation instead of real API
      // In production, this would connect to real WebSocket
      // progressWebSocketManager.connect(sessionId);
      
      // Start demo progress simulation
      startProgressDemo(sessionId, topic);
      logger.info('Demo progress simulation started', { sessionId }, LogCategory.RESEARCH);
      
      // Show progress widget
      progressWidgetVisible = true;
      logger.logUIEvent('progress_widget_shown', 'ProgressWidget', { sessionId });
      
      // Mock API call - would integrate with actual research API
      logger.debug('Research API call (simulated)', { topic, sessionId }, LogCategory.API);
      
      // Simulate API call for demo (comment out for real implementation)
      // const apiLogger = logApiCall('POST', `${API_BASE}/research/start`);
      // const response = await handleApiCall(() => fetch(`${API_BASE}/research/start`, {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ topic, session_id: sessionId })
      // }), { operation: 'start_research', sessionId });
      
      const newSession = {
        id: sessionId,
        title: topic,
        status: 'in_progress',
        created_at: new Date().toISOString(),
        papers_found: 0,
        queries_processed: 0
      };
      
      researchSessions = [newSession, ...researchSessions];
      newResearchTopic = '';
      
      // Open progress modal for immediate feedback
      progressModalOpen = true;
      logger.logUIEvent('progress_modal_opened', 'ProgressModal', { sessionId });
      
      logger.info('Research session started successfully', { sessionId, topic }, LogCategory.RESEARCH);
      
    } catch (error) {
      logger.error('Failed to start research session', error, LogCategory.RESEARCH, { sessionId, topic });
      
      errorHandler.handleError({
        type: 'api',
        severity: 'high',
        message: 'Failed to start research session',
        details: error,
        sessionId,
        context: { topic, operation: 'start_research' }
      });
      
      // Update progress with error
      if (progressStore) {
        progressStore.errorSession(sessionId, error?.message || 'Failed to start research');
      }
    } finally {
      isLoading = false;
      logger.endPerformanceTracking(`research_session_${sessionId}`, { topic }, { operation: 'start_research' });
    }
  }
  
  function formatDate(dateStr: string): string {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  }
  
  function getStatusBadgeClass(status: string): string {
    switch (status) {
      case 'completed': return 'badge-success';
      case 'in_progress': return 'badge-warning';
      case 'error': return 'badge-danger';
      default: return 'badge-info';
    }
  }
  
  function getStatusDotClass(status: string): string {
    switch (status) {
      case 'completed': return 'bg-green-500';
      case 'in_progress': return 'bg-amber-500 animate-pulse';
      case 'error': return 'bg-red-500';
      default: return 'bg-neutral-400';
    }
  }
  
  // Progress tracking event handlers
  function handleProgressModalClose() {
    progressModalOpen = false;
  }
  
  function handleProgressModalMinimize() {
    progressModalOpen = false;
    progressWidgetVisible = true;
  }
  
  function handleProgressWidgetExpand() {
    progressWidgetVisible = false;
    progressModalOpen = true;
  }
  
  function handleProgressWidgetClose() {
    progressWidgetVisible = false;
    // Keep the session running, just hide the UI
  }

  // PDF viewer event handlers
  function handleOpenPdf(event: CustomEvent<{ paper: Paper }>) {
    selectedPaper = event.detail.paper;
    isPdfViewerOpen = true;
  }

  function handleDownloadPdf(event: CustomEvent<{ paper: Paper }>) {
    const paper = event.detail.paper;
    if (paper.pdfUrl) {
      // Create download link
      const link = document.createElement('a');
      link.href = paper.pdfUrl;
      link.download = `${paper.title.substring(0, 50)}.pdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } else if (paper.url) {
      // Open source URL in new tab
      window.open(paper.url, '_blank');
    }
  }

  function handleClosePdfViewer() {
    isPdfViewerOpen = false;
    selectedPaper = null;
  }

  function handleTogglePdfFullscreen() {
    isPdfViewerFullscreen = !isPdfViewerFullscreen;
  }

  function handlePdfDownload(event: CustomEvent<{ paper: Paper }>) {
    handleDownloadPdf(event);
  }

  // Chat interface event handlers
  function handleChatMessage(event: CustomEvent<{ message: string; sessionId: string; researchSessionId?: string }>) {
    const { message, sessionId, researchSessionId } = event.detail;
    console.log('Chat message:', message, 'Session:', sessionId, 'Research:', researchSessionId);
    
    // In production, this would send the message to the backend API
    // For now, the ChatInterface handles demo responses internally
    
    // Example API call structure:
    // try {
    //   const response = await fetch(`${API_BASE}/chat/message`, {
    //     method: 'POST',
    //     headers: { 'Content-Type': 'application/json' },
    //     body: JSON.stringify({
    //       message,
    //       session_id: sessionId,
    //       research_session_id: researchSessionId
    //     })
    //   });
    //   const result = await response.json();
    //   // Handle response...
    // } catch (error) {
    //   console.error('Failed to send chat message:', error);
    // }
  }

  function handleNewChatSession() {
    console.log('Creating new chat session');
    // In production, this would create a new chat session via API
  }

  function handleClearChatSession(event: CustomEvent<{ sessionId: string }>) {
    const { sessionId } = event.detail;
    console.log('Clearing chat session:', sessionId);
    // In production, this would clear the chat session via API
  }

  // Report generator event handlers
  function handleGenerateReport(event: CustomEvent<{ config: any; papers: Paper[]; sessionId: string | null }>) {
    const { config, papers, sessionId } = event.detail;
    console.log('Generating report:', config, papers.length, sessionId);
    
    // In production, this would call the backend API to generate the report
    // const response = await fetch(`${API_BASE}/reports/generate`, {
    //   method: 'POST',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify({ config, papers, session_id: sessionId })
    // });
  }

  function handlePreviewReport(event: CustomEvent<{ config: any; papers: Paper[] }>) {
    const { config, papers } = event.detail;
    console.log('Previewing report:', config, papers.length);
    // The ReportGenerator handles preview internally
  }

  function handleExportReport(event: CustomEvent<{ config: any; format: string }>) {
    const { config, format } = event.detail;
    console.log('Exporting report:', config, format);
    
    // In production, this would trigger the actual file generation and download
    // const response = await fetch(`${API_BASE}/reports/export`, {
    //   method: 'POST', 
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify({ config, format })
    // });
  }
  
  // Reactive statements for progress store integration
  $: {
    // Subscribe to progress store changes
    if ($progressStore) {
      progressModalOpen = $progressStore.modalOpen;
      progressWidgetVisible = $progressStore.widgetVisible;
    }
  }
  
  // Cleanup on destroy
  onDestroy(() => {
    progressWebSocketManager.disconnectAll();
  });
</script>

<main class="min-h-screen bg-neutral-50">
  <!-- Header -->
  <header class="bg-white border-b border-neutral-200/50 px-6 py-4 fixed top-0 left-0 right-0 z-40">
    <div class="flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="p-2 bg-primary-100 rounded-xl">
          <Brain class="h-6 w-6 text-primary-600" />
        </div>
        <div>
          <h1 class="text-xl font-bold text-neutral-900">Research Agent</h1>
          <p class="text-sm text-neutral-500">Advanced Literature Research Platform</p>
        </div>
      </div>
      
      <!-- System Status -->
      <div class="flex items-center gap-4">
        <div class="flex items-center gap-2 px-3 py-1.5 bg-neutral-100 rounded-xl">
          <div class="status-dot {getStatusDotClass(systemHealth.status)}"></div>
          <span class="text-sm font-medium capitalize">{systemHealth.status}</span>
        </div>
        
        <button class="btn-ghost">
          <Settings class="h-4 w-4" />
        </button>
      </div>
    </div>
  </header>

  <!-- Layout Container -->
  <div class="flex pt-20 min-h-screen">
    <!-- Left Sidebar -->
    <aside class="w-80 bg-white border-r border-neutral-200/50 fixed left-0 top-20 h-[calc(100vh-5rem)] overflow-y-auto scrollbar-thin">
      <!-- Navigation -->
      <nav class="p-6 border-b border-neutral-200/50">
        <h3 class="text-sm font-semibold text-neutral-500 uppercase tracking-wider mb-4">Navigation</h3>
        <div class="flex flex-col gap-1">
          <button 
            class="sidebar-item {currentView === 'dashboard' ? 'sidebar-item-active' : ''}"
            on:click={() => currentView = 'dashboard'}
          >
            <BarChart3 class="h-4 w-4" />
            Dashboard
          </button>
          <button 
            class="sidebar-item {currentView === 'sessions' ? 'sidebar-item-active' : ''}"
            on:click={() => currentView = 'sessions'}
          >
            <History class="h-4 w-4" />
            Sessions
          </button>
          <button 
            class="sidebar-item {currentView === 'analytics' ? 'sidebar-item-active' : ''}"
            on:click={() => currentView = 'analytics'}
          >
            <TrendingUp class="h-4 w-4" />
            Analytics
          </button>
          <button 
            class="sidebar-item {currentView === 'chat' ? 'sidebar-item-active' : ''}"
            on:click={() => currentView = 'chat'}
          >
            <MessageCircle class="h-4 w-4" />
            Chat
          </button>
        </div>
      </nav>

      <!-- Recent Sessions in Sidebar -->
      <div class="p-6">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-sm font-semibold text-neutral-500 uppercase tracking-wider">Recent Sessions</h3>
          <button class="btn-ghost p-1" on:click={() => currentView = 'sessions'}>
            <Plus class="h-3 w-3" />
          </button>
        </div>
        
        <div class="space-y-3">
          {#each researchSessions.slice(0, 5) as session}
            <div class="card-hover p-3 cursor-pointer">
              <div class="flex items-start justify-between mb-2">
                <div class="flex items-center gap-2">
                  <BookOpen class="h-4 w-4 text-neutral-400 flex-shrink-0" />
                  <div class="status-dot {getStatusDotClass(session.status)}"></div>
                </div>
                <span class="text-xs text-neutral-400">{formatDate(session.created_at)}</span>
              </div>
              
              <h4 class="font-medium text-sm text-neutral-900 line-clamp-2 mb-2">{session.title}</h4>
              
              <div class="flex items-center justify-between text-xs text-neutral-500">
                <span>{session.papers_found} papers</span>
                <Badge variant={session.status === 'completed' ? 'default' : session.status === 'in_progress' ? 'secondary' : 'destructive'} class="text-xs px-2 py-1">
                  {session.status}
                </Badge>
              </div>
            </div>
          {/each}
        </div>
      </div>
    </aside>

    <!-- Main Content -->
    <div class="flex-1 ml-80 flex flex-col">

      <!-- Dashboard View -->
      {#if currentView === 'dashboard'}
        <div class="flex-1 p-8">
          <!-- Top Navigation (above content) -->
          <nav class="flex gap-2 mb-8 justify-center">
            <button 
              class="sidebar-item {currentView === 'dashboard' ? 'sidebar-item-active' : ''}"
              on:click={() => currentView = 'dashboard'}
            >
              <BarChart3 class="h-4 w-4" />
              Dashboard
            </button>
            <button 
              class="sidebar-item {currentView === 'sessions' ? 'sidebar-item-active' : ''}"
              on:click={() => currentView = 'sessions'}
            >
              <History class="h-4 w-4" />
              Sessions
            </button>
            <button 
              class="sidebar-item {currentView === 'analytics' ? 'sidebar-item-active' : ''}"
              on:click={() => currentView = 'analytics'}
            >
              <TrendingUp class="h-4 w-4" />
              Analytics
            </button>
            <button 
              class="sidebar-item {currentView === 'reports' ? 'sidebar-item-active' : ''}"
              on:click={() => currentView = 'reports'}
            >
              <FileText class="h-4 w-4" />
              Reports
            </button>
            <button 
              class="sidebar-item {currentView === 'chat' ? 'sidebar-item-active' : ''}"
              on:click={() => currentView = 'chat'}
            >
              <MessageCircle class="h-4 w-4" />
              Chat
            </button>
          </nav>

          <!-- Centered Research Input -->
          <div class="max-w-7xl mx-auto {$activeSession && ($activeSession.status === 'searching' || $activeSession.status === 'processing' || $activeSession.papers.length > 0) ? 'mb-8' : 'mt-16'}">
            <div class="text-center mb-8">
              <h2 class="text-3xl font-bold text-neutral-900 mb-3">Start New Research</h2>
              <p class="text-lg text-neutral-600">Enter a research topic to begin comprehensive literature analysis</p>
            </div>
            
            <Card.Root class="p-8 shadow-lg">
              <Card.Content>
                <div class="flex gap-4">
                  <div class="flex-1">
                    <Input 
                      bind:value={newResearchTopic}
                      placeholder="e.g., Machine learning applications in medical diagnosis"
                      onkeydown={(e) => e.key === 'Enter' && startNewResearch()}
                      class="h-14 text-lg"
                    />
                  </div>
                  <Button 
                    onclick={startNewResearch}
                    disabled={isLoading || !newResearchTopic.trim()}
                    class="h-14 px-8 text-lg bg-primary-600 hover:bg-primary-700 text-white font-semibold shadow-lg"
                  >
                    {#if isLoading}
                      <div class="inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                      Processing...
                    {:else}
                      <Play class="h-5 w-5 mr-2" />
                      Start Research
                    {/if}
                  </Button>
                </div>
              </Card.Content>
            </Card.Root>
          </div>

          <!-- Live Paper Discovery (show when research is active) -->
          {#if $activeSession && ($activeSession.status === 'searching' || $activeSession.status === 'processing' || $activeSession.papers.length > 0)}
            <div class="max-w-7xl mx-auto">
              <PaperDiscoveryFeed 
                papers={$activeSession.papers || []}
                sessionId={$activeSession.sessionId}
                isVisible={true}
                on:openPdf={handleOpenPdf}
                on:downloadPdf={handleDownloadPdf}
              />
            </div>
          {/if}

          <!-- Quick Analytics Preview (show when session is completed) -->
          {#if $activeSession && $activeSession.status === 'completed' && $activeSession.papers.length > 0}
            <div class="max-w-7xl mx-auto mb-8">
              <div class="section-header mb-4">
                <div>
                  <h3 class="section-title">Quick Analytics</h3>
                  <p class="section-subtitle">Summary insights from your completed research</p>
                </div>
                <button class="btn-secondary" on:click={() => currentView = 'analytics'}>
                  View Full Analytics
                </button>
              </div>
              
              <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Card.Root class="p-4">
                  <Card.Content class="p-0 text-center">
                    <div class="text-lg font-bold text-primary-600">{$activeSession.papers.length}</div>
                    <div class="text-sm text-neutral-500">Papers Found</div>
                    <div class="text-xs text-neutral-400 mt-1">Research corpus</div>
                  </Card.Content>
                </Card.Root>
                
                <Card.Root class="p-4">
                  <Card.Content class="p-0 text-center">
                    <div class="text-lg font-bold text-green-600">
                      {$activeSession.papers.reduce((sum, p) => sum + (p.citationCount || 0), 0)}
                    </div>
                    <div class="text-sm text-neutral-500">Total Citations</div>
                    <div class="text-xs text-neutral-400 mt-1">Impact metrics</div>
                  </Card.Content>
                </Card.Root>
                
                <Card.Root class="p-4">
                  <Card.Content class="p-0 text-center">
                    <div class="text-lg font-bold text-blue-600">
                      {new Set($activeSession.papers.flatMap(p => p.authors || [])).size}
                    </div>
                    <div class="text-sm text-neutral-500">Unique Authors</div>
                    <div class="text-xs text-neutral-400 mt-1">Research network</div>
                  </Card.Content>
                </Card.Root>
              </div>
            </div>
          {/if}
        </div>
      {/if}

      <!-- Sessions View -->
      {#if currentView === 'sessions'}
        <div class="p-8">
          <div class="section-header mb-6">
            <div>
              <h2 class="section-title">Research Sessions</h2>
              <p class="section-subtitle">Manage and explore your research history</p>
            </div>
            <button class="btn-primary">
              <Plus class="h-4 w-4" />
              New Session
            </button>
          </div>
      
          <Card.Root>
            <Table.Root>
              <Table.Header>
                <Table.Row>
                  <Table.Head>Research Topic</Table.Head>
                  <Table.Head>Status</Table.Head>
                  <Table.Head>Papers</Table.Head>
                  <Table.Head>Queries</Table.Head>
                  <Table.Head>Created</Table.Head>
                  <Table.Head>Actions</Table.Head>
                </Table.Row>
              </Table.Header>
              <Table.Body>
                {#each researchSessions as session}
                  <Table.Row>
                    <Table.Cell>
                      <div class="font-medium text-neutral-900">{session.title}</div>
                      <div class="text-sm text-neutral-500">ID: {session.id}</div>
                    </Table.Cell>
                    <Table.Cell>
                      <Badge variant={session.status === 'completed' ? 'default' : session.status === 'in_progress' ? 'secondary' : 'destructive'}>
                        {session.status}
                      </Badge>
                    </Table.Cell>
                    <Table.Cell class="text-neutral-600">{session.papers_found}</Table.Cell>
                    <Table.Cell class="text-neutral-600">{session.queries_processed}</Table.Cell>
                    <Table.Cell class="text-neutral-600">{formatDate(session.created_at)}</Table.Cell>
                    <Table.Cell>
                      <div class="flex gap-2">
                        <Button variant="ghost" size="sm">View</Button>
                        <Button variant="ghost" size="sm">Export</Button>
                        <Button variant="ghost" size="sm">Chat</Button>
                      </div>
                    </Table.Cell>
                  </Table.Row>
                {/each}
              </Table.Body>
            </Table.Root>
          </Card.Root>
        </div>
      {/if}

      <!-- Analytics View -->
      {#if currentView === 'analytics'}
        <div class="p-8">
          <div class="section-header mb-6">
            <div>
              <h2 class="section-title">Research Analytics</h2>
              <p class="section-subtitle">Visual insights and data analysis from your research findings</p>
            </div>
            
            {#if $activeSession && $activeSession.papers.length > 0}
              <div class="text-sm text-neutral-600">
                Analyzing: <span class="font-medium text-primary-600">{$activeSession.papers.length} paper{$activeSession.papers.length !== 1 ? 's' : ''}</span>
              </div>
            {/if}
          </div>
          
          <ResearchVisualization 
            papers={$activeSession?.papers || []}
            isVisible={true}
            title="Research Data Visualization"
          />
        </div>
      {/if}

      <!-- Reports View -->
      {#if currentView === 'reports'}
        <div class="p-8">
          <div class="section-header mb-6">
            <div>
              <h2 class="section-title">Research Reports</h2>
              <p class="section-subtitle">Generate comprehensive reports from your research findings</p>
            </div>
            
            {#if $activeSession && $activeSession.papers.length > 0}
              <div class="text-sm text-neutral-600">
                Ready to generate report from: <span class="font-medium text-primary-600">{$activeSession.papers.length} paper{$activeSession.papers.length !== 1 ? 's' : ''}</span>
              </div>
            {/if}
          </div>
          
          <ReportGenerator 
            papers={$activeSession?.papers || []}
            sessionId={$activeSession?.sessionId || null}
            researchTopic={$activeSession?.topic || ''}
            isVisible={true}
            on:generateReport={handleGenerateReport}
            on:previewReport={handlePreviewReport}
            on:exportReport={handleExportReport}
          />
        </div>
      {/if}

      <!-- Chat View -->
      {#if currentView === 'chat'}
        <div class="p-8">
          <div class="section-header mb-6">
            <div>
              <h2 class="section-title">Interactive Chat</h2>
              <p class="section-subtitle">Ask questions about your research findings</p>
            </div>
            
            {#if $activeSession}
              <div class="text-sm text-neutral-600">
                Connected to: <span class="font-medium text-primary-600">{$activeSession.topic}</span>
              </div>
            {/if}
          </div>
          
          <div class="card p-0 h-[600px]">
            <ChatInterface 
              sessionId={$activeSession?.sessionId || null}
              researchSessionId={$activeSession?.sessionId || null}
              isVisible={true}
              placeholder="Ask questions about your research findings..."
              on:sendMessage={handleChatMessage}
              on:newSession={handleNewChatSession}
              on:clearSession={handleClearChatSession}
            />
          </div>
        </div>
      {/if}
    </div>
  </div>

  <!-- Footer with Metrics -->
  <footer class="bg-white border-t border-neutral-200/50 p-6 mt-auto">
    <div class="max-w-7xl mx-auto">
      <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div class="text-center">
          <div class="text-2xl font-bold text-neutral-900">{mockMetrics.totalPapers}</div>
          <div class="text-xs text-neutral-500 uppercase tracking-wider mt-1">Papers Analyzed</div>
          <div class="text-sm font-medium text-green-600 mt-1">+12 this week</div>
        </div>
        
        <div class="text-center">
          <div class="text-2xl font-bold text-neutral-900">{mockMetrics.totalSessions}</div>
          <div class="text-xs text-neutral-500 uppercase tracking-wider mt-1">Research Sessions</div>
          <div class="text-sm font-medium text-green-600 mt-1">+3 this week</div>
        </div>
        
        <div class="text-center">
          <div class="text-2xl font-bold text-neutral-900">{mockMetrics.avgProcessingTime}</div>
          <div class="text-xs text-neutral-500 uppercase tracking-wider mt-1">Avg Processing Time</div>
          <div class="text-sm font-medium text-green-600 mt-1">-0.8m improvement</div>
        </div>
        
        <div class="text-center">
          <div class="text-2xl font-bold text-neutral-900">{mockMetrics.successRate}</div>
          <div class="text-xs text-neutral-500 uppercase tracking-wider mt-1">Success Rate</div>
          <div class="text-sm font-medium text-green-600 mt-1">+2% this month</div>
        </div>
      </div>
    </div>
  </footer>
</main>

<!-- Progress Tracking Components -->
<!-- Progress Modal -->
<ProgressModal 
  isOpen={progressModalOpen}
  sessionId={$activeSession?.sessionId || null}
  topic={$activeSession?.topic || ''}
  on:close={handleProgressModalClose}
  on:minimize={handleProgressModalMinimize}
/>

<!-- Progress Widget -->
<ProgressWidget 
  isVisible={progressWidgetVisible && $activeSession}
  sessionId={$activeSession?.sessionId || null}
  topic={$activeSession?.topic || ''}
  progress={$activeSession?.overallProgress || 0}
  status={$activeSession?.status === 'completed' ? 'completed' : 
          $activeSession?.status === 'error' ? 'error' : 'running'}
  currentStep={$activeSession?.currentStep || ''}
  on:expand={handleProgressWidgetExpand}
  on:close={handleProgressWidgetClose}
/>

<!-- PDF Viewer -->
<PdfViewer 
  paper={selectedPaper}
  isOpen={isPdfViewerOpen}
  isFullscreen={isPdfViewerFullscreen}
  on:close={handleClosePdfViewer}
  on:toggleFullscreen={handleTogglePdfFullscreen}
  on:download={handlePdfDownload}
/>

<style>
  /* Custom component styles specific to App */
  .list-item:hover {
    transform: translateY(-1px);
  }
  
  /* Line clamp utility */
  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
  
  /* Smooth transitions for view changes */
  main {
    transition: all 0.2s ease-in-out;
  }

  /* Sidebar improvements */
  aside {
    transition: transform 0.2s ease-in-out;
  }

  /* Dynamic layout for research input */
  .research-input-centered {
    transition: margin-top 0.3s ease-in-out;
  }

  /* Enhanced button and input styling */
  :global(.btn-primary) {
    background: linear-gradient(135deg, var(--primary-600), var(--primary-700));
    border: none;
    box-shadow: 0 4px 14px 0 rgba(0, 118, 255, 0.39);
    transition: all 0.2s ease-in-out;
  }

  :global(.btn-primary:hover) {
    background: linear-gradient(135deg, var(--primary-700), var(--primary-800));
    box-shadow: 0 6px 20px rgba(0, 118, 255, 0.5);
    transform: translateY(-1px);
  }

  /* Research input card enhancements */
  :global(.research-card) {
    border: 2px solid var(--primary-200);
    background: linear-gradient(135deg, white, var(--primary-50));
    transition: all 0.3s ease-in-out;
  }

  :global(.research-card:hover) {
    border-color: var(--primary-300);
    box-shadow: 0 10px 25px rgba(0, 118, 255, 0.1);
  }
</style>
