import { progressStore } from '../stores/progress';

/**
 * Demo function to simulate real-time progress updates
 * This would be replaced by actual WebSocket messages from the backend
 */
export function startProgressDemo(sessionId: string, topic: string) {
  // Initialize with demo steps
  const steps = [
    { id: 'init', name: 'Initializing Research Session', status: 'running' as const },
    { id: 'search', name: 'Searching Literature Sources', status: 'pending' as const },
    { id: 'extract', name: 'Extracting Papers', status: 'pending' as const },
    { id: 'process', name: 'Processing with 3-Tier RAG', status: 'pending' as const },
    { id: 'analyze', name: 'Analyzing Results', status: 'pending' as const },
    { id: 'synthesize', name: 'Synthesizing Final Report', status: 'pending' as const }
  ];

  // Add initial steps
  steps.forEach(step => progressStore.addStep(sessionId, step));

  // Simulate progress updates
  const progressSequence = [
    { delay: 1000, stepId: 'init', progress: 50, message: 'Connecting to literature sources...' },
    { delay: 2000, stepId: 'init', progress: 100, status: 'completed' as const },
    { delay: 2500, stepId: 'search', status: 'running' as const, progress: 0, message: 'Searching ArXiv, PubMed, and Sci-Hub...' },
    { delay: 4000, stepId: 'search', progress: 30, papersFound: 15, message: 'Found 15 relevant papers...' },
    { delay: 6000, stepId: 'search', progress: 70, papersFound: 32, message: 'Found 32 relevant papers...' },
    { delay: 8000, stepId: 'search', progress: 100, status: 'completed' as const, papersFound: 45 },
    { delay: 8500, stepId: 'extract', status: 'running' as const, progress: 0, message: 'Extracting full text and metadata...' },
    { delay: 10000, stepId: 'extract', progress: 60, message: 'Processing paper contents...' },
    { delay: 12000, stepId: 'extract', progress: 100, status: 'completed' as const },
    { delay: 12500, stepId: 'process', status: 'running' as const, progress: 0, message: 'Running LightRAG analysis...' },
    { delay: 15000, stepId: 'process', progress: 30, message: 'Processing with PaperQA2...' },
    { delay: 18000, stepId: 'process', progress: 70, message: 'Running GraphRAG synthesis...' },
    { delay: 20000, stepId: 'process', progress: 100, status: 'completed' as const, queriesProcessed: 12 },
    { delay: 20500, stepId: 'analyze', status: 'running' as const, progress: 0, message: 'Analyzing research gaps...' },
    { delay: 22000, stepId: 'analyze', progress: 80, message: 'Identifying key insights...' },
    { delay: 23000, stepId: 'analyze', progress: 100, status: 'completed' as const },
    { delay: 23500, stepId: 'synthesize', status: 'running' as const, progress: 0, message: 'Generating final report...' },
    { delay: 25000, stepId: 'synthesize', progress: 50, message: 'Compiling citations and references...' },
    { delay: 26500, stepId: 'synthesize', progress: 100, status: 'completed' as const },
    { delay: 27000, complete: true }
  ];

  // Execute the progress sequence
  progressSequence.forEach(update => {
    setTimeout(() => {
      if (update.complete) {
        progressStore.completeSession(sessionId);
        progressStore.updateSession(sessionId, {
          overallProgress: 100,
          currentStep: 'Research completed successfully!',
          papersFound: 45,
          queriesProcessed: 12,
          estimatedCompletion: new Date()
        });
      } else {
        // Update step
        progressStore.updateStep(sessionId, update.stepId, {
          status: update.status || 'running',
          progress: update.progress,
          message: update.message
        });

        // Update overall session progress
        const overallProgress = Math.min(95, (update.delay / 27000) * 100);
        const sessionUpdates: any = {
          overallProgress,
          currentStep: update.message || `Processing ${update.stepId}...`
        };

        if (update.papersFound) sessionUpdates.papersFound = update.papersFound;
        if (update.queriesProcessed) sessionUpdates.queriesProcessed = update.queriesProcessed;

        // Calculate estimated completion
        if (overallProgress < 100) {
          const remainingTime = ((100 - overallProgress) / 100) * (27000 - update.delay);
          sessionUpdates.estimatedCompletion = new Date(Date.now() + remainingTime);
        }

        progressStore.updateSession(sessionId, sessionUpdates);
      }
    }, update.delay);
  });
}

/**
 * Create a shorter demo for quick testing
 */
export function startQuickProgressDemo(sessionId: string, topic: string) {
  const steps = [
    { id: 'search', name: 'Searching Papers', status: 'running' as const },
    { id: 'process', name: 'Processing Results', status: 'pending' as const }
  ];

  steps.forEach(step => progressStore.addStep(sessionId, step));

  // Quick 10-second demo
  setTimeout(() => {
    progressStore.updateStep(sessionId, 'search', { progress: 100, status: 'completed' });
    progressStore.updateStep(sessionId, 'process', { status: 'running', progress: 0 });
    progressStore.updateSession(sessionId, { overallProgress: 50, currentStep: 'Processing results...' });
  }, 2000);

  setTimeout(() => {
    progressStore.updateStep(sessionId, 'process', { progress: 100, status: 'completed' });
    progressStore.completeSession(sessionId);
    progressStore.updateSession(sessionId, {
      overallProgress: 100,
      currentStep: 'Completed!',
      papersFound: 25,
      queriesProcessed: 8
    });
  }, 5000);
}

/**
 * Simulate an error scenario for testing
 */
export function startErrorProgressDemo(sessionId: string, topic: string) {
  const steps = [
    { id: 'init', name: 'Initializing', status: 'running' as const },
    { id: 'search', name: 'Searching Papers', status: 'pending' as const }
  ];

  steps.forEach(step => progressStore.addStep(sessionId, step));

  setTimeout(() => {
    progressStore.updateStep(sessionId, 'init', { progress: 100, status: 'completed' });
    progressStore.updateStep(sessionId, 'search', { status: 'running', progress: 30 });
    progressStore.updateSession(sessionId, { 
      overallProgress: 40, 
      currentStep: 'Searching literature sources...' 
    });
  }, 1000);

  setTimeout(() => {
    progressStore.updateStep(sessionId, 'search', { 
      status: 'error', 
      message: 'Connection to Sci-Hub failed' 
    });
    progressStore.errorSession(sessionId, 'Failed to connect to literature sources');
  }, 3000);
}