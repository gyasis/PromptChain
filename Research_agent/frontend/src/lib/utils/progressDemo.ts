import { progressStore, type Paper } from '../stores/progress';

// Sample papers for demonstration
const samplePapers: Paper[] = [
  {
    id: 'paper-1',
    title: 'Deep Learning Approaches for Medical Image Analysis: A Comprehensive Review',
    authors: ['Smith, J.A.', 'Johnson, M.B.', 'Williams, K.C.'],
    abstract: 'This paper provides a comprehensive review of deep learning methodologies applied to medical image analysis. We examine convolutional neural networks, transfer learning approaches, and recent advances in transformer-based architectures for medical imaging tasks including diagnosis, segmentation, and treatment planning.',
    source: 'arxiv' as const,
    url: 'https://arxiv.org/abs/2301.12345',
    pdfUrl: 'https://arxiv.org/pdf/2301.12345.pdf',
    publishedDate: '2023-01-15',
    citationCount: 127,
    tags: ['deep learning', 'medical imaging', 'CNN', 'computer vision'],
    status: 'found' as const
  },
  {
    id: 'paper-2', 
    title: 'Transformer Networks in Healthcare: Applications and Challenges',
    authors: ['Chen, L.', 'Kumar, R.', 'Anderson, P.', 'Martinez, S.'],
    abstract: 'We investigate the application of transformer architectures in healthcare domains, focusing on electronic health records, medical text processing, and clinical decision support systems. Our analysis covers recent breakthroughs and identifies key challenges for practical deployment.',
    source: 'pubmed' as const,
    url: 'https://pubmed.ncbi.nlm.nih.gov/36789012/',
    publishedDate: '2023-03-22',
    citationCount: 89,
    tags: ['transformers', 'healthcare', 'NLP', 'clinical decision support'],
    status: 'found' as const
  },
  {
    id: 'paper-3',
    title: 'Federated Learning for Privacy-Preserving Medical AI: A Systematic Study',
    authors: ['Zhang, Y.', 'Brown, T.M.', 'Davis, R.J.'],
    abstract: 'This systematic study examines federated learning approaches for developing medical AI systems while preserving patient privacy. We analyze communication efficiency, model performance, and privacy guarantees across different medical domains and datasets.',
    source: 'scihub' as const,
    url: 'https://sci-hub.se/10.1038/s41591-023-02456-7',
    pdfUrl: 'https://sci-hub.se/downloads/2023/03/28/zhang_federated_learning_medical_ai.pdf',
    publishedDate: '2023-02-08',
    citationCount: 156,
    tags: ['federated learning', 'privacy', 'medical AI', 'distributed systems'],
    status: 'found' as const
  },
  {
    id: 'paper-4',
    title: 'Explainable AI in Medical Diagnosis: Techniques and Applications',
    authors: ['Thompson, A.K.', 'Lee, S.H.', 'Wilson, M.D.', 'Garcia, C.R.', 'Patel, N.'],
    abstract: 'We present a comprehensive analysis of explainable AI techniques applied to medical diagnosis systems. The paper covers attention mechanisms, gradient-based explanations, and rule-based interpretability methods with case studies in radiology, pathology, and clinical decision making.',
    source: 'arxiv' as const,
    url: 'https://arxiv.org/abs/2302.34567',
    pdfUrl: 'https://arxiv.org/pdf/2302.34567.pdf',
    publishedDate: '2023-04-10',
    citationCount: 203,
    tags: ['explainable AI', 'medical diagnosis', 'interpretability', 'attention mechanisms'],
    status: 'found' as const
  },
  {
    id: 'paper-5',
    title: 'Multi-Modal Learning for Clinical Prediction: Combining Images, Text, and Structured Data',
    authors: ['Rodriguez, M.A.', 'Kim, J.S.', 'Taylor, B.L.'],
    abstract: 'This work explores multi-modal learning approaches that combine medical images, clinical notes, and structured electronic health record data for improved clinical prediction tasks. We demonstrate significant performance gains across multiple prediction scenarios.',
    source: 'pubmed' as const,
    url: 'https://pubmed.ncbi.nlm.nih.gov/37123456/',
    publishedDate: '2023-01-30',
    citationCount: 94,
    tags: ['multi-modal learning', 'clinical prediction', 'EHR', 'medical images'],
    status: 'found' as const
  }
];

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

  // Simulate progress updates with paper discovery
  const progressSequence = [
    { delay: 1000, stepId: 'init', progress: 50, message: 'Connecting to literature sources...' },
    { delay: 2000, stepId: 'init', progress: 100, status: 'completed' as const },
    { delay: 2500, stepId: 'search', status: 'running' as const, progress: 0, message: 'Searching ArXiv, PubMed, and Sci-Hub...' },
    
    // Add papers during search phase
    { delay: 3500, stepId: 'search', progress: 20, papersFound: 1, message: 'Found first paper on ArXiv...', addPaper: 0 },
    { delay: 4500, stepId: 'search', progress: 35, papersFound: 2, message: 'Found paper on PubMed...', addPaper: 1 },
    { delay: 5500, stepId: 'search', progress: 50, papersFound: 3, message: 'Found paper via Sci-Hub...', addPaper: 2 },
    { delay: 6500, stepId: 'search', progress: 70, papersFound: 4, message: 'Found additional paper...', addPaper: 3 },
    { delay: 7500, stepId: 'search', progress: 85, papersFound: 5, message: 'Found final paper...', addPaper: 4 },
    { delay: 8000, stepId: 'search', progress: 100, status: 'completed' as const, papersFound: 5 },
    
    { delay: 8500, stepId: 'extract', status: 'running' as const, progress: 0, message: 'Extracting full text and metadata...' },
    // Update paper status during extraction
    { delay: 9000, stepId: 'extract', progress: 20, message: 'Processing paper contents...', updatePaper: { index: 0, status: 'downloading' as const } },
    { delay: 9500, stepId: 'extract', progress: 40, message: 'Processing paper contents...', updatePaper: { index: 0, status: 'ready' as const }, updatePaper2: { index: 1, status: 'downloading' as const } },
    { delay: 10500, stepId: 'extract', progress: 60, message: 'Processing paper contents...', updatePaper: { index: 1, status: 'ready' as const }, updatePaper2: { index: 2, status: 'downloading' as const } },
    { delay: 11000, stepId: 'extract', progress: 80, message: 'Processing paper contents...', updatePaper: { index: 2, status: 'ready' as const }, updatePaper2: { index: 3, status: 'downloading' as const } },
    { delay: 11500, stepId: 'extract', progress: 90, message: 'Processing paper contents...', updatePaper: { index: 3, status: 'ready' as const }, updatePaper2: { index: 4, status: 'downloading' as const } },
    { delay: 12000, stepId: 'extract', progress: 100, status: 'completed' as const, updatePaper: { index: 4, status: 'ready' as const } },
    
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
  progressSequence.forEach((update: any) => {
    setTimeout(() => {
      if (update.complete) {
        progressStore.completeSession(sessionId);
        progressStore.updateSession(sessionId, {
          overallProgress: 100,
          currentStep: 'Research completed successfully!',
          papersFound: 5,
          queriesProcessed: 12,
          estimatedCompletion: new Date()
        });
      } else {
        // Add new papers
        if (update.addPaper !== undefined && samplePapers[update.addPaper]) {
          progressStore.addPaper(sessionId, { ...samplePapers[update.addPaper] });
        }

        // Update paper status
        if (update.updatePaper) {
          const paper = samplePapers[update.updatePaper.index];
          if (paper) {
            progressStore.updatePaper(sessionId, paper.id, { status: update.updatePaper.status });
          }
        }

        // Update second paper status (for simultaneous updates)
        if (update.updatePaper2) {
          const paper = samplePapers[update.updatePaper2.index];
          if (paper) {
            progressStore.updatePaper(sessionId, paper.id, { status: update.updatePaper2.status });
          }
        }

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