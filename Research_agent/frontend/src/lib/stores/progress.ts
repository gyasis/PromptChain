import { writable, derived, get } from 'svelte/store';

export interface ProgressStep {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  progress?: number;
  message?: string;
  startTime?: Date;
  endTime?: Date;
  substeps?: ProgressStep[];
}

export interface Paper {
  id: string;
  title: string;
  authors: string[];
  abstract: string;
  source: 'arxiv' | 'pubmed' | 'scihub';
  url?: string;
  pdfUrl?: string;
  publishedDate?: string;
  citationCount?: number;
  tags: string[];
  status: 'found' | 'downloading' | 'ready' | 'error';
}

export interface ProgressSession {
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
  papers: Paper[];
}

export interface ProgressState {
  sessions: Map<string, ProgressSession>;
  activeSessionId: string | null;
  modalOpen: boolean;
  widgetVisible: boolean;
}

// Create the main progress store
function createProgressStore() {
  const { subscribe, set, update } = writable<ProgressState>({
    sessions: new Map(),
    activeSessionId: null,
    modalOpen: false,
    widgetVisible: false
  });

  return {
    subscribe,
    
    // Session management
    startSession: (sessionId: string, topic: string) => {
      const newSession: ProgressSession = {
        sessionId,
        topic,
        status: 'initializing',
        overallProgress: 0,
        currentStep: 'Initializing research session...',
        steps: [],
        startTime: new Date(),
        papersFound: 0,
        queriesProcessed: 0,
        papers: []
      };

      update(state => ({
        ...state,
        sessions: new Map(state.sessions).set(sessionId, newSession),
        activeSessionId: sessionId,
        widgetVisible: true
      }));
    },

    updateSession: (sessionId: string, updates: Partial<ProgressSession>) => {
      update(state => {
        const sessions = new Map(state.sessions);
        const existingSession = sessions.get(sessionId);
        if (existingSession) {
          sessions.set(sessionId, { ...existingSession, ...updates });
        }
        return { ...state, sessions };
      });
    },

    updateStep: (sessionId: string, stepId: string, updates: Partial<ProgressStep>) => {
      update(state => {
        const sessions = new Map(state.sessions);
        const session = sessions.get(sessionId);
        if (session) {
          const updatedSteps = session.steps.map(step => 
            step.id === stepId ? { ...step, ...updates } : step
          );
          sessions.set(sessionId, { ...session, steps: updatedSteps });
        }
        return { ...state, sessions };
      });
    },

    addStep: (sessionId: string, step: ProgressStep) => {
      update(state => {
        const sessions = new Map(state.sessions);
        const session = sessions.get(sessionId);
        if (session) {
          sessions.set(sessionId, {
            ...session,
            steps: [...session.steps, step]
          });
        }
        return { ...state, sessions };
      });
    },

    completeSession: (sessionId: string) => {
      update(state => {
        const sessions = new Map(state.sessions);
        const session = sessions.get(sessionId);
        if (session) {
          sessions.set(sessionId, {
            ...session,
            status: 'completed',
            overallProgress: 100,
            currentStep: 'Research completed successfully!'
          });
        }
        return { ...state, sessions };
      });
    },

    errorSession: (sessionId: string, error: string) => {
      update(state => {
        const sessions = new Map(state.sessions);
        const session = sessions.get(sessionId);
        if (session) {
          sessions.set(sessionId, {
            ...session,
            status: 'error',
            currentStep: `Error: ${error}`
          });
        }
        return { ...state, sessions };
      });
    },

    // UI state management
    openModal: () => update(state => ({ ...state, modalOpen: true })),
    closeModal: () => update(state => ({ ...state, modalOpen: false })),
    showWidget: () => update(state => ({ ...state, widgetVisible: true })),
    hideWidget: () => update(state => ({ ...state, widgetVisible: false })),
    
    minimizeModal: () => update(state => ({ 
      ...state, 
      modalOpen: false, 
      widgetVisible: true 
    })),

    setActiveSession: (sessionId: string | null) => {
      update(state => ({ ...state, activeSessionId: sessionId }));
    },

    // Clear completed sessions (cleanup)
    clearCompletedSessions: () => {
      update(state => {
        const sessions = new Map();
        for (const [id, session] of state.sessions) {
          if (session.status !== 'completed') {
            sessions.set(id, session);
          }
        }
        return { ...state, sessions };
      });
    },

    // Paper management
    addPaper: (sessionId: string, paper: Paper) => {
      update(state => {
        const sessions = new Map(state.sessions);
        const session = sessions.get(sessionId);
        if (session) {
          const updatedPapers = [...session.papers, paper];
          sessions.set(sessionId, {
            ...session,
            papers: updatedPapers,
            papersFound: updatedPapers.length
          });
        }
        return { ...state, sessions };
      });
    },

    updatePaper: (sessionId: string, paperId: string, updates: Partial<Paper>) => {
      update(state => {
        const sessions = new Map(state.sessions);
        const session = sessions.get(sessionId);
        if (session) {
          const updatedPapers = session.papers.map(paper =>
            paper.id === paperId ? { ...paper, ...updates } : paper
          );
          sessions.set(sessionId, { ...session, papers: updatedPapers });
        }
        return { ...state, sessions };
      });
    },

    getPapersForSession: (sessionId: string): Paper[] => {
      const state = get(progressStore);
      const session = state.sessions.get(sessionId);
      return session?.papers || [];
    },

    // Reset all progress
    reset: () => set({
      sessions: new Map(),
      activeSessionId: null,
      modalOpen: false,
      widgetVisible: false
    })
  };
}

export const progressStore = createProgressStore();

// Derived stores for easier access
export const activeSession = derived(
  progressStore,
  $progress => $progress.activeSessionId 
    ? $progress.sessions.get($progress.activeSessionId) 
    : null
);

export const hasActiveSessions = derived(
  progressStore,
  $progress => Array.from($progress.sessions.values()).some(
    session => !['completed', 'error'].includes(session.status)
  )
);

export const completedSessionsCount = derived(
  progressStore,
  $progress => Array.from($progress.sessions.values()).filter(
    session => session.status === 'completed'
  ).length
);

// WebSocket management
export class ProgressWebSocketManager {
  private websockets: Map<string, WebSocket> = new Map();
  private reconnectAttempts: Map<string, number> = new Map();
  private maxReconnectAttempts = 5;

  connect(sessionId: string) {
    if (this.websockets.has(sessionId)) {
      return; // Already connected
    }

    const wsUrl = `ws://localhost:8078/ws/progress/${sessionId}`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log(`Progress WebSocket connected for session ${sessionId}`);
      this.reconnectAttempts.delete(sessionId);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'progress_update') {
          progressStore.updateSession(sessionId, data.session);
        } else if (data.type === 'step_update') {
          progressStore.updateStep(sessionId, data.stepId, data.step);
        } else if (data.type === 'step_added') {
          progressStore.addStep(sessionId, data.step);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    ws.onclose = () => {
      console.log(`Progress WebSocket disconnected for session ${sessionId}`);
      this.websockets.delete(sessionId);
      this.attemptReconnect(sessionId);
    };

    ws.onerror = (error) => {
      console.error(`WebSocket error for session ${sessionId}:`, error);
    };

    this.websockets.set(sessionId, ws);
  }

  disconnect(sessionId: string) {
    const ws = this.websockets.get(sessionId);
    if (ws) {
      ws.close();
      this.websockets.delete(sessionId);
      this.reconnectAttempts.delete(sessionId);
    }
  }

  disconnectAll() {
    for (const [sessionId] of this.websockets) {
      this.disconnect(sessionId);
    }
  }

  private attemptReconnect(sessionId: string) {
    const attempts = this.reconnectAttempts.get(sessionId) || 0;
    if (attempts < this.maxReconnectAttempts) {
      const delay = Math.pow(2, attempts) * 1000; // Exponential backoff
      setTimeout(() => {
        this.reconnectAttempts.set(sessionId, attempts + 1);
        this.connect(sessionId);
      }, delay);
    }
  }
}

export const progressWebSocketManager = new ProgressWebSocketManager();