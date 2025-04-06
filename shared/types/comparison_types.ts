// Result structure for a single mode (batch/sequential/mirror)
export interface ModeResult {
    total_tokens: number;
    prompt_tokens: number;
    response_tokens: number;
    token_efficiency: number;
    execution_time: number;
    estimated_cost: number;
    drafts: string[];
    final_answer: string;
}

// Complete comparison results
export interface ComparisonResults {
    batch: ModeResult;
    sequential: ModeResult;
    mirror: ModeResult;
    total_execution_time: number;
}

// WebSocket message types
export type MessageType = 
    | 'status'
    | 'progress'
    | 'results'
    | 'error';

// Status message payload
export interface StatusPayload {
    status: 'idle' | 'connected' | 'starting' | 'running' | 'completed' | 'error';
}

// Progress message payload
export interface ProgressPayload {
    mode: 'batch' | 'sequential' | 'mirror';
    progress: number;  // 0-100
}

// Error message payload
export interface ErrorPayload {
    message: string;
}

// Base WebSocket message structure
export interface WebSocketMessage {
    type: MessageType;
    payload: StatusPayload | ProgressPayload | ComparisonResults | ErrorPayload;
}

// Parameters for running a comparison
export interface ComparisonParameters {
    problem: string;
    model: string;
}

// Request message for running a comparison
export interface RunComparisonMessage {
    type: 'run_comparison';
    payload: ComparisonParameters;
} 