import React, { useState, useEffect } from 'react';
import { AIPlayground, ChatConsole } from '@cogniui/components';
import { WebSocketClient } from '@cogniui/websocket';
import ComparisonResults from './ComparisonResults';

const ChainOfDraftExperiment = () => {
    const [ws, setWs] = useState(null);
    const [status, setStatus] = useState('idle');
    const [results, setResults] = useState(null);
    const [progress, setProgress] = useState({
        batch: 0,
        sequential: 0,
        mirror: 0
    });

    // Initialize WebSocket connection
    useEffect(() => {
        const socket = new WebSocketClient('ws://your-server/comparison');
        
        socket.onMessage = (data) => {
            switch (data.type) {
                case 'status':
                    setStatus(data.payload.status);
                    break;
                case 'progress':
                    setProgress(prev => ({
                        ...prev,
                        [data.payload.mode]: data.payload.progress
                    }));
                    break;
                case 'results':
                    setResults(data.payload);
                    setStatus('completed');
                    break;
                case 'error':
                    setStatus('error');
                    // Handle error display
                    break;
            }
        };

        setWs(socket);
        return () => socket.close();
    }, []);

    // Configuration for the experiment
    const config = {
        title: 'Chain of Draft Comparison',
        description: 'Compare different Chain of Draft approaches',
        parameters: {
            problem: {
                type: 'text',
                label: 'Problem to Solve',
                placeholder: 'Enter your problem here...',
                required: true
            },
            model: {
                type: 'select',
                label: 'Model',
                options: [
                    { value: 'openai/gpt-4o-mini', label: 'GPT-4 Mini' },
                    { value: 'anthropic/claude-3-sonnet', label: 'Claude 3 Sonnet' }
                ],
                default: 'openai/gpt-4o-mini'
            }
        }
    };

    // Handle experiment submission
    const handleSubmit = (params) => {
        if (!ws) return;
        
        setStatus('running');
        setProgress({ batch: 0, sequential: 0, mirror: 0 });
        setResults(null);
        
        ws.send({
            type: 'run_comparison',
            payload: params
        });
    };

    return (
        <div className="chain-of-draft-experiment">
            <AIPlayground
                config={config}
                onSubmit={handleSubmit}
                status={status}
            />
            
            {status === 'running' && (
                <div className="progress-indicators">
                    <h3>Progress</h3>
                    <div className="progress-bars">
                        <div className="progress-item">
                            <label>Batch Mode</label>
                            <progress value={progress.batch} max="100" />
                            <span>{progress.batch}%</span>
                        </div>
                        <div className="progress-item">
                            <label>Sequential Mode</label>
                            <progress value={progress.sequential} max="100" />
                            <span>{progress.sequential}%</span>
                        </div>
                        <div className="progress-item">
                            <label>Mirror Mode</label>
                            <progress value={progress.mirror} max="100" />
                            <span>{progress.mirror}%</span>
                        </div>
                    </div>
                </div>
            )}
            
            {results && (
                <ComparisonResults results={results} />
            )}
            
            <style jsx>{`
                .chain-of-draft-experiment {
                    padding: 20px;
                    max-width: 1200px;
                    margin: 0 auto;
                }
                
                .progress-indicators {
                    margin: 20px 0;
                    padding: 20px;
                    background: var(--cogni-surface-bg);
                    border-radius: 8px;
                }
                
                .progress-bars {
                    display: flex;
                    flex-direction: column;
                    gap: 15px;
                }
                
                .progress-item {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
                
                .progress-item label {
                    width: 120px;
                }
                
                .progress-item progress {
                    flex: 1;
                    height: 20px;
                }
                
                .progress-item span {
                    width: 50px;
                    text-align: right;
                }
            `}</style>
        </div>
    );
};

export default ChainOfDraftExperiment; 