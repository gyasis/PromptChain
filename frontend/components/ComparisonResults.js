import React, { useState } from 'react';
import { Tabs, Tab, Table } from '@cogniui/components';

const ComparisonResults = ({ results }) => {
    const [activeTab, setActiveTab] = useState('metrics');
    
    const formatTokens = (num) => num.toLocaleString();
    const formatTime = (ms) => `${(ms / 1000).toFixed(2)}s`;
    const formatCost = (usd) => `$${usd.toFixed(4)}`;
    const formatPercent = (num) => `${num.toFixed(1)}%`;

    const renderMetricsTable = () => (
        <Table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Batch</th>
                    <th>Sequential</th>
                    <th>Mirror</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Total Tokens</td>
                    <td>{formatTokens(results.batch.total_tokens)}</td>
                    <td>{formatTokens(results.sequential.total_tokens)}</td>
                    <td>{formatTokens(results.mirror.total_tokens)}</td>
                </tr>
                <tr>
                    <td>Prompt Tokens</td>
                    <td>{formatTokens(results.batch.prompt_tokens)}</td>
                    <td>{formatTokens(results.sequential.prompt_tokens)}</td>
                    <td>{formatTokens(results.mirror.prompt_tokens)}</td>
                </tr>
                <tr>
                    <td>Response Tokens</td>
                    <td>{formatTokens(results.batch.response_tokens)}</td>
                    <td>{formatTokens(results.sequential.response_tokens)}</td>
                    <td>{formatTokens(results.mirror.response_tokens)}</td>
                </tr>
                <tr>
                    <td>Token Efficiency</td>
                    <td>{formatPercent(results.batch.token_efficiency)}</td>
                    <td>{formatPercent(results.sequential.token_efficiency)}</td>
                    <td>{formatPercent(results.mirror.token_efficiency)}</td>
                </tr>
                <tr>
                    <td>Execution Time</td>
                    <td>{formatTime(results.batch.execution_time)}</td>
                    <td>{formatTime(results.sequential.execution_time)}</td>
                    <td>{formatTime(results.mirror.execution_time)}</td>
                </tr>
                <tr>
                    <td>Estimated Cost</td>
                    <td>{formatCost(results.batch.estimated_cost)}</td>
                    <td>{formatCost(results.sequential.estimated_cost)}</td>
                    <td>{formatCost(results.mirror.estimated_cost)}</td>
                </tr>
            </tbody>
        </Table>
    );

    const renderDraftsComparison = () => (
        <div className="drafts-comparison">
            <div className="draft-column">
                <h4>Batch Drafts</h4>
                {results.batch.drafts.map((draft, i) => (
                    <div key={i} className="draft-card">
                        <div className="draft-header">Draft {i + 1}</div>
                        <pre>{draft}</pre>
                    </div>
                ))}
            </div>
            <div className="draft-column">
                <h4>Sequential Drafts</h4>
                {results.sequential.drafts.map((draft, i) => (
                    <div key={i} className="draft-card">
                        <div className="draft-header">Draft {i + 1}</div>
                        <pre>{draft}</pre>
                    </div>
                ))}
            </div>
            <div className="draft-column">
                <h4>Mirror Drafts</h4>
                {results.mirror.drafts.map((draft, i) => (
                    <div key={i} className="draft-card">
                        <div className="draft-header">Draft {i + 1}</div>
                        <pre>{draft}</pre>
                    </div>
                ))}
            </div>
        </div>
    );

    const renderFinalAnswers = () => (
        <div className="final-answers">
            <div className="answer-card">
                <h4>Batch Final Answer</h4>
                <pre>{results.batch.final_answer}</pre>
            </div>
            <div className="answer-card">
                <h4>Sequential Final Answer</h4>
                <pre>{results.sequential.final_answer}</pre>
            </div>
            <div className="answer-card">
                <h4>Mirror Final Answer</h4>
                <pre>{results.mirror.final_answer}</pre>
            </div>
        </div>
    );

    return (
        <div className="comparison-results">
            <h2>Comparison Results</h2>
            
            <Tabs activeTab={activeTab} onChange={setActiveTab}>
                <Tab id="metrics" label="Metrics & Costs">
                    {renderMetricsTable()}
                </Tab>
                <Tab id="drafts" label="Draft Comparison">
                    {renderDraftsComparison()}
                </Tab>
                <Tab id="answers" label="Final Answers">
                    {renderFinalAnswers()}
                </Tab>
            </Tabs>

            <style jsx>{`
                .comparison-results {
                    margin-top: 30px;
                    padding: 20px;
                    background: var(--cogni-surface-bg);
                    border-radius: 8px;
                }

                .drafts-comparison {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 20px;
                    margin-top: 20px;
                }

                .draft-column {
                    display: flex;
                    flex-direction: column;
                    gap: 15px;
                }

                .draft-card, .answer-card {
                    background: var(--cogni-surface-bg-alt);
                    border-radius: 6px;
                    overflow: hidden;
                }

                .draft-header {
                    padding: 8px 12px;
                    background: var(--cogni-primary);
                    color: white;
                    font-weight: 500;
                }

                pre {
                    padding: 12px;
                    margin: 0;
                    white-space: pre-wrap;
                    word-break: break-word;
                    font-size: 14px;
                    line-height: 1.5;
                    background: var(--cogni-surface-bg-alt);
                }

                .final-answers {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 20px;
                    margin-top: 20px;
                }

                h4 {
                    margin: 0;
                    padding: 8px 12px;
                    background: var(--cogni-primary);
                    color: white;
                }
            `}</style>
        </div>
    );
};

export default ComparisonResults; 