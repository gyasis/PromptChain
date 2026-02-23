-- PromptChain CLI Database Schema
-- SQLite database for session and agent persistence

-- Schema version tracking table
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at REAL NOT NULL,
    description TEXT
);

-- Insert initial schema version
INSERT OR IGNORE INTO schema_version (version, applied_at, description)
VALUES (1, strftime('%s', 'now'), 'Initial CLI schema');

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    created_at REAL NOT NULL,
    last_accessed REAL NOT NULL,
    working_directory TEXT NOT NULL,
    active_agent TEXT,
    default_model TEXT NOT NULL DEFAULT 'openai/gpt-4.1-mini-2025-04-14',
    auto_save_enabled INTEGER NOT NULL DEFAULT 1,
    auto_save_interval INTEGER NOT NULL DEFAULT 120,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

-- Indexes for session queries
CREATE INDEX IF NOT EXISTS idx_sessions_name ON sessions(name);
CREATE INDEX IF NOT EXISTS idx_sessions_last_accessed ON sessions(last_accessed DESC);

-- Agents table
CREATE TABLE IF NOT EXISTS agents (
    session_id TEXT NOT NULL,
    name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    last_used REAL,
    usage_count INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (session_id, name),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Index for agent queries
CREATE INDEX IF NOT EXISTS idx_agents_session ON agents(session_id);
CREATE INDEX IF NOT EXISTS idx_agents_last_used ON agents(session_id, last_used DESC);

-- =============================================================================
-- V3 Schema: Multi-Agent Communication Tables
-- =============================================================================

-- Task queue for agent delegation (FR-006 to FR-010)
CREATE TABLE IF NOT EXISTS task_queue (
    task_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    description TEXT NOT NULL,
    source_agent TEXT NOT NULL,
    target_agent TEXT NOT NULL,
    priority TEXT NOT NULL DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high')),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed', 'failed')),
    context_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    started_at REAL,
    completed_at REAL,
    result_json TEXT,
    error_message TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Indexes for task queue queries
CREATE INDEX IF NOT EXISTS idx_task_queue_session ON task_queue(session_id);
CREATE INDEX IF NOT EXISTS idx_task_queue_status ON task_queue(session_id, status);
CREATE INDEX IF NOT EXISTS idx_task_queue_target ON task_queue(session_id, target_agent, status);
CREATE INDEX IF NOT EXISTS idx_task_queue_priority ON task_queue(session_id, priority DESC, created_at ASC);

-- Blackboard for shared data (FR-011 to FR-015)
CREATE TABLE IF NOT EXISTS blackboard (
    session_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    written_by TEXT NOT NULL,
    written_at REAL NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (session_id, key),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Indexes for blackboard queries
CREATE INDEX IF NOT EXISTS idx_blackboard_session ON blackboard(session_id);
CREATE INDEX IF NOT EXISTS idx_blackboard_writer ON blackboard(session_id, written_by);

-- Workflow state tracking (FR-021 to FR-025)
CREATE TABLE IF NOT EXISTS workflow_state (
    workflow_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    stage TEXT NOT NULL DEFAULT 'planning' CHECK (stage IN ('planning', 'execution', 'review', 'complete')),
    agents_involved_json TEXT NOT NULL DEFAULT '[]',
    completed_tasks_json TEXT NOT NULL DEFAULT '[]',
    current_task TEXT,
    context_json TEXT NOT NULL DEFAULT '{}',
    started_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Indexes for workflow queries
CREATE INDEX IF NOT EXISTS idx_workflow_session ON workflow_state(session_id);
CREATE INDEX IF NOT EXISTS idx_workflow_stage ON workflow_state(session_id, stage);

-- Message log for agent communication (FR-016 to FR-020)
CREATE TABLE IF NOT EXISTS message_log (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    sender TEXT NOT NULL,
    receiver TEXT NOT NULL,
    message_type TEXT NOT NULL CHECK (message_type IN ('request', 'response', 'broadcast', 'delegation', 'status')),
    payload_json TEXT NOT NULL DEFAULT '{}',
    timestamp REAL NOT NULL,
    delivered INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Indexes for message log queries
CREATE INDEX IF NOT EXISTS idx_message_log_session ON message_log(session_id);
CREATE INDEX IF NOT EXISTS idx_message_log_receiver ON message_log(session_id, receiver, delivered);
CREATE INDEX IF NOT EXISTS idx_message_log_type ON message_log(session_id, message_type);

-- V3 schema version
INSERT OR IGNORE INTO schema_version (version, applied_at, description)
VALUES (3, strftime('%s', 'now'), 'Multi-agent communication tables');
