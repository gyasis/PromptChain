# PostgreSQL Setup for LightRAG: Complete Guide with Docker Compose

**Author:** PromptChain Research Team  
**Date:** 2025-01-09  
**Purpose:** Complete setup guide for PostgreSQL-based LightRAG with Docker, backup, and migration

---

## Table of Contents

1. [Overview](#overview)
2. [Docker Compose Setup](#docker-compose-setup)
3. [PostgreSQL Configuration](#postgresql-configuration)
4. [LightRAG PostgreSQL Integration](#lightrag-postgresql-integration)
5. [Backup & Restore Procedures](#backup-restore-procedures)
6. [Migration from JSON to PostgreSQL](#migration-from-json)
7. [Production Deployment](#production-deployment)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This guide sets up **PostgreSQL with pgvector** for LightRAG, providing:
- ✅ **Vector Storage**: pgvector extension for efficient vector similarity search
- ✅ **Knowledge Graph**: PostgreSQL tables for entities and relationships
- ✅ **Key-Value Store**: PostgreSQL tables for metadata
- ✅ **Docker Compose**: Easy deployment and management
- ✅ **Backup/Restore**: Complete data protection

**Why PostgreSQL for LightRAG?**
- Hybrid search (vector + SQL filtering)
- ACID guarantees for data consistency
- Scalable to millions of vectors
- Existing infrastructure integration
- Excellent backup/restore tools

---

## Docker Compose Setup

### Step 1: Create Docker Compose File

Create `docker-compose.yml` in your project root:

```yaml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg16  # PostgreSQL 16 with pgvector
    container_name: athena_lightrag_postgres
    environment:
      POSTGRES_USER: lightrag_user
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-change_me_in_production}
      POSTGRES_DB: athena_lightrag
      PGDATA: /var/lib/postgresql/data/pgdata
    ports:
      - "5432:5432"
    volumes:
      # Persistent data storage
      - postgres_data:/var/lib/postgresql/data
      # Backup directory (mounted from host)
      - ./backups:/backups
      # Init scripts
      - ./docker/postgres/init:/docker-entrypoint-initdb.d
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U lightrag_user -d athena_lightrag"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - lightrag_network

  # Optional: pgAdmin for database management
  pgadmin:
    image: dpage/pgadmin4:latest
    container_name: athena_lightrag_pgadmin
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@lightrag.local
      PGADMIN_DEFAULT_PASSWORD: ${PGADMIN_PASSWORD:-admin}
      PGADMIN_CONFIG_SERVER_MODE: 'False'
    ports:
      - "5050:80"
    volumes:
      - pgadmin_data:/var/lib/pgadmin
    depends_on:
      - postgres
    networks:
      - lightrag_network
    restart: unless-stopped

volumes:
  postgres_data:
    driver: local
  pgadmin_data:
    driver: local

networks:
  lightrag_network:
    driver: bridge
```

### Step 2: Create Environment File

Create `.env` file (add to `.gitignore`):

```bash
# PostgreSQL Configuration
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_USER=lightrag_user
POSTGRES_DB=athena_lightrag
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# pgAdmin Configuration
PGADMIN_PASSWORD=admin_password_here

# LightRAG Configuration
OPENAI_API_KEY=your_openai_key_here
```

### Step 3: Create Initialization Script

Create `docker/postgres/init/01-init-extensions.sql`:

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable additional useful extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text similarity

-- Verify extensions
SELECT extname, extversion 
FROM pg_extension 
WHERE extname IN ('vector', 'uuid-ossp', 'pg_trgm');
```

### Step 4: Start PostgreSQL

```bash
# Start services
docker-compose up -d

# Check logs
docker-compose logs -f postgres

# Verify PostgreSQL is running
docker-compose exec postgres psql -U lightrag_user -d athena_lightrag -c "SELECT version();"
docker-compose exec postgres psql -U lightrag_user -d athena_lightrag -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

---

## PostgreSQL Configuration

### Database Schema for LightRAG

LightRAG will create its own tables, but here's what they look like:

#### **Vector Storage Tables (pgvector)**

```sql
-- Entity vectors
CREATE TABLE IF NOT EXISTS vdb_entities (
    id SERIAL PRIMARY KEY,
    entity_id TEXT UNIQUE NOT NULL,
    embedding vector(1536),  -- Adjust dimension to match your embeddings
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Relationship vectors
CREATE TABLE IF NOT EXISTS vdb_relationships (
    id SERIAL PRIMARY KEY,
    relationship_id TEXT UNIQUE NOT NULL,
    embedding vector(1536),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Text chunk vectors
CREATE TABLE IF NOT EXISTS vdb_chunks (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    embedding vector(1536),
    text_content TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for vector similarity search
CREATE INDEX IF NOT EXISTS vdb_entities_embedding_idx 
ON vdb_entities USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS vdb_relationships_embedding_idx 
ON vdb_relationships USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS vdb_chunks_embedding_idx 
ON vdb_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

#### **Knowledge Graph Tables**

```sql
-- Entities table
CREATE TABLE IF NOT EXISTS kv_store_entities (
    id SERIAL PRIMARY KEY,
    entity_id TEXT UNIQUE NOT NULL,
    entity_name TEXT NOT NULL,
    entity_type TEXT,
    properties JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Relationships table
CREATE TABLE IF NOT EXISTS kv_store_relations (
    id SERIAL PRIMARY KEY,
    relation_id TEXT UNIQUE NOT NULL,
    source_entity_id TEXT NOT NULL,
    target_entity_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    properties JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (source_entity_id) REFERENCES kv_store_entities(entity_id),
    FOREIGN KEY (target_entity_id) REFERENCES kv_store_entities(entity_id)
);

-- Text chunks table
CREATE TABLE IF NOT EXISTS kv_store_chunks (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    text_content TEXT NOT NULL,
    source_document TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for graph queries
CREATE INDEX IF NOT EXISTS idx_relations_source ON kv_store_relations(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON kv_store_relations(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_relations_type ON kv_store_relations(relation_type);
CREATE INDEX IF NOT EXISTS idx_entities_type ON kv_store_entities(entity_type);
```

### PostgreSQL Tuning for LightRAG

Create `docker/postgres/init/02-postgresql-config.conf`:

```conf
# Memory Configuration
shared_buffers = 256MB          # 25% of RAM for small instances
effective_cache_size = 1GB      # 50-75% of RAM
work_mem = 16MB                 # Per operation memory
maintenance_work_mem = 128MB    # For VACUUM, CREATE INDEX

# Connection Settings
max_connections = 100
listen_addresses = '*'

# Vector Search Optimization
random_page_cost = 1.1          # Lower for SSD
effective_io_concurrency = 200  # For SSD

# Logging
log_statement = 'mod'            # Log DDL and DML
log_min_duration_statement = 1000  # Log slow queries (>1s)
```

Apply configuration in `docker-compose.yml`:

```yaml
services:
  postgres:
    # ... existing config ...
    command:
      - "postgres"
      - "-c"
      - "shared_buffers=256MB"
      - "-c"
      - "effective_cache_size=1GB"
      - "-c"
      - "work_mem=16MB"
      - "-c"
      - "maintenance_work_mem=128MB"
      - "-c"
      - "max_connections=100"
      - "-c"
      - "random_page_cost=1.1"
      - "-c"
      - "effective_io_concurrency=200"
```

---

## LightRAG PostgreSQL Integration

### Step 1: Install Dependencies

```bash
pip install lightrag-hku psycopg2-binary pgvector
```

### Step 2: Configure Environment Variables

Update your `.env` file or set environment variables:

```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_USER=lightrag_user
export POSTGRES_PASSWORD=your_password
export POSTGRES_DATABASE=athena_lightrag
```

### Step 3: Update LightRAG Configuration

Modify `lightrag_core.py` or create new configuration:

```python
#!/usr/bin/env python3
"""
LightRAG with PostgreSQL Storage Configuration
"""

import os
import asyncio
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from dotenv import load_dotenv

load_dotenv()

async def create_postgresql_lightrag():
    """Create LightRAG instance with PostgreSQL storage."""
    
    # LLM function
    def llm_model_func(prompt: str, system_prompt: str = None, **kwargs) -> str:
        return openai_complete_if_cache(
            model="gpt-4.1-mini",
            prompt=prompt,
            system_prompt=system_prompt,
            api_key=os.getenv("OPENAI_API_KEY"),
            **kwargs
        )
    
    # Embedding function
    def embedding_func(texts: list) -> list:
        return openai_embed(
            texts=texts,
            model="text-embedding-ada-002",
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    # Initialize LightRAG with PostgreSQL storage
    rag = LightRAG(
        working_dir="./athena_lightrag_db",  # Still used for config, not storage
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=1536,
            func=embedding_func
        ),
        # PostgreSQL storage backends
        kv_storage="PGKVStorage",
        vector_storage="PGVectorStorage",
        graph_storage="PGGraphStorage",
        doc_status_storage="PGDocStatusStorage",
        # Optional: Workspace for multi-tenancy
        workspace="athena_medical"
    )
    
    # Initialize storages (creates tables if they don't exist)
    await rag.initialize_storages()
    
    print("✅ LightRAG initialized with PostgreSQL storage!")
    return rag

if __name__ == "__main__":
    asyncio.run(create_postgresql_lightrag())
```

### Step 4: Update athena-lightrag Configuration

Modify `config.py` to support PostgreSQL:

```python
@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    # Storage backend type
    storage_backend: Literal["json", "postgresql"] = "postgresql"
    
    # JSON storage (if using)
    working_dir: str = "/home/gyasis/Documents/code/PromptChain/athena-lightrag/athena_lightrag_db"
    
    # PostgreSQL configuration
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_user: str = os.getenv("POSTGRES_USER", "lightrag_user")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "")
    postgres_database: str = os.getenv("POSTGRES_DATABASE", "athena_lightrag")
    postgres_workspace: str = os.getenv("POSTGRES_WORKSPACE", "athena_medical")
    
    def get_storage_config(self) -> dict:
        """Get storage configuration based on backend type."""
        if self.storage_backend == "postgresql":
            return {
                "kv_storage": "PGKVStorage",
                "vector_storage": "PGVectorStorage",
                "graph_storage": "PGGraphStorage",
                "doc_status_storage": "PGDocStatusStorage"
            }
        else:
            return {
                "kv_storage": "JSONKVStorage",
                "vector_storage": "NanoVectorDBStorage",
                "graph_storage": "NetworkXStorage"
            }
```

### Step 5: Update LightRAG Core Initialization

Modify `lightrag_core.py`:

```python
def _init_lightrag(self):
    """Initialize LightRAG instance with PostgreSQL storage."""
    logger.info(f"Initializing LightRAG with PostgreSQL storage")
    
    # Get storage configuration
    from config import get_config
    config = get_config()
    storage_config = config.database.get_storage_config()
    
    # LLM model function
    def llm_model_func(prompt: str, system_prompt: str = None, **kwargs) -> str:
        return openai_complete_if_cache(
            model=self.config.model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            api_key=self.api_key,
            **kwargs
        )
    
    # Embedding function
    def embedding_func(texts: list) -> list:
        return openai_embed(
            texts=texts,
            model=self.config.embedding_model,
            api_key=self.api_key
        )
    
    # Initialize LightRAG with PostgreSQL
    self.rag = LightRAG(
        working_dir=self.config.working_dir,
        llm_model_func=llm_model_func,
        llm_model_max_async=self.config.max_async,
        embedding_func=EmbeddingFunc(
            embedding_dim=self.config.embedding_dim,
            func=embedding_func
        ),
        **storage_config  # PostgreSQL storage backends
    )
    
    logger.info("LightRAG initialized with PostgreSQL storage successfully")
```

---

## Backup & Restore Procedures

### Automated Backup Script

Create `scripts/backup_postgres.sh`:

```bash
#!/bin/bash
# PostgreSQL Backup Script for LightRAG

set -e

# Configuration
BACKUP_DIR="./backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/athena_lightrag_${TIMESTAMP}.sql"
COMPRESSED_FILE="${BACKUP_FILE}.gz"
RETENTION_DAYS=30

# Create backup directory
mkdir -p "${BACKUP_DIR}"

# Load environment variables
source .env

echo "🔄 Starting PostgreSQL backup..."

# Create backup using pg_dump
docker-compose exec -T postgres pg_dump \
    -U "${POSTGRES_USER}" \
    -d "${POSTGRES_DB}" \
    --clean \
    --if-exists \
    --create \
    --format=plain \
    > "${BACKUP_FILE}"

# Compress backup
gzip "${BACKUP_FILE}"
echo "✅ Backup created: ${COMPRESSED_FILE}"

# Remove old backups (older than retention period)
find "${BACKUP_DIR}" -name "athena_lightrag_*.sql.gz" -type f -mtime +${RETENTION_DAYS} -delete
echo "🧹 Cleaned up backups older than ${RETENTION_DAYS} days"

# List recent backups
echo "📦 Recent backups:"
ls -lh "${BACKUP_DIR}"/*.sql.gz | tail -5
```

Make it executable:
```bash
chmod +x scripts/backup_postgres.sh
```

### Restore Script

Create `scripts/restore_postgres.sh`:

```bash
#!/bin/bash
# PostgreSQL Restore Script for LightRAG

set -e

# Check if backup file provided
if [ -z "$1" ]; then
    echo "❌ Error: Please provide backup file path"
    echo "Usage: ./restore_postgres.sh <backup_file.sql.gz>"
    exit 1
fi

BACKUP_FILE="$1"

# Check if file exists
if [ ! -f "${BACKUP_FILE}" ]; then
    echo "❌ Error: Backup file not found: ${BACKUP_FILE}"
    exit 1
fi

# Load environment variables
source .env

echo "🔄 Starting PostgreSQL restore from: ${BACKUP_FILE}"

# Confirm restore
read -p "⚠️  This will overwrite existing data. Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Restore cancelled"
    exit 1
fi

# Decompress if needed
if [[ "${BACKUP_FILE}" == *.gz ]]; then
    echo "📦 Decompressing backup..."
    TEMP_FILE=$(mktemp)
    gunzip -c "${BACKUP_FILE}" > "${TEMP_FILE}"
    BACKUP_FILE="${TEMP_FILE}"
fi

# Restore database
echo "🔄 Restoring database..."
docker-compose exec -T postgres psql \
    -U "${POSTGRES_USER}" \
    -d postgres \
    < "${BACKUP_FILE}"

# Cleanup temp file if created
if [ -n "${TEMP_FILE}" ]; then
    rm "${TEMP_FILE}"
fi

echo "✅ Database restored successfully!"
```

### Scheduled Backups with Cron

Add to crontab (`crontab -e`):

```bash
# Daily backup at 2 AM
0 2 * * * cd /home/gyasis/Documents/code/PromptChain/athena-lightrag && ./scripts/backup_postgres.sh >> /var/log/lightrag_backup.log 2>&1

# Weekly full backup on Sunday at 1 AM
0 1 * * 0 cd /home/gyasis/Documents/code/PromptChain/athena-lightrag && ./scripts/backup_postgres.sh >> /var/log/lightrag_backup.log 2>&1
```

### Manual Backup Commands

```bash
# Full database backup
docker-compose exec postgres pg_dump -U lightrag_user -d athena_lightrag > backup.sql

# Compressed backup
docker-compose exec postgres pg_dump -U lightrag_user -d athena_lightrag | gzip > backup.sql.gz

# Backup specific tables
docker-compose exec postgres pg_dump -U lightrag_user -d athena_lightrag -t vdb_entities -t vdb_relationships > vectors_backup.sql

# Restore from backup
docker-compose exec -T postgres psql -U lightrag_user -d athena_lightrag < backup.sql
```

### Transfer Backup to Another Server

```bash
# Compress and transfer
tar -czf lightrag_backup.tar.gz backups/
scp lightrag_backup.tar.gz user@remote-server:/path/to/destination/

# On remote server, extract and restore
tar -xzf lightrag_backup.tar.gz
./scripts/restore_postgres.sh backups/athena_lightrag_YYYYMMDD_HHMMSS.sql.gz
```

---

## Migration from JSON to PostgreSQL

### Migration Script

Create `scripts/migrate_json_to_postgresql.py`:

```python
#!/usr/bin/env python3
"""
Migration script: JSON (nano-vectordb) → PostgreSQL (pgvector)
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

class JSONToPostgreSQLMigrator:
    """Migrate LightRAG data from JSON to PostgreSQL."""
    
    def __init__(self, json_dir: str, postgres_config: dict):
        self.json_dir = Path(json_dir)
        self.postgres_config = postgres_config
        self.conn = None
    
    def connect(self):
        """Connect to PostgreSQL."""
        self.conn = psycopg2.connect(
            host=self.postgres_config['host'],
            port=self.postgres_config['port'],
            user=self.postgres_config['user'],
            password=self.postgres_config['password'],
            database=self.postgres_config['database']
        )
        self.conn.autocommit = False
    
    def create_tables(self):
        """Create PostgreSQL tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Enable extensions
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")
        
        # Vector tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vdb_entities (
                id SERIAL PRIMARY KEY,
                entity_id TEXT UNIQUE NOT NULL,
                embedding vector(1536),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vdb_relationships (
                id SERIAL PRIMARY KEY,
                relationship_id TEXT UNIQUE NOT NULL,
                embedding vector(1536),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vdb_chunks (
                id SERIAL PRIMARY KEY,
                chunk_id TEXT UNIQUE NOT NULL,
                embedding vector(1536),
                text_content TEXT,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        
        # KV store tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kv_store_entities (
                id SERIAL PRIMARY KEY,
                entity_id TEXT UNIQUE NOT NULL,
                entity_name TEXT NOT NULL,
                entity_type TEXT,
                properties JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kv_store_relations (
                id SERIAL PRIMARY KEY,
                relation_id TEXT UNIQUE NOT NULL,
                source_entity_id TEXT,
                target_entity_id TEXT,
                relation_type TEXT,
                properties JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kv_store_chunks (
                id SERIAL PRIMARY KEY,
                chunk_id TEXT UNIQUE NOT NULL,
                text_content TEXT NOT NULL,
                source_document TEXT,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS vdb_entities_embedding_idx 
            ON vdb_entities USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
        """)
        
        self.conn.commit()
        cursor.close()
        print("✅ Tables created")
    
    def migrate_vectors(self, vdb_file: str, table_name: str):
        """Migrate vector data from JSON to PostgreSQL."""
        file_path = self.json_dir / vdb_file
        
        if not file_path.exists():
            print(f"⚠️  File not found: {vdb_file}")
            return 0
        
        print(f"📦 Migrating {vdb_file} to {table_name}...")
        
        with open(file_path) as f:
            data = json.load(f)
        
        vectors = data.get('vectors', {})
        cursor = self.conn.cursor()
        
        migrated = 0
        batch_size = 100
        
        for entity_id, vector_data in vectors.items():
            embedding = vector_data.get('embedding', [])
            metadata = vector_data.get('metadata', {})
            
            # Convert to PostgreSQL vector format
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            
            try:
                if table_name == 'vdb_chunks':
                    text_content = metadata.get('text', '')
                    cursor.execute(f"""
                        INSERT INTO {table_name} (chunk_id, embedding, text_content, metadata)
                        VALUES (%s, %s::vector, %s, %s::jsonb)
                        ON CONFLICT (chunk_id) DO NOTHING
                    """, (entity_id, embedding_str, text_content, json.dumps(metadata)))
                else:
                    cursor.execute(f"""
                        INSERT INTO {table_name} ({table_name.split('_')[1]}_id, embedding, metadata)
                        VALUES (%s, %s::vector, %s::jsonb)
                        ON CONFLICT ({table_name.split('_')[1]}_id) DO NOTHING
                    """, (entity_id, embedding_str, json.dumps(metadata)))
                
                migrated += 1
                
                if migrated % batch_size == 0:
                    self.conn.commit()
                    print(f"  ✅ Migrated {migrated} vectors...")
            
            except Exception as e:
                print(f"  ❌ Error migrating {entity_id}: {e}")
                self.conn.rollback()
                continue
        
        self.conn.commit()
        cursor.close()
        print(f"✅ Migrated {migrated} vectors to {table_name}")
        return migrated
    
    def migrate_kv_store(self, kv_file: str, table_name: str):
        """Migrate key-value store data from JSON to PostgreSQL."""
        file_path = self.json_dir / kv_file
        
        if not file_path.exists():
            print(f"⚠️  File not found: {kv_file}")
            return 0
        
        print(f"📦 Migrating {kv_file} to {table_name}...")
        
        with open(file_path) as f:
            data = json.load(f)
        
        cursor = self.conn.cursor()
        migrated = 0
        batch_size = 100
        
        for key, value in data.items():
            try:
                if table_name == 'kv_store_entities':
                    cursor.execute("""
                        INSERT INTO kv_store_entities (entity_id, entity_name, entity_type, properties)
                        VALUES (%s, %s, %s, %s::jsonb)
                        ON CONFLICT (entity_id) DO NOTHING
                    """, (
                        key,
                        value.get('name', key),
                        value.get('type', 'unknown'),
                        json.dumps(value)
                    ))
                elif table_name == 'kv_store_relations':
                    cursor.execute("""
                        INSERT INTO kv_store_relations (relation_id, source_entity_id, target_entity_id, relation_type, properties)
                        VALUES (%s, %s, %s, %s, %s::jsonb)
                        ON CONFLICT (relation_id) DO NOTHING
                    """, (
                        key,
                        value.get('source', ''),
                        value.get('target', ''),
                        value.get('type', 'unknown'),
                        json.dumps(value)
                    ))
                elif table_name == 'kv_store_chunks':
                    cursor.execute("""
                        INSERT INTO kv_store_chunks (chunk_id, text_content, source_document, metadata)
                        VALUES (%s, %s, %s, %s::jsonb)
                        ON CONFLICT (chunk_id) DO NOTHING
                    """, (
                        key,
                        value.get('text', ''),
                        value.get('source', ''),
                        json.dumps(value)
                    ))
                
                migrated += 1
                
                if migrated % batch_size == 0:
                    self.conn.commit()
                    print(f"  ✅ Migrated {migrated} records...")
            
            except Exception as e:
                print(f"  ❌ Error migrating {key}: {e}")
                self.conn.rollback()
                continue
        
        self.conn.commit()
        cursor.close()
        print(f"✅ Migrated {migrated} records to {table_name}")
        return migrated
    
    def migrate_all(self):
        """Migrate all data from JSON to PostgreSQL."""
        print("🚀 Starting migration from JSON to PostgreSQL...")
        
        self.connect()
        self.create_tables()
        
        total_migrated = 0
        
        # Migrate vectors
        total_migrated += self.migrate_vectors('vdb_entities.json', 'vdb_entities')
        total_migrated += self.migrate_vectors('vdb_relationships.json', 'vdb_relationships')
        total_migrated += self.migrate_vectors('vdb_chunks.json', 'vdb_chunks')
        
        # Migrate KV stores
        total_migrated += self.migrate_kv_store('kv_store_full_entities.json', 'kv_store_entities')
        total_migrated += self.migrate_kv_store('kv_store_full_relations.json', 'kv_store_relations')
        total_migrated += self.migrate_kv_store('kv_store_text_chunks.json', 'kv_store_chunks')
        
        self.conn.close()
        
        print(f"✅ Migration complete! Total records migrated: {total_migrated}")
        return total_migrated

async def main():
    """Main migration function."""
    postgres_config = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': int(os.getenv('POSTGRES_PORT', '5432')),
        'user': os.getenv('POSTGRES_USER', 'lightrag_user'),
        'password': os.getenv('POSTGRES_PASSWORD', ''),
        'database': os.getenv('POSTGRES_DATABASE', 'athena_lightrag')
    }
    
    json_dir = os.getenv('LIGHTRAG_JSON_DIR', './athena_lightrag_db')
    
    migrator = JSONToPostgreSQLMigrator(json_dir, postgres_config)
    migrator.migrate_all()

if __name__ == "__main__":
    asyncio.run(main())
```

### Run Migration

```bash
# Set environment variables
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_USER=lightrag_user
export POSTGRES_PASSWORD=your_password
export POSTGRES_DATABASE=athena_lightrag
export LIGHTRAG_JSON_DIR=./athena_lightrag_db

# Run migration
python scripts/migrate_json_to_postgresql.py
```

---

## Production Deployment

### Security Hardening

1. **Change Default Passwords**
   ```bash
   # Generate strong password
   openssl rand -base64 32
   
   # Update .env file
   POSTGRES_PASSWORD=<strong_password>
   ```

2. **Network Security**
   ```yaml
   # In docker-compose.yml, restrict network access
   services:
     postgres:
       # ... existing config ...
       ports:
         - "127.0.0.1:5432:5432"  # Only localhost access
   ```

3. **SSL Configuration**
   ```yaml
   services:
     postgres:
       environment:
         POSTGRES_HOST_AUTH_METHOD: scram-sha-256
       command:
         - "postgres"
         - "-c"
         - "ssl=on"
         - "-c"
         - "ssl_cert_file=/var/lib/postgresql/ssl/server.crt"
         - "-c"
         - "ssl_key_file=/var/lib/postgresql/ssl/server.key"
   ```

### Monitoring

Create `scripts/monitor_postgres.sh`:

```bash
#!/bin/bash
# PostgreSQL Monitoring Script

docker-compose exec postgres psql -U lightrag_user -d athena_lightrag <<EOF
-- Database size
SELECT pg_size_pretty(pg_database_size('athena_lightrag')) AS database_size;

-- Table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Vector table counts
SELECT 
    'vdb_entities' AS table_name,
    COUNT(*) AS record_count
FROM vdb_entities
UNION ALL
SELECT 
    'vdb_relationships',
    COUNT(*)
FROM vdb_relationships
UNION ALL
SELECT 
    'vdb_chunks',
    COUNT(*)
FROM vdb_chunks;

-- Connection count
SELECT count(*) AS active_connections FROM pg_stat_activity;
EOF
```

---

## Troubleshooting

### Common Issues

1. **pgvector extension not found**
   ```bash
   # Verify extension is installed
   docker-compose exec postgres psql -U lightrag_user -d athena_lightrag -c "CREATE EXTENSION vector;"
   ```

2. **Connection refused**
   ```bash
   # Check if PostgreSQL is running
   docker-compose ps
   docker-compose logs postgres
   
   # Test connection
   docker-compose exec postgres psql -U lightrag_user -d athena_lightrag
   ```

3. **Vector dimension mismatch**
   ```sql
   -- Check current dimension
   SELECT atttypmod FROM pg_attribute 
   WHERE attrelid = 'vdb_entities'::regclass AND attname = 'embedding';
   
   -- Adjust if needed (requires table recreation)
   ALTER TABLE vdb_entities ALTER COLUMN embedding TYPE vector(1536);
   ```

4. **Performance issues**
   ```sql
   -- Analyze tables for better query planning
   ANALYZE vdb_entities;
   ANALYZE vdb_relationships;
   ANALYZE vdb_chunks;
   
   -- Reindex if needed
   REINDEX TABLE vdb_entities;
   ```

---

## Quick Reference

### Docker Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f postgres

# Access PostgreSQL CLI
docker-compose exec postgres psql -U lightrag_user -d athena_lightrag

# Backup
docker-compose exec postgres pg_dump -U lightrag_user athena_lightrag > backup.sql

# Restore
docker-compose exec -T postgres psql -U lightrag_user athena_lightrag < backup.sql
```

### Environment Variables

```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=lightrag_user
POSTGRES_PASSWORD=your_password
POSTGRES_DATABASE=athena_lightrag
POSTGRES_WORKSPACE=athena_medical
```

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-09  
**Maintained By:** PromptChain Research Team

