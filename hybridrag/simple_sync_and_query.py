#!/usr/bin/env python3
"""
Simple Sync and Query System
============================
Single process, sync once, then query. Simple and reliable.
"""

import os
import sys
import asyncio
import hashlib
import sqlite3
import json
from pathlib import Path
from typing import List, Set, Dict, Optional
from datetime import datetime
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config.config import load_config
from src.lightrag_core import create_lightrag_core
from src.search_interface import SearchInterface
from src.ingestion_pipeline import DocumentProcessor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleSyncSystem:
    """Simple sync and query system."""
    
    def __init__(self):
        """Initialize the system."""
        self.config = load_config()
        self.lightrag_core = create_lightrag_core(self.config)
        self.search_interface = SearchInterface(self.config, self.lightrag_core)
        self.processor = DocumentProcessor(self.config)
        self.processed_db = "processed_files.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize tracking database."""
        with sqlite3.connect(self.processed_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_files (
                    file_path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def get_file_hash(self, file_path: str) -> str:
        """Calculate file hash."""
        sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error hashing {file_path}: {e}")
            return ""
    
    def is_file_processed(self, file_path: str, file_hash: str) -> bool:
        """Check if file was already processed."""
        with sqlite3.connect(self.processed_db) as conn:
            cursor = conn.execute(
                "SELECT file_hash FROM processed_files WHERE file_path = ?",
                (file_path,)
            )
            row = cursor.fetchone()
            return row and row[0] == file_hash
    
    def mark_file_processed(self, file_path: str, file_hash: str):
        """Mark file as processed."""
        with sqlite3.connect(self.processed_db) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO processed_files (file_path, file_hash) VALUES (?, ?)",
                (file_path, file_hash)
            )
            conn.commit()
    
    def scan_folder(self, folder_path: str, recursive: bool = True) -> List[Dict]:
        """Scan folder and find files to process."""
        folder = Path(folder_path)
        if not folder.exists():
            logger.error(f"Folder doesn't exist: {folder}")
            return []
        
        files_to_process = []
        extensions = ['.txt', '.md', '.pdf', '.json', '.py', '.js', '.html', '.csv', '.yaml', '.yml']
        
        # Get files
        if recursive:
            file_paths = [f for ext in extensions for f in folder.rglob(f"*{ext}")]
        else:
            file_paths = [f for ext in extensions for f in folder.glob(f"*{ext}")]
        
        # Check each file
        for file_path in file_paths:
            if not file_path.is_file():
                continue
            
            # Skip hidden files (but allow hidden directories if explicitly watched)
            if file_path.name.startswith('.') and file_path.name not in ['.', '..']:
                continue
            
            try:
                file_hash = self.get_file_hash(str(file_path))
                
                if not self.is_file_processed(str(file_path), file_hash):
                    files_to_process.append({
                        'path': str(file_path),
                        'hash': file_hash,
                        'size': file_path.stat().st_size,
                        'extension': file_path.suffix
                    })
            except Exception as e:
                logger.error(f"Error checking {file_path}: {e}")
        
        return files_to_process
    
    async def sync_folder(self, folder_path: str, recursive: bool = True):
        """Sync folder with LightRAG database."""
        print(f"\n📂 Syncing folder: {folder_path}")
        print(f"🔄 Recursive: {'Yes' if recursive else 'No'}")
        
        # Scan for new files
        files_to_process = self.scan_folder(folder_path, recursive)
        
        if not files_to_process:
            print("✅ No new files to process - database is up to date!")
            return 0
        
        print(f"📊 Found {len(files_to_process)} new/modified files to process")
        
        # Process files
        success_count = 0
        error_count = 0
        
        for i, file_info in enumerate(files_to_process, 1):
            file_path = file_info['path']
            file_name = os.path.basename(file_path)
            
            # Progress indicator
            print(f"\r[{i}/{len(files_to_process)}] Processing: {file_name[:50]}...", end='', flush=True)
            
            try:
                # Read file content
                content = self.processor.read_file(file_path, file_info['extension'])
                
                if content:
                    # Add metadata
                    doc_content = f"# Document: {file_path}\n"
                    doc_content += f"# Type: {file_info['extension']}\n\n"
                    doc_content += content
                    
                    # Insert into LightRAG
                    success = await self.lightrag_core.ainsert(doc_content)
                    
                    if success:
                        self.mark_file_processed(file_path, file_info['hash'])
                        success_count += 1
                    else:
                        error_count += 1
                else:
                    logger.warning(f"Empty content: {file_path}")
                    error_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                error_count += 1
        
        print()  # New line after progress
        print(f"\n📈 Sync complete:")
        print(f"   ✅ Successfully processed: {success_count}")
        if error_count > 0:
            print(f"   ❌ Errors: {error_count}")
        
        return success_count
    
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        with sqlite3.connect(self.processed_db) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM processed_files")
            total_files = cursor.fetchone()[0]
        
        # Check LightRAG database
        lightrag_stats = self.lightrag_core.get_stats()
        
        return {
            'total_files': total_files,
            'lightrag': lightrag_stats
        }
    
    async def interactive_query_mode(self):
        """Interactive query mode."""
        print("\n" + "="*60)
        print("🔍 HybridRAG Query Interface")
        print("="*60)
        
        # Show database stats
        stats = self.get_database_stats()
        print(f"📊 Database: {stats['total_files']} documents indexed")
        print(f"📁 Location: {stats['lightrag']['working_directory']}")
        
        print("\nCommands:")
        print("  <question>           - Ask about your documents")
        print("  sync <folder>        - Sync new files from folder")
        print("  stats                - Show database statistics")
        print("  clear                - Clear screen")
        print("  quit                 - Exit")
        print("="*60)
        
        while True:
            try:
                user_input = input("\nQuery> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                elif user_input.lower() == 'clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    
                elif user_input.lower() == 'stats':
                    stats = self.get_database_stats()
                    print(f"📊 Total files: {stats['total_files']}")
                    print(f"📁 LightRAG: {stats['lightrag']}")
                    
                elif user_input.lower().startswith('sync '):
                    folder = user_input[5:].strip()
                    folder_path = Path(folder).expanduser().resolve()
                    if folder_path.exists() and folder_path.is_dir():
                        await self.sync_folder(str(folder_path))
                    else:
                        print(f"❌ Invalid folder: {folder}")
                        
                else:
                    # Search query
                    print(f"🔍 Searching...")
                    
                    result = await self.search_interface.simple_search(user_input)
                    
                    print(f"\n📊 Result (mode: {result.mode}, time: {result.execution_time:.2f}s)")
                    print("─" * 60)
                    
                    if result.error:
                        print(f"❌ Error: {result.error}")
                    else:
                        print(result.result)
                        if result.sources:
                            print(f"\n📚 Sources ({len(result.sources)}):")
                            for source in result.sources[:3]:
                                print(f"  • {source}")
                        print(f"\n💯 Confidence: {result.confidence_score:.1%}")
                        
            except KeyboardInterrupt:
                print("\n🛑 Use 'quit' to exit")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n👋 Goodbye!")

async def main():
    """Main entry point."""
    print("🚀 Simple HybridRAG System")
    print("=" * 50)
    
    system = SimpleSyncSystem()
    
    # Check for existing database
    stats = system.get_database_stats()
    
    if stats['total_files'] > 0:
        print(f"\n📊 Existing database found: {stats['total_files']} documents")
        print("\nOptions:")
        print("1. Use existing database")
        print("2. Sync new folder and add to database")
        print("3. Clear and start fresh")
        
        choice = input("\nChoice [1-3]: ").strip()
        
        if choice == "2":
            # Sync additional folder
            folder = input("Folder path to sync: ").strip()
            folder_path = Path(folder).expanduser().resolve()
            if folder_path.exists() and folder_path.is_dir():
                await system.sync_folder(str(folder_path))
            else:
                print(f"❌ Invalid folder: {folder}")
                
        elif choice == "3":
            # Clear database
            confirm = input("⚠️  Clear all data? [y/N]: ").strip().lower()
            if confirm == 'y':
                # Clear processed files DB
                with sqlite3.connect(system.processed_db) as conn:
                    conn.execute("DELETE FROM processed_files")
                    conn.commit()
                print("✅ Cleared tracking database")
                
                # Ask for new folder
                folder = input("Folder path to sync: ").strip()
                folder_path = Path(folder).expanduser().resolve()
                if folder_path.exists() and folder_path.is_dir():
                    await system.sync_folder(str(folder_path))
                else:
                    print(f"❌ Invalid folder: {folder}")
    else:
        # No existing database
        print("\n📊 No existing database found")
        folder = input("Folder path to sync: ").strip()
        
        if not folder:
            folder = "./data"
            print(f"Using default: {folder}")
        
        folder_path = Path(folder).expanduser().resolve()
        if folder_path.exists() and folder_path.is_dir():
            await system.sync_folder(str(folder_path))
        else:
            print(f"❌ Invalid folder: {folder}")
            return
    
    # Start query mode
    await system.interactive_query_mode()

if __name__ == "__main__":
    asyncio.run(main())