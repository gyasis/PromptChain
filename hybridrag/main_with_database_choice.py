#!/usr/bin/env python3
"""
HybridRAG with Database Choice
==============================
Enhanced startup that checks for existing database and gives user options.
"""

import os
import sys
import time
import signal
import shutil
from pathlib import Path
from typing import List
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config.config import load_config
from src.process_manager import ProcessManager

class HybridRAGWithChoice:
    """HybridRAG system with database choice options."""
    
    def __init__(self):
        self.config = load_config()
        self.process_manager = ProcessManager()
        self.running = True
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\n🛑 Shutting down...")
        self.running = False
        self.process_manager.shutdown()
        sys.exit(0)
    
    def check_existing_database(self) -> bool:
        """Check if LightRAG database exists."""
        lightrag_db = Path(self.config.lightrag.working_dir)
        if not lightrag_db.exists():
            return False
        
        # Check for LightRAG files
        db_files = list(lightrag_db.glob("*.json")) + list(lightrag_db.glob("*.graphml"))
        return len(db_files) > 0
    
    def get_database_stats(self) -> dict:
        """Get statistics about existing database."""
        lightrag_db = Path(self.config.lightrag.working_dir)
        stats = {"files": 0, "size_mb": 0}
        
        if lightrag_db.exists():
            db_files = list(lightrag_db.glob("*"))
            stats["files"] = len(db_files)
            total_size = sum(f.stat().st_size for f in db_files if f.is_file())
            stats["size_mb"] = total_size / (1024 * 1024)
        
        return stats
    
    def choose_database_action(self) -> str:
        """Let user choose what to do with existing database."""
        print("🚀 HybridRAG System - Database Setup")
        print("=" * 50)
        
        has_existing = self.check_existing_database()
        
        if has_existing:
            stats = self.get_database_stats()
            print(f"\n📊 Existing LightRAG database found:")
            print(f"   Location: {self.config.lightrag.working_dir}")
            print(f"   Files: {stats['files']}")
            print(f"   Size: {stats['size_mb']:.1f} MB")
            
            print(f"\n📋 Choose action:")
            print("1. Use existing database (start with current data)")
            print("2. Add to existing database (keep current + add new folder)")
            print("3. Start fresh (delete existing + create new)")
            print("4. Exit")
            
            while True:
                choice = input(f"\nChoice [1-4]: ").strip()
                
                if choice == "1":
                    print("✅ Using existing database")
                    return "use_existing"
                elif choice == "2":
                    print("✅ Will add to existing database")
                    return "add_to_existing"
                elif choice == "3":
                    confirm = input("⚠️  This will DELETE all existing data. Continue? [y/N]: ").strip().lower()
                    if confirm == 'y':
                        self.clear_database()
                        print("✅ Starting fresh")
                        return "start_fresh"
                    else:
                        print("Cancelled. Please choose again.")
                        continue
                elif choice == "4":
                    print("👋 Goodbye!")
                    sys.exit(0)
                else:
                    print("❌ Invalid choice. Please enter 1-4.")
        else:
            print(f"\n📊 No existing database found")
            print(f"   Will create new database at: {self.config.lightrag.working_dir}")
            return "start_fresh"
    
    def clear_database(self):
        """Clear existing LightRAG database."""
        lightrag_db = Path(self.config.lightrag.working_dir)
        if lightrag_db.exists():
            shutil.rmtree(lightrag_db)
            print(f"🗑️  Cleared database: {lightrag_db}")
        
        # Also clear processed files tracking
        processed_db = Path(self.config.ingestion.processed_files_db)
        if processed_db.exists():
            processed_db.unlink()
            print(f"🗑️  Cleared file tracking: {processed_db}")
    
    def choose_folders(self, action: str) -> List[str]:
        """Choose folders based on the action."""
        if action == "use_existing":
            print(f"\n🧠 Using existing database - no new ingestion needed")
            print(f"Ready for queries!")
            return []  # No new folders to watch
        
        print(f"\n📁 Choose folders to {'add to' if action == 'add_to_existing' else 'ingest into'} the database:")
        print("\nOptions:")
        print("1. Use sample data folder (./data)")
        print("2. Choose custom folder(s)")
        
        while True:
            choice = input("\nChoice [1-2]: ").strip()
            
            if choice == "1":
                data_folder = Path("./data").resolve()
                if data_folder.exists():
                    print(f"✅ Selected: {data_folder}")
                    return [str(data_folder)]
                else:
                    print("❌ Sample data folder doesn't exist")
                    continue
            elif choice == "2":
                return self.get_custom_folders()
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
    
    def get_custom_folders(self) -> List[str]:
        """Get custom folder paths."""
        print("\n📂 Enter folder paths:")
        folders = []
        
        while True:
            if folders:
                folder_input = input(f"Folder {len(folders)+1} (or Enter to finish): ").strip()
                if not folder_input:
                    break
            else:
                folder_input = input("Folder path: ").strip()
                if not folder_input:
                    print("❌ At least one folder required!")
                    continue
            
            # Expand user path
            folder_path = Path(folder_input).expanduser().resolve()
            
            if folder_path.exists() and folder_path.is_dir():
                folders.append(str(folder_path))
                print(f"✅ Added: {folder_path}")
            else:
                print(f"❌ Folder doesn't exist: {folder_path}")
                create = input("Create it? [y/N]: ").strip().lower()
                if create == 'y':
                    try:
                        folder_path.mkdir(parents=True, exist_ok=True)
                        folders.append(str(folder_path))
                        print(f"✅ Created and added: {folder_path}")
                    except Exception as e:
                        print(f"❌ Failed to create: {e}")
        
        return folders
    
    def start_query_mode(self):
        """Start query-only mode (no ingestion)."""
        print(f"\n🧠 Starting query mode...")
        
        try:
            # Initialize search interface
            from src.lightrag_core import create_lightrag_core
            from src.search_interface import SearchInterface
            
            lightrag_core = create_lightrag_core(self.config)
            search_interface = SearchInterface(self.config, lightrag_core)
            
            print(f"✅ Search interface ready")
            
            # Simple query loop
            print(f"\n" + "="*60)
            print("🔍 HybridRAG Query Mode")
            print("="*60)
            print("Commands:")
            print("  <question>    - Ask about your documents")
            print("  quit          - Exit")
            print("="*60)
            
            while self.running:
                try:
                    query = input(f"\nHybridRAG> ").strip()
                    
                    if not query:
                        continue
                    
                    if query.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    print(f"🔍 Searching: {query}")
                    
                    import asyncio
                    result = asyncio.run(search_interface.simple_search(query))
                    
                    print(f"\n📊 Result (mode: {result.mode}, time: {result.execution_time:.2f}s)")
                    print("─" * 60)
                    
                    if result.error:
                        print(f"❌ Error: {result.error}")
                    else:
                        print(result.result)
                        if result.sources:
                            print(f"\n📚 Sources: {len(result.sources)}")
                        print(f"💯 Confidence: {result.confidence_score:.1%}")
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
            
        except Exception as e:
            print(f"❌ Failed to start query mode: {e}")
    
    def start_ingestion_mode(self, watch_folders: List[str], continuous_mode: bool = False):
        """Start ingestion mode with folder watching.
        
        Args:
            watch_folders: List of folders to watch
            continuous_mode: If True, keep watching. If False, process once and move to query mode.
        """
        if not watch_folders:
            print(f"❌ No folders to watch")
            return
        
        # Ask about recursive
        recursive_input = input(f"\n🔄 Watch folders recursively? [Y/n]: ").strip().lower()
        recursive = recursive_input != 'n'
        print(f"{'✅ Recursive' if recursive else '❌ Non-recursive'} watching")
        
        print(f"\n🔧 Starting processes...")
        
        try:
            # Start watcher
            self.process_manager.start_watcher_process(watch_folders, recursive)
            print("✅ Folder watcher started")
            
            # Start ingestion  
            self.process_manager.start_ingestion_process(self.config)
            print("✅ Ingestion worker started")
            
            time.sleep(2)  # Let processes start
            
            # Check status
            status = self.process_manager.get_system_status()
            watcher_status = status['processes'].get('watcher', {}).get('status', 'unknown')
            ingestion_status = status['processes'].get('ingestion', {}).get('status', 'unknown')
            
            print(f"📊 Status: Watcher={watcher_status}, Ingestion={ingestion_status}")
            
            if watcher_status == 'running' and ingestion_status in ['running', 'starting']:
                print(f"\n✅ System ready!")
                print(f"📁 Processing: {', '.join(watch_folders)}")
                
                if continuous_mode:
                    print(f"⏱️  Add files to watched folders to see real-time ingestion")
                    print(f"🛑 Press Ctrl+C to exit")
                else:
                    print(f"⏳ Processing files once, then switching to query mode...")
                
                # Track if we've processed files
                initial_found = 0
                stable_count = 0
                last_processed = 0
                
                # Monitor ingestion
                while self.running:
                    time.sleep(3)
                    
                    # Show progress if available
                    progress = self.process_manager.get_ingestion_progress()
                    if progress and progress.total_files > 0:
                        print(f"📊 Processing: {progress.processed_files}/{progress.total_files} files | Current: {progress.current_file}")
                    
                    # Get current status
                    status = self.process_manager.get_system_status()
                    shared = status['shared_state']
                    found = shared.get('total_files_found', 0)
                    processed = shared.get('total_files_processed', 0)
                    ingestion_active = shared.get('ingestion_active', False)
                    
                    if found > 0 or processed > 0:
                        print(f"📈 Found: {found}, Processed: {processed}")
                    
                    # For one-time processing mode
                    if not continuous_mode:
                        # Track initial files found
                        if initial_found == 0 and found > 0:
                            initial_found = found
                            print(f"📦 Total files to process: {initial_found}")
                        
                        # Check if processing is complete
                        if initial_found > 0:
                            # All files processed and ingestion is inactive
                            if processed >= initial_found and not ingestion_active:
                                stable_count += 1
                                if stable_count >= 2:  # Wait for 2 cycles to ensure completion
                                    print(f"\n✅ Processing complete! {processed} files ingested.")
                                    print(f"🔄 Shutting down processes and switching to query mode...")
                                    self.process_manager.shutdown()
                                    time.sleep(2)
                                    self.start_query_mode()
                                    return
                            elif processed != last_processed:
                                # Reset stable count if still processing
                                stable_count = 0
                                last_processed = processed
                    
            else:
                print("❌ Failed to start processes properly")
                
        except KeyboardInterrupt:
            print(f"\n🛑 Interrupted")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            if not continuous_mode:
                # Ensure cleanup even on error
                self.process_manager.shutdown()
    
    async def sync_folders_once(self, watch_folders: List[str]):
        """Sync folders once without starting watcher processes."""
        print(f"\n📂 One-time sync of folders: {', '.join(watch_folders)}")
        
        # Ask about recursive
        recursive_input = input(f"\n🔄 Scan folders recursively? [Y/n]: ").strip().lower()
        recursive = recursive_input != 'n'
        print(f"{'✅ Recursive' if recursive else '❌ Non-recursive'} scanning")
        
        try:
            # Import the folder scanning logic directly
            from src.folder_watcher import FolderWatcher
            
            # Create a temporary folder watcher just for scanning
            temp_config = self.config
            temp_config.ingestion.watch_folders = watch_folders
            temp_config.ingestion.recursive = recursive
            
            watcher = FolderWatcher(temp_config)
            
            # Initialize LightRAG
            from src.lightrag_core import create_lightrag_core
            from src.ingestion_pipeline import DocumentProcessor
            
            lightrag_core = create_lightrag_core(self.config)
            processor = DocumentProcessor(self.config)
            
            print(f"\n🔍 Scanning for files...")
            
            # Scan folders for files to process
            files_to_process = watcher.scan_folders()
            
            if not files_to_process:
                print("✅ No new files to process - database is up to date!")
                return 0
            
            print(f"📊 Found {len(files_to_process)} new/modified files to process")
            
            # Process files directly without queuing
            success_count = 0
            error_count = 0
            
            for i, file_info in enumerate(files_to_process, 1):
                file_path = file_info.path
                file_name = os.path.basename(file_path)
                
                # Progress indicator
                print(f"\\r[{i}/{len(files_to_process)}] Processing: {file_name[:50]}...", end='', flush=True)
                
                try:
                    # Read file content
                    content = processor.read_file(file_path, file_info.extension)
                    
                    if content:
                        # Add metadata
                        doc_content = f"# Document: {file_path}\\n"
                        doc_content += f"# Type: {file_info.extension}\\n\\n"
                        doc_content += content
                        
                        # Insert into LightRAG
                        await lightrag_core.ainsert(doc_content)
                        
                        # Mark as processed
                        file_info.status = "processed"
                        file_info.ingested_at = datetime.now()
                        watcher.tracker.mark_file_processed(file_info)
                        
                        success_count += 1
                    else:
                        print(f"\\n⚠️  Empty content: {file_path}")
                        error_count += 1
                        
                except Exception as e:
                    print(f"\\n❌ Error processing {file_path}: {e}")
                    error_count += 1
            
            print()  # New line after progress
            print(f"\\n✅ Sync complete:")
            print(f"   📦 Successfully processed: {success_count}")
            if error_count > 0:
                print(f"   ❌ Errors: {error_count}")
            
            return success_count
            
        except Exception as e:
            print(f"❌ Error during sync: {e}")
            return 0
    
    async def run_async(self):
        """Run the system asynchronously."""
        try:
            # Choose database action
            action = self.choose_database_action()
            
            if action == "use_existing":
                # Query mode only
                self.start_query_mode()
            else:
                # Choose folders and start ingestion
                watch_folders = self.choose_folders(action)
                if watch_folders:
                    # Ask if user wants continuous monitoring or one-time sync
                    print(f"\n📋 Choose processing mode:")
                    print("1. Process files once, then switch to query mode (NO MONITORING)")
                    print("2. Continuous monitoring (watch for new files)")
                    
                    while True:
                        mode_choice = input(f"\nChoice [1-2]: ").strip()
                        if mode_choice == "1":
                            # One-time sync without watcher processes
                            processed_count = await self.sync_folders_once(watch_folders)
                            if processed_count >= 0:
                                print(f"\\n🔄 Switching to query mode...")
                                self.start_query_mode()
                            break
                        elif mode_choice == "2":
                            # Continuous monitoring with watcher processes
                            self.start_ingestion_mode(watch_folders, continuous_mode=True)
                            break
                        else:
                            print("❌ Invalid choice. Please enter 1 or 2.")
                else:
                    print("❌ No folders selected for ingestion")
                    
        except KeyboardInterrupt:
            print(f"\n🛑 Interrupted")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            print(f"🔄 Shutting down...")
            self.process_manager.shutdown()
            print(f"👋 Goodbye!")
    
    def run(self):
        """Run the system."""
        import asyncio
        asyncio.run(self.run_async())

def main():
    app = HybridRAGWithChoice()
    app.run()

if __name__ == "__main__":
    main()