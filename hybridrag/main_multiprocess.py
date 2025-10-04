#!/usr/bin/env python3
"""
HybridRAG Multiprocess Main Application
=======================================
Main application with separate processes for watching, ingestion, and querying.
"""

import os
import sys
import time
import threading
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config.config import load_config
from src.process_manager import ProcessManager, setup_signal_handlers
from src.lightrag_core import create_lightrag_core
from src.search_interface import SearchInterface

class HybridRAGApp:
    """Main HybridRAG application with multiprocess architecture."""
    
    def __init__(self):
        """Initialize the application."""
        self.config = load_config()
        self.process_manager = ProcessManager()
        self.lightrag_core = None
        self.search_interface = None
        self.status_thread = None
        self.running = False
        
        print("🚀 HybridRAG Multiprocess System")
        print("=" * 50)
    
    def setup_folders(self) -> List[str]:
        """Setup watch folders - always ask user to choose."""
        default_folders = self.config.ingestion.watch_folders
        
        print("\n📁 Folder Selection")
        print("Choose folders to watch for documents to ingest:")
        
        # Show default options
        existing_defaults = [f for f in default_folders if Path(f).exists()]
        if existing_defaults:
            print(f"\nDefault options:")
            for i, folder in enumerate(existing_defaults, 1):
                print(f"  {i}. {folder}")
            print(f"  {len(existing_defaults)+1}. Custom folder(s)")
            
            choice = input(f"\nChoice [1-{len(existing_defaults)+1}]: ").strip()
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(existing_defaults):
                    selected_folder = existing_defaults[choice_num - 1]
                    print(f"✅ Selected: {selected_folder}")
                    watch_folders = [selected_folder]
                else:
                    # Custom folders
                    watch_folders = self._get_custom_folders()
            except ValueError:
                # Invalid input, go to custom
                watch_folders = self._get_custom_folders()
        else:
            # No defaults exist, must choose custom
            watch_folders = self._get_custom_folders()
        
        # Ask about recursive watching
        recursive_input = input("\n🔄 Watch folders recursively? [Y/n]: ").strip().lower()
        recursive = recursive_input != 'n'
        
        self.config.ingestion.recursive = recursive
        print(f"    {'✅ Recursive' if recursive else '❌ Non-recursive'} watching enabled")
        
        return watch_folders
    
    def _get_custom_folders(self) -> List[str]:
        """Get custom folder paths from user input."""
        print("\nEnter custom folder paths:")
        folders = []
        while True:
            if folders:
                folder_input = input(f"  Folder {len(folders)+1} (or press Enter to finish): ").strip()
            else:
                folder_input = input(f"  Folder path: ").strip()
            
            if not folder_input:
                if folders:
                    break
                else:
                    print("❌ At least one folder is required!")
                    continue
            
            # Expand user path (~ to home directory)
            folder_path = Path(folder_input).expanduser().resolve()
            
            if folder_path.exists() and folder_path.is_dir():
                folders.append(str(folder_path))
                print(f"    ✅ Added: {folder_path}")
            else:
                print(f"    ❌ Invalid folder: {folder_path}")
                create_choice = input(f"    Create folder? [y/N]: ").strip().lower()
                if create_choice == 'y':
                    try:
                        folder_path.mkdir(parents=True, exist_ok=True)
                        folders.append(str(folder_path))
                        print(f"    ✅ Created and added: {folder_path}")
                    except Exception as e:
                        print(f"    ❌ Failed to create folder: {e}")
        
        return folders
    
    def start_background_processes(self, watch_folders: List[str]):
        """Start the background watcher and ingestion processes."""
        print("\n🔧 Starting background processes...")
        
        try:
            # Start folder watcher
            print("  📁 Starting folder watcher...")
            self.process_manager.start_watcher_process(
                watch_folders, 
                self.config.ingestion.recursive
            )
            
            # Start ingestion worker
            print("  ⚡ Starting ingestion worker...")
            self.process_manager.start_ingestion_process(self.config)
            
            # Wait a moment for processes to start
            time.sleep(2)
            
            # Check if processes started successfully
            status = self.process_manager.get_system_status()
            watcher_status = status['processes'].get('watcher', {}).get('status', 'unknown')
            ingestion_status = status['processes'].get('ingestion', {}).get('status', 'unknown')
            
            if watcher_status == 'running' and ingestion_status == 'running':
                print("  ✅ Background processes started successfully")
                return True
            else:
                print(f"  ❌ Process startup failed: watcher={watcher_status}, ingestion={ingestion_status}")
                return False
                
        except Exception as e:
            print(f"  ❌ Failed to start background processes: {e}")
            return False
    
    def initialize_search(self):
        """Initialize the search interface."""
        print("\n🧠 Initializing search interface...")
        
        try:
            # Create LightRAG core for main process
            self.lightrag_core = create_lightrag_core(self.config)
            
            # Create search interface
            self.search_interface = SearchInterface(self.config, self.lightrag_core)
            
            print("  ✅ Search interface ready")
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to initialize search: {e}")
            return False
    
    def start_status_monitor(self):
        """Start background thread to monitor process status."""
        def status_monitor():
            """Background status monitoring."""
            last_progress_display = 0
            
            while self.running:
                try:
                    # Update process status
                    self.process_manager.update_status()
                    
                    # Check for ingestion progress
                    progress = self.process_manager.get_ingestion_progress()
                    if progress and progress.total_files > 0:
                        current_time = time.time()
                        
                        # Show progress every 5 seconds
                        if current_time - last_progress_display > 5:
                            percent = (progress.processed_files / progress.total_files) * 100
                            print(f"\n📊 Ingestion Progress: {progress.processed_files}/{progress.total_files} ({percent:.1f}%)")
                            print(f"    Current: {progress.current_file}")
                            print(f"    Rate: {progress.processing_rate:.1f} files/min")
                            if progress.estimated_remaining > 0:
                                print(f"    ETA: {progress.estimated_remaining:.1f} minutes")
                            if progress.errors > 0:
                                print(f"    Errors: {progress.errors}")
                            print("HybridRAG> ", end="", flush=True)
                            
                            last_progress_display = current_time
                    
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"\n⚠️ Status monitor error: {e}")
                    time.sleep(5)
        
        self.status_thread = threading.Thread(target=status_monitor, daemon=True)
        self.status_thread.start()
    
    def interactive_query_loop(self):
        """Main interactive query loop."""
        print("\n" + "="*60)
        print("🔍 HybridRAG Interactive Query Interface")
        print("="*60)
        print("Commands:")
        print("  <query>                  - Ask a question")
        print("  agentic <query>          - Multi-hop reasoning query") 
        print("  multi <query1> | <query2> - Multiple query search")
        print("  status                   - Show system status")
        print("  stats                    - Show search statistics")
        print("  history                  - Show search history")
        print("  health                   - System health check")
        print("  clear                    - Clear screen")
        print("  help                     - Show this help")
        print("  quit / exit              - Exit application")
        print("="*60)
        
        self.running = True
        self.start_status_monitor()
        
        while self.running:
            try:
                user_input = input("\nHybridRAG> ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'help':
                    self.show_help()
                elif user_input.lower() == 'clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                elif user_input.lower() == 'status':
                    self.show_system_status()
                elif user_input.lower() == 'stats':
                    self.show_search_stats()
                elif user_input.lower() == 'history':
                    self.show_search_history()
                elif user_input.lower() == 'health':
                    self.show_health_status()
                elif user_input.startswith('agentic '):
                    query = user_input[8:].strip()
                    if query:
                        self.handle_agentic_query(query)
                    else:
                        print("❌ Usage: agentic <query>")
                elif user_input.startswith('multi '):
                    query = user_input[6:].strip()
                    if query:
                        self.handle_multi_query(query)
                    else:
                        print("❌ Usage: multi <query1> | <query2> | ...")
                else:
                    # Regular search query
                    self.handle_simple_query(user_input)
                    
            except KeyboardInterrupt:
                print("\n\n🛑 Ctrl+C pressed, shutting down...")
                break
            except EOFError:
                print("\n\n👋 EOF received, shutting down...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def handle_simple_query(self, query: str):
        """Handle a simple search query."""
        print(f"🔍 Searching: {query}")
        
        try:
            import asyncio
            result = asyncio.run(self.search_interface.simple_search(query))
            self.display_search_result(result)
        except Exception as e:
            print(f"❌ Search failed: {e}")
    
    def handle_agentic_query(self, query: str):
        """Handle an agentic search query."""
        print(f"🧠 Agentic search: {query}")
        
        try:
            import asyncio
            result = asyncio.run(self.search_interface.agentic_search(query))
            self.display_search_result(result)
        except Exception as e:
            print(f"❌ Agentic search failed: {e}")
    
    def handle_multi_query(self, query: str):
        """Handle multiple query search."""
        queries = [q.strip() for q in query.split('|')]
        print(f"🔀 Multi-query search: {len(queries)} queries")
        
        try:
            import asyncio
            result = asyncio.run(self.search_interface.multi_query_search(queries))
            self.display_search_result(result)
        except Exception as e:
            print(f"❌ Multi-query search failed: {e}")
    
    def display_search_result(self, result):
        """Display a formatted search result."""
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
            
            if result.agentic_steps:
                print(f"\n🧠 Reasoning steps: {len(result.agentic_steps)}")
            
            print(f"\n💯 Confidence: {result.confidence_score:.1%}")
    
    def show_help(self):
        """Show help information."""
        print("\n📖 HybridRAG Help")
        print("─" * 30)
        print("Query Types:")
        print("  • Simple: Just type your question")
        print("  • Agentic: 'agentic <query>' for multi-hop reasoning")
        print("  • Multi: 'multi <q1> | <q2>' for multiple queries")
        print("\nQuery Modes (automatic selection):")
        print("  • Local: Focus on specific entities")
        print("  • Global: High-level overviews")
        print("  • Hybrid: Combined approach (default)")
        print("\nSystem Commands:")
        print("  • status: Process status")
        print("  • stats: Search statistics")
        print("  • health: System health")
        print("  • clear: Clear screen")
    
    def show_system_status(self):
        """Show detailed system status."""
        print("\n📊 System Status")
        print("─" * 40)
        
        status = self.process_manager.get_system_status()
        
        # Process status
        for name, proc_status in status['processes'].items():
            status_emoji = {"running": "✅", "error": "❌", "stopped": "⚪"}.get(proc_status['status'], "❓")
            print(f"{status_emoji} {name.capitalize()}: {proc_status['status']} (PID: {proc_status['pid']})")
            
            if proc_status.get('error'):
                print(f"    Error: {proc_status['error']}")
            
            # Show relevant stats
            stats = proc_status.get('stats', {})
            if name == 'watcher' and stats:
                print(f"    Files found: {stats.get('files_found', 0)}")
                print(f"    Total tracked: {stats.get('total_tracked', 0)}")
            elif name == 'ingestion' and stats:
                print(f"    Processed: {stats.get('processed_files', 0)}")
                print(f"    Pending: {stats.get('pending_files', 0)}")
                print(f"    Errors: {stats.get('error_files', 0)}")
        
        # Shared state
        shared = status['shared_state']
        print(f"\n📈 Overall Stats:")
        print(f"  Files found: {shared.get('total_files_found', 0)}")
        print(f"  Files processed: {shared.get('total_files_processed', 0)}")
        print(f"  Ingestion active: {'Yes' if shared.get('ingestion_active') else 'No'}")
        print(f"  LightRAG ready: {'Yes' if shared.get('lightrag_ready') else 'No'}")
        
        # Queue sizes
        queues = status['queue_sizes']
        print(f"\n📋 Queue Status:")
        print(f"  Files to process: {queues.get('watcher_to_ingestion', 0)}")
        print(f"  Progress updates: {queues.get('ingestion_to_main', 0)}")
    
    def show_search_stats(self):
        """Show search statistics."""
        if not self.search_interface:
            print("❌ Search interface not ready")
            return
        
        print("\n📈 Search Statistics")
        print("─" * 40)
        
        stats = self.search_interface.get_stats()
        print(f"Total queries: {stats['total_queries']}")
        print(f"Successful: {stats['successful_queries']}")
        print(f"Avg execution time: {stats['avg_execution_time']:.2f}s")
        print(f"Avg confidence: {stats['avg_confidence']:.1%}")
        print(f"Agentic enabled: {stats['agentic_enabled']}")
        
        if stats['mode_distribution']:
            print("\nMode distribution:")
            for mode, count in stats['mode_distribution'].items():
                print(f"  {mode}: {count}")
    
    def show_search_history(self):
        """Show search history."""
        if not self.search_interface:
            print("❌ Search interface not ready")
            return
        
        print("\n📜 Search History")
        print("─" * 40)
        
        history = self.search_interface.get_search_history(limit=10)
        if not history:
            print("No search history yet.")
            return
        
        for i, query in enumerate(history, 1):
            status = "✅" if not query['has_error'] else "❌"
            print(f"{i:2d}. {status} {query['query'][:50]}...")
            print(f"     Mode: {query['mode']}, Time: {query['execution_time']:.2f}s")
    
    def show_health_status(self):
        """Show system health."""
        if not self.lightrag_core:
            print("❌ LightRAG not ready")
            return
        
        print("\n🏥 System Health Check")
        print("─" * 40)
        
        try:
            import asyncio
            health = asyncio.run(self.lightrag_core.health_check())
            
            status_emoji = {"healthy": "✅", "degraded": "⚠️", "unhealthy": "❌"}.get(health['status'], "❓")
            print(f"Overall: {status_emoji} {health['status'].upper()}")
            
            for check, status in health.get('checks', {}).items():
                check_emoji = "✅" if status == "ok" else "❌"
                print(f"{check_emoji} {check}: {status}")
            
            if health.get('error'):
                print(f"❌ Error: {health['error']}")
                
        except Exception as e:
            print(f"❌ Health check failed: {e}")
    
    def shutdown(self):
        """Shutdown the application."""
        print("\n🛑 Shutting down HybridRAG...")
        
        self.running = False
        
        # Wait for status thread
        if self.status_thread and self.status_thread.is_alive():
            self.status_thread.join(timeout=2)
        
        # Shutdown process manager
        self.process_manager.shutdown()
        
        print("👋 HybridRAG shutdown complete")
    
    def run(self):
        """Run the application."""
        try:
            # Setup signal handlers
            setup_signal_handlers(self.process_manager)
            
            # Setup folders
            watch_folders = self.setup_folders()
            
            # Start background processes
            if not self.start_background_processes(watch_folders):
                print("❌ Failed to start background processes")
                return 1
            
            # Initialize search
            if not self.initialize_search():
                print("❌ Failed to initialize search")
                return 1
            
            print("\n✅ HybridRAG system ready!")
            print("💡 Tip: Try asking 'What documents are available?' to test the system")
            
            # Start interactive loop
            self.interactive_query_loop()
            
        except KeyboardInterrupt:
            print("\n\n🛑 Interrupted by user")
        except Exception as e:
            print(f"\n❌ Application error: {e}")
            return 1
        finally:
            self.shutdown()
        
        return 0

def main():
    """Main entry point."""
    app = HybridRAGApp()
    return app.run()

if __name__ == "__main__":
    import sys
    sys.exit(main())