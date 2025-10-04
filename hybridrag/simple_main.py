#!/usr/bin/env python3
"""
Simple HybridRAG System
=======================
Simplified version that works properly with folder selection and shutdown.
"""

import os
import sys
import time
import signal
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config.config import load_config
from src.process_manager import ProcessManager

class SimpleHybridRAG:
    """Simplified HybridRAG system."""
    
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
    
    def choose_folders(self) -> List[str]:
        """Let user choose folders to watch."""
        print("🚀 HybridRAG System - Folder Setup")
        print("=" * 50)
        print("\n📁 Choose folders to watch for document ingestion:")
        print("\nOptions:")
        print("1. Use sample data folder (./data)")
        print("2. Choose custom folder(s)")
        
        while True:
            choice = input("\nChoice [1-2]: ").strip()
            
            if choice == "1":
                data_folder = Path("./data").resolve()
                if data_folder.exists():
                    print(f"✅ Using: {data_folder}")
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
        print("\n📂 Enter folder paths to watch:")
        print("(Supports ~ for home directory)")
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
    
    def run(self):
        """Run the system."""
        try:
            # Choose folders
            watch_folders = self.choose_folders()
            
            # Ask about recursive
            recursive_input = input("\n🔄 Watch recursively? [Y/n]: ").strip().lower()
            recursive = recursive_input != 'n'
            print(f"{'✅ Recursive' if recursive else '❌ Non-recursive'} watching")
            
            print("\n🔧 Starting processes...")
            
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
            
            if watcher_status == 'running' and ingestion_status == 'running':
                print("\n✅ System ready! Press Ctrl+C to exit.")
                
                # Simple monitoring loop
                while self.running:
                    time.sleep(5)
                    
                    # Show progress if available
                    progress = self.process_manager.get_ingestion_progress()
                    if progress and progress.total_files > 0:
                        print(f"📊 Processing: {progress.processed_files}/{progress.total_files} files")
                    
                    # Show status occasionally
                    status = self.process_manager.get_system_status()
                    shared = status['shared_state']
                    print(f"📈 Found: {shared.get('total_files_found', 0)}, Processed: {shared.get('total_files_processed', 0)}")
                    
            else:
                print("❌ Failed to start processes properly")
                
        except KeyboardInterrupt:
            print("\n🛑 Interrupted")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            print("🔄 Shutting down...")
            self.process_manager.shutdown()
            print("👋 Goodbye!")

def main():
    app = SimpleHybridRAG()
    app.run()

if __name__ == "__main__":
    main()