#!/usr/bin/env python3
"""
Quick Test - Just folder watching and file detection
"""

import os
import sys
import time
import signal
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config.config import load_config
from src.process_manager import ProcessManager

def main():
    print("🧪 Quick Test - Folder Watching Only")
    print("=" * 50)
    
    # Use sample data folder for testing
    folder_input = "./data"
    print(f"Using test folder: {folder_input}")
    
    folder_path = Path(folder_input).expanduser().resolve()
    if not folder_path.exists():
        print(f"❌ Folder doesn't exist: {folder_path}")
        return
    
    print(f"✅ Watching: {folder_path}")
    
    # Setup process manager
    config = load_config()
    process_manager = ProcessManager()
    
    def signal_handler(signum, frame):
        print(f"\n🛑 Shutting down...")
        process_manager.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Start only watcher process
        process_manager.start_watcher_process([str(folder_path)], recursive=True)
        print("✅ Folder watcher started")
        
        time.sleep(2)
        
        # Monitor for 30 seconds
        print("\n📁 Monitoring for file changes (30 seconds)...")
        print("Add/modify files in the watched folder to see detection")
        
        for i in range(30):
            status = process_manager.get_system_status()
            watcher_status = status['processes'].get('watcher', {})
            
            if watcher_status.get('status') == 'running':
                stats = watcher_status.get('stats', {})
                files_found = stats.get('files_found', 0)
                total_tracked = stats.get('total_tracked', 0)
                
                print(f"\r⏱️  {30-i}s | Files found: {files_found} | Total tracked: {total_tracked}", end="", flush=True)
            else:
                print(f"\r❌ Watcher status: {watcher_status.get('status', 'unknown')}", end="", flush=True)
            
            time.sleep(1)
        
        print(f"\n\n📊 Final Results:")
        final_status = process_manager.get_system_status()
        shared = final_status['shared_state']
        queues = final_status['queue_sizes']
        
        print(f"  Files found: {shared.get('total_files_found', 0)}")
        print(f"  Files queued: {queues.get('watcher_to_ingestion', 0)}")
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
    finally:
        process_manager.shutdown()
        print("👋 Test complete!")

if __name__ == "__main__":
    main()