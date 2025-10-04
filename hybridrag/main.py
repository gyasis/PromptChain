#!/usr/bin/env python3
"""
HybridRAG Main Orchestration Script
==================================
Main entry point for the hybrid RAG system with ingestion and search capabilities.
"""

import asyncio
import logging
import argparse
import signal
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config.config import load_config, HybridRAGConfig
from src.folder_watcher import FolderWatcher
from src.ingestion_pipeline import IngestionPipeline
from src.lightrag_core import create_lightrag_core
from src.search_interface import SearchInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hybridrag.log')
    ]
)
logger = logging.getLogger(__name__)

class HybridRAGSystem:
    """Main orchestrator for the hybrid RAG system."""
    
    def __init__(self, config: HybridRAGConfig):
        """Initialize the hybrid RAG system."""
        self.config = config
        self.lightrag_core = None
        self.folder_watcher = None
        self.ingestion_pipeline = None
        self.search_interface = None
        self._shutdown_event = asyncio.Event()
        
        logger.info("HybridRAGSystem initializing...")
    
    async def initialize(self):
        """Initialize all components."""
        try:
            # Initialize LightRAG core
            logger.info("Initializing LightRAG core...")
            self.lightrag_core = create_lightrag_core(self.config)
            
            # Initialize folder watcher
            logger.info("Initializing folder watcher...")
            self.folder_watcher = FolderWatcher(self.config)
            
            # Initialize ingestion pipeline
            logger.info("Initializing ingestion pipeline...")
            self.ingestion_pipeline = IngestionPipeline(self.config, self.lightrag_core)
            
            # Initialize search interface
            logger.info("Initializing search interface...")
            self.search_interface = SearchInterface(self.config, self.lightrag_core)
            
            logger.info("HybridRAGSystem initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize HybridRAGSystem: {e}")
            raise
    
    async def start_ingestion_mode(self):
        """Start in ingestion mode - watch folders and process files."""
        logger.info("Starting ingestion mode")
        
        try:
            # Start both folder watcher and ingestion pipeline
            tasks = [
                asyncio.create_task(self.folder_watcher.start()),
                asyncio.create_task(self.ingestion_pipeline.start())
            ]
            
            # Wait for shutdown signal
            await self._shutdown_event.wait()
            
            # Cancel tasks
            for task in tasks:
                task.cancel()
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error in ingestion mode: {e}")
        finally:
            await self.shutdown()
    
    async def start_search_mode(self):
        """Start in search mode - interactive query interface."""
        logger.info("Starting search mode")
        
        try:
            print("\n" + "="*60)
            print("🔍 HybridRAG Search Interface")
            print("="*60)
            print("Commands:")
            print("  search <query>           - Simple search")
            print("  agentic <query>          - Multi-hop reasoning search")
            print("  multi <query1> | <query2> - Multiple query search")
            print("  stats                    - Show statistics")
            print("  history                  - Show search history")
            print("  health                   - System health check")
            print("  quit                     - Exit")
            print("="*60)
            
            while not self._shutdown_event.is_set():
                try:
                    user_input = input("\nHybridRAG> ").strip()
                    
                    if not user_input:
                        continue
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    await self._handle_search_command(user_input)
                    
                except KeyboardInterrupt:
                    break
                except EOFError:
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
            
        except Exception as e:
            logger.error(f"Error in search mode: {e}")
        finally:
            await self.shutdown()
    
    async def _handle_search_command(self, command: str):
        """Handle search commands."""
        parts = command.split(' ', 1)
        cmd = parts[0].lower()
        query = parts[1] if len(parts) > 1 else ""
        
        try:
            if cmd == 'search':
                if not query:
                    print("❌ Usage: search <query>")
                    return
                
                print(f"🔍 Searching: {query}")
                result = await self.search_interface.simple_search(query)
                self._print_search_result(result)
                
            elif cmd == 'agentic':
                if not query:
                    print("❌ Usage: agentic <query>")
                    return
                
                print(f"🧠 Agentic search: {query}")
                result = await self.search_interface.agentic_search(query)
                self._print_search_result(result)
                
            elif cmd == 'multi':
                if not query:
                    print("❌ Usage: multi <query1> | <query2> | ...")
                    return
                
                queries = [q.strip() for q in query.split('|')]
                print(f"🔀 Multi-query search: {len(queries)} queries")
                result = await self.search_interface.multi_query_search(queries)
                self._print_search_result(result)
                
            elif cmd == 'stats':
                stats = self.search_interface.get_stats()
                self._print_stats(stats)
                
            elif cmd == 'history':
                history = self.search_interface.get_search_history()
                self._print_history(history)
                
            elif cmd == 'health':
                health = await self.lightrag_core.health_check()
                self._print_health(health)
                
            else:
                print(f"❌ Unknown command: {cmd}")
                
        except Exception as e:
            print(f"❌ Command failed: {e}")
    
    def _print_search_result(self, result):
        """Print formatted search result."""
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
    
    def _print_stats(self, stats):
        """Print system statistics."""
        print(f"\n📈 System Statistics")
        print("─" * 40)
        print(f"Total queries: {stats['total_queries']}")
        print(f"Successful: {stats['successful_queries']}")
        print(f"Avg execution time: {stats['avg_execution_time']:.2f}s")
        print(f"Avg confidence: {stats['avg_confidence']:.1%}")
        print(f"Agentic enabled: {stats['agentic_enabled']}")
        
        if stats['mode_distribution']:
            print("\nMode distribution:")
            for mode, count in stats['mode_distribution'].items():
                print(f"  {mode}: {count}")
    
    def _print_history(self, history):
        """Print search history."""
        print(f"\n📜 Search History ({len(history)} queries)")
        print("─" * 60)
        
        for i, query in enumerate(history[-10:], 1):
            status = "✅" if not query['has_error'] else "❌"
            print(f"{i:2d}. {status} {query['query'][:50]}...")
            print(f"     Mode: {query['mode']}, Time: {query['execution_time']:.2f}s")
    
    def _print_health(self, health):
        """Print system health."""
        status_emoji = {"healthy": "✅", "degraded": "⚠️", "unhealthy": "❌"}
        emoji = status_emoji.get(health['status'], "❓")
        
        print(f"\n🏥 System Health: {emoji} {health['status'].upper()}")
        print("─" * 40)
        
        for check, status in health.get('checks', {}).items():
            check_emoji = "✅" if status == "ok" else "❌"
            print(f"{check_emoji} {check}: {status}")
        
        if health.get('error'):
            print(f"❌ Error: {health['error']}")
    
    async def one_shot_query(self, query: str, mode: str = "simple") -> Dict[str, Any]:
        """Execute a single query and return result."""
        try:
            if mode == "agentic":
                result = await self.search_interface.agentic_search(query)
            else:
                result = await self.search_interface.simple_search(query)
            
            return {
                "query": result.query,
                "result": result.result,
                "mode": result.mode,
                "execution_time": result.execution_time,
                "confidence_score": result.confidence_score,
                "sources": result.sources,
                "error": result.error
            }
            
        except Exception as e:
            return {
                "query": query,
                "result": "",
                "error": str(e)
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status."""
        try:
            status = {
                "lightrag": self.lightrag_core.get_stats() if self.lightrag_core else None,
                "search": self.search_interface.get_stats() if self.search_interface else None,
                "ingestion": self.ingestion_pipeline.get_stats() if self.ingestion_pipeline else None,
                "watcher": self.folder_watcher.get_queue_stats() if self.folder_watcher else None
            }
            
            if self.lightrag_core:
                health = await self.lightrag_core.health_check()
                status["health"] = health
            
            return status
            
        except Exception as e:
            return {"error": str(e)}
    
    async def shutdown(self):
        """Shutdown all components gracefully."""
        logger.info("Shutting down HybridRAGSystem...")
        
        try:
            if self.folder_watcher:
                await self.folder_watcher.stop()
            
            if self.ingestion_pipeline:
                await self.ingestion_pipeline.stop()
                
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        logger.info("HybridRAGSystem shutdown complete")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self._shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="HybridRAG System")
    parser.add_argument(
        "mode", 
        choices=["ingestion", "search", "query", "status"],
        help="Operation mode"
    )
    parser.add_argument(
        "--query", 
        help="Query for one-shot mode"
    )
    parser.add_argument(
        "--agentic", 
        action="store_true",
        help="Use agentic search for query mode"
    )
    parser.add_argument(
        "--config", 
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create system
    system = HybridRAGSystem(config)
    
    try:
        await system.initialize()
        
        if args.mode == "ingestion":
            system.setup_signal_handlers()
            await system.start_ingestion_mode()
            
        elif args.mode == "search":
            system.setup_signal_handlers()
            await system.start_search_mode()
            
        elif args.mode == "query":
            if not args.query:
                print("❌ Query mode requires --query parameter")
                return 1
            
            mode = "agentic" if args.agentic else "simple"
            result = await system.one_shot_query(args.query, mode)
            print(json.dumps(result, indent=2))
            
        elif args.mode == "status":
            status = await system.get_system_status()
            print(json.dumps(status, indent=2))
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        return 1
    finally:
        await system.shutdown()
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))