#!/usr/bin/env python3
"""
LightRAG Query Demo Interface
==============================
Interactive demo for querying the Athena medical database using LightRAG.

Author: HybridRAG System
Date: 2024
"""

import asyncio
import logging
from typing import Optional
from pathlib import Path
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from dotenv import load_dotenv
import os
from colorama import init, Fore, Style

# Initialize colorama for colored output
init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class LightRAGQueryInterface:
    """Interactive query interface for LightRAG Athena database."""
    
    def __init__(self, working_dir: str = "./athena_lightrag_db"):
        """
        Initialize the query interface.
        
        Args:
            working_dir: Path to LightRAG database
        """
        self.working_dir = working_dir
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Check if database exists
        if not Path(working_dir).exists():
            raise FileNotFoundError(f"LightRAG database not found at {working_dir}. Run ingestion first.")
        
        self._init_lightrag()
        self.rag_initialized = False
    
    def _init_lightrag(self):
        """Initialize LightRAG instance."""
        logger.info(f"Loading LightRAG database from {self.working_dir}")
        
        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=lambda prompt, system_prompt=None, history_messages=[], **kwargs: 
                openai_complete_if_cache(
                    "gpt-4o-mini",
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=self.api_key,
                    **kwargs
                ),
            embedding_func=EmbeddingFunc(
                embedding_dim=1536,
                func=lambda texts: openai_embed(
                    texts,
                    model="text-embedding-ada-002",
                    api_key=self.api_key
                ),
            ),
        )
        logger.info("LightRAG loaded successfully")
    
    async def _ensure_initialized(self):
        """Ensure LightRAG storages are initialized."""
        if not self.rag_initialized:
            logger.info("Initializing LightRAG storages for first query...")
            await self.rag.initialize_storages()
            self.rag_initialized = True
            logger.info("LightRAG storages initialized successfully")
    
    async def query(
        self, 
        query_text: str, 
        mode: str = "hybrid",
        only_need_context: bool = False,
        top_k: int = 60,
        max_entity_tokens: int = 6000,
        max_relation_tokens: int = 8000
    ):
        """
        Execute a query against the LightRAG database.
        
        Args:
            query_text: The query string
            mode: Query mode (local, global, hybrid, naive)
            only_need_context: If True, only return context without generating response
            top_k: Number of top items to retrieve
            max_entity_tokens: Maximum tokens for entity context
            max_relation_tokens: Maximum tokens for relationship context
            
        Returns:
            Query results
        """
        # Ensure storages are initialized
        await self._ensure_initialized()
        
        # Create QueryParam object
        query_param = QueryParam(
            mode=mode,
            only_need_context=only_need_context,
            top_k=top_k,
            max_entity_tokens=max_entity_tokens,
            max_relation_tokens=max_relation_tokens,
            response_type="Multiple Paragraphs"
        )
        
        try:
            result = await self.rag.aquery(query_text, param=query_param)
            return result
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return None
    
    def print_header(self):
        """Print the demo header."""
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"{Fore.CYAN}       LightRAG Athena Medical Database Query Interface")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}Available query modes:{Style.RESET_ALL}")
        print(f"  • {Fore.GREEN}local{Style.RESET_ALL}   - Focus on specific entity relationships")
        print(f"  • {Fore.GREEN}global{Style.RESET_ALL}  - High-level overview and summaries")
        print(f"  • {Fore.GREEN}hybrid{Style.RESET_ALL}  - Combination of local and global (default)")
        print(f"  • {Fore.GREEN}naive{Style.RESET_ALL}   - Simple retrieval without graph structure")
        print(f"\n{Fore.YELLOW}Commands:{Style.RESET_ALL}")
        print(f"  • Type your query and press Enter")
        print(f"  • Use '/mode <mode>' to change query mode")
        print(f"  • Use '/context' to get only context without response")
        print(f"  • Use '/help' for more commands")
        print(f"  • Type 'exit' or 'quit' to leave\n")
    
    def print_help(self):
        """Print help information."""
        print(f"\n{Fore.CYAN}=== Help ==={Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}Query Examples:{Style.RESET_ALL}")
        print("  • What tables are related to patient appointments?")
        print("  • Describe the anesthesia case management tables")
        print("  • Show me all collector category tables")
        print("  • What is the structure of allowable schedule categories?")
        print("  • List tables with high row counts")
        print(f"\n{Fore.YELLOW}Commands:{Style.RESET_ALL}")
        print(f"  {Fore.GREEN}/mode <mode>{Style.RESET_ALL}  - Change query mode (local/global/hybrid/naive)")
        print(f"  {Fore.GREEN}/context{Style.RESET_ALL}      - Toggle context-only mode")
        print(f"  {Fore.GREEN}/topk <n>{Style.RESET_ALL}     - Set number of results (default: 60)")
        print(f"  {Fore.GREEN}/clear{Style.RESET_ALL}        - Clear screen")
        print(f"  {Fore.GREEN}/help{Style.RESET_ALL}         - Show this help")
        print(f"  {Fore.GREEN}exit/quit{Style.RESET_ALL}     - Exit the demo\n")
    
    async def run_interactive(self):
        """Run the interactive query interface."""
        self.print_header()
        
        # Default settings
        current_mode = "hybrid"
        context_only = False
        top_k = 60
        
        while True:
            try:
                # Get user input
                prompt = f"{Fore.BLUE}[{current_mode}]{Style.RESET_ALL} > "
                user_input = input(prompt).strip()
                
                # Check for exit
                if user_input.lower() in ['exit', 'quit']:
                    print(f"\n{Fore.YELLOW}Thank you for using LightRAG Query Interface!{Style.RESET_ALL}")
                    break
                
                # Check for commands
                if user_input.startswith('/'):
                    parts = user_input.split()
                    command = parts[0].lower()
                    
                    if command == '/help':
                        self.print_help()
                        continue
                    
                    elif command == '/clear':
                        os.system('clear' if os.name == 'posix' else 'cls')
                        self.print_header()
                        continue
                    
                    elif command == '/mode' and len(parts) > 1:
                        new_mode = parts[1].lower()
                        if new_mode in ['local', 'global', 'hybrid', 'naive']:
                            current_mode = new_mode
                            print(f"{Fore.GREEN}✓ Query mode changed to: {current_mode}{Style.RESET_ALL}")
                        else:
                            print(f"{Fore.RED}✗ Invalid mode. Use: local, global, hybrid, or naive{Style.RESET_ALL}")
                        continue
                    
                    elif command == '/context':
                        context_only = not context_only
                        status = "enabled" if context_only else "disabled"
                        print(f"{Fore.GREEN}✓ Context-only mode {status}{Style.RESET_ALL}")
                        continue
                    
                    elif command == '/topk' and len(parts) > 1:
                        try:
                            top_k = int(parts[1])
                            print(f"{Fore.GREEN}✓ Top-K set to: {top_k}{Style.RESET_ALL}")
                        except ValueError:
                            print(f"{Fore.RED}✗ Invalid number{Style.RESET_ALL}")
                        continue
                    
                    else:
                        print(f"{Fore.RED}✗ Unknown command. Use /help for available commands{Style.RESET_ALL}")
                        continue
                
                # Skip empty queries
                if not user_input:
                    continue
                
                # Execute query
                print(f"\n{Fore.CYAN}Searching...{Style.RESET_ALL}")
                
                result = await self.query(
                    user_input,
                    mode=current_mode,
                    only_need_context=context_only,
                    top_k=top_k
                )
                
                if result:
                    print(f"\n{Fore.GREEN}=== Results ==={Style.RESET_ALL}")
                    print(result)
                    print(f"\n{Fore.GREEN}{'='*50}{Style.RESET_ALL}\n")
                else:
                    print(f"\n{Fore.RED}No results found or an error occurred.{Style.RESET_ALL}\n")
                
            except KeyboardInterrupt:
                print(f"\n\n{Fore.YELLOW}Interrupted. Type 'exit' to quit.{Style.RESET_ALL}")
                continue
            except Exception as e:
                print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}\n")
                logger.error(f"Interactive query error: {e}")


async def main():
    """Main function to run the query interface."""
    try:
        interface = LightRAGQueryInterface(working_dir="./athena_lightrag_db")
        await interface.run_interactive()
    except FileNotFoundError as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Please run the ingestion script first:{Style.RESET_ALL}")
        print(f"  python deeplake_to_lightrag.py")
    except Exception as e:
        print(f"{Fore.RED}Fatal error: {e}{Style.RESET_ALL}")
        logger.error(f"Fatal error: {e}")


if __name__ == "__main__":
    asyncio.run(main())