import asyncio
from crawl4ai import AsyncWebCrawler
import tiktoken
import base64
import requests
from urllib.parse import urljoin
import argparse
import contextlib
import sys
import subprocess
import signal
import os
import time

class AdvancedPageScraper:
    def __init__(self, include_images=True):
        self.include_images = include_images
        self.crawler = None  # Initialize crawler to None
        self.crawler_initialized = False

    @staticmethod
    def count_tokens(text):
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        return len(tokens)

    @staticmethod
    async def convert_image_to_base64(image_url):
        try:
            response = requests.get(image_url)
            if response.status_code == 200:
                return base64.b64encode(response.content).decode('utf-8')
        except Exception as e:
            print(f"[ERROR] Failed to convert image {image_url}: {e}", file=sys.stderr)
        return None

    def _kill_playwright_processes(self):
        """Kill any lingering Playwright/Chromium processes that might interfere."""
        try:
            # Kill chromium/chrome processes that might be lingering
            if os.name == 'posix':  # Linux/macOS
                subprocess.run(['pkill', '-f', 'chromium'], capture_output=True)
                subprocess.run(['pkill', '-f', 'chrome'], capture_output=True)
                subprocess.run(['pkill', '-f', 'playwright'], capture_output=True)
            elif os.name == 'nt':  # Windows
                subprocess.run(['taskkill', '/F', '/IM', 'chrome.exe'], capture_output=True)
                subprocess.run(['taskkill', '/F', '/IM', 'chromium.exe'], capture_output=True)
            print("[INFO] Cleaned up any lingering browser processes", file=sys.stderr)
        except Exception as e:
            print(f"[WARNING] Error cleaning up processes: {e}", file=sys.stderr)

    async def _initialize_crawler(self):
        """Initializes the AsyncWebCrawler if it hasn't been already."""
        if not self.crawler_initialized:
            print("[INFO] Cleaning up any lingering processes before initialization...", file=sys.stderr)
            self._kill_playwright_processes()
            time.sleep(1)  # Give processes time to fully terminate
            
            print("[INFO] Initializing new browser instance...", file=sys.stderr)
            self.crawler = AsyncWebCrawler(verbose=False)
            try:
                await self.crawler.__aenter__()
                self.crawler_initialized = True
                print("[INFO] Browser initialized successfully", file=sys.stderr)
            except Exception as e:
                print(f"[ERROR] Failed to initialize crawler: {e}", file=sys.stderr)
                print("[INFO] Attempting aggressive cleanup and retry...", file=sys.stderr)
                self._kill_playwright_processes()
                time.sleep(2)  # Wait longer for cleanup
                self.crawler = None  # Ensure crawler is None if initialization fails
                self.crawler_initialized = False
                raise  # Re-raise the exception to signal initialization failure

    async def _cleanup_crawler(self):
        """Safely cleans up the crawler."""
        if self.crawler:
            print("[INFO] Cleaning up browser instance...", file=sys.stderr)
            try:
                await self.crawler.__aexit__(None, None, None)
                print("[INFO] Browser cleanup completed successfully", file=sys.stderr)
            except Exception as e:
                print(f"[ERROR] Error during crawler cleanup: {e}", file=sys.stderr)
            finally:
                self.crawler = None
                self.crawler_initialized = False
                # Force cleanup of any lingering processes
                self._kill_playwright_processes()
                time.sleep(0.5)  # Brief pause to ensure cleanup

    async def close(self):
        """Closes the crawler when the AdvancedPageScraper is no longer needed."""
        await self._cleanup_crawler()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def scrape_url(self, url):
        """Scrapes a URL using a persistent AsyncWebCrawler instance."""
        print(f"\n[INFO] Starting scrape of: {url}", file=sys.stderr)
        print(f"[INFO] Image processing is {'enabled' if self.include_images else 'disabled'}", file=sys.stderr)

        if not self.crawler_initialized:
            try:
                await self._initialize_crawler()
            except Exception:
                # Initialization failed, return a default result
                return {'content': None, 'links': [], 'token_count': 0}

        if not self.crawler:
            print("[ERROR] Crawler is not initialized.", file=sys.stderr)
            return {'content': None, 'links': [], 'token_count': 0}

        try:
            result = await self.crawler.arun(url=url, bypass_cache=True, magic=True, timeout=60000)
        except Exception as e:
            print(f"[ERROR] Failed to crawl {url}: {e}", file=sys.stderr)
            # Attempt to recover from crawler errors by re-initializing
            print("[WARNING] Browser crash detected, attempting recovery...", file=sys.stderr)
            await self._cleanup_crawler()  # Clean up the potentially broken crawler
            try:
                await self._initialize_crawler()  # Try to re-initialize
            except Exception as reinit_e:
                print(f"[ERROR] Failed to re-initialize crawler: {reinit_e}", file=sys.stderr)
                return {'content': None, 'links': [], 'token_count': 0}  # Return default result if re-initialization fails
            if self.crawler:
                try:
                    result = await self.crawler.arun(url=url, bypass_cache=True, magic=True, timeout=60000)
                except Exception as e2:
                    print(f"[ERROR] Failed to crawl {url} after re-initialization: {e2}", file=sys.stderr)
                    return {'content': None, 'links': [], 'token_count': 0}
            else:
                return {'content': None, 'links': [], 'token_count': 0}

        # Try to get the main content
        fit_content = getattr(result, 'fit_markdown', None)
        if fit_content is None:
            fit_content = getattr(result, 'markdown', None)
        if fit_content is None:
            fit_content = getattr(result, 'html', None)

        if not fit_content:
            print("[WARNING] No content was extracted from the page!", file=sys.stderr)

        # Optionally process images
        if self.include_images and hasattr(result, 'media') and result.media:
            images = result.media.get("images", [])
            print(f"[DEBUG] Processing {len(images)} images", file=sys.stderr)
            for image in images:
                base64_data = await self.convert_image_to_base64(image['src'])
                if base64_data:
                    img_markdown = f"![{image.get('alt', '')}]({image['src']})"
                    base64_markdown = f"![{image.get('alt', '')}](data:image/png;base64,{base64_data})"
                    fit_content = fit_content.replace(img_markdown, base64_markdown)

        # Get links (internal and external)
        links = []
        if hasattr(result, 'links') and result.links:
            for link_type in result.links:
                for link in result.links[link_type]:
                    href = link.get('href')
                    if href:
                        links.append(urljoin(url, href))

        token_count = self.count_tokens(fit_content) if fit_content else 0
        print(f"[INFO] Scraping completed:", file=sys.stderr)
        print(f"[INFO] - Content length: {len(fit_content) if fit_content else 0} characters", file=sys.stderr)
        print(f"[INFO] - Token count: {token_count}", file=sys.stderr)
        print(f"[INFO] - Links found: {len(links)}", file=sys.stderr)

        return {
            'content': fit_content,
            'links': links,
            'token_count': token_count
        }

async def interactive_mode():
    """Interactive menu for the scraper"""
    print("\n=== Web Scraper Interactive Mode ===")

    # Get URL
    url = input("\nEnter the URL to scrape: ").strip()

    # Image processing choice
    while True:
        choice = input("\nInclude images? (y/n): ").lower()
        if choice in ['y', 'n']:
            include_images = (choice == 'y')
            break
        print("Please enter 'y' or 'n'")

    print("\nStarting scrape...")
    async with AdvancedPageScraper(include_images=include_images) as scraper:
        result = await scraper.scrape_url(url)

    # Output options
    while True:
        print("\nWhat would you like to see?", file=sys.stderr)
        print("1. Content", file=sys.stderr)
        print("2. Token Count", file=sys.stderr)
        print("3. Discovered Links", file=sys.stderr)
        print("4. All", file=sys.stderr)
        print("5. Exit", file=sys.stderr)

        choice = input("\nEnter your choice (1-5): ")

        if choice == '1':
            print("\n=== Content ===", file=sys.stderr)
            print(result['content'], file=sys.stderr)
        elif choice == '2':
            print("\n=== Token Count ===", file=sys.stderr)
            print(f"Number of tokens: {result['token_count']}", file=sys.stderr)
        elif choice == '3':
            print("\n=== Discovered Links ===", file=sys.stderr)
            for link in result['links']:
                print(link, file=sys.stderr)
        elif choice == '4':
            print("\n=== Content ===", file=sys.stderr)
            print(result['content'], file=sys.stderr)
            print("\n=== Token Count ===", file=sys.stderr)
            print(f"Number of tokens: {result['token_count']}", file=sys.stderr)
            print("\n=== Discovered Links ===", file=sys.stderr)
            for link in result['links']:
                print(link, file=sys.stderr)
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.", file=sys.stderr)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, help='URL to scrape')
    parser.add_argument('--no-images', action='store_true', help='Disable image processing')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    args = parser.parse_args()

    if args.interactive:
        await interactive_mode()
    elif args.url:
        async with AdvancedPageScraper(include_images=not args.no_images) as scraper:
            result = await scraper.scrape_url(args.url)

            print("Extracted Content:", result['content'], file=sys.stderr)
            print(f"Number of tokens: {result['token_count']}", file=sys.stderr)
            print("Discovered Links:", file=sys.stderr)
            for link in result['links']:
                print(link, file=sys.stderr)
    else:
        print("Error: Please provide --url or use --interactive mode", file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(main())

