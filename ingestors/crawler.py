import asyncio
import json
import os
import re
import argparse
from crawl4ai import AsyncWebCrawler
from urllib.parse import urljoin
from tqdm import tqdm  # Import tqdm

class CustomCrawler:
    def __init__(self, start_url, output_dir):
        self.start_url = start_url
        self.output_dir = output_dir

    async def crawl(self):
        # Returns (fit_content, links) for the given start_url
        async with AsyncWebCrawler(verbose=True) as crawler:
            return await self._crawl_page(crawler, self.start_url)

    async def _crawl_page(self, crawler, url):
        # Ensure the directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Generate filename and truncate if necessary
        filename = os.path.join(self.output_dir, re.sub(r'[^\w\-_\. ]', '-', url)[:255] + ".md")

        # Check if the file already exists before crawling
        if os.path.exists(filename):
            print(f"[ALERT] File {filename} already exists. Returning no new links.")
            # Return no content and no links
            return None, []

        # Execute JavaScript if needed
        js_code = [
            "const loadMoreButton = Array.from(document.querySelectorAll('button')).find(button => button.textContent.includes('Load More')); loadMoreButton && loadMoreButton.click();"
        ]

        # Crawl the page with magic mode enabled and increased timeout
        try:
            result = await crawler.arun(url=url, js_code=js_code, bypass_cache=True, magic=True, timeout=60000)
        except Exception as e:
            print(f"[ERROR] Failed to crawl {url}: {e}")
            return None, []

        # Extract links and content
        links = result.links
        fit_content = result.fit_markdown  # Main content in Markdown format

        if fit_content is not None:
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(fit_content)
        else:
            print(f"[ALERT] No fit content extracted from {url}. Not saving file.")

        # Resolve relative links to absolute URLs
        internal_links = links.get("internal", [])
        absolute_links = [urljoin(url, link['href']) for link in internal_links]

        return fit_content, absolute_links


class Crawler:
    STATE_FILE = 'crawler_state.json'

    def __init__(self, output_dir='./output', headless=True, workers=2, skip_proxy=True, initial_links_file='initial_links.json', filter_str=''):
        self.output_dir = output_dir
        self.headless = headless
        self.workers = workers
        self.skip_proxy = skip_proxy
        self.initial_links_file = initial_links_file
        self.filter_str = filter_str
        self.seen_urls = set()  # Track processed URLs

        self.visited_urls = set()
        self.skipped_urls = set()
        self.pending_urls = set()

        self.load_state()
        self.load_initial_links(initial_links_file)
        self.pbar = None  # Will be initialized in run()

    def load_initial_links(self, filename):
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                for link in data.get('initial_links', []):
                    if self.filter_str in link:
                        self.pending_urls.add(link)
            print(f"Loaded {len(self.pending_urls)} initial links from {filename}.")
        else:
            print(f"No initial links file {filename} found.")

    def load_state(self):
        if os.path.exists(self.STATE_FILE):
            with open(self.STATE_FILE, 'r') as f:
                data = json.load(f)
                self.visited_urls = set(data.get('visited_urls', []))
                self.skipped_urls = set(data.get('skipped_urls', []))
                self.pending_urls = set(data.get('pending_urls', []))
            print(f"Loaded state from {self.STATE_FILE}.")
        else:
            print("No existing state file found. Starting fresh.")

    def save_state(self):
        data = {
            'visited_urls': list(self.visited_urls),
            'skipped_urls': list(self.skipped_urls),
            'pending_urls': list(self.pending_urls)
        }
        with open(self.STATE_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"State saved to {self.STATE_FILE}.")

    def generate_filename(self, url):
        safe_url = re.sub(r'[^\w\-_\. ]', '-', url)[:255]
        filename = os.path.join(self.output_dir, f"{safe_url}.md")
        return filename

    def clean_url(self, url: str) -> str:
        """Clean URL by removing fragments and normalizing."""
        # Remove fragment/anchor part of URL (everything after #)
        base_url = url.split('#')[0]
        # Remove trailing slash for consistency
        return base_url.rstrip('/')

    async def process_url(self, url):
        # Clean URL before processing
        url = self.clean_url(url)
        
        # Skip if we've already processed this URL
        if url in self.seen_urls:
            return
        
        self.seen_urls.add(url)
        if url in self.visited_urls or url in self.skipped_urls:
            return

        filename = self.generate_filename(url)
        if os.path.exists(filename):
            print(f"[ALERT] File {filename} already exists. Skipping {url}.")
            self.skipped_urls.add(url)
            self.save_state()
            if self.pbar:
                self.pbar.update(1)
            return

        print(f"[CRAWL] Processing: {url}")
        custom_crawler = CustomCrawler(url, self.output_dir)
        fit_content, discovered_links = await custom_crawler.crawl()

        # Mark the current URL as visited
        self.visited_urls.add(url)

        # Add discovered links to pending if they match filter and are not visited/skipped
        for link in discovered_links:
            if self.filter_str in link and link not in self.visited_urls and link not in self.skipped_urls:
                self.pending_urls.add(link)

        self.save_state()
        if self.pbar:
            self.pbar.update(1)

    async def worker(self, worker_id):
        while True:
            if not self.pending_urls:
                break
            url = self.pending_urls.pop()
            await self.process_url(url)
            await asyncio.sleep(0.1)  # small delay to avoid tight loops

    async def run(self):
        # Initialize tqdm based on the current pending_urls count at the start
        initial_pending_count = len(self.pending_urls)
        self.pbar = tqdm(total=initial_pending_count, desc="Processing links")

        # Clean initial links before processing
        initial_links = [self.clean_url(url) for url in self.pending_urls]
        # Remove duplicates after cleaning
        initial_links = list(set(initial_links))

        while True:
            # If no pending URLs, break out of the loop immediately
            if not self.pending_urls:
                break

            tasks = []
            count = min(self.workers, len(self.pending_urls))

            # Create worker tasks
            for i in range(count):
                task = asyncio.create_task(self.worker(i))
                tasks.append(task)

            # Wait for all workers in this batch to complete
            await asyncio.gather(*tasks)

            # After workers have finished, check if there are no more pending URLs
            if not self.pending_urls:
                # No pending URLs left, break the loop
                break

        self.pbar.close()
        print("Crawling complete.")

def parse_args():
    parser = argparse.ArgumentParser(description="Advanced Crawler")
    parser.add_argument('--output_dir', required=True, help='Directory to store output.')
    parser.add_argument('--headless', type=lambda x: (str(x).lower() == 'true'), default=True, help='Headless browser.')
    parser.add_argument('--workers', type=int, default=6, help='Number of workers.')
    parser.add_argument('--skip_proxy', type=lambda x: (str(x).lower() == 'true'), default=False, help='Skip proxies.')
    parser.add_argument('--initial_links_file', default='initial_links.json', help='JSON file with initial links.')
    parser.add_argument('--filter_str', default='', help='Filter substring for links.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    crawler = Crawler(
        output_dir=args.output_dir,
        headless=args.headless,
        workers=args.workers,
        skip_proxy=args.skip_proxy,
        initial_links_file=args.initial_links_file,
        filter_str=args.filter_str
    )
    asyncio.run(crawler.run())