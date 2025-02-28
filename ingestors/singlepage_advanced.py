import asyncio
from ingestors.crawler import CustomCrawler  # Import from the ingestors package
import tiktoken
import os
import base64
import requests
from urllib.parse import urljoin
import argparse

def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)


async def convert_image_to_base64(image_url):
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            return base64.b64encode(response.content).decode('utf-8')
    except Exception as e:
        print(f"[ERROR] Failed to convert image {image_url}: {e}")
    return None

async def patched_crawl_page(self, crawler, url):
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

    # Extract content and process images
    links = result.links
    fit_content = result.fit_markdown

    # Modified image processing section
    if self.include_images and hasattr(result, 'media') and result.media:
        for image in result.media.get("images", []):
            base64_data = await convert_image_to_base64(image['src'])
            if base64_data:
                img_markdown = f"![{image.get('alt', '')}]({image['src']})"
                base64_markdown = f"![{image.get('alt', '')}](data:image/png;base64,{base64_data})"
                fit_content = fit_content.replace(img_markdown, base64_markdown)

    # Resolve relative links to absolute URLs
    internal_links = links.get("internal", [])
    absolute_links = [urljoin(url, link['href']) for link in internal_links]

    return fit_content, absolute_links

# Monkey patch the _crawl_page method
CustomCrawler._crawl_page = patched_crawl_page

original_init = CustomCrawler.__init__

def patched_init(self, start_url, output_dir=None, include_images=True):
    original_init(self, start_url, output_dir)
    self.include_images = include_images

CustomCrawler.__init__ = patched_init

async def main():
    # Add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-images', action='store_true', help='Disable image processing')
    args = parser.parse_args()

    start_url = "https://towardsdatascience.com/synthetic-data-generation-with-llms/"

    # Pass the image parameter to CustomCrawler
    custom_crawler = CustomCrawler(start_url, include_images=not args.no_images)
    fit_content, links = await custom_crawler.crawl()

    # Print the extracted content and links
    print("Extracted Content:", fit_content)
    # Count tokens
    token_count = count_tokens(fit_content)
    print(f"Number of tokens in singlepage_advanced.py: {token_count}")
    print("Discovered Links:")
    for link in links:
        print(link)

if __name__ == "__main__":
    asyncio.run(main())

