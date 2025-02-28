# %%
import requests
from bs4 import BeautifulSoup
import os
import tempfile
import fitz  # PyMuPDF
from weasyprint import HTML
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ContentExtractor:
    def __init__(self, url):
        self.url = url
        self.markdown_content = ''
        self.pdf_text = ''
        self.pdf_filename = ''
        self.title = ''

    def fetch_webpage_content(self):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36'}
        logging.info("Fetching the webpage content")
        response = requests.get(self.url, headers=headers)
        response.raise_for_status()
        logging.info("Webpage content fetched successfully")
        return response.content

    def parse_html_content(self, html_content):
        logging.info("Parsing HTML content")
        soup = BeautifulSoup(html_content, 'html.parser')
        article = soup.find('article')
        if not article:
            logging.error("Main content not found")
            raise ValueError('Main content not found.')
        
        # Extract the title
        title_tag = soup.find('h1', class_='entry-title')
        if title_tag:
            self.title = title_tag.get_text().strip()
        else:
            logging.warning("Title not found in the HTML content")
        
        return article

    def prepare_markdown_content(self, article):
        logging.info("Preparing Markdown content")
        markdown_content = ''
        for paragraph in article.find_all('p'):
            markdown_content += paragraph.get_text() + '\n\n'
        for img in article.find_all('img'):
            img_src = img.get('src')
            img_alt = img.get('alt', 'Image')
            markdown_content += f'![{img_alt}]({img_src})\n\n'
        for link in article.find_all('a'):
            link_href = link.get('href')
            link_text = link.get_text() or link_href
            markdown_content += f'[{link_text}]({link_href})\n'
        self.markdown_content = markdown_content

    def create_pdf_from_html(self, soup):
        logging.info("Creating PDF from HTML content")
        pdf_creation_start = time.time()
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            self.pdf_filename = temp_pdf.name
        HTML(string=str(soup)).write_pdf(self.pdf_filename)
        logging.info(f"PDF created successfully in {time.time() - pdf_creation_start:.2f} seconds")

    def extract_text_from_pdf(self):
        logging.info("Extracting text from the PDF")
        text_extraction_start = time.time()
        pdf_text = ''
        with fitz.open(self.pdf_filename) as doc:
            for page in doc:
                pdf_text += page.get_text("text") + '\n\n'
        logging.info(f"Text extracted from PDF in {time.time() - text_extraction_start:.2f} seconds")
        self.pdf_text = pdf_text

    def extract_content(self):
        start_time = time.time()
        logging.info("Starting the extraction process")
        try:
            html_content = self.fetch_webpage_content()
            article = self.parse_html_content(html_content)
            self.prepare_markdown_content(article)
            self.create_pdf_from_html(article)
            self.extract_text_from_pdf()
            combined_output = f"Markdown Content:\n\n{self.markdown_content}\n\nPDF Text:\n\n{self.pdf_text.strip()}"
            logging.info(f"Extraction process completed in {time.time() - start_time:.2f} seconds")
            print(combined_output)
            return self.markdown_content, self.pdf_text.strip(), self.pdf_filename, self.title
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return f'An error occurred: {e}', '', '', ''
        finally:
            self.pdf_filename = ''

# Example usage:
# extractor = ContentExtractor('https://example.com')
# markdown_content, pdf_text, pdf_filename, title = extractor.extract_content()
# ... rest of the file remains unchanged ...
# %%
