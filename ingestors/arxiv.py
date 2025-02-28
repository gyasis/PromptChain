import arxiv
from pathlib import Path
import re
import tempfile
import pymupdf4llm

class ArxivProcessor:
    @staticmethod
    def extract_identifier(url_or_id):
        """
        Extract the arXiv identifier from a URL or return the identifier if already provided.

        Args:
            url_or_id (str): The arXiv URL or identifier.

        Returns:
            str: The arXiv identifier.
        """
        match = re.search(r'arxiv\.org/abs/(\d+\.\d+)', url_or_id)
        if match:
            return match.group(1)
        return url_or_id

    @staticmethod
    def download_and_process_arxiv_pdf(identifier):
        """
        Download an arXiv PDF, convert it to Markdown, and return the Markdown content.

        Args:
            identifier (str): The arXiv URL or identifier.

        Returns:
            str: The Markdown content of the paper, or None if an error occurs
        """
        try:
            # Extract identifier if URL is provided
            identifier = ArxivProcessor.extract_identifier(identifier)

            # Create a client and search for the paper
            client = arxiv.Client()
            search = arxiv.Search(id_list=[identifier])
            paper = next(client.results(search))

            # Create a temporary directory
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Define the PDF path
                pdf_path = Path(tmp_dir) / f"{identifier}.pdf"
                
                # Download the PDF to the temporary directory with a custom filename
                paper.download_pdf(dirpath=tmp_dir, filename=f"{identifier}.pdf")

                # Convert PDF to Markdown
                md_content = pymupdf4llm.to_markdown(str(pdf_path))

            return md_content

        except StopIteration:
            print(f"No paper found with identifier: {identifier}")
            return None
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None

    @staticmethod
    def get_arxiv_abstract(identifier):
        """
        Retrieve the abstract of an arXiv paper based on its identifier.

        Args:
            identifier (str): The arXiv URL or identifier.

        Returns:
            str: The abstract of the paper, or None if an error occurs
        """
        try:
            # Extract identifier if URL is provided
            identifier = ArxivProcessor.extract_identifier(identifier)

            # Search for the paper
            client = arxiv.Client()
            results = client.results(arxiv.Search(id_list=[identifier]))
            paper = next(results)

            # Return the abstract
            return paper.summary

        except StopIteration:
            print(f"No paper found with identifier: {identifier}")
            return None
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None

if __name__ == "__main__":
    # Example usage
    input_value = input("Enter the arXiv URL or identifier: ")
    md_content = ArxivProcessor.download_and_process_arxiv_pdf(input_value)
    
    if md_content:
        print("Markdown content:")
        print(md_content)
    else:
        print("Failed to process the PDF.")
# %%
# import arxiv
# client = arxiv.Client()


# %%
