import os
import requests
import logging
from urllib.parse import urlparse
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def crawl_github_files(repo_url, token=None, max_file_size=1 * 1024 * 1024):
    """
    Crawl 'system.md' files from the 'patterns' folder and all files from the 'strategies' folder in a GitHub repository.
    """
    logging.info("Starting to crawl GitHub repository.")
    
    # Parse GitHub URL to extract owner, repo, commit/branch, and path
    parsed_url = urlparse(repo_url)
    path_parts = parsed_url.path.strip('/').split('/')
    
    if len(path_parts) < 2:
        logging.error(f"Invalid GitHub URL: {repo_url}")
        raise ValueError(f"Invalid GitHub URL: {repo_url}")
    
    owner = path_parts[0]
    repo = path_parts[1].replace('.git', '')
    
    if len(path_parts) > 3 and path_parts[2] == 'tree':
        ref = path_parts[3]
    else:
        ref = "main"
    
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}" 
    
    files = {}

    def fetch_contents(path):
        logging.info(f"Fetching contents from path: {path}")
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        params = {"ref": ref}
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            logging.error(f"Error fetching {path}: {response.status_code} - {response.text}")
            return
        
        contents = response.json()
        if not isinstance(contents, list):
            contents = [contents]
        
        for item in contents:
            item_path = item["path"]
            file_extension = os.path.splitext(item["name"])[1].lower()
            
            # Check if the file is a .md or .json file
            if item["type"] == "file" and file_extension in {".md", ".json"}:
                file_size = item.get("size", 0)
                if file_size > max_file_size:
                    logging.warning(f"Skipping {item_path} due to size limit.")
                    continue
                
                if "download_url" in item and item["download_url"]:
                    file_url = item["download_url"]
                    file_response = requests.get(file_url, headers=headers)
                    
                    if file_response.status_code == 200:
                        content = file_response.text
                        files[item_path] = content
                        logging.info(f"Downloaded file: {item_path}")
                    else:
                        logging.error(f"Failed to download {item_path}: {file_response.status_code}")
            
            elif item["type"] == "dir":
                fetch_contents(item_path)
    
    # Fetch 'system.md' files from the 'patterns' folder
    fetch_contents('patterns')
    # Fetch all files from the 'strategies' folder
    fetch_contents('strategies')
    
    logging.info("Completed crawling GitHub repository.")
    return files

def process_patterns(files, output_dir):
    """
    Process the patterns files and save them to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    for file_path, content in files.items():
        if file_path.endswith('system.md'):
            # Extract the parent directory name to use as the new file name
            parent_dir_name = os.path.basename(os.path.dirname(file_path))
            new_file_name = f"{parent_dir_name}.md"
            new_file_path = os.path.join(output_dir, new_file_name)
            
            # Check for potential name collisions
            if os.path.exists(new_file_path):
                logging.warning(f"File {new_file_name} already exists. Appending a unique identifier.")
                base_name, ext = os.path.splitext(new_file_name)
                new_file_name = f"{base_name}_{os.path.basename(file_path)}{ext}"
                new_file_path = os.path.join(output_dir, new_file_name)
            
            with open(new_file_path, 'w') as f:
                f.write(content)
            logging.info(f"Processed and saved: {new_file_path}")

def copy_strategies(files, output_dir):
    """
    Copy the strategies files to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    for file_path, content in files.items():
        if file_path.startswith('strategies/'):
            new_file_path = os.path.join(output_dir, os.path.basename(file_path))
            with open(new_file_path, 'w') as f:
                f.write(content)
            logging.info(f"Copied: {new_file_path}")

def main():
    load_dotenv()
    github_token = os.getenv("GITHUB_TOKEN")
    
    repo_url = "https://github.com/danielmiessler/fabric.git"
    
    files = crawl_github_files(
        repo_url=repo_url,
        token=github_token
    )
    
    logging.info(f"Downloaded {len(files)} files.")
    
    # Process patterns and copy strategies
    process_patterns(files, './prompts')
    copy_strategies(files, './prompts/strategies')

if __name__ == "__main__":
    main()