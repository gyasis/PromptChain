import os
import sys
from pathlib import Path

# Get the absolute path to the project root
project_root = Path(__file__).parent.absolute()

# Add both project root and tests directory to Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / 'tests') not in sys.path:
    sys.path.insert(0, str(project_root / 'tests'))

# Now we can import from extras directly
from extras.gemini_multimedia import GeminiMultimedia
from promptchain import PromptChain
from ingestors.youtube_subtitles_processor.processor import YouTubeSubtitlesProcessor
from ingestors.arxiv import ArxivProcessor
from ingestors.crawler import CustomCrawler
from ingestors.marktechpost import ContentExtractor
import argparse
import asyncio
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown
import logging
import mimetypes
from urllib.parse import urlparse
import re
from datetime import datetime

def setup_logging():
    """Configure logging to both file and console."""
    logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    log_file = os.path.join(logs_dir, f'multimodal_ingest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('LiteLLM').setLevel(logging.WARNING)

class MultimodalIngestor:
    def __init__(self):
        self.console = Console()
        self.logger = logging.getLogger(__name__)

    def detect_input_type(self, input_path: str) -> str:
        """
        Detect the type of input (local file, URL, or specific platform)
        """
        # Check if it's a URL
        parsed = urlparse(input_path)
        if parsed.scheme in ['http', 'https']:
            # YouTube URL detection
            if 'youtube.com' in parsed.netloc or 'youtu.be' in parsed.netloc:
                return 'youtube'
            # arXiv URL detection
            elif 'arxiv.org' in parsed.netloc:
                return 'arxiv'
            # General webpage
            else:
                return 'webpage'
        
        # Local file detection
        if os.path.exists(input_path):
            mime_type, _ = mimetypes.guess_type(input_path)
            if mime_type:
                if mime_type.startswith('video/'):
                    return 'video'
                elif mime_type.startswith('audio/'):
                    return 'audio'
                elif mime_type.startswith('application/pdf'):
                    return 'pdf'
            
        return 'unknown'

    async def process_youtube(self, url: str, download_video: bool = False, video_prompt: str = None) -> str:
        """
        Process YouTube content with options for subtitles or full video processing
        
        Args:
            url (str): YouTube URL
            download_video (bool): Whether to also process video with Gemini
            video_prompt (str): Prompt for video processing if download_video is True
        """
        try:
            # Always get subtitles first
            processor = YouTubeSubtitlesProcessor(video_url=url, return_text=True)
            subtitle_content = await processor.process()
            
            if download_video:
                # Additionally process the video with Gemini
                import yt_dlp
                
                ydl_opts = {
                    'format': 'best[height<=720]',  # Limit resolution for faster processing
                    'outtmpl': '%(title)s.%(ext)s'
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    video_path = ydl.prepare_filename(info)
                    
                # Process video with Gemini
                prompt = video_prompt or "Analyze this video and describe its visual content, gestures, and non-verbal elements"
                with GeminiMultimedia() as gm:
                    video_content = gm.process_media(file_path=video_path, prompt=prompt)
                    
                # Clean up downloaded video
                if os.path.exists(video_path):
                    os.remove(video_path)
                
                # Combine subtitle and video analysis
                combined_content = f"""
# Transcript:
{subtitle_content}

# Visual Analysis:
{video_content}
"""
                return combined_content
            
            return subtitle_content
            
        except Exception as e:
            self.logger.error(f"Error processing YouTube content: {str(e)}")
            raise

    async def process_arxiv(self, url: str) -> str:
        """Process arXiv paper"""
        return ArxivProcessor.download_and_process_arxiv_pdf(url)

    async def process_webpage(self, url: str) -> str:
        """Process general webpage"""
        crawler = CustomCrawler(url, output_dir=None)
        content, _ = await crawler.crawl()
        return content

    async def process_media(self, file_path: str, prompt: str) -> str:
        """Process media files using Gemini"""
        with GeminiMultimedia() as gm:
            return gm.process_media(file_path=file_path, prompt=prompt)

    def run_analysis_chain(self, content: str, analysis_prompt: str = "educational content") -> str:
        """Run the analysis chain on the content"""
        analysis_chain = PromptChain(
            models=["openai/gpt-4o-mini"],
            instructions=[
                # Step 1: Initial structuring while preserving all content
                f"""Structure this content into a clear lesson format while preserving ALL original information:
                1. Add clear section headings
                2. Organize into logical segments
                3. DO NOT remove or summarize any content
                4. Focus area: {analysis_prompt}
                
                Format each section like this:
                # [Section Title]
                [Original content goes here, exactly as provided]
                
                Input: {{input}}""",
                
                # Step 2: Add explanations while keeping structure
                """Enhance the structured content by adding explanations for key terms:
                1. Keep ALL existing content and structure exactly as is
                2. After each key term or concept, add:
                    *[Term Explanation: Brief explanation here]*
                3. DO NOT modify or remove any original content
                
                Input: {input}""",
                
                # Step 3: Add examples while preserving everything
                """Add concrete examples while maintaining all previous content:
                1. Keep ALL existing content exactly as is (including structure and explanations)
                2. After relevant sections, add:
                    > Example: [Practical example here]
                3. DO NOT change or remove anything from the previous output
                
                Input: {input}"""
            ],
            full_history=True  # Keep this to build upon previous steps
        )
        
        try:
            result = analysis_chain.process_prompt(content)
            
            # Get the final output that includes all transformations
            if isinstance(result, dict) and 'history' in result:
                # Return the last step's output which should contain everything
                return result['history'][-1]['output']
            elif isinstance(result, list):
                return result[-1].get('output', str(result[-1])) if result else ""
            
            return str(result)
            
        except Exception as e:
            self.logger.error(f"Error in analysis chain: {str(e)}")
            return f"Error processing content: {str(e)}"

    async def process_input(self, input_path: str, media_prompt: str = None, analysis_prompt: str = "educational content", download_video: bool = False):
        """Main processing function"""
        input_type = self.detect_input_type(input_path)
        
        # Process content based on type
        content = None
        
        try:
            if input_type == 'youtube':
                content = await self.process_youtube(input_path, download_video, media_prompt)
            elif input_type == 'arxiv':
                content = await self.process_arxiv(input_path)
            elif input_type == 'webpage':
                content = await self.process_webpage(input_path)
            elif input_type in ['video', 'audio']:
                if not media_prompt:
                    media_prompt = "Transcribe and analyze this media file"
                content = await self.process_media(input_path, media_prompt)
            else:
                raise ValueError(f"Unsupported input type: {input_type}")

            if content:
                # Display raw content
                self.console.print("\n[bold cyan]Extracted Content:[/bold cyan]")
                self.console.print(Panel(content, title="Raw Content", border_style="cyan"))

                # Run analysis chain
                analyzed_content = self.run_analysis_chain(content, analysis_prompt)
                
                # Display analyzed content
                self.console.print("\n[bold green]Analyzed Content:[/bold green]")
                self.console.print(Markdown(analyzed_content))
                
                return analyzed_content
            
        except Exception as e:
            self.logger.error(f"Error processing input: {str(e)}", exc_info=True)
            self.console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
            return None

async def main():
    console = Console()
    
    parser = argparse.ArgumentParser(description="Process multimodal inputs")
    parser.add_argument("--input", type=str, help="Input file or URL")
    parser.add_argument("--media-prompt", type=str, help="Prompt for media processing")
    parser.add_argument("--analysis-prompt", type=str, help="Prompt for content analysis")
    parser.add_argument("--download-video", action="store_true", help="Download and process YouTube videos instead of subtitles")
    parser.add_argument("--interactive", action="store_true", help="Use interactive menu")
    args = parser.parse_args()

    setup_logging()
    ingestor = MultimodalIngestor()
    
    try:
        if args.interactive:
            # Interactive menu
            console.print("\n[bold cyan]Multimodal Content Processor[/bold cyan]")
            console.print("\nSelect input type:")
            console.print("1. YouTube Video")
            console.print("2. Local Media File")
            console.print("3. arXiv Paper")
            console.print("4. Web Page")
            
            choice = Prompt.ask(
                "\n[bold yellow]Enter your choice[/bold yellow]",
                choices=["1", "2", "3", "4"],
                default="1"
            )
            
            if choice == "1":
                input_path = Prompt.ask("\n[bold yellow]Enter YouTube URL[/bold yellow]")
                
                process_choice = Prompt.ask(
                    "\n[bold yellow]How would you like to process the video?[/bold yellow]",
                    choices=["1", "2"],
                    default="1",
                    show_choices=False,
                    show_default=False
                )
                
                console.print("\n1. Extract subtitles only (faster)")
                console.print("2. Process full video + subtitles (more comprehensive)")
                
                download_video = process_choice == "2"
                
                if download_video:
                    media_prompt = Prompt.ask(
                        "\n[bold yellow]Enter prompt for video analysis[/bold yellow]",
                        default="Analyze this video and describe its visual content, gestures, and non-verbal elements"
                    )
                else:
                    media_prompt = None
                    
            elif choice == "2":
                input_path = Prompt.ask("\n[bold yellow]Enter path to media file[/bold yellow]")
                media_prompt = Prompt.ask(
                    "\n[bold yellow]Enter prompt for media processing[/bold yellow]",
                    default="Transcribe and analyze this media file"
                )
                download_video = False
                
            elif choice == "3":
                input_path = Prompt.ask("\n[bold yellow]Enter arXiv URL or ID[/bold yellow]")
                media_prompt = None
                download_video = False
                
            elif choice == "4":
                input_path = Prompt.ask("\n[bold yellow]Enter webpage URL[/bold yellow]")
                media_prompt = None
                download_video = False
            
            analysis_prompt = Prompt.ask(
                "\n[bold yellow]Enter focus for content analysis[/bold yellow]",
                default="educational content"
            )
            
        else:
            # Command line arguments
            input_path = args.input or Prompt.ask("\n[bold yellow]Enter input path or URL[/bold yellow]")
            input_type = ingestor.detect_input_type(input_path)
            
            media_prompt = None
            download_video = args.download_video
            
            if input_type == 'youtube' and download_video:
                media_prompt = args.media_prompt or Prompt.ask(
                    "[bold yellow]Enter prompt for video processing[/bold yellow]",
                    default="Analyze this video and describe its visual content, gestures, and non-verbal elements"
                )
            elif input_type in ['video', 'audio']:
                media_prompt = args.media_prompt or Prompt.ask(
                    "[bold yellow]Enter prompt for media processing[/bold yellow]",
                    default="Transcribe and analyze this media file"
                )
            
            analysis_prompt = args.analysis_prompt or Prompt.ask(
                "[bold yellow]Enter focus for content analysis[/bold yellow]",
                default="educational content"
            )
        
        # Process the input
        console.print("\n[bold blue]Processing content...[/bold blue]")
        await ingestor.process_input(
            input_path,
            media_prompt=media_prompt,
            analysis_prompt=analysis_prompt,
            download_video=download_video
        )
        
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}", exc_info=True)
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 