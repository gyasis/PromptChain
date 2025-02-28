import os
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
import mimetypes
import tempfile
import atexit
import logging
from typing import Union, List, Optional
from PIL import Image
import argparse
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel





class GeminiMultimedia:
    """
    A comprehensive wrapper for handling multimedia interactions with Gemini API.
    Supports images, audio, and video files with automatic cleanup.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the GeminiMultimedia wrapper.
        
        Args:
            api_key (str, optional): Gemini API key. If not provided, will look for GOOGLE_GENERATIVE_AI_API_KEY in env.
        """
        # Load environment variables from workspace root
        load_dotenv(dotenv_path='/home/gyasis/Documents/code/Applied_AI/.env')
        
        # Configure Gemini
        self.api_key = api_key or os.getenv('GOOGLE_GENERATIVE_AI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key not found. Please provide it or set GOOGLE_GENERATIVE_AI_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
        
        # Initialize state
        self.uploaded_files = []
        self.temp_files = []
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize models with correct Gemini model names
        self.models = {
            'vision': 'gemini-pro-vision',     # For images
            'text': 'gemini-pro',              # For text
            'audio': 'gemini-1.5-pro',         # For audio processing
            'video': 'gemini-1.5-pro'          # For video processing
        }
        
        # Model capabilities
        self.model_capabilities = {
            'gemini-2.0-flash-exp': ['text', 'image', 'audio', 'video'],
            'gemini-1.5-flash': ['text', 'image', 'audio', 'video'],
            'gemini-1.5-flash-8b': ['text', 'image', 'audio', 'video'],
            'gemini-1.5-pro': ['text', 'image', 'audio', 'video'],
            'gemini-pro-vision': ['text', 'image'],
            'gemini-pro': ['text']
        }

        # Supported audio formats with all common MIME types
        self.supported_audio_formats = {
            'audio/wav': '.wav',
            'audio/x-wav': '.wav',
            'audio/mp3': '.mp3',
            'audio/mpeg': '.mp3',      # Standard MIME type for MP3
            'audio/mpeg3': '.mp3',     # Alternative MP3 MIME type
            'audio/x-mpeg-3': '.mp3',  # Another MP3 variant
            'audio/aiff': '.aiff',
            'audio/x-aiff': '.aiff',
            'audio/aac': '.aac',
            'audio/aacp': '.aac',
            'audio/ogg': '.ogg',
            'audio/vorbis': '.ogg',
            'audio/flac': '.flac',
            'audio/x-flac': '.flac'
        }
        
        # Audio processing settings
        self.audio_settings = {
            'max_duration_hours': 9.5,  # Maximum supported length in hours
            'tokens_per_second': 25,    # Gemini uses 25 tokens per second of audio
            'max_file_size_gb': 2,      # Maximum file size in GB
            'downsampling_rate': 16000  # Downsampling rate in Hz (16 Kbps)
        }
        
        # Verify models are available
        self._verify_models()
    
    def _verify_models(self):
        """Verify that the required models are available."""
        try:
            available_models = [model.name for model in genai.list_models()]
            self.logger.info(f"Available models: {available_models}")
            
            for purpose, model in self.models.items():
                if not any(model in m for m in available_models):
                    self.logger.warning(f"Model {model} for {purpose} may not be available")
        except Exception as e:
            self.logger.warning(f"Could not verify models: {str(e)}")

    def process_media(self, 
                     file_path: Union[str, Path], 
                     prompt: str,
                     model_name: Optional[str] = None,
                     max_tokens: int = 8192,
                     temperature: float = 0.7) -> str:
        """
        Process any media file (image, audio, or video) with Gemini API.
        
        Args:
            file_path (Union[str, Path]): Path to the media file
            prompt (str): Prompt/question about the media
            model_name (str, optional): Specific model to use. If None, will auto-select based on media type
            max_tokens (int): Maximum tokens for response
            temperature (float): Temperature for generation
            
        Returns:
            str: Generated response from Gemini
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        mime_type = mimetypes.guess_type(file_path)[0]
        if not mime_type:
            raise ValueError(f"Could not determine mime type for file: {file_path}")
            
        media_type = mime_type.split('/')[0]
        self.logger.info(f"Processing {media_type} file: {file_path}")
        
        # Validate audio format if applicable
        if media_type == 'audio':
            if mime_type not in self.supported_audio_formats:
                raise ValueError(f"Unsupported audio format: {mime_type}. Supported formats: {list(self.supported_audio_formats.keys())}")
        
        # Select appropriate model if not specified
        if not model_name:
            if media_type == 'image':
                model_name = self.models['vision']
            elif media_type == 'audio':
                model_name = self.models['audio']
            elif media_type == 'video':
                model_name = self.models['video']
            else:
                model_name = self.models['text']
                
        # Verify model capabilities
        if media_type not in self.model_capabilities.get(model_name, []):
            self.logger.warning(f"Model {model_name} may not fully support {media_type} processing")
            
        try:
            if media_type == 'image':
                return self._process_image(file_path, prompt, model_name, max_tokens, temperature)
            elif media_type in ['audio', 'video']:
                return self._process_av_media(file_path, prompt, model_name, max_tokens, temperature)
            else:
                raise ValueError(f"Unsupported media type: {media_type}")
        except Exception as e:
            self.logger.error(f"Error processing {media_type} file: {str(e)}")
            raise

    def _process_image(self, 
                      image_path: Path, 
                      prompt: str,
                      model_name: str,
                      max_tokens: int,
                      temperature: float) -> str:
        """Process image with Gemini Vision API."""
        try:
            image = Image.open(image_path)
            model = genai.GenerativeModel(model_name)
            
            response = model.generate_content(
                [prompt, image],
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature
                }
            )
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            raise

    def _process_av_media(self, 
                         file_path: Path, 
                         prompt: str,
                         model_name: str,
                         max_tokens: int,
                         temperature: float) -> str:
        """Process audio/video with Gemini API using File API for large files."""
        try:
            file_size = file_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            self.logger.info(f"Processing media file of size: {file_size_mb:.2f} MB")
            
            # Check file size limits
            if file_size > self.audio_settings['max_file_size_gb'] * 1024 * 1024 * 1024:
                raise ValueError(f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed size of {self.audio_settings['max_file_size_gb']} GB")
            
            mime_type = mimetypes.guess_type(file_path)[0]
            
            # Handle MP3 and other audio formats with multiple MIME types
            if mime_type in self.supported_audio_formats:
                self.logger.info(f"Detected supported audio format: {mime_type}")
            else:
                # Try to infer format from extension
                ext = file_path.suffix.lower()
                if any(ext == supported_ext for supported_ext in self.supported_audio_formats.values()):
                    self.logger.info(f"Inferred audio format from extension: {ext}")
                else:
                    raise ValueError(f"Unsupported audio format: {mime_type}. Supported formats: {list(self.supported_audio_formats.keys())}")
            
            if file_size > 20 * 1024 * 1024:  # 20MB
                # Use File API for large files
                self.logger.info("Using File API for large file")
                file_ref = genai.upload_file(str(file_path))
                self.uploaded_files.append(file_ref)
                
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    [prompt, file_ref],
                    generation_config={
                        "max_output_tokens": max_tokens,
                        "temperature": temperature
                    }
                )
            else:
                # For small files, use inline data
                self.logger.info("Using inline data for small file")
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    [prompt, {'mime_type': mime_type, 'data': file_data}],
                    generation_config={
                        "max_output_tokens": max_tokens,
                        "temperature": temperature
                    }
                )
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Error processing audio/video: {str(e)}")
            raise

    def cleanup(self):
        """Clean up uploaded files and temporary files."""
        # Clean up uploaded files
        for file_ref in self.uploaded_files:
            try:
                genai.delete_file(file_ref.name)
                self.logger.info(f"Cleaned up uploaded file: {file_ref.name}")
            except Exception as e:
                self.logger.warning(f"Error cleaning up file {file_ref.name}: {str(e)}")
        
        # Clean up temporary files
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    self.logger.info(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                self.logger.warning(f"Error cleaning up temp file {temp_file}: {str(e)}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process media files with Gemini API")
    parser.add_argument("--model", type=str, default="gemini-1.5-pro",
                       help="Model to use (default: gemini-1.5-pro)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation (default: 0.7)")
    parser.add_argument("--max-tokens", type=int, default=8192,
                       help="Max tokens for response (default: 8192)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Initialize rich console
    console = Console()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        with GeminiMultimedia() as gm:
            console.print(Panel.fit(
                "[bold green]Gemini Multimedia Processor[/bold green]\n"
                f"[cyan]Model:[/cyan] {args.model}\n"
                f"[cyan]Temperature:[/cyan] {args.temperature}\n"
                f"[cyan]Max Tokens:[/cyan] {args.max_tokens}\n"
                f"[cyan]Debug Mode:[/cyan] {'Enabled' if args.debug else 'Disabled'}\n"
                "\nType 'exit' or 'quit' to end the session",
                title="Configuration",
                border_style="blue"
            ))
            
            while True:
                # Get file path
                file_path = Prompt.ask("\n[bold yellow]Enter path to media file[/bold yellow]")
                if file_path.lower() in ['exit', 'quit']:
                    break
                
                # Validate file exists
                if not os.path.exists(file_path):
                    console.print(f"[red]File not found: {file_path}[/red]")
                    continue
                
                # Get prompt
                prompt = Prompt.ask("[bold yellow]Enter your prompt/question about the media[/bold yellow]")
                if prompt.lower() in ['exit', 'quit']:
                    break
                
                try:
                    console.print("\n[bold blue]Processing media...[/bold blue]")
                    response = gm.process_media(
                        file_path,
                        prompt,
                        model_name=args.model,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens
                    )
                    
                    console.print(Panel(
                        response,
                        title="[bold green]Response[/bold green]",
                        border_style="green"
                    ))
                    
                except Exception as e:
                    console.print(f"[red]Error processing media: {str(e)}[/red]")
                
                # Ask if user wants to continue
                if not Prompt.ask("\n[yellow]Process another file?[/yellow]", choices=["y", "n"], default="y") == "y":
                    break
                    
            console.print("\n[bold green]Session ended. Goodbye![/bold green]")
            
    except Exception as e:
        console.print(f"[red]Fatal error: {str(e)}[/red]") 