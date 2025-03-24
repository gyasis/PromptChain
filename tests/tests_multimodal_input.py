from extras.gemini_multimedia import GeminiMultimedia
from promptchain import PromptChain
import argparse
from rich.console import Console
from rich.prompt import Prompt
import logging
import os
from datetime import datetime
from rich.panel import Panel
from rich.markdown import Markdown

def setup_logging():
    """Configure logging to both file and console."""
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a log file with timestamp
    log_file = os.path.join(logs_dir, f'multimodal_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will print to console
        ]
    )
    
    # Suppress specific loggers
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('LiteLLM').setLevel(logging.WARNING)

def preprocess_multimodal_input(media_file_path: str, media_prompt: str) -> str:
    """
    Preprocess the multimodal input using GeminiMultimedia.
    
    Args:
        media_file_path (str): Path to the media file.
        media_prompt (str): Prompt/question about the media.
    
    Returns:
        str: Extracted text from the media.
    """
    console = Console()
    console.print("\n[bold blue]Preprocessing media file...[/bold blue]")
    
    with GeminiMultimedia() as gm:
        extracted_text = gm.process_media(
            file_path=media_file_path,
            prompt=media_prompt
        )
    
    console.print("[green]Media preprocessing complete![/green]")
    
    # Display the transcript
    console.print("\n[bold cyan]Transcript from media file:[/bold cyan]")
    console.print(Panel(extracted_text, title="Transcript", border_style="cyan"))
    
    return extracted_text

def run_analysis_chain(extracted_text: str, analysis_prompt: str = "medical education") -> str:
    """
    Run the analysis chain on the extracted text.
    
    Args:
        extracted_text (str): Text extracted from the media.
        analysis_prompt (str, optional): Prompt for analyzing the extracted text.
            Defaults to "medical education".
    
    Returns:
        str: Final analysis result.
    """
    console = Console()
    console.print("\n[bold blue]Running analysis chain...[/bold blue]")
    
    # Create analysis chain with default medical education focus
    analysis_chain = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=[
            f"""Format this text into a markdown structured teaching lesson. DO NOT ADD ANYTHING ELSE TO THE TEXT. DO NOTE LEAVE ANYTHING OUT: {{input}}\n\nFocus on: {analysis_prompt}",
            "Expand on the less talked about but highly specialized or significant medical terms important for the overall subject by italicizing new content and explain the term and giving a deeper background:

            Examples of expanded content:

            - medications or categories of medications - giving common medicatoisn and mechanism of actions, expnad on anmes, MOA, side effects, etc.
            -diseasee pathologies - giving a detailed explanation of the disease and the pathologies associated with the disease sort of like a quick amboss
            - flesh out any anacronym explanations
            - any procedures or any other content that is not usually covered in a basic medical education setting but is recognized as more specialied. for example 12 lead ekc shoud be ignored and not expanded on. but a central venous catheter should be expanded on and pic lines should be expanded on
            
             the output should be a word for word recreation of the text with extra content boxes interdispered to explain a medical term when it is mentioned in the text.DO NOTE LEAVE ANYTHING OUT FROM THE PREVIOUS INPUT WHILE INLCUDE THE ADDITIONAL CONTENT: {input}"""
        ],
        full_history=False
    )
    
    # Process the extracted text through the chain
    result = analysis_chain.process_prompt(extracted_text)
    
    # Handle different result types
    if isinstance(result, dict):
        if 'history' in result:
            # Display each step's output
            for i, step in enumerate(result['history'], 1):
                console.print(f"\n[bold yellow]Step {i} Output:[/bold yellow]")
                console.print(Panel(step['output'], title=f"Step {i}", border_style="yellow"))
            final_output = result['output']
        else:
            final_output = result.get('output', str(result))
    elif isinstance(result, list):
        # If result is a list, take the last item as the final output
        final_output = result[-1].get('output', str(result[-1])) if result else ""
    else:
        final_output = str(result)
    
    console.print("[green]Analysis complete![/green]")
    return final_output

if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process media files with prompt chain")
    parser.add_argument("--file", type=str, help="Path to media file")
    args = parser.parse_args()
    
    console = Console()
    
    try:
        # Get inputs interactively if not provided
        media_file_path = args.file or Prompt.ask(
            "\n[bold yellow]Enter path to media file[/bold yellow]"
        )
        
        media_prompt = Prompt.ask(
            "[bold yellow]Enter prompt for media processing[/bold yellow]",
            default="Transcribe this audio file word for word"
        )
        
        # Preprocess the media file to get transcript
        extracted_text = preprocess_multimodal_input(
            media_file_path=media_file_path,
            media_prompt=media_prompt
        )
        
        # Run the analysis chain with default settings
        final_result = run_analysis_chain(extracted_text)
        
        # Display results using Rich's Markdown rendering
        console.print("\n[bold green]Final Analysis:[/bold green]")
        console.print(Markdown(final_result))
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}") 