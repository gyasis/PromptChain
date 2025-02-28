# __main__.py
import argparse
from .processor import YouTubeSubtitlesProcessor


def main():
    """
    Downloads subtitles from a YouTube video, converts them to text, and either copies the text content to the clipboard
    or returns it as an output depending on the provided flag.
    """
    parser = argparse.ArgumentParser(
        description="Download and process YouTube subtitles",
        epilog=(
            "Usage examples:\n"
            '  python -m youtube_subtitles_processor "https://www.youtube.com/watch?v=example_video_id"\n'
            '  python -m youtube_subtitles_processor "https://www.youtube.com/watch?v=example_video_id" --text\n'
            "\n"
            "To use as a library module:\n"
            "  from youtube_subtitles_processor.processor import YouTubeSubtitlesProcessor\n"
            '  processor = YouTubeSubtitlesProcessor(video_url="https://www.youtube.com/watch?v=example_video_id", return_text=True)\n'
            "  processor.process()"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("video_url", nargs="?", help="The URL of the YouTube video")
    parser.add_argument(
        "--text",
        action="store_true",
        help="Print the text output instead of copying to clipboard",
    )

    args = parser.parse_args()

    processor = YouTubeSubtitlesProcessor(
        video_url=args.video_url, return_text=args.text
    )
    processor.process()


if __name__ == "__main__":
    main()
