# processor.py
import subprocess
import sys
import re
import os
import pyperclip
import asyncio


class YouTubeSubtitlesProcessor:
    def __init__(self, video_url=None, return_text=False):
        self.video_url = video_url
        self.return_text = return_text
        self.processed_text = None

    def remove_final_timecodes(self, text):
        """
        Remove any remaining timecodes like '00:40', '00:41', etc.
        Also, remove any double newline characters.
        """
        text = re.sub(r"\b\d{2}:\d{2}\b", "", text).strip()
        text = re.sub(r"\n\s*\n", "\n", text)  # Remove double newlines
        return text

    def remove_tags(self, text):
        """
        Remove VTT markup tags
        """
        tags = [
            r"</c>",
            r"<c(\.color\w+)?>",
            r"<\d{2}:\d{2}:\d{2}\.\d{3}>",
        ]

        for pat in tags:
            text = re.sub(pat, "", text)

        # Extract timestamp, only keep HH:MM
        text = re.sub(
            r"(\d{2}:\d{2}):\d{2}\.\d{3} --> .* align:start position:0%", r"\g<1>", text
        )

        text = re.sub(r"^\s+$", "", text, flags=re.MULTILINE)
        return text

    def remove_header(self, lines):
        """
        Remove VTT file header
        """
        pos = -1
        for mark in ("##", "Language: en"):
            if mark in lines:
                pos = lines.index(mark)
        lines = lines[pos + 1 :]
        return lines

    def merge_duplicates(self, lines):
        """
        Remove duplicated subtitles. Duplicates are always adjacent.
        """
        last_timestamp = ""
        last_cap = ""
        for line in lines:
            if line == "":
                continue
            if re.match(r"^\d{2}:\d{2}$", line):
                if line != last_timestamp:
                    yield line
                    last_timestamp = line
            else:
                if line != last_cap:
                    yield line
                    last_cap = line

    def merge_short_lines(self, lines):
        buffer = ""
        for line in lines:
            if line == "" or re.match(r"^\d{2}:\d{2}$", line):
                yield "\n" + line
                continue

            if len(line + buffer) < 80:
                buffer += " " + line
            else:
                yield buffer.strip()
                buffer = line
        yield buffer

    def convert_vtt_to_text(self, vtt_file_name):
        """
        Convert VTT file to text and return it as a string
        """
        with open(vtt_file_name, encoding="utf-8") as f:
            text = f.read()
        text = self.remove_tags(text)
        lines = text.splitlines()
        lines = self.remove_header(lines)
        lines = self.merge_duplicates(lines)
        lines = list(lines)
        lines = self.merge_short_lines(lines)
        lines = list(lines)

        result = "\n".join(lines)
        return result

    def download_subtitles(self):
        """
        Download subtitles using yt-dlp
        """
        command = [
            "yt-dlp",
            "--write-auto-subs",
            "--skip-download",
            "--sub-format",
            "vtt",
            self.video_url,
        ]
        subprocess.run(command, check=True)

    def get_video_title(self):
        """
        Get the video title using yt-dlp
        """
        command = ["yt-dlp", "--get-title", self.video_url]
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=True)
        title = result.stdout.strip()
        return title

    async def process(self):
        if not self.video_url:
            self.video_url = input("Please enter the YouTube URL: ")

        # Get the video title
        video_title = self.get_video_title()

        # Download subtitles
        self.download_subtitles()

        # Find the downloaded VTT file
        vtt_files = [f for f in os.listdir(".") if f.endswith(".vtt")]
        if not vtt_files:
            print("No VTT files found.")
            sys.exit(1)

        # Convert VTT to text and process it in memory
        text_contents = []
        for vtt_file in vtt_files:
            text_content = self.convert_vtt_to_text(vtt_file)
            text_content = self.remove_final_timecodes(text_content)
            text_contents.append(text_content)

            if self.return_text:
                print(text_content)
            else:
                # Copy the content to the clipboard
                pyperclip.copy(text_content)
                print(f"Subtitles from {vtt_file} have been copied to the clipboard.")

        # Store the processed text for manipulation
        self.processed_text = "\n".join(text_contents)

        # Delete the VTT files
        for vtt_file in vtt_files:
            os.remove(vtt_file)
            print(f"File {vtt_file} has been deleted.")

        return self.processed_text


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
            "  processed_text = processor.process()\n"
            "  print(processed_text)  # Or manipulate the text as needed"
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
