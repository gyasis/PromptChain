import subprocess
import shlex
import sys
import os

class RipgrepSearcher:
    """
    A class to perform searches using the ripgrep (rg) command-line tool.

    Requires ripgrep (rg) to be installed and accessible in the system PATH,
    or the path to the rg executable must be provided.
    """

    def __init__(self, rg_path='rg'):
        """
        Initializes the RipgrepSearcher.

        Args:
            rg_path (str): The path to the ripgrep (rg) executable.
                           Defaults to 'rg', assuming it's in the PATH.
        """
        self.rg_path = rg_path
        self._check_ripgrep()

    def _check_ripgrep(self):
        """Verify that ripgrep is available."""
        try:
            # Use '--version' which typically has exit code 0
            subprocess.run([self.rg_path, '--version'], check=True, capture_output=True, text=True)
        except FileNotFoundError:
            print(f"Error: '{self.rg_path}' command not found.", file=sys.stderr)
            print("Please install ripgrep (e.g., 'apt install ripgrep' or 'brew install ripgrep') or provide the correct path.", file=sys.stderr)
            raise
        except subprocess.CalledProcessError as e:
            # This might happen if rg exists but --version fails for some reason
            print(f"Error: '{self.rg_path}' command failed during version check.", file=sys.stderr)
            if e.stderr:
                print(f"Stderr: {e.stderr}", file=sys.stderr)
            raise

    def search(
        self,
        query: str,
        search_path: str = '.',
        case_sensitive: bool | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        fixed_strings: bool = False,
        word_regexp: bool = False,
        multiline: bool = False,
        # Consider adding more rg flags as needed, e.g.:
        # context_lines: int = 0, (-C)
        # count_matches: bool = False, (-c)
        # print_filenames_only: bool = False, (-l)
    ) -> list[str]:
        """
        Performs a search using ripgrep.

        Args:
            query (str): The pattern (regex or literal) to search for.
            search_path (str): The directory or file to search within. Defaults to '.'.
            case_sensitive (bool | None):
                True: Force case-sensitive search.
                False: Force case-insensitive search (-i).
                None: Use smart case (-S, default rg behavior).
            include_patterns (list[str] | None): List of glob patterns for files to include (-g).
            exclude_patterns (list[str] | None): List of glob patterns for files to exclude (-g !pattern).
            fixed_strings (bool): Treat the query as literal strings (-F). Defaults to False.
            word_regexp (bool): Match only whole words (-w). Defaults to False.
            multiline (bool): Allow matches to span multiple lines (-U). Defaults to False.

        Returns:
            list[str]: A list of matching lines from the ripgrep output. Returns an empty list
                       if no matches are found (rg exit code 1).

        Raises:
            subprocess.CalledProcessError: If the ripgrep command fails with an exit code > 1.
            FileNotFoundError: If rg_path is invalid (should be caught by __init__).
        """
        cmd = [self.rg_path]

        # --- Handle Flags ---
        if case_sensitive is False:
            cmd.append('-i')
        elif case_sensitive is None: # Smart case is default unless query has uppercase
            cmd.append('-S')
        # if case_sensitive is True, no flag needed (rg default)

        if fixed_strings:
            cmd.append('-F')
        if word_regexp:
            cmd.append('-w')
        if multiline:
            cmd.append('-U')

        # Add include globs
        if include_patterns:
            for pattern in include_patterns:
                cmd.extend(['-g', pattern])

        # Add exclude globs
        if exclude_patterns:
            for pattern in exclude_patterns:
                # Format for rg exclude glob: -g '!<pattern>'
                cmd.extend(['-g', f'!{pattern}'])

        # Add required arguments (ensure query comes before path if path is specified)
        cmd.append(query)
        cmd.append(search_path)

        # --- Execute Command ---
        try:
            # Using text=True for automatic encoding handling based on locale
            # Use --with-filename and --line-number for consistent output format
            result = subprocess.run(
                cmd + ['--with-filename', '--line-number'],
                check=False,         # Don't raise error on exit code 1 (no matches)
                capture_output=True, # Capture stdout/stderr
                text=True,           # Decode stdout/stderr as text
                encoding=sys.getdefaultencoding() # Be explicit about encoding
            )

            # Check for actual errors (exit code > 1)
            if result.returncode > 1:
                 # Create a CalledProcessError manually for consistent error handling
                error = subprocess.CalledProcessError(
                    result.returncode, cmd, output=result.stdout, stderr=result.stderr
                )
                print(f"Error running ripgrep: {' '.join(shlex.quote(c) for c in cmd)}", file=sys.stderr)
                print(f"Exit Code: {error.returncode}", file=sys.stderr)
                if error.stdout:
                    print(f"Stdout: {error.stdout}", file=sys.stderr)
                if error.stderr:
                    print(f"Stderr: {error.stderr}", file=sys.stderr)
                raise error # Raise the constructed error

            # Split output into lines, removing potential trailing newline
            # Filter out empty lines which might occur
            output_lines = [line for line in result.stdout.strip().splitlines() if line]
            return output_lines

        except FileNotFoundError:
             # This case should be caught by __init__, but handle defensively
            print(f"Error: '{self.rg_path}' not found during search execution.", file=sys.stderr)
            raise


# --- Example Usage ---
if __name__ == "__main__":
    # Example assumes 'rg' is installed and in PATH
    try:
        searcher = RipgrepSearcher()
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Failed to initialize RipgrepSearcher. Ensure 'rg' is installed and in PATH.")
        sys.exit(1)


    # Create dummy files for testing in a temporary directory
    test_dir = "rg_temp_test_dir"
    try:
        os.makedirs(test_dir, exist_ok=True)
        file1_path = os.path.join(test_dir, "file1.txt")
        file2_path = os.path.join(test_dir, "file2.log")
        file3_path = os.path.join(test_dir, "file3.txt")

        with open(file1_path, "w", encoding='utf-8') as f:
            f.write("Hello World\n")
            f.write("hello there\n")
            f.write("Another line with hello\n")
        with open(file2_path, "w", encoding='utf-8') as f:
            f.write("Log entry: hello\n")
            f.write("DEBUG: hello again\n")
        with open(file3_path, "w", encoding='utf-8') as f:
            f.write("HELLO world pattern\n")
            f.write("File with Ümlauts\n") # Test non-ASCII

        print(f"--- Test Files Created in ./{test_dir} ---")

        print("--- Basic Search (Smart Case) ---")
        try:
            matches = searcher.search("hello", search_path=test_dir)
            for line in matches: print(line)
        except Exception as e: print(f"Search failed: {e}", file=sys.stderr)

        print("--- Case Insensitive Search ---")
        try:
            matches = searcher.search("hello", search_path=test_dir, case_sensitive=False)
            for line in matches: print(line)
        except Exception as e: print(f"Search failed: {e}", file=sys.stderr)

        print("--- Case Sensitive Search ---")
        try:
            matches = searcher.search("hello", search_path=test_dir, case_sensitive=True)
            for line in matches: print(line)
        except Exception as e: print(f"Search failed: {e}", file=sys.stderr)

        print("--- Include only .txt ---")
        try:
            matches = searcher.search("hello", search_path=test_dir, case_sensitive=False, include_patterns=["*.txt"])
            for line in matches: print(line)
        except Exception as e: print(f"Search failed: {e}", file=sys.stderr)

        print("--- Exclude .log ---")
        try:
            matches = searcher.search("hello", search_path=test_dir, case_sensitive=False, exclude_patterns=["*.log"])
            for line in matches: print(line)
        except Exception as e: print(f"Search failed: {e}", file=sys.stderr)

        print("--- Fixed Strings Search ('Hello World') ---")
        try:
            matches = searcher.search("Hello World", search_path=test_dir, fixed_strings=True)
            for line in matches: print(line)
        except Exception as e: print(f"Search failed: {e}", file=sys.stderr)

        print("--- Regex Search ('^Log.*:') ---")
        try:
            matches = searcher.search("^Log.*:", search_path=test_dir) # Regex pattern
            for line in matches: print(line)
        except Exception as e: print(f"Search failed: {e}", file=sys.stderr)

        print("\n--- Word Boundary Search ('hello') ---")
        try:
            matches = searcher.search("hello", search_path=test_dir, case_sensitive=False, word_regexp=True)
            for line in matches: print(line)
        except Exception as e: print(f"Search failed: {e}", file=sys.stderr)

        print("\n--- Search non-ASCII ('Ümlauts') ---")
        try:
            matches = searcher.search("Ümlauts", search_path=test_dir, case_sensitive=False)
            for line in matches: print(line)
        except Exception as e: print(f"Search failed: {e}", file=sys.stderr)

        print("\n--- Search Non-existent pattern ---")
        try:
            matches = searcher.search("thispatternshouldnotexistanywhere", search_path=test_dir)
            if not matches:
                print("(No matches found - expected)")
            else:
                print("Error: Unexpected matches found for non-existent pattern:")
                for line in matches: print(line)
        except Exception as e: print(f"Search failed: {e}", file=sys.stderr)

    finally:
        # Clean up dummy files/dir
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir)
            print(f"--- Cleaned up ./{test_dir} ---") 