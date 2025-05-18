# Git Commit Helper

A Node.js wrapper for the Git Commit Helper Python script that automates the git commit process using AI for analysis.

## Features

- Analyzes git diffs to generate meaningful commit messages
- Creates detailed technical summaries of changes
- Integrates with project context from memory-bank
- Recommends semantic version increments (MAJOR, MINOR, PATCH)
- Provides options to commit and push automatically

## Prerequisites

- Node.js 14 or higher
- Python 3.6 or higher
- Git

## Installation

### Global Installation (Recommended)

```bash
npm install -g git-commit-helper
```

This will make the `git-commit-helper` command available globally.

### Local Installation

```bash
npm install git-commit-helper
```

## Usage

Navigate to a git repository and run:

```bash
git-commit-helper
```

This will:
1. Analyze your staged changes
2. Generate a detailed commit message
3. Let you edit the message if needed
4. Commit the changes (unless `--no-commit` is specified)

### Options

- `--push`: Automatically push changes after committing
- `--no-commit`: Generate a commit message without actually committing
- `--debug`: Enable debug logging for troubleshooting

## Example

```bash
# Stage files
git add file1.js file2.js

# Generate commit message and commit
git-commit-helper

# Generate commit message, commit, and push
git-commit-helper --push

# Generate message only without committing
git-commit-helper --no-commit
```

## How It Works

This package is a Node.js wrapper around the Python-based Git Commit Helper. It:

1. Handles command-line arguments
2. Invokes the Python script with appropriate parameters
3. Handles output and error formatting

The Python script does the heavy lifting of analyzing changes, generating the commit message, and handling git operations.

## License

MIT 
noteId: "f30b855032c511f0a51a37b27c0392c3"
tags: []

---

 