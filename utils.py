"""
Utility functions for CodeReader
Shared helper functions used across modules.
"""

import os
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


def load_env_and_validate() -> str:
    """
    Load environment variables and validate API key.

    Returns:
        API key string

    Raises:
        SystemExit: If API key is not found
    """
    load_dotenv()

    api_key = os.getenv('COHERE_API_KEY')
    if not api_key:
        print("❌ Error: COHERE_API_KEY not found in environment variables")
        print("   Please create a .env file with your Cohere API key")
        print("\n   Steps:")
        print("   1. Copy .env.example to .env: cp .env.example .env")
        print("   2. Edit .env and add your Cohere API key")
        print("   3. Get your API key from: https://dashboard.cohere.com/api-keys")
        sys.exit(1)

    return api_key


def validate_directory(path: str) -> Path:
    """
    Validate that a directory exists and is accessible.

    Args:
        path: Directory path to validate

    Returns:
        Resolved Path object

    Raises:
        SystemExit: If directory is invalid or inaccessible
    """
    dir_path = Path(path).resolve()

    if not dir_path.exists():
        print(f"❌ Error: Directory '{path}' does not exist")
        sys.exit(1)

    if not dir_path.is_dir():
        print(f"❌ Error: '{path}' is not a directory")
        sys.exit(1)

    try:
        # Test if directory is readable
        list(dir_path.iterdir())
    except PermissionError:
        print(f"❌ Error: No permission to read directory '{path}'")
        sys.exit(1)

    return dir_path


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def print_header(title: str, width: int = 60) -> None:
    """
    Print a formatted header.

    Args:
        title: Header title
        width: Width of the header
    """
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_section(title: str, width: int = 60) -> None:
    """
    Print a formatted section divider.

    Args:
        title: Section title
        width: Width of the divider
    """
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


def safe_read_file(file_path: Path, max_size_mb: int = 10) -> Optional[str]:
    """
    Safely read a file with size and encoding checks.

    Args:
        file_path: Path to file
        max_size_mb: Maximum file size in MB

    Returns:
        File contents or None if reading fails
    """
    try:
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > max_size_mb * 1024 * 1024:
            print(f"⚠️  Warning: Skipping large file {file_path.name} "
                  f"({format_file_size(file_size)})")
            return None

        # Try multiple encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        print(f"⚠️  Warning: Could not decode {file_path.name} with any encoding")
        return None

    except PermissionError:
        print(f"⚠️  Warning: No permission to read {file_path}")
        return None
    except Exception as e:
        print(f"⚠️  Warning: Error reading {file_path}: {e}")
        return None


def count_tokens_estimate(text: str) -> int:
    """
    Estimate token count for text (rough approximation).

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    # Rough estimate: 1 token ≈ 4 characters
    return len(text) // 4


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def print_progress(current: int, total: int, prefix: str = "Progress:") -> None:
    """
    Print a simple progress indicator.

    Args:
        current: Current progress
        total: Total items
        prefix: Prefix text
    """
    percentage = (current / total) * 100 if total > 0 else 0
    print(f"   {prefix} {current}/{total} ({percentage:.1f}%)")


