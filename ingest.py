#!/usr/bin/env python3
"""
Ingestion Engine for CodeReader
Recursively crawls, chunks, embeds, and stores code files in ChromaDB.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
import cohere
import chromadb
from chromadb.config import Settings

# Load environment variables
load_dotenv()

# Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BATCH_SIZE = 96
COLLECTION_NAME = "codebase_chunks"
CHROMA_DB_PATH = "./chroma_db"
COHERE_EMBED_MODEL = "embed-english-v3.0"

# File extensions to process (common code and text files)
ALLOWED_EXTENSIONS = {
    '.py', '.pyw',  # Python
    '.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs',  # JavaScript/TypeScript
    '.java', '.kt', '.scala',  # JVM languages
    '.cpp', '.c', '.h', '.hpp', '.cc', '.cxx',  # C/C++
    '.go',  # Go
    '.rs',  # Rust
    '.rb',  # Ruby
    '.php',  # PHP
    '.swift',  # Swift
    '.m', '.mm',  # Objective-C
    '.cs',  # C#
    '.html', '.htm', '.css', '.scss', '.sass', '.less',  # Web
    '.vue', '.svelte',  # Frontend frameworks
    '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',  # Config
    '.md', '.rst', '.txt',  # Documentation
    '.sql',  # SQL
    '.sh', '.bash', '.zsh', '.fish',  # Shell scripts
    '.xml',  # XML
    '.r', '.R',  # R
    '.lua',  # Lua
    '.dart',  # Dart
}

# Directories and files to ignore
IGNORE_DIRS = {
    '.git', '__pycache__', 'node_modules', '.venv', 'venv',
    'env', '.env', 'build', 'dist', '.next', '.nuxt',
    'target', 'bin', 'obj', '.idea', '.vscode',
}

IGNORE_FILES = {
    '.env', '.env.local', '.env.production', '.env.development',
    '.DS_Store', 'Thumbs.db',
}


def should_process_file(file_path: Path) -> bool:
    """
    Determine if a file should be processed based on extension and name.

    Args:
        file_path: Path to the file

    Returns:
        True if file should be processed, False otherwise
    """
    if file_path.name in IGNORE_FILES:
        return False

    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False

    return True


def crawl_directory(root_dir: str) -> List[Path]:
    """
    Recursively crawl directory and collect processable files.

    Args:
        root_dir: Root directory to crawl

    Returns:
        List of file paths to process
    """
    root_path = Path(root_dir).resolve()
    files_to_process = []

    print(f"🔍 Crawling directory: {root_path}")

    for item in root_path.rglob('*'):
        try:
            # Skip ignored directories
            if any(ignored in item.parts for ignored in IGNORE_DIRS):
                continue

            # Skip if not a file
            if not item.is_file():
                continue

            # Check if should process
            if should_process_file(item):
                files_to_process.append(item)

        except (PermissionError, OSError) as e:
            print(f"⚠️  Warning: Cannot access {item}: {e}")
            continue

    print(f"✅ Found {len(files_to_process)} files to process")
    return files_to_process


def read_file_safely(file_path: Path) -> Optional[str]:
    """
    Safely read file contents with multiple encoding attempts.

    Args:
        file_path: Path to file

    Returns:
        File contents as string, or None if reading fails
    """
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, PermissionError):
            continue
        except Exception as e:
            print(f"⚠️  Error reading {file_path}: {e}")
            return None

    print(f"⚠️  Could not decode {file_path} with any encoding")
    return None


def chunk_text(text: str, file_path: Path, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[Dict[str, any]]:
    """
    Split text into overlapping chunks with metadata.

    Args:
        text: Text content to chunk
        file_path: Path to source file
        chunk_size: Size of each chunk in characters
        overlap: Number of overlapping characters between chunks

    Returns:
        List of chunk dictionaries with text and metadata
    """
    if not text or not text.strip():
        return []

    chunks = []
    text_length = len(text)
    start = 0
    chunk_index = 0

    # Calculate approximate line number for start position
    lines = text.split('\n')

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text = text[start:end].strip()

        if chunk_text:
            # Calculate approximate line numbers
            chars_before = text[:start]
            start_line = chars_before.count('\n') + 1
            chars_in_chunk = text[start:end]
            end_line = start_line + chars_in_chunk.count('\n')

            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'file_path': str(file_path),
                    'chunk_index': chunk_index,
                    'start_line': start_line,
                    'end_line': end_line,
                    'file_name': file_path.name,
                    'file_extension': file_path.suffix,
                }
            })
            chunk_index += 1

        # Move forward with overlap
        start += (chunk_size - overlap)

    return chunks


def process_files(file_paths: List[Path]) -> List[Dict[str, any]]:
    """
    Process files into chunks with metadata.

    Args:
        file_paths: List of file paths to process

    Returns:
        List of all chunks from all files
    """
    all_chunks = []
    processed_count = 0

    print(f"\n📄 Processing files and creating chunks...")

    for file_path in file_paths:
        try:
            content = read_file_safely(file_path)
            if content is None:
                continue

            chunks = chunk_text(content, file_path)
            all_chunks.extend(chunks)
            processed_count += 1

            if processed_count % 10 == 0:
                print(f"   Processed {processed_count}/{len(file_paths)} files...")

        except Exception as e:
            print(f"⚠️  Error processing {file_path}: {e}")
            continue

    print(f"✅ Created {len(all_chunks)} chunks from {processed_count} files")
    return all_chunks


def embed_chunks_batch(chunks: List[Dict[str, any]], cohere_client: cohere.Client) -> List[Dict[str, any]]:
    """
    Embed chunks in batches using Cohere API.

    Args:
        chunks: List of chunk dictionaries
        cohere_client: Initialized Cohere client

    Returns:
        List of chunks with embeddings added
    """
    print(f"\n🧠 Embedding {len(chunks)} chunks in batches of {BATCH_SIZE}...")

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_texts = [chunk['text'] for chunk in batch]

        try:
            # Embed with input_type="search_document" for ingestion
            response = cohere_client.embed(
                texts=batch_texts,
                model=COHERE_EMBED_MODEL,
                input_type="search_document",
                embedding_types=["float"]
            )

            # Add embeddings to chunks
            for j, embedding in enumerate(response.embeddings.float):
                chunks[i + j]['embedding'] = embedding

            print(f"   Embedded batch {i//BATCH_SIZE + 1}/{(len(chunks)-1)//BATCH_SIZE + 1}")

        except Exception as e:
            print(f"❌ Error embedding batch starting at index {i}: {e}")
            # Skip this batch but continue with others
            continue

    # Filter out chunks without embeddings (failed batches)
    embedded_chunks = [c for c in chunks if 'embedding' in c]
    print(f"✅ Successfully embedded {len(embedded_chunks)} chunks")

    return embedded_chunks


def store_in_chromadb(chunks: List[Dict[str, any]]) -> None:
    """
    Store embedded chunks in ChromaDB.

    Args:
        chunks: List of chunks with embeddings and metadata
    """
    print(f"\n💾 Storing chunks in ChromaDB...")

    try:
        # Initialize persistent ChromaDB client
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

        # Get or create collection
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
            # Delete existing collection to start fresh
            client.delete_collection(name=COLLECTION_NAME)
            print(f"   Deleted existing collection: {COLLECTION_NAME}")
        except:
            pass

        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Codebase chunks for RAG"}
        )

        # Prepare data for ChromaDB
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        embeddings = [chunk['embedding'] for chunk in chunks]
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]

        # Add to collection in batches (ChromaDB has its own batch size limits)
        batch_size = 1000
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            print(f"   Stored {end_idx}/{len(chunks)} chunks...")

        print(f"✅ Successfully stored {len(chunks)} chunks in ChromaDB")
        print(f"   Collection: {COLLECTION_NAME}")
        print(f"   Location: {CHROMA_DB_PATH}")

    except Exception as e:
        print(f"❌ Error storing in ChromaDB: {e}")
        raise


def ingest_codebase(target_dir: str) -> None:
    """
    Main ingestion pipeline.

    Args:
        target_dir: Directory to ingest
    """
    print("=" * 60)
    print("🚀 CodeReader Ingestion Engine")
    print("=" * 60)

    # Check for API key
    api_key = os.getenv('COHERE_API_KEY')
    if not api_key:
        print("❌ Error: COHERE_API_KEY not found in environment variables")
        print("   Please create a .env file with your Cohere API key")
        sys.exit(1)

    try:
        # Initialize Cohere client
        cohere_client = cohere.Client(api_key=api_key)

        # Step 1: Crawl directory
        file_paths = crawl_directory(target_dir)
        if not file_paths:
            print("⚠️  No files found to process")
            return

        # Step 2: Process files into chunks
        chunks = process_files(file_paths)
        if not chunks:
            print("⚠️  No chunks created")
            return

        # Step 3: Embed chunks
        embedded_chunks = embed_chunks_batch(chunks, cohere_client)
        if not embedded_chunks:
            print("❌ No chunks were successfully embedded")
            return

        # Step 4: Store in ChromaDB
        store_in_chromadb(embedded_chunks)

        print("\n" + "=" * 60)
        print("✨ Ingestion complete! You can now run chat.py")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️  Ingestion interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error during ingestion: {e}")
        sys.exit(1)


def main():
    """Entry point for CLI."""
    # Default to current directory, or take first argument
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    if not os.path.isdir(target_dir):
        print(f"❌ Error: '{target_dir}' is not a valid directory")
        sys.exit(1)

    ingest_codebase(target_dir)


if __name__ == "__main__":
    main()


