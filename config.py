"""
Configuration module for CodeReader
Centralized configuration for easy customization.
"""

# Chunking Configuration
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 50  # Overlapping characters between chunks

# API Configuration
BATCH_SIZE = 96  # Max batch size for Cohere embedding API
COHERE_EMBED_MODEL = "embed-english-v3.0"
COHERE_CHAT_MODEL = "command-r-plus-08-2024"  # Current model as of 2026
# Alternative: "command-a-03-2025" (strongest performing, but may cost more)

# ChromaDB Configuration
COLLECTION_NAME = "codebase_chunks"
CHROMA_DB_PATH = "./chroma_db"

# Retrieval Configuration
TOP_K_RESULTS = 5  # Number of chunks to retrieve for each query

# File Processing Configuration
ALLOWED_EXTENSIONS = {
    # Python
    '.py', '.pyw',
    # JavaScript/TypeScript
    '.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs',
    # JVM languages
    '.java', '.kt', '.scala',
    # C/C++
    '.cpp', '.c', '.h', '.hpp', '.cc', '.cxx',
    # Other compiled languages
    '.go',  # Go
    '.rs',  # Rust
    '.cs',  # C#
    '.swift',  # Swift
    # Dynamic languages
    '.rb',  # Ruby
    '.php',  # PHP
    '.lua',  # Lua
    '.r', '.R',  # R
    '.dart',  # Dart
    # Web technologies
    '.html', '.htm', '.css', '.scss', '.sass', '.less',
    '.vue', '.svelte',
    # Configuration files
    '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
    # Documentation
    '.md', '.rst', '.txt',
    # Database
    '.sql',
    # Shell scripts
    '.sh', '.bash', '.zsh', '.fish',
    # Markup
    '.xml',
    # Apple
    '.m', '.mm',  # Objective-C
}

# Directories to ignore during crawling
IGNORE_DIRS = {
    '.git', '__pycache__', 'node_modules', '.venv', 'venv',
    'env', '.env', 'build', 'dist', '.next', '.nuxt',
    'target', 'bin', 'obj', '.idea', '.vscode',
    '.pytest_cache', '.mypy_cache', '.tox', 'coverage',
}

# Files to ignore during crawling
IGNORE_FILES = {
    '.env', '.env.local', '.env.production', '.env.development',
    '.DS_Store', 'Thumbs.db', '.gitignore', '.dockerignore',
}


