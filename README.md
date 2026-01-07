# CodeReader - RAG-Powered Codebase Chat

A robust CLI tool that lets you chat with your local codebase using RAG (Retrieval-Augmented Generation).

## Features

- 🔍 Recursively ingests code files from your project
- 🧠 Uses Cohere embeddings and Command R model
- 💾 Persistent vector storage with ChromaDB
- 💬 Interactive chat interface with source citations
- 🎯 Zero-config design - works out of the box

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your API key:**
   ```bash
   cp .env.example .env
   # Edit .env and add your Cohere API key
   ```

## Usage

### 1. Ingest Your Codebase

Run this in your project directory to index all code files:

```bash
python ingest.py
```

By default, it indexes the current directory (`.`). You can specify a different path:

```bash
python ingest.py /path/to/your/project
```

### 2. Chat with Your Code

Start an interactive chat session:

```bash
python chat.py
```

Ask questions about your codebase:
- "How does the authentication work?"
- "Where is the database connection configured?"
- "What does the User class do?"

Type `exit` or `quit` to end the session.

## Configuration

### Supported File Types

The tool automatically processes these file extensions:
- Python: `.py`, `.pyw`
- JavaScript/TypeScript: `.js`, `.jsx`, `.ts`, `.tsx`
- Web: `.html`, `.css`, `.scss`, `.vue`
- Config: `.json`, `.yaml`, `.yml`, `.toml`, `.ini`, `.cfg`
- Documentation: `.md`, `.txt`, `.rst`
- And many more...

### Ignored Directories

The following are automatically excluded:
- `.git`, `__pycache__`, `node_modules`
- `.env` files and other sensitive files
- Binary files and images

### Chunking Strategy

- Chunk size: 500 characters
- Overlap: 50 characters
- Ensures context preservation across chunks

## Technical Stack

- **LLM/Embeddings**: Cohere API (embed-english-v3.0, command-r-plus)
- **Vector DB**: ChromaDB (persistent storage in `./chroma_db/`)
- **Language**: Python 3.10+

## Error Handling

The tool includes robust error handling for:
- File permission errors
- API rate limits
- Network issues
- Invalid file encodings

## License

MIT


