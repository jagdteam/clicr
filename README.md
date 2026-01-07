# Clicr - RAG-Powered Codebase Chat

Clicr (short for CLI Reader) is a robust CLI tool that lets you chat with your local codebase using RAG (Retrieval-Augmented Generation).

## Features

- 🔍 Recursively ingests code files from your project
- 🧠 Uses Cohere embeddings and Command R model
- 💾 Persistent vector storage with ChromaDB
- 💬 Interactive chat interface with source citations
- 🎯 Zero-config design - works out of the box

## Prerequisites

- Python 3.10 or higher
- A Cohere API key ([Get one free here](https://dashboard.cohere.com/api-keys))

## Setup

### 1. Create and Activate Virtual Environment

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

> **Note:** On macOS, use `python3` instead of `python`. You'll need to activate the virtual environment every time you open a new terminal session.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Create a `.env` file in the project root:

```bash
echo "COHERE_API_KEY=your_actual_api_key_here" > .env
```

Or manually create a `.env` file with this content:

```
COHERE_API_KEY=your_actual_api_key_here
```

Replace `your_actual_api_key_here` with your real API key from https://dashboard.cohere.com/api-keys

### 4. Validate Setup (Optional but Recommended)

```bash
python validate.py
```

This checks your Python version, dependencies, and API connection.

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

You can customize these settings in `config.py`.

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

## Helper Script

For convenience, you can use the provided helper script to automatically activate the virtual environment:

```bash
./activate_and_run.sh ingest.py
./activate_and_run.sh chat.py
```

## Troubleshooting

### "COHERE_API_KEY not found"

Make sure you've created a `.env` file with your API key.

### "command not found: python"

On macOS, use `python3` instead of `python`.

### "externally-managed-environment" error

You need to use a virtual environment (see Setup step 1).

### "No files found to process"

Check that you're in the right directory and it contains code files.

### "Could not find collection"

Run `python ingest.py` first to index your codebase.

## Additional Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Quick 3-minute setup guide with examples
- **[SETUP.md](SETUP.md)** - Detailed macOS-specific setup instructions

## License

MIT
