# CodeReader - Quick Start Guide

Get up and running with CodeReader in 3 minutes!

## Prerequisites

- Python 3.10 or higher
- A Cohere API key ([Get one free here](https://dashboard.cohere.com/api-keys))

## Installation

### 1. Navigate to Project

```bash
cd /Users/giuseppi/Documents/jagd/CodeReader
```

### 2. Create & Activate Virtual Environment (macOS)

```bash
python3 -m venv venv
source venv/bin/activate
```

**Note:** On macOS, use `python3` instead of `python`. You'll need to activate the virtual environment every time you open a new terminal.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Your API Key

Create a `.env` file in the project root:

```bash
echo "COHERE_API_KEY=your_actual_api_key_here" > .env
```

Or manually create `.env` with this content:

```
COHERE_API_KEY=your_actual_api_key_here
```

**Get your API key from:** https://dashboard.cohere.com/api-keys

### 5. Validate Setup (Optional)

Run the validation script to ensure everything is configured correctly:

```bash
python validate.py
```

**Important:** Make sure your virtual environment is activated (you should see `(venv)` in your terminal prompt)

This will check:
- ✅ Python version
- ✅ Dependencies installed
- ✅ Project files present
- ✅ API key configured
- ✅ Cohere API connection

## Usage

### Step 1: Ingest Your Codebase

Navigate to the directory you want to index and run:

```bash
python /path/to/CodeReader/ingest.py
```

Or to index a specific directory:

```bash
python /path/to/CodeReader/ingest.py /path/to/your/project
```

This will:
1. 🔍 Crawl all code files
2. ✂️ Split them into chunks
3. 🧠 Generate embeddings
4. 💾 Store in ChromaDB

**Time estimate:** ~30 seconds for a small project, 2-5 minutes for large projects.

### Step 2: Chat with Your Code

Start the interactive chat interface:

```bash
python /path/to/CodeReader/chat.py
```

Then ask questions like:
- "How does authentication work in this codebase?"
- "Where is the database connection configured?"
- "What does the User class do?"
- "Show me how API requests are handled"

## Example Session

```
$ python ingest.py
============================================================
🚀 CodeReader Ingestion Engine
============================================================
🔍 Crawling directory: /Users/you/myproject
✅ Found 47 files to process

📄 Processing files and creating chunks...
✅ Created 234 chunks from 47 files

🧠 Embedding 234 chunks in batches of 96...
✅ Successfully embedded 234 chunks

💾 Storing chunks in ChromaDB...
✅ Successfully stored 234 chunks in ChromaDB

============================================================
✨ Ingestion complete! You can now run chat.py
============================================================

$ python chat.py
============================================================
💬 CodeReader Chat Interface
============================================================
Ask questions about your codebase!
Type 'exit' or 'quit' to end the session.
============================================================

You: How does authentication work?
🔍 Searching codebase...
🧠 Generating answer...

============================================================
💡 Answer:
============================================================
The authentication is handled using JWT tokens. The login
endpoint validates credentials and returns a token...

------------------------------------------------------------
📚 Citations:
------------------------------------------------------------
   • src/auth/jwt.py (lines 12-45)
   • src/routes/auth.py (lines 23-67)

------------------------------------------------------------
📄 Sources:
------------------------------------------------------------
   • src/auth/jwt.py
   • src/routes/auth.py
   • src/middleware/auth.py
============================================================
```

## Troubleshooting

### "COHERE_API_KEY not found"

Make sure you've created a `.env` file with your API key.

### "No files found to process"

Check that you're in the right directory and it contains code files.

### "Could not find collection"

Run `python ingest.py` first to index your codebase.

### Rate limit errors

Wait a few moments and try again. Consider reducing BATCH_SIZE in `config.py`.

## Configuration

Edit `config.py` to customize:
- Chunk size and overlap
- Number of results retrieved
- File extensions to process
- Directories to ignore

## Tips

1. **Re-index after major changes**: Run `ingest.py` again after adding new files or making significant changes.

2. **Specific questions work best**: Instead of "tell me about this code", ask "how does the payment processing work?"

3. **Check sources**: Always review the source files cited to verify the answer.

4. **Multiple projects**: Run `ingest.py` in each project directory. ChromaDB will be created locally in each project.

## Need Help?

Check the main [README.md](README.md) for detailed documentation.


