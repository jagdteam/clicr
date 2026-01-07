# CodeReader Setup Instructions (macOS)

## Quick Setup for macOS Users

On macOS, Python 3 is accessed via `python3` (not `python`) and you need to use a virtual environment.

### Step 1: Create Virtual Environment

```bash
cd /Users/giuseppi/Documents/jagd/CodeReader
python3 -m venv venv
```

### Step 2: Activate Virtual Environment

```bash
source venv/bin/activate
```

**Important:** You need to activate the virtual environment every time you open a new terminal session!

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

(Note: Inside the virtual environment, you can use `pip` instead of `pip3`)

### Step 4: Create .env File

Create a file named `.env` in the project directory with your Cohere API key:

```bash
echo "COHERE_API_KEY=your_actual_api_key_here" > .env
```

Or create it manually:
1. Create a file called `.env`
2. Add this line: `COHERE_API_KEY=your_actual_api_key_here`
3. Replace `your_actual_api_key_here` with your real API key from https://dashboard.cohere.com/api-keys

### Step 5: Verify Setup

```bash
python validate.py
```

## Usage

Every time you want to use CodeReader:

1. **Activate the virtual environment:**
   ```bash
   cd /Users/giuseppi/Documents/jagd/CodeReader
   source venv/bin/activate
   ```

2. **Run ingest or chat:**
   ```bash
   python ingest.py     # Index your codebase
   python chat.py       # Chat with your code
   ```

3. **Deactivate when done (optional):**
   ```bash
   deactivate
   ```

## Quick Reference

| Command | What it does |
|---------|-------------|
| `python3 -m venv venv` | Create virtual environment (one-time) |
| `source venv/bin/activate` | Activate virtual environment (every session) |
| `pip install -r requirements.txt` | Install dependencies (one-time) |
| `python validate.py` | Check setup |
| `python ingest.py` | Index codebase |
| `python chat.py` | Start chat interface |
| `deactivate` | Exit virtual environment |

## Troubleshooting

### "command not found: python"
Use `python3` instead of `python` on macOS.

### "externally-managed-environment" error
You need to use a virtual environment (see Step 1 above).

### Virtual environment not working
Make sure you've activated it: `source venv/bin/activate`
You should see `(venv)` at the start of your terminal prompt.

### Dependencies not found when running scripts
Activate the virtual environment first: `source venv/bin/activate`

