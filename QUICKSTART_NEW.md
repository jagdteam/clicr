# Quick Start Guide - New Features

## 🚀 Getting Started with the New Interface

### Step 1: Launch the Main Menu

```bash
python main.py
```

You'll see:

```
============================================================
📚 Clicr - RAG-Powered Codebase Chat
============================================================

Select an option:

  1. 💬 Start Chat Session
  2. 🔄 Watch Mode (Incremental Updates)
  3. 📤 Export Chat Session
  4. 📜 View Query History
  5. 📋 View Chat Sessions
  6. 🔧 Settings & Info
  0. 🚪 Exit

Enter your choice:
```

---

## 💬 Feature 1: Chat with Conversation History

### Start a Chat Session

**From Main Menu:**
1. Enter `1` to start chat
2. Optionally name your session
3. Start asking questions!

**Example Session:**

```
Enter session name (or press Enter for default): Feature Research

✅ Connected to ChromaDB collection: codebase_chunks
   Total chunks in database: 1,247
📝 Session started: 20260107_143022

============================================================
💬 CodeReader Chat Interface
============================================================
Ask questions about your codebase!
Type 'exit' or 'quit' to end the session.
Type '/export' to export this session to markdown.
============================================================

You: What does the authentication module do?

🔍 Searching codebase...
🧠 Generating answer...

============================================================
💡 Answer:
============================================================
The authentication module handles user login and session
management. It uses JWT tokens for stateless authentication
and includes password hashing with bcrypt...

------------------------------------------------------------
📄 Sources:
------------------------------------------------------------
   • auth/login.py
   • auth/session.py
   • utils/crypto.py
============================================================

You: How are the tokens validated?

🔍 Searching codebase...
🧠 Generating answer...

[Notice: This answer uses context from your previous question!]

You: /export

✅ Session exported to: ./chat_history/20260107_143022_export.md

You: exit

💾 Session saved: 20260107_143022
   View history with: python main.py --view-session 20260107_143022

👋 Goodbye!
```

**Key Features:**
- ✅ Multi-turn context (remembers previous questions)
- ✅ Auto-saved conversation history
- ✅ Export anytime with `/export`
- ✅ Session IDs for later reference

---

## 🔄 Feature 2: Watch Mode (Incremental Updates)

### Keep Your Database in Sync

**From Main Menu:**
1. Enter `2` for Watch Mode
2. Choose directory to watch (or press Enter for current)
3. Set check interval (default: 10 seconds)

**Example:**

```
Enter directory to watch (or press Enter for current): [Enter]
Enter check interval in seconds (default: 10): 15

============================================================
👀 CodeReader Watch Mode
============================================================
Watching: C:\MyProject
Check interval: 15 seconds
Press Ctrl+C to stop watching
============================================================

============================================================
🚀 CodeReader Ingestion Engine (Incremental)
============================================================
🔍 Crawling directory: C:\MyProject
✅ Found 157 files to process
📝 Found 3 modified files (out of 157 total)

📄 Processing files and creating chunks...
   Processed 3/3 files...
✅ Created 47 chunks from 3 files

🧠 Embedding 47 chunks in batches of 96...
   Embedded batch 1/1
✅ Successfully embedded 47 chunks

💾 Updating ChromaDB incrementally...
   Removing old chunks for 3 modified files...
   Deleted 43 old chunks
   Adding 47 new chunks...
   Added 47/47 chunks...
✅ Database updated! Total chunks: 1,251

============================================================
✨ Ingestion complete! You can now run chat.py
============================================================

⏰ Next check in 15 seconds...

[Monitoring continues...]
```

**Use Cases:**
- Active development (auto-sync code changes)
- Team collaboration (stay updated)
- CI/CD integration (background indexing)

---

## 📤 Feature 3: Export Chat Sessions

### Share Your Conversations

**From Main Menu:**
1. Enter `3` for Export
2. Select session from list
3. Optionally specify output filename

**Example:**

```
============================================================
📤 Export Chat Session
============================================================

Available sessions:

  1. Feature Research (ID: 20260107_143022, Created: 2026-01-07 14:30)
  2. Bug Investigation (ID: 20260107_101500, Created: 2026-01-07 10:15)
  3. Code Review (ID: 20260106_165422, Created: 2026-01-06 16:54)

Enter session number to export (or 'q' to cancel): 1

Enter output filename (or press Enter for default): [Enter]

✅ Session exported successfully!
   File: ./chat_history/20260107_143022_export.md
```

**Exported Markdown Preview:**

```markdown
# Feature Research

**Created:** 2026-01-07T14:30:22
**Session ID:** 20260107_143022

---

## 🙋 User (2026-01-07 14:30:25)

What does the authentication module do?

## 🤖 Assistant (2026-01-07 14:30:28)

The authentication module handles user login and session
management. It uses JWT tokens for stateless authentication
and includes password hashing with bcrypt...

**Sources:**
- `auth/login.py`
- `auth/session.py`
- `utils/crypto.py`

[... rest of conversation ...]
```

---

## 📜 Feature 4: Query History

### Track Your Research

**From Main Menu:**
1. Enter `4` for Query History
2. Choose to view recent or search

**Example - View Recent:**

```
============================================================
📜 Query History
============================================================

Options:
  1. View recent queries
  2. Search queries

Select option (or 'q' to cancel): 1

How many queries to show? (default: 20): [Enter]

Showing last 20 queries:

1. [2026-01-07 14:30:25]
   Query: What does the authentication module do?
   Response: The authentication module handles user login...
   Sources: auth/login.py, auth/session.py, utils/crypto.py

2. [2026-01-07 14:31:10]
   Query: How are the tokens validated?
   Response: Token validation occurs in the middleware...
   Sources: auth/middleware.py, auth/jwt.py

[... more queries ...]
```

**Example - Search:**

```
Select option (or 'q' to cancel): 2

Enter search keyword: database

Found 7 matching queries:

1. [2026-01-07 13:15:42]
   Query: How do we connect to the database?
   Response: The database connection is established using...

2. [2026-01-07 11:22:18]
   Query: What database migrations exist?
   Response: There are currently 15 migrations in the...

[... more results ...]
```

---

## 📋 Bonus: Session Management

**From Main Menu:**
Enter `5` to view all sessions

**Features:**
- View session details (messages, timestamps)
- Delete old sessions
- Quick overview of all conversations

**Example:**

```
============================================================
📋 Chat Sessions
============================================================

Total sessions: 12

1. Feature Research
   ID: 20260107_143022
   Created: 2026-01-07 14:30:22

2. Bug Investigation
   ID: 20260107_101500
   Created: 2026-01-07 10:15:00

[... more sessions ...]

Options:
  1. View session details
  2. Delete session

Select option (or 'q' to cancel): 1

Enter session number to view: 1

============================================================
Session: Feature Research
============================================================
ID: 20260107_143022
Created: 2026-01-07T14:30:22
Messages: 8

Recent messages:

  🙋 You [14:30:25]: What does the authentication module do?
  
  🤖 Assistant [14:30:28]: The authentication module handles user login...
  
  🙋 You [14:31:10]: How are the tokens validated?
  
  🤖 Assistant [14:31:13]: Token validation occurs in the middleware...

[... more messages ...]
```

---

## 🎯 Quick Command Reference

### Direct Commands

```bash
# Start menu
python main.py

# Start chat (standalone)
python chat.py

# Start named session
python chat.py --session "My Session"

# Incremental update (one-time)
python ingest.py --incremental

# Watch mode
python ingest.py --watch

# Watch with custom interval
python ingest.py --watch --interval 30

# View specific session
python main.py --view-session 20260107_143022
```

### In-Chat Commands

```
/export     - Export current session to Markdown
exit        - End session and save
quit        - Same as exit
q           - Same as exit
```

---

## 💡 Pro Tips

### 1. Name Your Sessions
```bash
python chat.py --session "Sprint 3 Planning"
```
Helps you find sessions later!

### 2. Use Watch Mode During Development
```bash
python ingest.py --watch --interval 20
```
Keep database synced automatically!

### 3. Export Important Conversations
Use `/export` during chat to save research for documentation.

### 4. Search Query History
Find past answers without re-asking questions.

### 5. Incremental Updates Save Time
```bash
python ingest.py --incremental
```
80% faster than full re-indexing!

---

## 🐛 Common Issues

**Q: Menu shows [Chat] instead of emojis**  
A: Your terminal doesn't support emojis. Functionality is the same!

**Q: "Session not found" error**  
A: Ensure `./chat_history/` exists and has write permissions.

**Q: Watch mode not detecting changes**  
A: Increase interval (try 15-30 seconds) and check file extensions.

**Q: Export fails**  
A: Check write permissions in `./chat_history/` directory.

---

## 🎉 You're Ready!

Start with:
```bash
python main.py
```

Enjoy the new features! 🚀

