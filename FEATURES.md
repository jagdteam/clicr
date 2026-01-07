# Clicr - New Features Guide

This guide covers the recently added features to Clicr.

## 🎯 Main Menu Interface

Launch the interactive menu:

```bash
python main.py
```

### Menu Options

1. **💬 Start Chat Session** - Begin a new interactive chat with conversation history
2. **🔄 Watch Mode** - Monitor your codebase for changes and auto-update
3. **📤 Export Chat Session** - Export any session to Markdown format
4. **📜 View Query History** - Browse and search past queries
5. **📋 View Chat Sessions** - Manage all your chat sessions
6. **🔧 Settings & Info** - View configuration and system status

---

## 📝 Conversation History

### What It Does

Every chat session now maintains full conversation history, enabling:
- Context-aware follow-up questions
- Multi-turn conversations
- Session persistence and replay

### How to Use

**Start a named session:**

```bash
python chat.py --session "Debugging Auth Flow"
```

**During chat:**
- Your questions and the assistant's responses are automatically saved
- Follow-up questions use context from previous messages
- The assistant remembers what you discussed

**View past sessions:**

```bash
python main.py
# Select option 5: View Chat Sessions
```

### Example Conversation

```
You: What does the authentication function do?
Assistant: [Explains the auth function]

You: How does it handle errors?
Assistant: [Uses context from previous answer to explain error handling]
```

---

## 🔄 Incremental Updates & Watch Mode

### Incremental Updates

Only re-index files that have changed since last ingestion.

**Usage:**

```bash
# One-time incremental update
python ingest.py --incremental
```

**Benefits:**
- Faster processing (only modified files)
- Lower API costs
- Maintains existing embeddings

### Watch Mode

Continuously monitor your codebase and auto-update when files change.

**Usage:**

```bash
# Watch with 10-second intervals (default)
python ingest.py --watch

# Custom interval (30 seconds)
python ingest.py --watch --interval 30
```

**Use Cases:**
- Active development - keep database synced automatically
- Team collaboration - always have latest code indexed
- Continuous integration - run in background during development

**To stop:** Press `Ctrl+C`

---

## 📤 Export Chat Sessions

### What It Does

Export any chat session to a formatted Markdown document.

### How to Use

**During a chat session:**

```
You: [Your question]
Assistant: [Response]

You: /export
✅ Session exported to: ./chat_history/20260107_143022_export.md
```

**From the main menu:**

```bash
python main.py
# Select option 3: Export Chat Session
# Choose the session to export
```

**Via command line:**

```bash
python main.py --view-session 20260107_143022
```

### Export Format

The exported Markdown includes:
- Session metadata (name, ID, timestamp)
- Full conversation history
- Timestamps for each message
- Source file citations

**Example output:**

```markdown
# Debugging Auth Flow

**Created:** 2026-01-07 14:30:22
**Session ID:** 20260107_143022

---

## 🙋 User (14:30:25)

What does the authentication function do?

## 🤖 Assistant (14:30:28)

The authentication function validates user credentials...

**Sources:**
- `auth.py`
- `utils.py`
```

---

## 📜 Query History

### What It Does

Automatically logs all queries with:
- Query text
- Response preview
- Source files used
- Timestamp

### How to Use

**View recent queries:**

```bash
python main.py
# Select option 4: View Query History
# Select option 1: View recent queries
```

**Search queries:**

```bash
python main.py
# Select option 4: View Query History
# Select option 2: Search queries
# Enter keyword: "authentication"
```

### Use Cases

- Find similar past questions
- Review what was asked about specific topics
- Track research progress
- Identify frequently asked questions

---

## 💡 Best Practices

### Session Management

1. **Use descriptive session names** for easy identification
2. **Export important sessions** for documentation
3. **Review query history** to avoid redundant questions

### Incremental Updates

1. **Use watch mode during active development**
2. **Run incremental updates** after pulling changes
3. **Full re-index** only when changing config (chunk size, etc.)

### Conversation Strategy

1. **Ask follow-up questions** - leverage conversation history
2. **Reference previous answers** - "Can you explain that in more detail?"
3. **Build on context** - multi-turn conversations are more efficient

---

## 🔧 Technical Details

### Storage Locations

- **Chat History:** `./chat_history/`
  - `sessions.json` - Session metadata
  - `queries.json` - Query log (last 100 queries)
  - `[session_id].json` - Individual session files
  - `[session_id]_export.md` - Exported Markdown files

- **Database:** `./chroma_db/`
  - `file_hashes.txt` - File modification tracking (for incremental updates)
  - ChromaDB vector storage

### Data Privacy

- All data is stored locally
- No telemetry or external logging
- API calls only to Cohere for embeddings/chat
- Delete sessions anytime from the main menu

### Performance Tips

- **Incremental mode** saves ~80% time for small changes
- **Watch mode interval**: 10-30 seconds recommended
- **Query history** is capped at 100 entries (oldest removed)
- **Session storage** is unlimited (manual cleanup via menu)

---

## 🐛 Troubleshooting

### "Session not found"

Sessions are created automatically when starting chat. If you see this error:
- Check `./chat_history/` exists
- Ensure you have write permissions
- Try starting a new session

### Watch mode not detecting changes

- Ensure sufficient interval (10+ seconds)
- Check file is not in ignored directories
- Verify file extension is in `ALLOWED_EXTENSIONS`

### Export fails

- Ensure `./chat_history/` directory exists
- Check write permissions
- Verify session ID is correct

---

## 📚 Additional Resources

- **[README.md](README.md)** - Main documentation
- **[QUICKSTART.md](QUICKSTART.md)** - Quick setup guide
- **[SETUP.md](SETUP.md)** - Detailed setup instructions
- **[config.py](config.py)** - Configuration options

---

## 🎉 Summary

With these new features, Clicr now offers:

✅ **Persistent Conversations** - Never lose context across questions  
✅ **Smart Updates** - Only process what changed  
✅ **Auto-Sync** - Watch mode keeps database current  
✅ **History Tracking** - Review and search past queries  
✅ **Easy Export** - Share conversations as Markdown  

Enjoy coding with Clicr! 🚀

