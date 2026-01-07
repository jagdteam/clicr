#!/usr/bin/env python3
"""
Main CLI Interface for CodeReader
Provides a menu-driven interface to access all features.
"""

import os
import sys
from datetime import datetime
from typing import Optional
from pathlib import Path

# Fix Windows encoding issue
if sys.platform == 'win32':
    try:
        # Try to set UTF-8 encoding for Windows console
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from dotenv import load_dotenv
from history import HistoryManager
from chat import CodebaseChat
from ingest import ingest_codebase, watch_mode

# Load environment variables
load_dotenv()

# Detect if emojis are supported
USE_EMOJIS = True
try:
    test_string = "📚💬🔄"
    test_string.encode(sys.stdout.encoding or 'utf-8')
except (UnicodeEncodeError, AttributeError):
    USE_EMOJIS = False


def emoji(emoji_char: str, fallback: str) -> str:
    """
    Return emoji or fallback text based on system support.

    Args:
        emoji_char: Emoji character
        fallback: Fallback text if emoji not supported

    Returns:
        Emoji or fallback string
    """
    return emoji_char if USE_EMOJIS else fallback


def clear_screen() -> None:
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header() -> None:
    """Print the main header."""
    print("\n" + "=" * 60)
    print(f"{emoji('📚', '[Clicr]')} Clicr - RAG-Powered Codebase Chat")
    print("=" * 60)


def print_menu() -> None:
    """Print the main menu."""
    print("\nSelect an option:")
    print()
    print(f"  1. {emoji('💬', '[Chat]')} Start Chat Session")
    print(f"  2. {emoji('🔄', '[Watch]')} Watch Mode (Incremental Updates)")
    print(f"  3. {emoji('📤', '[Export]')} Export Chat Session")
    print(f"  4. {emoji('📜', '[History]')} View Query History")
    print(f"  5. {emoji('📋', '[Sessions]')} View Chat Sessions")
    print(f"  6. {emoji('🔧', '[Settings]')} Settings & Info")
    print(f"  0. {emoji('🚪', '[Exit]')} Exit")
    print()


def start_chat_session() -> None:
    """Start a new chat session."""
    clear_screen()
    print_header()
    print(f"\n{emoji('💬', '[Chat]')} Starting Chat Session...")
    print()
    
    session_name = input("Enter session name (or press Enter for default): ").strip()
    
    if not session_name:
        session_name = None
    
    print()
    
    try:
        chat = CodebaseChat(enable_history=True, session_name=session_name)
        chat.chat_loop()
    except Exception as e:
        print(f"\n{emoji('❌', '[ERROR]')} Error starting chat: {e}")
        input("\nPress Enter to continue...")


def watch_mode_interface() -> None:
    """Interface for watch mode."""
    clear_screen()
    print_header()
    print(f"\n{emoji('👀', '[Watch]')} Watch Mode - Incremental Updates")
    print("=" * 60)
    print("\nThis will monitor your codebase for changes and")
    print("automatically update the database when files are modified.")
    print()
    
    target_dir = input("Enter directory to watch (or press Enter for current): ").strip()
    if not target_dir:
        target_dir = "."
    
    interval_input = input("Enter check interval in seconds (default: 10): ").strip()
    
    try:
        interval = int(interval_input) if interval_input else 10
    except ValueError:
        print(f"{emoji('⚠️', '[WARNING]')} Invalid interval, using default (10 seconds)")
        interval = 10
    
    print()
    
    try:
        watch_mode(target_dir, interval)
    except Exception as e:
        print(f"\n{emoji('❌', '[ERROR]')} Error in watch mode: {e}")
        input("\nPress Enter to continue...")


def export_session_interface() -> None:
    """Interface for exporting chat sessions."""
    clear_screen()
    print_header()
    print(f"\n{emoji('📤', '[Export]')} Export Chat Session")
    print("=" * 60)
    
    history_manager = HistoryManager()
    sessions = history_manager.list_sessions()
    
    if not sessions:
        print(f"\n{emoji('⚠️', '[WARNING]')} No chat sessions found.")
        input("\nPress Enter to continue...")
        return
    
    print("\nAvailable sessions:")
    print()
    
    for i, session in enumerate(sessions[-10:], 1):  # Show last 10 sessions
        created = datetime.fromisoformat(session['created_at']).strftime("%Y-%m-%d %H:%M")
        print(f"  {i}. {session['name']} (ID: {session['id']}, Created: {created})")
    
    print()
    choice = input("Enter session number to export (or 'q' to cancel): ").strip()
    
    if choice.lower() == 'q':
        return
    
    try:
        session_idx = int(choice) - 1
        if 0 <= session_idx < len(sessions[-10:]):
            selected_session = sessions[-10:][session_idx]
            
            output_file = input("\nEnter output filename (or press Enter for default): ").strip()
            if not output_file:
                output_file = None
            
            export_path = history_manager.export_session_markdown(
                selected_session['id'], output_file
            )
            print(f"\n{emoji('✅', '[SUCCESS]')} Session exported successfully!")
            print(f"   File: {export_path}")
        else:
            print(f"\n{emoji('⚠️', '[WARNING]')} Invalid session number")
    except ValueError:
        print(f"\n{emoji('⚠️', '[WARNING]')} Invalid input")
    except Exception as e:
        print(f"\n{emoji('❌', '[ERROR]')} Error exporting session: {e}")
    
    input("\nPress Enter to continue...")


def view_query_history() -> None:
    """View query history."""
    clear_screen()
    print_header()
    print(f"\n{emoji('📜', '[History]')} Query History")
    print("=" * 60)
    
    history_manager = HistoryManager()
    
    print("\nOptions:")
    print("  1. View recent queries")
    print("  2. Search queries")
    print()
    
    choice = input("Select option (or 'q' to cancel): ").strip()
    
    if choice == '1':
        limit_input = input("\nHow many queries to show? (default: 20): ").strip()
        try:
            limit = int(limit_input) if limit_input else 20
        except ValueError:
            limit = 20
        
        queries = history_manager.get_query_history(limit)
        
        if not queries:
            print(f"\n{emoji('⚠️', '[WARNING]')} No queries found in history.")
        else:
            print(f"\nShowing last {len(queries)} queries:")
            print()
            
            for i, query in enumerate(queries, 1):
                timestamp = datetime.fromisoformat(query['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
                print(f"{i}. [{timestamp}]")
                print(f"   Query: {query['query']}")
                print(f"   Response: {query['response_preview'][:100]}...")
                print(f"   Sources: {', '.join(query['sources'][:3])}")
                if len(query['sources']) > 3:
                    print(f"            (and {len(query['sources']) - 3} more)")
                print()
    
    elif choice == '2':
        keyword = input("\nEnter search keyword: ").strip()
        
        if keyword:
            queries = history_manager.search_queries(keyword)
            
            if not queries:
                print(f"\n{emoji('⚠️', '[WARNING]')} No queries found matching '{keyword}'")
            else:
                print(f"\nFound {len(queries)} matching queries:")
                print()
                
                for i, query in enumerate(queries[:20], 1):  # Show first 20 results
                    timestamp = datetime.fromisoformat(query['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
                    print(f"{i}. [{timestamp}]")
                    print(f"   Query: {query['query']}")
                    print(f"   Response: {query['response_preview'][:100]}...")
                    print()
    
    input("\nPress Enter to continue...")


def view_chat_sessions() -> None:
    """View all chat sessions."""
    clear_screen()
    print_header()
    print(f"\n{emoji('📋', '[Sessions]')} Chat Sessions")
    print("=" * 60)
    
    history_manager = HistoryManager()
    sessions = history_manager.list_sessions()
    
    if not sessions:
        print(f"\n{emoji('⚠️', '[WARNING]')} No chat sessions found.")
        input("\nPress Enter to continue...")
        return
    
    print(f"\nTotal sessions: {len(sessions)}")
    print()
    
    # Show all sessions, most recent first
    for i, session in enumerate(reversed(sessions), 1):
        created = datetime.fromisoformat(session['created_at']).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{i}. {session['name']}")
        print(f"   ID: {session['id']}")
        print(f"   Created: {created}")
        print()
    
    print("\nOptions:")
    print("  1. View session details")
    print("  2. Delete session")
    print()
    
    choice = input("Select option (or 'q' to cancel): ").strip()
    
    if choice == '1':
        session_num = input("\nEnter session number to view: ").strip()
        try:
            session_idx = len(sessions) - int(session_num)  # Reverse index
            if 0 <= session_idx < len(sessions):
                selected_session = sessions[session_idx]
                session_data = history_manager.get_session(selected_session['id'])
                
                print(f"\n{'=' * 60}")
                print(f"Session: {session_data['name']}")
                print(f"{'=' * 60}")
                print(f"ID: {session_data['id']}")
                print(f"Created: {session_data['created_at']}")
                print(f"Messages: {len(session_data['messages'])}")
                print()
                
                if session_data['messages']:
                    print("Recent messages:")
                    for msg in session_data['messages'][-6:]:  # Show last 6 messages
                        role = f"{emoji('🙋', '[You]')} You" if msg['role'] == 'user' else f"{emoji('🤖', '[Bot]')} Assistant"
                        timestamp = datetime.fromisoformat(msg['timestamp']).strftime("%H:%M:%S")
                        content_preview = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                        print(f"\n  {role} [{timestamp}]: {content_preview}")
            else:
                print(f"\n{emoji('⚠️', '[WARNING]')} Invalid session number")
        except ValueError:
            print(f"\n{emoji('⚠️', '[WARNING]')} Invalid input")
    
    elif choice == '2':
        session_num = input("\nEnter session number to delete: ").strip()
        try:
            session_idx = len(sessions) - int(session_num)  # Reverse index
            if 0 <= session_idx < len(sessions):
                selected_session = sessions[session_idx]
                confirm = input(f"Delete session '{selected_session['name']}'? (yes/no): ").strip().lower()
                
                if confirm == 'yes':
                    if history_manager.delete_session(selected_session['id']):
                        print(f"\n{emoji('✅', '[SUCCESS]')} Session deleted successfully")
                    else:
                        print(f"\n{emoji('❌', '[ERROR]')} Failed to delete session")
                else:
                    print(f"\n{emoji('⚠️', '[WARNING]')} Deletion cancelled")
            else:
                print(f"\n{emoji('⚠️', '[WARNING]')} Invalid session number")
        except ValueError:
            print(f"\n{emoji('⚠️', '[WARNING]')} Invalid input")
    
    input("\nPress Enter to continue...")


def show_settings() -> None:
    """Show settings and information."""
    clear_screen()
    print_header()
    print(f"\n{emoji('🔧', '[Settings]')} Settings & Information")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv('COHERE_API_KEY')
    if api_key:
        print(f"\n{emoji('✅', '[OK]')} API Key: {api_key[:8]}...{api_key[-4:]}")
    else:
        print(f"\n{emoji('❌', '[ERROR]')} API Key: Not configured")
    
    # Check database
    from config import CHROMA_DB_PATH, COLLECTION_NAME
    db_path = Path(CHROMA_DB_PATH)
    
    if db_path.exists():
        print(f"{emoji('✅', '[OK]')} Database: {CHROMA_DB_PATH}")
        
        try:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            collection = client.get_collection(name=COLLECTION_NAME)
            count = collection.count()
            print(f"   Total chunks: {count:,}")
        except Exception as e:
            print(f"   {emoji('⚠️', '[WARNING]')} Could not read collection: {e}")
    else:
        print(f"{emoji('⚠️', '[WARNING]')} Database: Not initialized")
        print(f"   Run ingestion first: python ingest.py")
    
    # Check history
    history_dir = Path("./chat_history")
    if history_dir.exists():
        history_manager = HistoryManager()
        sessions = history_manager.list_sessions()
        print(f"\n{emoji('✅', '[OK]')} History: {len(sessions)} chat sessions")
    else:
        print(f"\n{emoji('⚠️', '[WARNING]')} History: No sessions yet")
    
    print("\n" + "=" * 60)
    print("\nConfiguration files:")
    print("  • .env - API key configuration")
    print("  • config.py - Application settings")
    print("  • chat_history/ - Chat session storage")
    print("  • chroma_db/ - Vector database storage")
    
    input("\nPress Enter to continue...")


def main() -> None:
    """Main entry point."""
    # Check if command line arguments were provided
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        # Support direct session viewing
        if arg == '--view-session' and len(sys.argv) > 2:
            session_id = sys.argv[2]
            history_manager = HistoryManager()
            session = history_manager.get_session(session_id)
            
            if session:
                print(f"\n{'=' * 60}")
                print(f"Session: {session['name']}")
                print(f"{'=' * 60}")
                print(f"ID: {session['id']}")
                print(f"Created: {session['created_at']}")
                print(f"Messages: {len(session['messages'])}\n")
                
                for msg in session['messages']:
                    role = "You" if msg['role'] == 'user' else "Assistant"
                    timestamp = datetime.fromisoformat(msg['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
                    print(f"\n[{timestamp}] {role}:")
                    print(msg['content'])
                    
                    if 'sources' in msg and msg['sources']:
                        print("\nSources:")
                        for source in msg['sources']:
                            print(f"  • {source}")
                
                print("\n" + "=" * 60)
            else:
                print(f"{emoji('❌', '[ERROR]')} Session not found: {session_id}")
            
            return
    
    # Interactive menu mode
    while True:
        clear_screen()
        print_header()
        print_menu()
        
        choice = input("Enter your choice: ").strip()
        
        if choice == '1':
            start_chat_session()
        elif choice == '2':
            watch_mode_interface()
        elif choice == '3':
            export_session_interface()
        elif choice == '4':
            view_query_history()
        elif choice == '5':
            view_chat_sessions()
        elif choice == '6':
            show_settings()
        elif choice == '0':
            print(f"\n{emoji('👋', 'Goodbye!')} Goodbye!")
            sys.exit(0)
        else:
            print(f"\n{emoji('⚠️', '[WARNING]')} Invalid choice. Please try again.")
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{emoji('👋', 'Goodbye!')} Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n{emoji('❌', '[ERROR]')} Fatal error: {e}")
        sys.exit(1)

