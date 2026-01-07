#!/usr/bin/env python3
"""
Chat Interface for CodeReader
Interactive CLI for querying the codebase using RAG.
"""

import os
import sys
from typing import List, Dict, Optional
from dotenv import load_dotenv
import cohere
import chromadb

# Load environment variables
load_dotenv()

# Configuration
COLLECTION_NAME = "codebase_chunks"
CHROMA_DB_PATH = "./chroma_db"
COHERE_EMBED_MODEL = "embed-english-v3.0"
COHERE_CHAT_MODEL = "command-r-plus-08-2024"  # Current model as of 2026
TOP_K_RESULTS = 5


class CodebaseChat:
    """Main chat interface for interacting with the codebase."""

    def __init__(self):
        """Initialize the chat interface with Cohere and ChromaDB clients."""
        # Check for API key
        self.api_key = os.getenv('COHERE_API_KEY')
        if not self.api_key:
            print("❌ Error: COHERE_API_KEY not found in environment variables")
            print("   Please create a .env file with your Cohere API key")
            sys.exit(1)

        try:
            # Initialize Cohere client
            self.cohere_client = cohere.Client(api_key=self.api_key)

            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

            # Get collection
            try:
                self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
                collection_count = self.collection.count()
                print(f"✅ Connected to ChromaDB collection: {COLLECTION_NAME}")
                print(f"   Total chunks in database: {collection_count}")
            except Exception as e:
                print(f"❌ Error: Could not find collection '{COLLECTION_NAME}'")
                print("   Please run 'python ingest.py' first to index your codebase")
                sys.exit(1)

        except Exception as e:
            print(f"❌ Error initializing chat interface: {e}")
            sys.exit(1)

    def embed_query(self, query: str) -> List[float]:
        """
        Embed user query using Cohere.

        Args:
            query: User's question

        Returns:
            Query embedding vector
        """
        try:
            response = self.cohere_client.embed(
                texts=[query],
                model=COHERE_EMBED_MODEL,
                input_type="search_query",  # Critical for v3 models
                embedding_types=["float"]
            )
            return response.embeddings.float[0]
        except Exception as e:
            print(f"❌ Error embedding query: {e}")
            return None

    def retrieve_relevant_chunks(self, query_embedding: List[float],
                                 top_k: int = TOP_K_RESULTS) -> Optional[Dict]:
        """
        Query ChromaDB for relevant code chunks.

        Args:
            query_embedding: Embedded query vector
            top_k: Number of top results to retrieve

        Returns:
            Dictionary containing documents, metadatas, and distances
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            return results
        except Exception as e:
            print(f"❌ Error querying ChromaDB: {e}")
            return None

    def format_documents_for_chat(self, results: Dict) -> List[Dict[str, str]]:
        """
        Format retrieved chunks as documents for Cohere chat API.

        Args:
            results: Results from ChromaDB query

        Returns:
            List of document dictionaries for Cohere API
        """
        documents = []

        if not results or not results['documents'][0]:
            return documents

        for i, (doc, metadata) in enumerate(zip(results['documents'][0],
                                                 results['metadatas'][0])):
            documents.append({
                'id': f"doc_{i}",
                'text': doc,
                'title': f"{metadata['file_name']} (lines {metadata['start_line']}-{metadata['end_line']})",
            })

        return documents

    def generate_answer(self, query: str, documents: List[Dict[str, str]]) -> Optional[Dict]:
        """
        Generate answer using Cohere's chat API with retrieved documents.

        Args:
            query: User's question
            documents: Retrieved and formatted documents

        Returns:
            Chat response object
        """
        try:
            # Build context-aware message
            response = self.cohere_client.chat(
                message=query,
                model=COHERE_CHAT_MODEL,
                documents=documents,  # Formal documents parameter for citations
                temperature=0.3,
            )
            return response
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return None

    def print_response(self, response, results: Dict) -> None:
        """
        Pretty print the chat response with citations.

        Args:
            response: Cohere chat response
            results: ChromaDB query results for source information
        """
        print("\n" + "=" * 60)
        print("💡 Answer:")
        print("=" * 60)
        print(response.text)

        # Print citations if available
        if hasattr(response, 'citations') and response.citations:
            print("\n" + "-" * 60)
            print("📚 Citations:")
            print("-" * 60)

            cited_docs = set()
            for citation in response.citations:
                if hasattr(citation, 'document_ids'):
                    for doc_id in citation.document_ids:
                        cited_docs.add(doc_id)

            # Map document IDs to source files
            for doc_id in sorted(cited_docs):
                try:
                    doc_idx = int(doc_id.split('_')[1])
                    metadata = results['metadatas'][0][doc_idx]
                    print(f"   • {metadata['file_path']} "
                          f"(lines {metadata['start_line']}-{metadata['end_line']})")
                except (IndexError, ValueError):
                    continue

        # Always show source files used
        print("\n" + "-" * 60)
        print("📄 Sources:")
        print("-" * 60)

        seen_files = set()
        for metadata in results['metadatas'][0]:
            file_path = metadata['file_path']
            if file_path not in seen_files:
                seen_files.add(file_path)
                print(f"   • {file_path}")

        print("=" * 60 + "\n")

    def chat_loop(self) -> None:
        """Main interactive chat loop."""
        print("\n" + "=" * 60)
        print("💬 CodeReader Chat Interface")
        print("=" * 60)
        print("Ask questions about your codebase!")
        print("Type 'exit' or 'quit' to end the session.")
        print("=" * 60 + "\n")

        while True:
            try:
                # Get user input
                query = input("You: ").strip()

                # Check for exit commands
                if query.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye!")
                    break

                # Skip empty queries
                if not query:
                    continue

                print("\n🔍 Searching codebase...")

                # Step 1: Embed query
                query_embedding = self.embed_query(query)
                if query_embedding is None:
                    print("⚠️  Failed to embed query. Please try again.")
                    continue

                # Step 2: Retrieve relevant chunks
                results = self.retrieve_relevant_chunks(query_embedding)
                if not results or not results['documents'][0]:
                    print("⚠️  No relevant code found. Try rephrasing your question.")
                    continue

                # Step 3: Format documents for chat API
                documents = self.format_documents_for_chat(results)

                print("🧠 Generating answer...")

                # Step 4: Generate answer
                response = self.generate_answer(query, documents)
                if response is None:
                    print("⚠️  Failed to generate answer. Please try again.")
                    continue

                # Step 5: Print response with citations
                self.print_response(response, results)

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n⚠️  Error: {e}")
                print("Please try again.\n")
                continue


def main():
    """Entry point for CLI."""
    try:
        chat = CodebaseChat()
        chat.chat_loop()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


