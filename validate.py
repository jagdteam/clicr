#!/usr/bin/env python3
"""
Validation script for CodeReader setup
Checks dependencies, API keys, and configuration.
"""

import sys
import os
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.10+"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} "
              f"(Requires Python 3.10+)")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    print("\n📦 Checking dependencies...")

    required_packages = {
        'cohere': 'cohere',
        'chromadb': 'chromadb',
        'dotenv': 'python-dotenv',
    }

    all_installed = True

    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name} (not installed)")
            all_installed = False

    if not all_installed:
        print("\n   Install missing packages with:")
        print("   pip install -r requirements.txt")

    return all_installed


def check_env_file():
    """Check if .env file exists and has API key"""
    print("\n🔑 Checking environment configuration...")

    env_path = Path('.env')

    if not env_path.exists():
        print("   ❌ .env file not found")
        print("\n   Create .env file with:")
        print("   echo 'COHERE_API_KEY=your_api_key_here' > .env")
        return False

    print("   ✅ .env file exists")

    # Check if API key is set
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv('COHERE_API_KEY')
    if not api_key or api_key == 'your_cohere_api_key_here':
        print("   ❌ COHERE_API_KEY not set or using placeholder value")
        print("\n   Get your API key from:")
        print("   https://dashboard.cohere.com/api-keys")
        return False

    print(f"   ✅ COHERE_API_KEY is set ({api_key[:8]}...)")
    return True


def check_cohere_connection():
    """Test connection to Cohere API"""
    print("\n🌐 Testing Cohere API connection...")

    try:
        import cohere
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.getenv('COHERE_API_KEY')

        if not api_key:
            print("   ⏭️  Skipped (no API key)")
            return False

        client = cohere.Client(api_key=api_key)

        # Try a simple embed call
        response = client.embed(
            texts=["test"],
            model="embed-english-v3.0",
            input_type="search_query",
            embedding_types=["float"]
        )

        print("   ✅ Successfully connected to Cohere API")
        return True

    except Exception as e:
        print(f"   ❌ Failed to connect: {e}")
        return False


def check_project_files():
    """Check if required project files exist"""
    print("\n📄 Checking project files...")

    required_files = ['ingest.py', 'chat.py', 'config.py', 'utils.py', 'requirements.txt']
    all_exist = True

    for file_name in required_files:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"   ✅ {file_name}")
        else:
            print(f"   ❌ {file_name} (missing)")
            all_exist = False

    return all_exist


def main():
    """Run all validation checks"""
    print("=" * 60)
    print("🔍 CodeReader Setup Validation")
    print("=" * 60)

    checks = [
        check_python_version(),
        check_dependencies(),
        check_project_files(),
        check_env_file(),
        check_cohere_connection(),
    ]

    print("\n" + "=" * 60)

    if all(checks):
        print("✅ All checks passed! You're ready to use CodeReader.")
        print("=" * 60)
        print("\n📋 Next steps:")
        print("   1. Run: python ingest.py")
        print("   2. Run: python chat.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()


