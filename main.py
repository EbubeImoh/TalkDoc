"""
main.py
Entry point for the Document Q&A System. Launches the user interface.
"""

import os
import sys
from dotenv import load_dotenv

def main():
    """Main entry point for the Document Q&A System."""
    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    # Load environment variables
    load_dotenv()
    
    # Check required environment variables
    required_vars = ["PINECONE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file.")
        sys.exit(1)
    
    # Import and run the UI directly
    try:
        from app.ui import main as ui_main
        ui_main()
    except ImportError as e:
        print(f"❌ Error importing UI module: {str(e)}")
        print("Please ensure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching app: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 