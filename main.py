"""
main.py
Entry point for the Document Q&A System. Launches the user interface.
"""

import os
import sys
from dotenv import load_dotenv

def main():
    """Main entry point for the Document Q&A System."""
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
    
    # Launch Streamlit app
    try:
        import streamlit.web.cli as stcli
        import sys
        
        # Set Streamlit configuration
        os.environ['STREAMLIT_SERVER_PORT'] = '8501'
        os.environ['STREAMLIT_SERVER_ADDRESS'] = 'localhost'
        
        # Launch the app
        sys.argv = ["streamlit", "run", "app/ui.py", "--server.port=8501"]
        sys.exit(stcli.main())
        
    except ImportError:
        print("❌ Streamlit not found. Please install it with: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching app: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 