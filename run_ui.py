"""
Quick launcher for IPMA Web UI Dashboard
========================================
Run:
    python run_ui.py
"""

import sys
import webbrowser
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.server import start_server

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000
    print("\nStarting IPMA Industrial Agent Dashboard...")
    print(f"Opening browser at http://{host}:{port} ...\n")
    try:
        # Launch server
        start_server(host=host, port=port)
    except KeyboardInterrupt:
        print("\nIPMA server stopped.")
