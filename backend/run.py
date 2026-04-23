#!/usr/bin/env python
"""
Entry point for running the Flask application.
Handles Python path setup for module imports.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path so "backend.src" imports work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import and run the app
from backend.src.api.app import create_app

if __name__ == "__main__":
    app = create_app(config_name="development")
    port = int(os.getenv("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=True)
