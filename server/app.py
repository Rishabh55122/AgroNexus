"""
server/app.py

OpenEnv multi-mode deployment entry point.
The openenv validate command checks for this file.
Exposes the same FastAPI app as api/main.py.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

__all__ = ["app"]
