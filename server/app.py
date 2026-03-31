"""
server/app.py — OpenEnv multi-mode deployment entry point
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app
import uvicorn

__all__ = ["app"]


def main():
    """Main entry point required by OpenEnv multi-mode deployment spec."""
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    main()
