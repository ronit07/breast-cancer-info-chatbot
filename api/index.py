"""Vercel entry point: serves the chat page and the /ask endpoint from server.py."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server import H as handler  # noqa: E402,F401
