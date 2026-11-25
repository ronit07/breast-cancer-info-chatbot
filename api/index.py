"""Vercel entry point: a plain WSGI app around the same Bot used by server.py."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server import bot, PAGE


def app(environ, start_response):
    if environ["REQUEST_METHOD"] == "POST":
        try:
            n = int(environ.get("CONTENT_LENGTH") or 0)
            data = json.loads(environ["wsgi.input"].read(n) or b"{}")
            out = bot.ask(str(data.get("q", ""))[:500], data.get("history", [])[-5:])
        except Exception:
            out = {"answer": "Sorry, something went wrong with that question.", "sources": [], "mode": "declined", "score": 0}
        body, ctype = json.dumps(out).encode(), "application/json"
    else:
        body, ctype = PAGE.encode(), "text/html; charset=utf-8"
    start_response("200 OK", [("Content-Type", ctype), ("Content-Length", str(len(body)))])
    return [body]
