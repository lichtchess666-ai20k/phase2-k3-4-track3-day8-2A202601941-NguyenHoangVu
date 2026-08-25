"""Pytest bootstrap: load .env before test modules are imported.

tests/test_graph_smoke.py decides whether to skip by reading the API key at import
time, so the key must already be in os.environ when collection starts.
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass
