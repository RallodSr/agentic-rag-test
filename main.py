"""Entry point: run the two-agent RAG pipeline on sample queries.

Usage:
    python main.py                     # run the built-in sample queries
    python main.py "your question"     # run a single custom query
"""

from __future__ import annotations

import re
import sys
import time

from dotenv import load_dotenv

load_dotenv()  # must run before importing modules that read env vars

from graph import build_graph  # noqa: E402

RATE_LIMIT_RETRIES = 5


def _invoke_with_rate_limit_retry(app, state: dict) -> dict:
    """The free tier allows only a few requests per minute; on 429, wait for
    the delay the API suggests and retry the whole query."""
    for attempt in range(RATE_LIMIT_RETRIES):
        try:
            return app.invoke(state)
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if "RESOURCE_EXHAUSTED" not in message and "429" not in message:
                raise
            # The quota window is per minute; the API's suggested delay is
            # often too short when a query needs several calls, so wait at
            # least a full window.
            match = re.search(r"retry in ([\d.]+)s", message)
            delay = max(float(match.group(1)) + 2 if match else 0.0, 60.0)
            print(
                f"[main] rate limit hit; waiting {delay:.0f}s before retrying "
                f"({attempt + 1}/{RATE_LIMIT_RETRIES})"
            )
            time.sleep(delay)
    raise RuntimeError("Rate limit retries exhausted")

SAMPLE_QUERIES = [
    "What is the policy on international travel?",
    "How many days of annual leave do employees get, and can unused days be carried over?",
    "What are the rules for working remotely?",
    "What is the company's policy on pets in the office?",  # not in the KB
]


def run_query(app, query: str) -> None:
    print("=" * 70)
    print(f"QUERY: {query}\n")
    result = _invoke_with_rate_limit_retry(
        app, {"query": query, "snippets": "", "answer": ""}
    )
    print("--- Retrieved snippets (Data Retriever) ---")
    print(result["snippets"])
    print("\n--- Final answer (Report Generator) ---")
    print(result["answer"])
    print()


def main() -> None:
    app = build_graph()
    queries = sys.argv[1:] or SAMPLE_QUERIES
    for query in queries:
        run_query(app, query)


if __name__ == "__main__":
    main()
