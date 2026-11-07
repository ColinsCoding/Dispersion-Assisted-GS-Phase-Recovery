#!/usr/bin/env python3
"""
vecdb_win.py — local semantic search over caps.jsonl history, fully local
via Ollama embeddings. Companion module to live_cas_ollama_win.py.

Embeds each utterance with an Ollama embedding model and stores the vectors
in a small SQLite file next to caps.jsonl. search() does brute-force cosine
similarity in-memory — caps.jsonl-scale history (thousands of rows, not
millions) never needs an ANN index.

Setup (once):
  ollama pull nomic-embed-text
"""

import json
import os

import numpy as np
import requests
import sqlite3

OLLAMA_URL  = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")


def _db_path(log_path: str) -> str:
    return os.path.splitext(log_path)[0] + "_vec.sqlite3"


def _connect(log_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(log_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS utterances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL UNIQUE,
            text TEXT NOT NULL,
            outcome TEXT NOT NULL,
            result TEXT,
            embedding BLOB NOT NULL
        )
    """)
    return conn


def embed(text: str) -> np.ndarray | None:
    """One Ollama embedding call. Returns None (with a printed warning) if the
    model isn't pulled or Ollama is unreachable — semantic search is optional,
    never fatal to the caller. Generous timeout: the first call after a pull
    or an idle keep_alive expiry has to cold-load the model into memory,
    which on this commit-tight machine can take well over 10s even though
    warm calls answer in ~1s."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        return np.array(resp.json()["embedding"], dtype=np.float32)
    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"[vecdb]   embedding failed ({e}) — pull it with: ollama pull {EMBED_MODEL}",
              flush=True)
        return None


def add(log_path: str, ts: str, text: str, outcome: str, result: str | None) -> None:
    vec = embed(text)
    if vec is None:
        return
    conn = _connect(log_path)
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO utterances (ts, text, outcome, result, embedding) "
            "VALUES (?, ?, ?, ?, ?)",
            (ts, text, outcome, result, vec.tobytes()),
        )
    conn.close()


def search(log_path: str, query: str, k: int = 5, outcome: str | None = "cas"):
    """Cosine-similarity search. Returns [(score, ts, text, result), ...],
    highest score first."""
    qvec = embed(query)
    if qvec is None:
        return []
    conn = _connect(log_path)
    sql = "SELECT ts, text, outcome, result, embedding FROM utterances"
    params: tuple = ()
    if outcome:
        sql += " WHERE outcome = ?"
        params = (outcome,)
    rows = conn.execute(sql, params).fetchall()
    conn.close()

    qnorm = qvec / (np.linalg.norm(qvec) + 1e-10)
    scored = []
    for ts, text, _out, result, blob in rows:
        vec = np.frombuffer(blob, dtype=np.float32)
        vnorm = vec / (np.linalg.norm(vec) + 1e-10)
        scored.append((float(np.dot(qnorm, vnorm)), ts, text, result))
    scored.sort(key=lambda r: r[0], reverse=True)
    return scored[:k]


def backfill(log_path: str) -> int:
    """Embed caps.jsonl rows not yet in the vector DB. Only outcome=="cas" rows
    are indexed, matching live_cas_ollama_win.py's log_utterance() policy —
    filtered/skip/no_extract rows are noise search would never want to hit
    anyway. Returns count added."""
    if not os.path.exists(log_path):
        return 0
    conn = _connect(log_path)
    existing = {row[0] for row in conn.execute("SELECT ts FROM utterances")}
    conn.close()

    added = 0
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = row.get("ts")
            if not ts or ts in existing or row.get("outcome") != "cas":
                continue
            add(log_path, ts, row.get("text", ""), row.get("outcome", ""), row.get("result"))
            added += 1
    return added
