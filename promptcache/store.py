"""SQLite-backed cache store: exact lookup, scoped semantic search, TTL, eviction, stats."""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exact_key TEXT NOT NULL UNIQUE,
    scope TEXT NOT NULL,
    bucket TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    query_text TEXT NOT NULL DEFAULT '',
    embedding BLOB,
    response BLOB NOT NULL,
    status INTEGER NOT NULL,
    content_type TEXT NOT NULL,
    created REAL NOT NULL,
    expires REAL NOT NULL,
    hits INTEGER NOT NULL DEFAULT 0,
    last_hit REAL NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_bucket ON entries(scope, bucket);
CREATE INDEX IF NOT EXISTS idx_expires ON entries(expires);
CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value REAL NOT NULL DEFAULT 0);
"""

COUNTERS = (
    "hits_exact",
    "hits_semantic",
    "misses",
    "bypasses",
    "stores",
    "saved_input_tokens",
    "saved_output_tokens",
)


@dataclass
class Entry:
    id: int
    response: bytes
    status: int
    content_type: str
    created: float
    hits: int
    query_text: str
    input_tokens: int
    output_tokens: int


_COLS = "id, response, status, content_type, created, hits, query_text, input_tokens, output_tokens"


class Store:
    def __init__(
        self,
        path: str = "promptcache.db",
        max_entries: int = 10_000,
        clock: Callable[[], float] = time.time,
    ):
        self.max_entries = max_entries
        self.clock = clock
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._db.execute("PRAGMA journal_mode=WAL")
        for c in COUNTERS:
            self._db.execute("INSERT OR IGNORE INTO counters(name, value) VALUES (?, 0)", (c,))
        self._db.commit()

    # -- lookups ---------------------------------------------------------
    def get_exact(self, exact_key: str) -> Entry | None:
        now = self.clock()
        with self._lock:
            row = self._db.execute(
                f"SELECT {_COLS} FROM entries WHERE exact_key=? AND expires>?", (exact_key, now)
            ).fetchone()
            if row:
                self._touch(row[0], now)
        return Entry(*row) if row else None

    def search_semantic(
        self, scope: str, bucket: str, vec: np.ndarray, threshold: float
    ) -> tuple[Entry, float] | None:
        now = self.clock()
        with self._lock:
            rows = self._db.execute(
                "SELECT id, embedding FROM entries "
                "WHERE scope=? AND bucket=? AND embedding IS NOT NULL AND expires>?",
                (scope, bucket, now),
            ).fetchall()
            if not rows:
                return None
            mat = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
            if mat.shape[1] != vec.shape[0]:
                return None  # embedder was swapped; old vectors are incomparable
            scores = mat @ vec
            best = int(np.argmax(scores))
            if float(scores[best]) < threshold:
                return None
            eid = rows[best][0]
            row = self._db.execute(f"SELECT {_COLS} FROM entries WHERE id=?", (eid,)).fetchone()
            self._touch(eid, now)
        return Entry(*row), float(scores[best])

    def _touch(self, eid: int, now: float) -> None:
        self._db.execute("UPDATE entries SET hits=hits+1, last_hit=? WHERE id=?", (now, eid))
        self._db.commit()

    # -- writes ----------------------------------------------------------
    def put(
        self,
        *,
        exact_key: str,
        scope: str,
        bucket: str,
        provider: str,
        model: str,
        query_text: str,
        embedding: np.ndarray | None,
        response: bytes,
        status: int,
        content_type: str,
        ttl: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        now = self.clock()
        emb = embedding.astype(np.float32).tobytes() if embedding is not None else None
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO entries(exact_key, scope, bucket, provider, model, query_text,"
                " embedding, response, status, content_type, created, expires, hits, last_hit,"
                " input_tokens, output_tokens) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0,?,?,?)",
                (exact_key, scope, bucket, provider, model, query_text, emb, response, status,
                 content_type, now, now + ttl, now, input_tokens, output_tokens),
            )
            self._evict(now)
            self._db.commit()
        self.incr("stores")

    def _evict(self, now: float) -> None:
        self._db.execute("DELETE FROM entries WHERE expires<=?", (now,))
        (n,) = self._db.execute("SELECT COUNT(*) FROM entries").fetchone()
        if n > self.max_entries:
            self._db.execute(
                "DELETE FROM entries WHERE id IN "
                "(SELECT id FROM entries ORDER BY last_hit ASC LIMIT ?)",
                (n - self.max_entries,),
            )

    def clear(self, scope: str | None = None, model: str | None = None) -> int:
        where, args = [], []
        if scope:
            where.append("scope=?"); args.append(scope)
        if model:
            where.append("model=?"); args.append(model)
        sql = "DELETE FROM entries" + (" WHERE " + " AND ".join(where) if where else "")
        with self._lock:
            n = self._db.execute(sql, args).rowcount
            self._db.commit()
        return n

    # -- stats -----------------------------------------------------------
    def incr(self, name: str, amount: float = 1) -> None:
        with self._lock:
            self._db.execute("UPDATE counters SET value=value+? WHERE name=?", (amount, name))
            self._db.commit()

    def stats(self) -> dict:
        with self._lock:
            c = dict(self._db.execute("SELECT name, value FROM counters").fetchall())
            (n,) = self._db.execute(
                "SELECT COUNT(*) FROM entries WHERE expires>?", (self.clock(),)
            ).fetchone()
        hits = c["hits_exact"] + c["hits_semantic"]
        lookups = hits + c["misses"]
        return {
            "entries": n,
            "hits_exact": int(c["hits_exact"]),
            "hits_semantic": int(c["hits_semantic"]),
            "misses": int(c["misses"]),
            "bypasses": int(c["bypasses"]),
            "hit_rate": round(hits / lookups, 4) if lookups else 0.0,
            "saved_input_tokens": int(c["saved_input_tokens"]),
            "saved_output_tokens": int(c["saved_output_tokens"]),
        }
