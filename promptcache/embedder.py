"""Embedders for semantic lookup.

HashingEmbedder is dependency-free and deterministic. It captures *lexical*
similarity (shared words and character n-grams), so it catches rephrasings,
typos and casing, but not true paraphrases. For real semantic matching, point
EMBEDDINGS_URL at any OpenAI-compatible /embeddings endpoint.
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import httpx
import numpy as np


class Embedder(Protocol):
    def embed(self, text: str) -> np.ndarray: ...


def _slot(token: str, dim: int) -> tuple[int, float]:
    d = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    n = int.from_bytes(d, "little")
    return n % dim, 1.0 if (n >> 63) & 1 else -1.0


class HashingEmbedder:
    def __init__(self, dim: int = 1024):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        t = re.sub(r"[^\w\s]", " ", text.lower())
        words = t.split()
        feats: list[tuple[str, float]] = [(f"w:{w}", 2.0) for w in words]
        feats += [(f"b:{a}_{b}", 1.5) for a, b in zip(words, words[1:])]
        padded = f" {' '.join(words)} "
        for n in (3, 4):
            feats += [(f"c{n}:{padded[i:i + n]}", 1.0) for i in range(len(padded) - n + 1)]
        v = np.zeros(self.dim, dtype=np.float32)
        for tok, w in feats:
            i, s = _slot(tok, self.dim)
            v[i] += s * w
        norm = np.linalg.norm(v)
        return v / norm if norm else v


class OpenAICompatEmbedder:
    """Calls POST {url} with {"model", "input"}; expects data[0].embedding."""

    def __init__(self, url: str, model: str, api_key: str = "", timeout: float = 10.0):
        self.url, self.model, self.api_key, self.timeout = url, model, api_key, timeout

    def embed(self, text: str) -> np.ndarray:
        headers = {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = httpx.post(
            self.url, json={"model": self.model, "input": text}, headers=headers, timeout=self.timeout
        )
        r.raise_for_status()
        v = np.asarray(r.json()["data"][0]["embedding"], dtype=np.float32)
        norm = np.linalg.norm(v)
        return v / norm if norm else v
