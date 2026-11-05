"""Caching gateway: drop-in proxy for Anthropic /v1/messages and OpenAI /v1/chat/completions.

Point your SDK's base URL at this server. Cache status is reported in response
headers: x-cache = HIT-EXACT | HIT-SEMANTIC | MISS | BYPASS.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from .embedder import Embedder, HashingEmbedder, OpenAICompatEmbedder
from .normalize import build_key, numbers_in, scope_for, usage_of
from .store import Store

log = logging.getLogger("promptcache")

_DROP_REQ = {"host", "content-length", "connection", "accept-encoding", "transfer-encoding", "x-cache-bypass"}
_DROP_RESP = {"content-length", "transfer-encoding", "connection"}


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    anthropic_upstream: str = field(default_factory=lambda: os.getenv("UPSTREAM_ANTHROPIC", "https://api.anthropic.com"))
    openai_upstream: str = field(default_factory=lambda: os.getenv("UPSTREAM_OPENAI", "https://api.openai.com"))
    db_path: str = field(default_factory=lambda: os.getenv("CACHE_DB", "promptcache.db"))
    ttl_seconds: float = field(default_factory=lambda: float(os.getenv("CACHE_TTL_SECONDS", "86400")))
    max_entries: int = field(default_factory=lambda: int(os.getenv("CACHE_MAX_ENTRIES", "10000")))
    semantic: bool = field(default_factory=lambda: _env_bool("SEMANTIC_ENABLED", True))
    semantic_threshold: float = field(default_factory=lambda: float(os.getenv("SEMANTIC_THRESHOLD", "0.93")))
    max_query_chars: int = field(default_factory=lambda: int(os.getenv("SEMANTIC_MAX_QUERY_CHARS", "4000")))
    scope_mode: str = field(default_factory=lambda: os.getenv("CACHE_SCOPE", "key"))  # key | global
    admin_token: str = field(default_factory=lambda: os.getenv("ADMIN_TOKEN", ""))
    embeddings_url: str = field(default_factory=lambda: os.getenv("EMBEDDINGS_URL", ""))
    embeddings_model: str = field(default_factory=lambda: os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small"))
    embeddings_key: str = field(default_factory=lambda: os.getenv("EMBEDDINGS_API_KEY", ""))


def create_app(
    config: Config | None = None,
    store: Store | None = None,
    embedder: Embedder | None = None,
    client: httpx.AsyncClient | None = None,
) -> FastAPI:
    cfg = config or Config()
    store = store or Store(cfg.db_path, cfg.max_entries)
    if embedder is None:
        embedder = (
            OpenAICompatEmbedder(cfg.embeddings_url, cfg.embeddings_model, cfg.embeddings_key)
            if cfg.embeddings_url
            else HashingEmbedder()
        )
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
    app = FastAPI(title="promptcache-gateway")

    upstream = {"anthropic": cfg.anthropic_upstream.rstrip("/"), "openai": cfg.openai_upstream.rstrip("/")}

    def fwd_headers(request: Request) -> dict:
        return {k: v for k, v in request.headers.items() if k.lower() not in _DROP_REQ}

    async def passthrough_stream(url: str, raw: bytes, headers: dict) -> Response:
        req = http.build_request("POST", url, content=raw, headers=headers)
        try:
            resp = await http.send(req, stream=True)
        except httpx.RequestError as e:
            raise HTTPException(502, f"upstream unreachable: {e.__class__.__name__}")
        out = {k: v for k, v in resp.headers.items() if k.lower() not in _DROP_RESP}
        out["x-cache"] = "BYPASS"
        return StreamingResponse(
            resp.aiter_raw(), status_code=resp.status_code, headers=out,
            background=BackgroundTask(resp.aclose),
        )

    async def handle(request: Request, provider: str, path: str) -> Response:
        raw = await request.body()
        try:
            body = json.loads(raw)
            assert isinstance(body, dict)
        except Exception:
            raise HTTPException(400, "request body must be a JSON object")

        url = upstream[provider] + path
        headers = fwd_headers(request)
        bypass = request.headers.get("x-cache-bypass", "").lower() in ("1", "true", "yes")

        # Streaming and explicit bypass are forwarded untouched, never cached.
        if body.get("stream") or bypass:
            await run_in_threadpool(store.incr, "bypasses")
            if body.get("stream"):
                return await passthrough_stream(url, raw, headers)

        auth = request.headers.get("x-api-key") or request.headers.get("authorization") or ""
        key = None if bypass else build_key(provider, body, scope_for(auth, cfg.scope_mode))

        vec = None
        if key:
            entry = await run_in_threadpool(store.get_exact, key.exact_key)
            if entry:
                return await _hit(entry, "HIT-EXACT", None)

            if cfg.semantic and key.query_text and len(key.query_text) <= cfg.max_query_chars:
                try:
                    vec = await run_in_threadpool(embedder.embed, key.query_text)
                    found = await run_in_threadpool(
                        store.search_semantic, key.scope, key.bucket, vec, cfg.semantic_threshold
                    )
                    if found and numbers_in(found[0].query_text) == numbers_in(key.query_text):
                        return await _hit(found[0], "HIT-SEMANTIC", found[1])
                except Exception:
                    log.exception("semantic lookup failed; falling through to upstream")
                    vec = None

        try:
            resp = await http.post(url, content=raw, headers=headers)
        except httpx.RequestError as e:
            raise HTTPException(502, f"upstream unreachable: {e.__class__.__name__}")

        out = {k: v for k, v in resp.headers.items() if k.lower() not in _DROP_RESP | {"content-encoding"}}
        out["x-cache"] = "BYPASS" if bypass else "MISS"

        if key and resp.status_code == 200:
            try:
                parsed = resp.json()
                tin, tout = usage_of(provider, parsed)
                await run_in_threadpool(
                    lambda: store.put(
                        exact_key=key.exact_key, scope=key.scope, bucket=key.bucket,
                        provider=provider, model=key.model, query_text=key.query_text,
                        embedding=vec, response=resp.content, status=200,
                        content_type=resp.headers.get("content-type", "application/json"),
                        ttl=cfg.ttl_seconds, input_tokens=tin, output_tokens=tout,
                    )
                )
            except Exception:
                log.exception("failed to store response")
        if key:
            await run_in_threadpool(store.incr, "misses")
        return Response(resp.content, status_code=resp.status_code, headers=out)

    async def _hit(entry, kind: str, score: float | None) -> Response:
        await run_in_threadpool(store.incr, "hits_exact" if kind == "HIT-EXACT" else "hits_semantic")
        await run_in_threadpool(store.incr, "saved_input_tokens", entry.input_tokens)
        await run_in_threadpool(store.incr, "saved_output_tokens", entry.output_tokens)
        h = {"x-cache": kind, "x-cache-age": str(int(store.clock() - entry.created))}
        if score is not None:
            h["x-cache-similarity"] = f"{score:.4f}"
        return Response(entry.response, status_code=entry.status, media_type=entry.content_type, headers=h)

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request):
        return await handle(request, "anthropic", "/v1/messages")

    @app.post("/v1/chat/completions")
    async def openai_chat(request: Request):
        return await handle(request, "openai", "/v1/chat/completions")

    def _admin(request: Request) -> None:
        if cfg.admin_token and request.headers.get("x-admin-token") != cfg.admin_token:
            raise HTTPException(401, "admin token required")

    @app.get("/cache/stats")
    async def stats(request: Request):
        _admin(request)
        return await run_in_threadpool(store.stats)

    @app.post("/cache/clear")
    async def clear(request: Request, model: str | None = None):
        _admin(request)
        n = await run_in_threadpool(store.clear, None, model)
        return {"deleted": n}

    @app.get("/health")
    async def health():
        return {"ok": True, "time": time.time()}

    return app


def app_factory() -> FastAPI:
    """uvicorn promptcache.app:app_factory --factory"""
    return create_app()
