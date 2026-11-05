# promptcache-gateway

A drop-in caching proxy for LLM APIs. Point your SDK's base URL at it; identical or
near-identical requests are answered from a local SQLite cache instead of the provider.

Supports `POST /v1/messages` (Anthropic) and `POST /v1/chat/completions` (OpenAI-compatible).

## Run

    pip install -r requirements.txt
    uvicorn promptcache.app:app_factory --factory --port 8080

    # Anthropic SDK
    client = Anthropic(base_url="http://localhost:8080")
    # OpenAI SDK
    client = OpenAI(base_url="http://localhost:8080/v1")

Your API key is forwarded to the provider untouched. Every response carries
`x-cache: HIT-EXACT | HIT-SEMANTIC | MISS | BYPASS` (semantic hits also carry
`x-cache-similarity`).

## How a lookup works

1. **Exact**: SHA-256 over tenant + model + all sampling/tool params + system + full messages.
   Whitespace at string edges, `cache_control` hints, and `"hi"` vs `[{"type":"text","text":"hi"}]`
   are normalized away so they don't fragment the cache.
2. **Semantic** (fallback): embed the final user message and compare only against entries in the
   same *bucket* — same tenant, model, params, system prompt, tools, and prior conversation. A similar
   prompt can never be served across a different context. A number guard also rejects hits whose
   numbers differ ($500 vs $5000).
3. **Miss**: forward upstream; cache only HTTP 200 JSON responses.

Never cached: streaming requests, requests with `x-cache-bypass: 1`, errors, and semantic matching
is skipped when the last turn contains images or tool results.

## Config (env vars)

| Var | Default | Meaning |
|---|---|---|
| `UPSTREAM_ANTHROPIC` / `UPSTREAM_OPENAI` | provider URLs | where misses go |
| `CACHE_DB` | `promptcache.db` | SQLite path |
| `CACHE_TTL_SECONDS` | 86400 | entry lifetime |
| `CACHE_MAX_ENTRIES` | 10000 | LRU eviction beyond this |
| `CACHE_SCOPE` | `key` | `key` = isolate per API key, `global` = share |
| `SEMANTIC_ENABLED` | true | turn the fallback off for exact-only |
| `SEMANTIC_THRESHOLD` | 0.93 | cosine cutoff |
| `EMBEDDINGS_URL` / `EMBEDDINGS_MODEL` / `EMBEDDINGS_API_KEY` | unset | OpenAI-compatible embeddings endpoint |
| `ADMIN_TOKEN` | unset | if set, `/cache/*` requires `x-admin-token` |

## Embedder: read this before enabling semantic hits

The default embedder is a dependency-free hashing embedder. It is **lexical**: it reliably
matches case, punctuation and spacing variants (similarity 1.0), but scores a one-word typo
around 0.83 and an entity swap ("France" -> "Germany") around 0.78. There is no threshold that
separates paraphrases from different questions with it. For real paraphrase matching, set
`EMBEDDINGS_URL` to an embedding model and tune `SEMANTIC_THRESHOLD` on your own traffic
(start high, 0.95+, and watch `/cache/stats` plus spot-check hits).

## Admin

    GET  /cache/stats          hit rate, entries, tokens saved
    POST /cache/clear?model=X  invalidate (all, or one model)
    GET  /health

## Known limits (v1)

- Streaming is passed through, not cached or replayed.
- Only the two chat endpoints are proxied (no `count_tokens`, embeddings, batches).
- Semantic search is brute-force within a bucket; fine for thousands of entries per bucket,
  swap in an ANN index beyond that.
- Cached responses replay the original `id`/timestamps.

## Test

    pytest -q
