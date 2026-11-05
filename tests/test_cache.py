import json

import httpx
import pytest
from fastapi.testclient import TestClient

from promptcache.app import Config, create_app
from promptcache.embedder import HashingEmbedder
from promptcache.store import Store


class FakeClock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


@pytest.fixture
def env(tmp_path):
    calls = []
    state = {"status": 200}

    def upstream(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if state["status"] != 200:
            return httpx.Response(state["status"], json={"error": "boom"})
        if "chat/completions" in request.url.path:
            return httpx.Response(200, json={
                "id": f"cmpl_{len(calls)}",
                "choices": [{"message": {"role": "assistant", "content": f"answer {len(calls)}"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 7},
            })
        if body.get("stream"):
            return httpx.Response(200, stream=httpx.ByteStream(b"data: hello\n\n"),
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={
            "id": f"msg_{len(calls)}",
            "content": [{"type": "text", "text": f"answer {len(calls)}"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })

    clock = FakeClock()
    store = Store(str(tmp_path / "t.db"), max_entries=100, clock=clock)
    client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    cfg = Config(semantic_threshold=0.93, ttl_seconds=60)
    app = create_app(cfg, store=store, embedder=HashingEmbedder(), client=client)
    return TestClient(app), calls, clock, state


def req(text="What is the capital of France?", key="sk-a", **over):
    body = {"model": "claude-x", "max_tokens": 100, "system": "Be brief.",
            "messages": [{"role": "user", "content": text}]}
    body.update(over)
    return {"json": body, "headers": {"x-api-key": key}}


def test_exact_hit(env):
    c, calls, _, _ = env
    r1 = c.post("/v1/messages", **req())
    r2 = c.post("/v1/messages", **req())
    assert r1.headers["x-cache"] == "MISS"
    assert r2.headers["x-cache"] == "HIT-EXACT"
    assert r1.json() == r2.json()
    assert len(calls) == 1


def test_whitespace_and_cache_control_do_not_fragment(env):
    c, calls, _, _ = env
    c.post("/v1/messages", **req("Hello there"))
    r = c.post("/v1/messages", **req("  Hello there \n", system="Be brief."))
    assert r.headers["x-cache"] == "HIT-EXACT"
    blocks = [{"type": "text", "text": "Hello there", "cache_control": {"type": "ephemeral"}}]
    r = c.post("/v1/messages", **req(blocks))
    assert r.headers["x-cache"] == "HIT-EXACT"
    assert len(calls) == 1


def test_semantic_hit_and_miss(env):
    # The built-in hashing embedder promises case/punctuation/spacing tolerance only.
    c, calls, _, _ = env
    c.post("/v1/messages", **req("What is the capital of France?"))
    near = c.post("/v1/messages", **req("what is the capital of FRANCE ??"))
    assert near.headers["x-cache"] == "HIT-SEMANTIC"
    assert float(near.headers["x-cache-similarity"]) >= 0.93
    for other in ("Explain how photosynthesis works in detail",
                  "What is the capital of Germany?"):  # entity swap must not hit
        assert c.post("/v1/messages", **req(other)).headers["x-cache"] == "MISS"
    assert len(calls) == 3


def test_number_guard_blocks_near_duplicates(env):
    c, calls, _, _ = env
    c.post("/v1/messages", **req("Refund policy for orders over $500"))
    r = c.post("/v1/messages", **req("Refund policy for orders over $5000"))
    assert r.headers["x-cache"] == "MISS"
    assert len(calls) == 2


def test_semantic_never_crosses_context(env):
    c, calls, _, _ = env
    c.post("/v1/messages", **req("What is the capital of France?"))
    for over in ({"system": "Answer in French."}, {"temperature": 0.9}, {"model": "claude-y"}):
        r = c.post("/v1/messages", **req("what's the capital of france", **over))
        assert r.headers["x-cache"] == "MISS", over
    # different conversation history
    hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "what's the capital of france"}]
    r = c.post("/v1/messages", **req(messages=hist))
    assert r.headers["x-cache"] == "MISS"


def test_tenant_isolation(env):
    c, calls, _, _ = env
    c.post("/v1/messages", **req(key="sk-a"))
    r = c.post("/v1/messages", **req(key="sk-b"))
    assert r.headers["x-cache"] == "MISS"
    assert len(calls) == 2


def test_images_never_semantic(env):
    c, calls, _, _ = env
    def img(data):
        return [{"type": "text", "text": "describe this"},
                {"type": "image", "source": {"type": "base64", "data": data}}]
    c.post("/v1/messages", **req(img("AAAA")))
    r = c.post("/v1/messages", **req(img("BBBB")))
    assert r.headers["x-cache"] == "MISS"


def test_ttl_expiry(env):
    c, calls, clock, _ = env
    c.post("/v1/messages", **req())
    clock.t += 61
    r = c.post("/v1/messages", **req())
    assert r.headers["x-cache"] == "MISS"
    assert len(calls) == 2


def test_errors_not_cached(env):
    c, calls, _, state = env
    state["status"] = 500
    assert c.post("/v1/messages", **req()).status_code == 500
    state["status"] = 200
    r = c.post("/v1/messages", **req())
    assert r.status_code == 200 and r.headers["x-cache"] == "MISS"


def test_stream_and_bypass_passthrough(env):
    c, calls, _, _ = env
    r = c.post("/v1/messages", **req(stream=True))
    assert r.headers["x-cache"] == "BYPASS" and b"hello" in r.content
    c.post("/v1/messages", **req())
    r = c.post("/v1/messages", headers={**req()["headers"], "x-cache-bypass": "1"}, json=req()["json"])
    assert r.headers["x-cache"] == "BYPASS"
    assert len(calls) == 3


def test_openai_route(env):
    c, calls, _, _ = env
    body = {"model": "gpt-x", "messages": [{"role": "system", "content": "s"},
                                           {"role": "user", "content": "ping"}]}
    h = {"authorization": "Bearer k"}
    assert c.post("/v1/chat/completions", json=body, headers=h).headers["x-cache"] == "MISS"
    assert c.post("/v1/chat/completions", json=body, headers=h).headers["x-cache"] == "HIT-EXACT"


def test_stats_and_clear(env):
    c, _, _, _ = env
    c.post("/v1/messages", **req())
    c.post("/v1/messages", **req())
    s = c.get("/cache/stats").json()
    assert s["hits_exact"] == 1 and s["misses"] == 1 and s["entries"] == 1
    assert s["saved_input_tokens"] == 10 and s["saved_output_tokens"] == 5
    assert c.post("/cache/clear").json()["deleted"] == 1
    assert c.post("/v1/messages", **req()).headers["x-cache"] == "MISS"


def test_eviction(tmp_path):
    clock = FakeClock()
    s = Store(str(tmp_path / "e.db"), max_entries=3, clock=clock)
    for i in range(5):
        clock.t += 1
        s.put(exact_key=f"k{i}", scope="s", bucket="b", provider="p", model="m", query_text="",
              embedding=None, response=b"{}", status=200, content_type="application/json", ttl=999)
    assert s.stats()["entries"] == 3
    assert s.get_exact("k0") is None and s.get_exact("k4") is not None
