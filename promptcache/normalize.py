"""Turn a chat-completion request body into cache keys.

Two keys come out of every request:

* exact_key  - hash of everything that can change the answer. Same key, same answer.
* bucket     - hash of everything EXCEPT the final user message. Semantic lookups
               only ever search inside one bucket, so a "similar" prompt can never
               be served across a different system prompt, model, tool set,
               sampling setting, tenant, or conversation history.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

# Fields that do not change the model's answer.
_EXCLUDE = {"messages", "system", "stream", "stream_options", "metadata", "user"}


def _strip(o: Any) -> Any:
    """Drop provider cache hints and edge whitespace so they don't fragment keys."""
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k != "cache_control"}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    if isinstance(o, str):
        return o.strip()
    return o


def _blocks(content: Any) -> Any:
    """'hi' and [{'type':'text','text':'hi'}] mean the same thing; store one form."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return content


def _norm_messages(messages: list) -> list:
    out = []
    for m in messages:
        m = dict(m) if isinstance(m, dict) else m
        if isinstance(m, dict) and "content" in m:
            m["content"] = _blocks(m["content"])
        out.append(m)
    return out


_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def numbers_in(text: str) -> list[str]:
    """Numbers are where near-duplicate prompts most often differ in meaning ($500 vs $5000)."""
    return sorted(_NUM.findall(text))


def _canon(o: Any) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _h(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def scope_for(auth_value: str, mode: str = "key") -> str:
    """Tenant isolation. 'key' = one cache namespace per API key; 'global' = shared."""
    if mode == "global":
        return "global"
    return hashlib.sha256(auth_value.encode("utf-8")).hexdigest()[:16]


def query_text(content: Any) -> str:
    """Text of the final user turn, or '' if it isn't pure text (images, tool results...).

    An empty string disables semantic matching for that request, because two
    prompts with identical text but different attachments are not 'similar'.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for b in content:
            if not isinstance(b, dict) or b.get("type") not in ("text", "input_text"):
                return ""
            parts.append(b.get("text", ""))
        return "\n".join(parts).strip()
    return ""


@dataclass
class CacheKey:
    scope: str
    provider: str
    model: str
    exact_key: str
    bucket: str
    query_text: str


def build_key(provider: str, body: dict, scope: str) -> CacheKey | None:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    msgs = _strip(_norm_messages(messages))
    params = _canon(_strip({k: v for k, v in body.items() if k not in _EXCLUDE}))
    system = _canon(_strip(_blocks(body.get("system"))))
    last = msgs[-1]
    q = query_text(last.get("content")) if last.get("role") == "user" else ""
    return CacheKey(
        scope=scope,
        provider=provider,
        model=str(body.get("model", "")),
        exact_key=_h(scope, provider, params, system, _canon(msgs)),
        bucket=_h(scope, provider, params, system, _canon(msgs[:-1])),
        query_text=q,
    )


def usage_of(provider: str, response: dict) -> tuple[int, int]:
    u = response.get("usage") or {}
    if provider == "anthropic":
        return int(u.get("input_tokens", 0) or 0), int(u.get("output_tokens", 0) or 0)
    return int(u.get("prompt_tokens", 0) or 0), int(u.get("completion_tokens", 0) or 0)
