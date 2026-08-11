#!/usr/bin/env python3
"""vcc-semantic.py — semantic search over VCC JSONL exports (sidecar to VCC.py).

Usage:
  python vcc-semantic.py export.jsonl --query "text" [--limit N]
      [--provider api|onnx] [--api-url URL] [--api-key KEY]
      [--model PATH] [--cache PATH] [--dims N] [--no-cache]

Embeds one vector per searchable export record (message granularity) via
an HTTP embeddings API (default) or local onnxruntime all-MiniLM-L6-v2,
then ranks records by cosine similarity to the query.

Line refs point at the JSONL export (1-based record number), NOT the
compiled .txt view. Grep the .txt with fromLine or read the export directly.

Failure contract: any error -> "ERROR: <msg>" on stderr, exit code 1,
nothing on stdout.
"""

import argparse
import hashlib
import io
import json
import math
import os
import sys
import urllib.error
import urllib.request

_BATCH = 32
_MODEL_API_DEFAULT = "harrier-oss-v1-0.6B-Embed"


def _cache_dir():
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(base, "vcc", "all-MiniLM-L6-v2")


_ONNX_DEFAULT = os.path.join(_cache_dir(), "model.onnx")
_ONNX_URL = (
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/"
    "resolve/main/onnx/model_qint8_avx512_vnni.onnx"
)
_TOK_URL = (
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/"
    "resolve/main/tokenizer.json"
)


def _rel_path(fp):
    try:
        return os.path.relpath(fp)
    except ValueError:
        return os.path.basename(fp)


def _text_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _record_text(r):
    """Searchable text for one export record (message granularity)."""
    parts = []
    msg = r.get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for b in content:
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text":
                t = b.get("text", "")
                if t:
                    parts.append(t)
            elif bt == "tool_use":
                name = b.get("name", "unknown")
                inp = b.get("input", {})
                parts.append(
                    "tool_call " + name + "\n" + json.dumps(inp, ensure_ascii=False)
                )
    return "\n".join(p for p in parts if p)


def _preview(text, n=6):
    return text.split("\n")[:n]


def _download(url, dest):
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    sys.stderr.write(f"vcc-semantic: downloading {url} -> {dest}\n")
    sys.stderr.flush()
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)


def _api_embed(texts, api_url, api_key, model):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    vecs = []
    for i in range(0, len(texts), _BATCH):
        chunk = texts[i : i + _BATCH]
        body = json.dumps({"model": model, "input": chunk}).encode("utf-8")
        req = urllib.request.Request(api_url, data=body, headers=headers, method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=60)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"embeddings api HTTP {e.code}: {e.reason}")
        if resp.status != 200:
            raise RuntimeError(f"embeddings api returned HTTP {resp.status}")
        data = json.loads(resp.read().decode("utf-8"))
        items = data.get("data") or []
        if len(items) != len(chunk):
            raise RuntimeError(
                f"embeddings api returned {len(items)} vectors for {len(chunk)} inputs"
            )
        for it in items:
            e = it.get("embedding")
            if not e:
                raise RuntimeError("embeddings api response missing embedding")
            vecs.append(e)
    return vecs


def _onnx_embed(texts, model_path):
    try:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as e:
        raise RuntimeError(
            "onnx provider needs numpy, onnxruntime, tokenizers: " + str(e)
        )

    if not os.path.exists(model_path):
        _download(_ONNX_URL, model_path)
    tok_path = os.path.join(
        os.path.dirname(os.path.abspath(model_path)), "tokenizer.json"
    )
    if not os.path.exists(tok_path):
        _download(_TOK_URL, tok_path)

    tok = Tokenizer.from_file(tok_path)
    cur_trunc = tok.truncation
    cur_max = cur_trunc.get("max_length") if isinstance(cur_trunc, dict) else None
    if cur_max is None or cur_max > 512:
        tok.enable_truncation(512)
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]

    vecs = []
    for i in range(0, len(texts), _BATCH):
        chunk = texts[i : i + _BATCH]
        enc = tok.encode_batch(chunk)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        res = sess.run(None, feeds)
        last = res[0]  # (B, T, D)
        m3 = mask.astype(np.float32)[:, :, None]
        summed = np.sum(last * m3, axis=1)
        cnt = np.maximum(np.sum(mask, axis=1, dtype=np.float32)[:, None], 1e-9)
        pooled = summed / cnt
        pooled = pooled / np.maximum(
            np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12
        )
        vecs.extend(pooled.tolist())
    return vecs


def _sims(vecs, qvec):
    """Cosine similarity of every corpus vec vs query vec (descending caller sorts)."""
    try:
        import numpy as np

        m = np.asarray(vecs, dtype=np.float32)
        q = np.asarray(qvec, dtype=np.float32)
        qn = np.linalg.norm(q)
        q = q / qn if qn else q
        m = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
        return (m @ q).tolist()
    except ImportError:
        qn = math.sqrt(sum(x * x for x in qvec))
        q = [x / qn for x in qvec] if qn else qvec
        out = []
        for v in vecs:
            vn = math.sqrt(sum(x * x for x in v))
            vv = [x / vn for x in v] if vn else v
            out.append(sum(a * b for a, b in zip(vv, q)))
        return out


def main():
    p = argparse.ArgumentParser(
        description="Semantic search over VCC JSONL exports"
    )
    p.add_argument("export", help="path to *_export.jsonl")
    p.add_argument("--query", required=True)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--provider", choices=["api", "onnx"], default="api")
    p.add_argument("--api-url", default="http://127.0.0.1:8012/v1/embeddings")
    p.add_argument("--api-key", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--cache", default=None)
    p.add_argument("--dims", type=int, default=None)
    p.add_argument("--map", default=None,
                   help="path to section map json (default: next to export)")
    p.add_argument("--no-cache", action="store_true")
    a = p.parse_args()

    export_path = a.export
    if not os.path.exists(export_path):
        raise RuntimeError(f"export not found: {export_path}")
    if a.provider == "api":
        model = a.model or _MODEL_API_DEFAULT
    else:
        model = a.model or os.environ.get("VCC_SEMANTIC_MODEL") or _ONNX_DEFAULT
    expect_dims = a.dims or (384 if a.provider == "onnx" else None)

    # ── corpus: one item per searchable export record ──
    items = []
    with open(export_path, encoding="utf-8") as f:
        for ln, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                r = json.loads(raw)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"bad JSON at line {ln}: {e}")
            text = _record_text(r)
            if not text.strip():
                continue
            msg_id = (r.get("message") or {}).get("id")
            items.append({"line": ln, "text": text, "text_hash": _text_hash(text), "msg_id": msg_id})
    if not items:
        print("No searchable records in export.")
        return

    # ── cache ──
    cache_path = a.cache or (export_path + ".emb.json")
    cache_map = None
    if not a.no_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                stored = json.load(f)
        except Exception:
            stored = None
        if (
            isinstance(stored, dict)
            and stored.get("provider") == a.provider
            and (expect_dims is None or stored.get("dims") == expect_dims)
        ):
            cache_map = {
                it.get("line"): it
                for it in stored.get("items", [])
                if isinstance(it, dict)
            }
            sys.stderr.write("vcc-semantic: using cache\n")
            sys.stderr.flush()
        else:
            sys.stderr.write(
                "vcc-semantic: rebuilding cache (provider/dims mismatch)\n"
            )
            sys.stderr.flush()

    # ── section map: msg_id → .txt line span ──
    map_path = a.map or (os.path.splitext(export_path)[0] + ".map.json")
    sec_map = {}
    if os.path.exists(map_path):
        try:
            _sm = json.load(open(map_path, encoding="utf-8"))
            sec_map = _sm.get("sections", {})
            sys.stderr.write(f"vcc-semantic: using section map ({len(sec_map)} ids)\n")
            sys.stderr.flush()
        except Exception:
            sec_map = {}

    # ── embeddings (reuse cached vecs by line+text_hash) ──
    vecs = [None] * len(items)
    todo = []
    for k, it in enumerate(items):
        c = cache_map.get(it["line"]) if cache_map else None
        if (
            c
            and c.get("text_hash") == it["text_hash"]
            and isinstance(c.get("vec"), list)
            and c["vec"]
        ):
            vecs[k] = c["vec"]
        else:
            todo.append(k)
    if todo:
        todo_texts = [items[k]["text"] for k in todo]
        if a.provider == "api":
            new_vecs = _api_embed(todo_texts, a.api_url, a.api_key, model)
        else:
            new_vecs = _onnx_embed(todo_texts, model)
        for j, k in enumerate(todo):
            vecs[k] = new_vecs[j]
    dims = len(vecs[0]) if vecs else (expect_dims or 0)

    # ── query embedding + rank ──
    if a.provider == "api":
        qvec = _api_embed([a.query], a.api_url, a.api_key, model)[0]
    else:
        qvec = _onnx_embed([a.query], model)[0]
    sims = _sims(vecs, qvec)
    ranked = sorted(range(len(items)), key=lambda k: sims[k], reverse=True)

    # ── persist cache ──
    if not a.no_cache:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "provider": a.provider,
                        "dims": dims,
                        "items": [
                            {
                                "line": it["line"],
                                "text_hash": it["text_hash"],
                                "msg_id": it.get("msg_id"),
                                "text": "\n".join(_preview(it["text"])),
                                "vec": vecs[k],
                            }
                            for k, it in enumerate(items)
                        ],
                    },
                    f,
                )
        except OSError as e:
            sys.stderr.write(f"vcc-semantic: cache write failed: {e}\n")
            sys.stderr.flush()

    # ── output ──
    limit = a.limit if a.limit and a.limit > 0 else 0
    short = _rel_path(export_path)
    out = []
    count = 0
    for k in ranked:
        sc = sims[k]
        if sc <= 0:
            continue
        if limit and count >= limit:
            break
        if count:
            out.append("")
        it = items[k]
        ref = None
        mids = sec_map.get(it.get("msg_id"))
        if mids:
            m0 = mids[0]
            ref = f"({_rel_path(m0['txt'])}:{m0['start']}-{m0['end']})"
        if ref is None:
            ref = f"({short}:{it['line']})"
        out.append(f"{ref} [semantic] score={sc:.2f}")
        for line in _preview(items[k]["text"]):
            out.append("  " + line)
        count += 1
    if out:
        print("\n".join(out))


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    try:
        main()
    except Exception as e:
        sys.stderr.write(f"ERROR: {e}\n")
        sys.exit(1)
