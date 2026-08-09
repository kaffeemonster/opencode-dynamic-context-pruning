#!/usr/bin/env python3
"""View-oriented Conversation Compiler (VCC) - compile Claude JSONL logs into adaptive views.

Produces per conversation chain:
  .txt       Full transcript (lossless rendering of JSONL)
  .min.txt   Brief mode (tool_call summaries with line refs, no separators)
  .view.txt  View mode (search-focused, only with --grep)

Usage:
  python VCC.py conversation.jsonl              # .txt + .min.txt
  python VCC.py conversation.jsonl --grep "kw"  # + .view.txt + stdout search hits
  python VCC.py conversation.jsonl -t 128       # truncation limit (tokens, default 128)
  python VCC.py conversation.jsonl -tu 256      # user message truncation limit (default 256)
  python VCC.py conversation.jsonl -o outdir    # output directory
  python VCC.py project/*.jsonl --grep "kw"     # multi-file search
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import glob as globmod
import math

# ── dict emitter ──

def _emit_dict(data, indent=0):
    parts = []
    prefix = " " * indent
    for k, v in data.items():
        if isinstance(v, dict):
            parts.append(prefix + k + ":")
            parts.append(_emit_dict(v, indent + 2))
        elif isinstance(v, list):
            parts.append(prefix + k + ":")
            for item in v:
                if isinstance(item, dict):
                    parts.append(prefix + "  -")
                    parts.append(_emit_dict(item, indent + 4))
                else:
                    parts.append(prefix + "  - " + str(item))
        else:
            s = str(v)
            if "\n" in s:
                parts.append(prefix + k + ": |")
                p = " " * (indent + 2)
                for line in s.split("\n"):
                    parts.append((p + line) if line else "")
            else:
                parts.append(prefix + k + ": " + s)
    return "\n".join(parts)

# ── tokenizer ──

_TOK_RE = re.compile(
    r'[a-zA-Z]+'           # letters (grouped)
    r'|[0-9]+'             # digits (grouped)
    r'|[^\sa-zA-Z0-9]'     # single char: any non-whitespace non-letter non-digit
    r'|\s+'                # whitespace (preserved, not counted)
)
_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")

def _tokenize(text):
    return _TOK_RE.findall(text)

# ── truncation (token-based) ──

def _trunc(text, limit, ref=""):
    if not limit or not text:
        return text
    tokens = _tokenize(text)
    count, cut = 0, len(tokens)
    for i, t in enumerate(tokens):
        if t.strip():
            count += 1
            if count > limit:
                cut = i
                break
    if cut >= len(tokens):
        return text
    return "".join(tokens[:cut]) + (f"...(truncated from {ref})" if ref else "...(truncated)")

# ── match lines ──

def match_lines(lines, regex, ref_fn="x.txt", start_line=1):
    if not lines:
        return []
    end_line = start_line + len(lines) - 1
    from_ref = f"...(from {ref_fn}:{start_line}-{end_line})"

    matched = []
    for i, line in enumerate(lines):
        if regex.search(line):
            matched.append((start_line + i, line))
    if not matched:
        return [from_ref]

    block_ref = f"({ref_fn}:{start_line}-{end_line})"
    result = [block_ref]
    for ln, lt in matched:
        result.append(f"  {ln}: {lt}")
    return result


# ── lexer ──

SEP = "══════════════════════════════"
_DISCARD_T = {"queue-operation", "file-history-snapshot", "last-prompt", "progress"}
_DISCARD_S = {"stop_hook_summary", "api_error", "bridge_status", "informational", "local_command"}

def _short(fn):
    n, e = os.path.splitext(fn)
    return ("#" + n[-6:] + e) if len(n) > 12 else fn

def _short_tid(tid):
    return tid[-6:] if len(tid) > 6 else tid

def lex(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _collect_stats(chain):
    """Extract usage/timing/model stats from a chain of records, including subagents."""
    from collections import defaultdict
    from datetime import datetime
    totals = defaultdict(int)
    models = set()
    timestamps = []
    api_calls = 0
    tool_uses = 0
    subagent_tokens = 0
    for r in chain:
        msg = r.get("message", {})
        ts = r.get("timestamp")
        if ts:
            timestamps.append(ts)
        usage = msg.get("usage")
        if usage:
            api_calls += 1
            for k, v in usage.items():
                if isinstance(v, int):
                    totals[k] += v
        model = msg.get("model")
        if model:
            models.add(model)
        content = msg.get("content", [])
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    tool_uses += 1
        # subagent usage from toolUseResult
        tur = r.get("toolUseResult")
        if isinstance(tur, dict) and tur.get("agentId"):
            try:
                subagent_tokens += int(tur.get("totalTokens", 0))
            except (ValueError, TypeError):
                pass
    duration = None
    if len(timestamps) >= 2:
        try:
            t0 = datetime.fromisoformat(min(timestamps).replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(max(timestamps).replace("Z", "+00:00"))
            duration = int((t1 - t0).total_seconds())
        except Exception:
            pass
    if api_calls == 0:
        return None
    lines = [SEP, "[stats]", ""]
    lines.append(f"model: {', '.join(sorted(models))}")
    lines.append(f"api_calls: {api_calls}  tool_uses: {tool_uses}")
    if duration is not None:
        m, s = divmod(duration, 60)
        lines.append(f"duration: {m}m{s:02d}s" if m else f"duration: {s}s")
    inp = totals.get("input_tokens", 0)
    cr = totals.get("cache_read_input_tokens", 0)
    cc = totals.get("cache_creation_input_tokens", 0)
    out = totals.get("output_tokens", 0)
    own = inp + cr + cc + out
    own_eff = int(inp * 1.0 + cr * 0.1 + cc * 1.25 + out * 5.0)

    def _fmt_block(label, raw, eff):
        return f"{label}: {raw:,} (effective: {eff:,})"

    if subagent_tokens:
        lines.append("")
        lines.append("Subagents:")
        lines.append(f"  total: {subagent_tokens:,}")
        lines.append("")
        lines.append("Main:")

    parts = []
    if inp: parts.append(f"input: {inp:,}")
    if cr: parts.append(f"cache_read: {cr:,}")
    if cc: parts.append(f"cache_create: {cc:,}")
    pfx = "  " if subagent_tokens else ""
    if parts:
        lines.append(f"{pfx}{'  '.join(parts)}")
    lines.append(f"{pfx}output: {out:,}")
    lines.append(f"{pfx}{_fmt_block('total', own, own_eff)}")

    if subagent_tokens:
        all_raw = own + subagent_tokens
        all_eff = own_eff + subagent_tokens
        lines.append("")
        lines.append("All:")
        lines.append(f"  {_fmt_block('total', all_raw, all_eff)}")
    return lines

def _sanitize(text):
    if not text:
        return text
    if "\r" in text:
        text = text.replace("\r", "")
    if "\x1b" in text:
        text = _ANSI_RE.sub("", text)
    return _CTRL_RE.sub("", text)

def _preprocess_tool_text(text, tool_name):
    text = _sanitize(text)
    if tool_name != "Read":
        return text
    lines = []
    for line in text.split("\n"):
        if "→" in line:
            head, tail = line.split("→", 1)
            if head.strip().isdigit():
                line = tail
        lines.append(line)
    return "\n".join(lines)

def _discard(r):
    t = r.get("type")
    return t in _DISCARD_T or (t == "system" and r.get("subtype") in _DISCARD_S)

def merge_chunks(recs):
    merged = []
    active_mid = None
    active_idx = None
    for r in recs:
        if r.get("type") == "assistant":
            m = r.get("message", {})
            mid = m.get("id")
            if mid and mid == active_mid and active_idx is not None:
                merged[active_idx]["message"]["content"].extend(m.get("content", []))
                if m.get("stop_reason"):
                    merged[active_idx]["message"]["stop_reason"] = m["stop_reason"]
            else:
                merged.append(r)
                if mid:
                    active_mid = mid
                    active_idx = len(merged) - 1
                else:
                    active_mid = None
                    active_idx = None
        else:
            merged.append(r)
            if not _discard(r):
                active_mid = None
                active_idx = None
    return merged

def split_chains(recs):
    kept = [r for r in recs if not _discard(r)]
    chains, cur = [], []
    for r in kept:
        if r.get("type") == "system" and r.get("subtype") == "compact_boundary":
            if cur: chains.append(cur)
            cur = []
        else:
            cur.append(r)
    if cur: chains.append(cur)
    return chains

# ── image / doc ──

def _media_ext(media_type, default_ext):
    if "/" not in media_type:
        return default_ext
    ext = media_type.split("/", 1)[1].split("+", 1)[0]
    if ext == "jpeg":
        return "jpg"
    return ext or default_ext

def _extract_base64(source, outdir, data_prefix, data_index, stem, default_mt, default_ext):
    mt = source.get("media_type", default_mt)
    fn = f"{data_prefix}_{stem}_{data_index}.{_media_ext(mt, default_ext)}"
    with open(os.path.join(outdir, fn), "wb") as f:
        f.write(base64.b64decode(source.get("data", "")))
    return fn

def _extract_img(source, outdir, data_prefix, data_index):
    return _extract_base64(source, outdir, data_prefix, data_index,
                           "img", "image/png", "png")

def _extract_doc(source, outdir, data_prefix, data_index):
    return _extract_base64(source, outdir, data_prefix, data_index,
                           "doc", "application/octet-stream", "bin")


# ── tool_call summary ──

_TOOL_SUMMARY_FIELDS = {
    "Read": "file_path", "Edit": "file_path", "Write": "file_path",
    "Glob": "pattern", "Grep": "pattern",
    "Agent": "description", "Skill": "skill",
}

def _tool_summary(name, inp):
    """Build one-line summary: * Name "param" """
    field = _TOOL_SUMMARY_FIELDS.get(name)
    if field and field in inp:
        return f'* {name} "{inp[field]}"'
    if name == "Bash":
        val = inp.get("description") or inp.get("command", "")
        if val:
            # truncate long commands
            if not inp.get("description") and len(val) > 60:
                val = val[:57] + "..."
            return f'* {name} "{val}"'
    return f"* {name}"


# ── IR node ──

def _node(typ, content, **kw):
    o = {"type": typ, "content": content,
         "searchable": kw.pop("searchable", False)}
    o.update(kw)
    return o

# ── parser ──

def parse(chain, outdir, data_prefix, data_ctr):
    ir = []
    sec = 0
    blk = 0
    active_timestamp = None

    tid_name = {}
    for r in chain:
        if r.get("type") == "assistant":
            for b in r.get("message", {}).get("content", []):
                if b.get("type") == "tool_use":
                    tid_name[b.get("id", "")] = b.get("name", "unknown")

    def _emit_sep():
        if sec > 0:
            ir.append(_node("meta", ["", SEP]))

    def _emit_header(h):
        ir.append(_node("meta_header", [h, ""], _sec=sec,
                        _event_timestamp=active_timestamp))

    def _emit_blocks(blocks, text_type):
        nonlocal blk
        has_any = False
        for b in blocks:
            bt = b.get("type")
            if bt == "thinking":
                txt = _sanitize(b.get("thinking", ""))
                if not txt: continue
                ir.append(_node("meta", [">>>thinking"], _sec=sec, _blk=blk))
                ir.append(_node("thinking", txt.split("\n"), searchable=True,
                                 _sec=sec, _blk=blk))
                ir.append(_node("meta", ["<<<thinking"], _sec=sec, _blk=blk))
                blk += 1; has_any = True

            elif bt == "redacted_thinking":
                ir.append(_node("meta", [">>>redacted_thinking"], _sec=sec, _blk=blk))
                ir.append(_node("redacted_thinking",
                                 ["[content redacted by model provider]"],
                                 searchable=True, _sec=sec, _blk=blk))
                ir.append(_node("meta", ["<<<redacted_thinking"], _sec=sec, _blk=blk))
                blk += 1; has_any = True

            elif bt == "text":
                txt = _sanitize(b.get("text", ""))
                if not txt: continue
                ir.append(_node(text_type, txt.split("\n"), searchable=True,
                                 _sec=sec, _blk=blk))
                blk += 1; has_any = True

            elif bt == "tool_use":
                name = b.get("name", "unknown")
                tid = b.get("id", "")
                inp = b.get("input", {})
                hl = f">>>tool_call {name}:{_short_tid(tid)}"
                summary = _tool_summary(name, inp)
                ir.append(_node("meta", [hl], _sec=sec, _blk=blk,
                                 _tool_summary=summary))
                if inp:
                    ir.append(_node("tool_call", _emit_dict(inp).split("\n"),
                                     searchable=True, _sec=sec, _blk=blk))
                ir.append(_node("meta", ["<<<tool_call"], _sec=sec, _blk=blk))
                blk += 1; has_any = True

            elif bt == "image":
                src = b.get("source", {})
                if src.get("type") == "base64":
                    fn = _extract_img(src, outdir, data_prefix, data_ctr[0])
                    data_ctr[0] += 1
                    ir.append(_node(f"{text_type}_image", [f"[image: {fn}]"],
                                     searchable=True, _sec=sec, _blk=blk))
                    blk += 1; has_any = True

            elif bt == "document":
                src = b.get("source", {})
                label = "[document]"
                if src.get("type") == "base64":
                    fn = _extract_doc(src, outdir, data_prefix, data_ctr[0])
                    data_ctr[0] += 1
                    label = f"[document: {fn}]"
                ir.append(_node(f"{text_type}_document", [label],
                                 searchable=True, _sec=sec, _blk=blk))
                blk += 1; has_any = True
        return has_any

    for r in chain:
        rt = r.get("type")
        active_timestamp = r.get("timestamp")

        if rt == "system":
            if r.get("subtype") == "compact_boundary": continue
            content = r.get("content", "") or r.get("message", {}).get("content", "")
            if not content: continue
            _emit_sep(); _emit_header("[system]")
            if isinstance(content, list):
                _emit_blocks(content, "system")
            else:
                ir.append(_node("system", _sanitize(content).split("\n"), searchable=True,
                                 _sec=sec, _blk=blk))
                blk += 1
            sec += 1

        elif rt == "user":
            if r.get("isCompactSummary"):
                content = r.get("message", {}).get("content", "")
                nlines = content.count("\n") + 1 if content else 0
                _emit_sep(); _emit_header("[user]")
                ir.append(_node("user", [f"[compact summary — {nlines} lines]"], searchable=False,
                                 _sec=sec, _blk=blk))
                blk += 1; sec += 1
                continue
            content = r.get("message", {}).get("content", "")
            if isinstance(content, str):
                if content:
                    _emit_sep(); _emit_header("[user]")
                    ir.append(_node("user", _sanitize(content).split("\n"), searchable=True,
                                     _sec=sec, _blk=blk))
                    blk += 1; sec += 1
            elif isinstance(content, list):
                tblocks = [b for b in content if b.get("type") != "tool_result"]
                tresults = [b for b in content if b.get("type") == "tool_result"]
                if tblocks:
                    mark = len(ir)
                    saved_data = data_ctr[0]
                    saved_blk = blk
                    _emit_sep(); _emit_header("[user]")
                    if _emit_blocks(tblocks, "user"):
                        sec += 1
                    else:
                        del ir[mark:]
                        data_ctr[0] = saved_data
                        blk = saved_blk
                for tr in tresults:
                    tuid = tr.get("tool_use_id", "")
                    is_err = tr.get("is_error", False)
                    nm = tid_name.get(tuid, "unknown")
                    role = "tool_error" if is_err else "tool"
                    btype = "tool_error" if is_err else "tool_result"
                    _emit_sep(); _emit_header(f"[{role}] {nm}:{_short_tid(tuid)}")
                    tc = tr.get("content", "")
                    parts = []
                    if isinstance(tc, str):
                        parts.append(_preprocess_tool_text(tc, nm))
                    elif isinstance(tc, list):
                        for item in tc:
                            if item.get("type") == "text":
                                parts.append(_preprocess_tool_text(item.get("text", ""), nm))
                            elif item.get("type") == "image":
                                src = item.get("source", {})
                                if src.get("type") == "base64":
                                    fn = _extract_img(src, outdir, data_prefix, data_ctr[0])
                                    data_ctr[0] += 1
                                    parts.append(f"[image: {fn}]")
                            elif item.get("type") == "document":
                                src = item.get("source", {})
                                if src.get("type") == "base64":
                                    fn = _extract_doc(src, outdir, data_prefix, data_ctr[0])
                                    data_ctr[0] += 1
                                    parts.append(f"[document: {fn}]")
                    ir.append(_node(btype, _sanitize("\n\n".join(parts)).split("\n"),
                                     searchable=True, _sec=sec, _blk=blk))
                    blk += 1; sec += 1

        elif rt == "assistant":
            blocks = r.get("message", {}).get("content", [])
            has = any(
                (b.get("type") == "thinking" and b.get("thinking")) or
                b.get("type") == "redacted_thinking" or
                (b.get("type") == "text" and b.get("text")) or
                b.get("type") == "tool_use" or
                b.get("type") == "image" or
                b.get("type") == "document"
                for b in blocks)
            if has:
                _emit_sep(); _emit_header("[assistant]")
                _emit_blocks(blocks, "assistant")
                sec += 1

    ir.append(_node("meta", [""]))  # trailing newline
    return ir


# ── IR walk ──

def _is_tool_summary(o):
    return o.get("_tool_summary") is not None

def _walk(ir, key="content"):
    prev_blk = None
    prev_o = None
    for o in ir:
        c = o.get(key)
        if c is None:
            continue
        if not c:
            continue
        blk = o.get("_blk")
        if blk is not None and prev_blk is not None and blk != prev_blk:
            if key == "content":
                blank = True
            else:
                # non-content: suppress blank between consecutive tool summaries
                blank = not (_is_tool_summary(prev_o) and _is_tool_summary(o))
        else:
            blank = False
        yield o, c, blank
        if blk is not None:
            prev_blk = blk
            prev_o = o
        elif SEP in o.get("content", []):
            prev_blk = None
            prev_o = None


# ── line assignment ──

def assign_lines(ir):
    line = 0
    for o, c, blank in _walk(ir, "content"):
        if blank: line += 1
        o["start_line"] = line
        line += len(c)
        o["end_line"] = line - 1
    return line


# ── lowering helpers ──

def _is_truncatable(o):
    t = o["type"]
    if t in ("meta", "meta_header", "thinking", "redacted_thinking"):
        return False
    if t.endswith("_image") or t.endswith("_document"):
        return False
    return True

def _is_thinking(o):
    return o["type"] in ("thinking", "redacted_thinking")

def _sec_roles(ir):
    """Map sec -> role from meta_header content."""
    roles = {}
    for o in ir:
        if o["type"] == "meta_header":
            s = o.get("_sec")
            if s is None: continue
            h = o["content"][0]
            if h.startswith("[tool_error]"):
                roles[s] = "tool_error"
            elif h.startswith("[tool]"):
                roles[s] = "tool"
            elif h.startswith("[assistant]"):
                roles[s] = "assistant"
            elif h.startswith("[user]"):
                roles[s] = "user"
            elif h.startswith("[system]"):
                roles[s] = "system"
    return roles


# ── brief-mode filtering ──

_BRIEF_STRIP_RE = re.compile(
    r'<(ide_opened_file|ide_selection|system-reminder|command-message)>.*?</\1>\s*',
    re.DOTALL)
_BRIEF_UNWRAP_RE = re.compile(r'</?(?:command-name|command-args)>')
_META_USER_RE = re.compile(
    r'^\s*<(?:task-notification|local-command-caveat|local-command-stdout'
    r'|local-command-stderr)\b')
_SKILL_USER_RE = re.compile(r'^Base directory for this skill:')
_BRIEF_HIDE_TOOLS = {"TodoWrite", "ToolSearch"}
_BRIEF_HIDE_EXACT = {"Continue from where you left off.", "No response requested."}

def _strip_noise_xml(text):
    """Strip known noise XML from user text for brief mode."""
    text = _BRIEF_STRIP_RE.sub('', text)
    text = _BRIEF_UNWRAP_RE.sub('', text)
    return text.strip()

def _user_hidden_in_brief(ir, sec):
    """Check if a user section should be entirely hidden in brief mode."""
    blocks = [o for o in ir if o.get("_sec") == sec and o.get("searchable")]
    if not blocks:
        return False
    for o in blocks:
        text = "\n".join(o["content"]).strip()
        if not text:
            continue
        if _META_USER_RE.match(text):
            continue
        if _SKILL_USER_RE.match(text):
            continue
        if not _strip_noise_xml(text):
            continue
        return False
    return True

def _section_hidden_exact(ir, sec):
    """Check if all searchable content in a section is an exact-match hide string."""
    blocks = [o for o in ir if o.get("_sec") == sec and o.get("searchable")
              and o["type"] not in ("thinking", "redacted_thinking")]
    if not blocks:
        return False
    for o in blocks:
        text = "\n".join(o["content"]).strip()
        if text not in _BRIEF_HIDE_EXACT:
            return False
    return True


# ── lowering: brief ──

def _tid_result_ranges(ir):
    """Map short_tid → (start_line, end_line) for tool_result sections."""
    # Find which sec corresponds to which short_tid
    tid_sec = {}
    for o in ir:
        if o["type"] == "meta_header":
            c = o["content"]
            if c and len(c) >= 1:
                h = c[0]
                # [tool] name:AABBCC or [tool_error] name:AABBCC
                if h.startswith("[tool]") or h.startswith("[tool_error]"):
                    parts = h.split(":")
                    if len(parts) >= 2:
                        tid_sec[parts[-1]] = o.get("_sec")
    # Collect line ranges per sec
    sec_range = {}
    for o in ir:
        s = o.get("_sec")
        if s is None:
            continue
        sl = o.get("start_line")
        el = o.get("end_line")
        if sl is not None and el is not None:
            if s not in sec_range:
                sec_range[s] = [sl, el]
            else:
                sec_range[s][0] = min(sec_range[s][0], sl)
                sec_range[s][1] = max(sec_range[s][1], el)
    # Build tid → range (only content nodes, skip the separator/header)
    result = {}
    for tid, sec in tid_sec.items():
        if sec in sec_range:
            result[tid] = tuple(sec_range[sec])
    return result


def lower_brief(ir, truncate, filename="", truncate_user=256):
    short = _short(filename)
    roles = _sec_roles(ir)
    tid_ranges = _tid_result_ranges(ir)

    # Sections visible in truncation: not tool/tool_error/system
    visible_secs = {sec for sec, role in roles.items()
                     if role not in ("tool", "tool_error", "system")}

    # Determine which assistant sections to merge:
    # An assistant section merges if the previous VISIBLE section is also assistant.
    all_secs = sorted(roles.keys())
    merge_secs = set()
    for i in range(1, len(all_secs)):
        cur_sec = all_secs[i]
        if not (cur_sec in visible_secs and
                roles.get(cur_sec) == "assistant"):
            continue
        # Find previous visible section
        for j in range(i - 1, -1, -1):
            if all_secs[j] in visible_secs:
                if roles.get(all_secs[j]) == "assistant":
                    merge_secs.add(cur_sec)
                break

    # Track which secs only have thinking content (no visible non-thinking blocks)
    sec_has_nonthink = set()
    for o in ir:
        s = o.get("_sec")
        if s is None: continue
        if o["type"] not in ("meta", "meta_header", "thinking", "redacted_thinking"):
            sec_has_nonthink.add(s)

    # Sections that are all-thinking should still be hidden in truncation
    # (they have no visible content after thinking is removed)
    for sec in list(visible_secs):
        if sec not in sec_has_nonthink:
            visible_secs.discard(sec)
        elif roles.get(sec) == "user" and _user_hidden_in_brief(ir, sec):
            visible_secs.discard(sec)
        elif _section_hidden_exact(ir, sec):
            visible_secs.discard(sec)

    # Recompute merge_secs after possible visibility changes
    all_secs_2 = sorted(roles.keys())
    merge_secs = set()
    for i in range(1, len(all_secs_2)):
        cur_sec = all_secs_2[i]
        if not (cur_sec in visible_secs and
                roles.get(cur_sec) == "assistant"):
            continue
        for j in range(i - 1, -1, -1):
            if all_secs_2[j] in visible_secs:
                if roles.get(all_secs_2[j]) == "assistant":
                    merge_secs.add(cur_sec)
                break

    for idx, o in enumerate(ir):
        s = o.get("_sec")

        # Separator: replace with blank line in brief mode
        if s is None and o["type"] == "meta" and SEP in o.get("content", []):
            ns = None
            for j in range(idx + 1, len(ir)):
                ns = ir[j].get("_sec")
                if ns is not None:
                    break
            if ns is None or ns not in visible_secs:
                o["content_brief"] = None
            elif ns in merge_secs:
                o["content_brief"] = None
            elif not any(sec in visible_secs for sec in range(ns)):
                o["content_brief"] = None
            else:
                o["content_brief"] = [""]
            continue

        # Section not visible → hide
        if s is not None and s not in visible_secs:
            o["content_brief"] = None
            continue

        # Merged assistant: hide separator and header
        if s in merge_secs and o["type"] == "meta_header":
            o["content_brief"] = None
            continue

        # Thinking / redacted_thinking → hide (including their >>> <<< metas)
        if _is_thinking(o):
            o["content_brief"] = None
            continue
        if o["type"] == "meta":
            c = o["content"]
            if c and (c[0].startswith(">>>thinking") or c[0].startswith("<<<thinking") or
                      c[0].startswith(">>>redacted_thinking") or c[0].startswith("<<<redacted_thinking")):
                o["content_brief"] = None
                continue

        # Tool_call three-piece: collapse to single-line summary with line ref
        if o["type"] == "meta" and o.get("content", []):
            c0 = o["content"][0]
            if c0.startswith(">>>tool_call"):
                # Hide noise tools (internal bookkeeping)
                tool_name = c0.split()[1].split(":")[0] if len(c0.split()) > 1 else ""
                if tool_name in _BRIEF_HIDE_TOOLS:
                    o["content_brief"] = None
                    continue
                summary = o.get("_tool_summary", f"* unknown")
                s = o.get("start_line")
                e = o.get("end_line")
                for j in (idx + 1, idx + 2):
                    if j < len(ir) and ir[j].get("_blk") == o.get("_blk"):
                        je = ir[j].get("end_line")
                        if je is not None:
                            e = je
                if s is not None and e is not None:
                    # Extract short_tid and look up tool_result range
                    parts = c0.split()[1].split(":") if len(c0.split()) > 1 else []
                    stid = parts[-1] if len(parts) >= 2 else ""
                    rr = tid_ranges.get(stid)
                    if rr:
                        summary = f"{summary} ({short}:{s+1}-{e+1},{rr[0]+1}-{rr[1]+1})"
                    else:
                        summary = f"{summary} ({short}:{s+1}-{e+1})"
                o["content_brief"] = [summary]
                continue
            if c0 == "<<<tool_call":
                o["content_brief"] = None
                continue

        if o["type"] == "tool_call":
            o["content_brief"] = None
            continue

        # meta_header → copy
        if o["type"] == "meta_header":
            o["content_brief"] = list(o["content"])
            continue

        # meta → copy
        if o["type"] == "meta":
            o["content_brief"] = list(o["content"])
            continue

        # Truncatable content
        if _is_truncatable(o):
            node_start = o.get("start_line", 0) + 1
            node_end = o.get("end_line", o.get("start_line", 0)) + 1
            ref = f"{short}:{node_start}-{node_end}"
            text = "\n".join(o["content"])
            if o["type"] == "user":
                text = _strip_noise_xml(text)
                if not text.strip():
                    o["content_brief"] = None
                    continue
            lim = truncate_user if o["type"] == "user" else truncate
            lines = (_trunc(text, lim, ref) if lim else text).split("\n")
            # Strip leading blank lines
            while lines and not lines[0]:
                lines.pop(0)
            o["content_brief"] = lines
            continue

        # Non-truncatable (images, documents, etc) → copy
        o["content_brief"] = list(o["content"])


# ── lowering: view ──

def lower_view(ir, filename="", grep_pattern=None, limit=0, brief=False):
    if not grep_pattern:
        # No grep: view is same as truncated (shouldn't normally be called)
        for o in ir:
            o["content_view"] = o.get("content_brief")
        return

    short = _short(filename)

    # Per-node match check
    def _node_matches(o):
        if not o["searchable"]:
            return False
        src = o["content_brief"] if brief else o["content"]
        if not src:
            return False
        for line in src:
            if grep_pattern.search(line):
                return True
        return False

    # Pass 1: determine visibility for each searchable block
    block_visible = {}  # blk -> bool
    count = 0
    for o in ir:
        blk = o.get("_blk")
        if blk is None or blk in block_visible:
            continue
        if o["searchable"] and _node_matches(o):
            block_visible[blk] = True
            count += 1
            if limit and count >= limit:
                break

    # Derive which sections have any visible block (for header/separator logic)
    sec_has_visible = set()
    for o in ir:
        blk = o.get("_blk")
        if blk is not None and block_visible.get(blk):
            s = o.get("_sec")
            if s is not None:
                sec_has_visible.add(s)

    # Pass 2: set content_view for each node
    for idx, o in enumerate(ir):
        s = o.get("_sec")
        blk = o.get("_blk")

        # Separator: show only between two sections that have visible blocks
        if s is None and o["type"] == "meta" and SEP in o.get("content", []):
            next_vis = False
            for j in range(idx + 1, len(ir)):
                ns = ir[j].get("_sec")
                if ns is not None:
                    next_vis = ns in sec_has_visible
                    break
            prev_vis = False
            seen = set()
            for j in range(idx - 1, -1, -1):
                ps = ir[j].get("_sec")
                if ps is not None and ps not in seen:
                    seen.add(ps)
                    if ps in sec_has_visible:
                        prev_vis = True
                        break
            o["content_view"] = list(o["content"]) if (next_vis and prev_vis) else None
            continue

        # meta_header: show if section has any visible block
        if o["type"] == "meta_header":
            o["content_view"] = list(o["content"]) if s in sec_has_visible else None
            continue

        # Thinking / tool_call metas: show if same blk matched
        if o["type"] == "meta" and o.get("content", []):
            c0 = o["content"][0]
            if c0.startswith(">>>thinking") or c0.startswith("<<<thinking") or \
               c0.startswith(">>>redacted_thinking") or c0.startswith("<<<redacted_thinking") or \
               c0.startswith(">>>tool_call ") or c0 == "<<<tool_call":
                o["content_view"] = list(o["content"]) if block_visible.get(blk) else None
                continue

        # Other meta → show if section has visible blocks
        if o["type"] == "meta":
            o["content_view"] = list(o["content"]) if s in sec_has_visible else None
            continue

        # Searchable content blocks: show only if this block matches
        if o["searchable"]:
            if limit and not block_visible.get(blk):
                o["content_view"] = None
                continue
            if _node_matches(o):
                node_start = o.get("start_line", 0) + 1
                o["content_view"] = match_lines(
                    o["content"], grep_pattern, short, node_start)
            else:
                o["content_view"] = None
            continue

        # Non-searchable (images, docs, etc) → hide
        o["content_view"] = None


# ── codegen ──

def emit(ir, key="content"):
    lines = []
    for o, c, blank in _walk(ir, key):
        if blank: lines.append("")
        lines.extend(c)
    return lines


# ── grep ──

def _rel_path(fp):
    try:
        return os.path.relpath(fp)
    except ValueError:
        return os.path.abspath(fp)

def grep_search(results, pattern, limit=0, brief=False, order="newest"):
    first = True
    count = 0
    oldest = order == "oldest"
    for filepath, ir in (results if oldest else reversed(results)):
        short = _rel_path(filepath)
        section_ts = {
            o.get("_sec"): o.get("_event_timestamp")
            for o in ir
            if o.get("type") == "meta_header" and o.get("_sec") is not None
        }
        for o in (ir if oldest else reversed(ir)):
            if not o["searchable"]: continue
            src = o["content_brief"] if brief else o["content"]
            lines = match_lines(src, pattern, short, o.get("start_line", 0) + 1)
            if len(lines) <= 1:
                continue
            count += 1
            if not first: print()
            first = False
            ts = section_ts.get(o.get("_sec"))
            ts_suffix = f" event={ts}" if ts else ""
            print(f"{lines[0]} [{o['type']}]{ts_suffix}")
            for lt in lines[1:]:
                print(lt)
            if limit and count >= limit:
                return


# ── BM25 text search ──

_STOPWORDS_EN = frozenset(
    "a an and are as at be but by for from has have he her his i if in is it its "
    "of on or our she so that the their them then there they this to was we were "
    "what when which who will with you your not no do does did can could would "
    "should may might must about into over under up down out off again once also "
    "just very than too more most some any each own same other only new now "
    "here where why how all because before after while during"
    .split()
)

_STOPWORDS_DE = frozenset(
    "aber als am an auch auf aus bei beim bin bis bist das dass dem den der des "
    "die dieser diese dieses diesen diesem doch dort du durch ein eine einem einen "
    "einer eines er es euch euer für gegen gewesen gibt habe haben hat hatte "
    "hier hin hinter ich ihm ihn ihre ihr im in ist ja jeder jedes jetzt kann "
    "können konnte kein keine keinen keinem keiner machen man mich mir mit muss "
    "müssen nach nicht nichts noch nun nur ob oder ohne sehr sein seine sich sie "
    "sind so soll sollte sondern sonst über um und uns unter vom von vor war "
    "waren warum was weg weil weiter welche welchem welchen welcher welches wenn "
    "wer werde werden wie wieder will wir wird wo wohl wollen womit wozu zu zum "
    "zur zwar zwischen"
    .split()
)

_STOPWORDS = _STOPWORDS_EN | _STOPWORDS_DE

def _is_cjk(ch):
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF or  # CJK Unified
        0x3400 <= cp <= 0x4DBF or  # Ext A
        0x3040 <= cp <= 0x30FF or  # Hiragana/Katakana
        0xAC00 <= cp <= 0xD7AF     # Hangul
    )

def _tokenize_bm25(text):
    """Lowercase, split on non-alnum into words; CJK runs → char bigrams."""
    tokens = []
    word_buf = []
    cjk_buf = []
    for ch in text.lower():
        if _is_cjk(ch):
            if word_buf:
                tokens.append("".join(word_buf))
                word_buf = []
            cjk_buf.append(ch)
            continue
        if cjk_buf:
            run = "".join(cjk_buf)
            if len(run) == 1:
                tokens.append(run)
            else:
                tokens.extend(run[i:i+2] for i in range(len(run) - 1))
            cjk_buf = []
        if ch.isalnum():
            word_buf.append(ch)
        else:
            if word_buf:
                tokens.append("".join(word_buf))
                word_buf = []
    if cjk_buf:
        run = "".join(cjk_buf)
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i:i+2] for i in range(len(run) - 1))
    if word_buf:
        tokens.append("".join(word_buf))
    return tokens

def _stem_basic(w):
    """Very basic suffix-stripping stemmer (stdlib-pure)."""
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ing"):
        return w[:-3]
    if w.endswith("ed"):
        return w[:-2]
    if w.endswith("ly"):
        return w[:-2]
    if w.endswith("es") and not w.endswith(("ses", "xes", "zes", "ches", "shes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us")):
        return w[:-1]
    return w

def _stem_german(w):
    """Very basic German suffix stripper (stdlib-pure). Over-stemming is
    acceptable: query and document must only agree with each other."""
    if len(w) <= 3:
        return w
    if w.startswith("ge") and len(w) > 5:
        w = w[2:]              # gemacht -> macht
    if w.endswith("ung"):
        w = w[:-3]
    if w.endswith("en") or w.endswith("ern"):
        w = w[:-2]
    elif w.endswith("e"):
        w = w[:-1]
    elif w.endswith("est"):
        w = w[:-3]
    elif w.endswith("st") and len(w) > 4:
        w = w[:-2]             # machst -> mach
    elif w.endswith("te") and len(w) > 4:
        w = w[:-2]             # machte -> mach
    elif w.endswith("er"):
        w = w[:-2]             # Spieler -> Spiel
    elif w.endswith("es"):
        w = w[:-2]
    if w.endswith("t") and len(w) > 3 and not w.endswith(("hat", "mit", "ist", "bit")):
        w = w[:-1]             # macht -> mach (guarded)
    return w

def _analyze(text):
    out = []
    for t in _tokenize_bm25(text):
        if t in _STOPWORDS:
            continue
        out.append(_stem_basic(_stem_german(t)))
    return out

def _bm25f_score(tf_map, dl, avgdl, query_terms, idf_cache, k1=1.2, b=0.75):
    """BM25+ (delta=1.0) for ONE field of ONE doc."""
    s = 0.0
    for t in query_terms:
        f = tf_map.get(t, 0)
        if not f:
            continue
        idf = idf_cache.get(t, 0.0)
        s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl)) + idf * 1.0
    return s


def _bm25f_scores(docs, query_terms, field_weights, k1=1.2, b=0.75):
    """
    docs: list of (tokens_full, tokens_brief) tuples (either may be []).
    field_weights: dict {'full': float, 'brief': float} (default {'full':1.0,'brief':1.4}).
    Returns list of floats (same order as docs).
    """
    n = len(docs)
    if n == 0 or not query_terms:
        return [0.0] * n
    # Per-field corpora for df/idf and avgdl
    corp_full = [d[0] for d in docs]
    corp_brief = [d[1] for d in docs]
    avg_full = sum(len(d) for d in corp_full) / max(n, 1)
    avg_brief = sum(len(d) for d in corp_brief) / max(n, 1)
    df_full, df_brief = {}, {}
    for d in corp_full:
        for t in set(d):
            df_full[t] = df_full.get(t, 0) + 1
    for d in corp_brief:
        for t in set(d):
            df_brief[t] = df_brief.get(t, 0) + 1
    def _idf(n_docs, df):
        return {t: math.log(1 + (n_docs - f + 0.5) / (f + 0.5)) for t, f in df.items()}
    idf_full = _idf(n, df_full)
    idf_brief = _idf(n, df_brief)
    out = []
    for i, (tok_full, tok_brief) in enumerate(docs):
        s = 0.0
        if tok_full and avg_full > 0 and field_weights.get('full', 0):
            tf_full = {}
            for t in tok_full:
                tf_full[t] = tf_full.get(t, 0) + 1
            s += field_weights['full'] * _bm25f_score(
                tf_full, len(tok_full), avg_full, query_terms, idf_full, k1, b)
        if tok_brief and avg_brief > 0 and field_weights.get('brief', 0):
            tf_brief = {}
            for t in tok_brief:
                tf_brief[t] = tf_brief.get(t, 0) + 1
            s += field_weights['brief'] * _bm25f_score(
                tf_brief, len(tok_brief), avg_brief, query_terms, idf_brief, k1, b)
        out.append(s)
    return out


def _bm25l_score(tf_map, dl, avgdl, query_terms, idf_cache, delta=0.5, k1=1.2, b=0.75):
    """BM25L: length-normalize tf first, then saturate (no b inside)."""
    s = 0.0
    denom = dl / avgdl + b if avgdl > 0 else 1.0
    for t in query_terms:
        f = tf_map.get(t, 0)
        if not f:
            continue
        tf_star = delta + f * (1 + b) / denom
        idf = idf_cache.get(t, 0.0)
        s += idf * (tf_star * (k1 + 1)) / (tf_star + k1)
    return s


def _bm25l_scores(docs, query_terms, field_weights, delta=0.5, k1=1.2, b=0.75):
    """BM25L multi-field variant; same signature shape as _bm25f_scores."""
    n = len(docs)
    if n == 0 or not query_terms:
        return [0.0] * n
    corp_full = [d[0] for d in docs]
    corp_brief = [d[1] for d in docs]
    avg_full = sum(len(d) for d in corp_full) / max(n, 1)
    avg_brief = sum(len(d) for d in corp_brief) / max(n, 1)
    df_full, df_brief = {}, {}
    for d in corp_full:
        for t in set(d):
            df_full[t] = df_full.get(t, 0) + 1
    for d in corp_brief:
        for t in set(d):
            df_brief[t] = df_brief.get(t, 0) + 1
    def _idf(n_docs, df):
        return {t: math.log(1 + (n_docs - f + 0.5) / (f + 0.5)) for t, f in df.items()}
    idf_full = _idf(n, df_full)
    idf_brief = _idf(n, df_brief)
    out = []
    for i, (tok_full, tok_brief) in enumerate(docs):
        s = 0.0
        if tok_full and avg_full > 0 and field_weights.get('full', 0):
            tf_full = {}
            for t in tok_full:
                tf_full[t] = tf_full.get(t, 0) + 1
            s += field_weights['full'] * _bm25l_score(
                tf_full, len(tok_full), avg_full, query_terms, idf_full, delta, k1, b)
        if tok_brief and avg_brief > 0 and field_weights.get('brief', 0):
            tf_brief = {}
            for t in tok_brief:
                tf_brief[t] = tf_brief.get(t, 0) + 1
            s += field_weights['brief'] * _bm25l_score(
                tf_brief, len(tok_brief), avg_brief, query_terms, idf_brief, delta, k1, b)
        out.append(s)
    return out

# pi-vcc-style importance priors (query-independent, additive boost)
_PRIOR_SCALE = 0.2          # keep prior modest vs BM25 score (~0-10)
_EDIT_TOOL_RE = re.compile(r"^(edit|write|multiedit|quick_edit|target_edit|apply_patch)$", re.I)
_READ_TOOL_RE = re.compile(r"^(read|glob|grep|ls|find|search|view)$", re.I)
_WORKFLOW_TOOL_RE = re.compile(
    r"^(git|gh|npm|npx|yarn|pnpm|cargo|make|docker|docker-compose|kubectl|terraform|"
    r"pip|pip3|apt|apt-get|dnf|pacman|emerge|systemctl|ssh|scp|rsync|curl|wget)$", re.I)
_TEST_CMD_RE = re.compile(
    r"^(pytest|python -m pytest|go test|npm test|npm run test|cargo test|make test|"
    r"yarn test|pnpm test|mix test|rake test|ctest|mvn test|gradle test|zig test)\b")
_TRIVIAL_BASH_RE = re.compile(r"^(ls|pwd|echo|cd|cat|head|tail|wc|date|whoami|env|"
    r"git status|git diff|git log)\b")

def _node_prior(o, cur_tool, content_lines):
    """Query-independent importance prior, pi-vcc-inspired."""
    typ = o["type"]
    if typ == "user":
        return 18
    if typ == "assistant":
        return 10
    if typ in ("thinking", "redacted_thinking"):
        return -8
    if typ == "system":
        return 5
    if typ == "tool_error":
        return 24
    if typ == "tool_result":
        p = 1
        if len(content_lines) > 300:
            p -= 8
        return p
    if typ == "tool_call":
        name = (cur_tool or "").lower()
        if _EDIT_TOOL_RE.match(name):
            return 34
        if name == "bash":
            first = (content_lines[0] if content_lines else "").strip()
            if _TEST_CMD_RE.match(first):
                return 26
            if _TRIVIAL_BASH_RE.match(first) and len(" ".join(content_lines)) < 80:
                return -16
            return 12
        if _WORKFLOW_TOOL_RE.match(name):
            return 14
        if _READ_TOOL_RE.match(name):
            return 6
        return 12
    return 0

def bm25_search(results, query, limit=0, brief=False, fusion=False, bm25l=False):
    query_terms = _analyze(query)
    if not query_terms:
        print("No searchable terms in query.")
        return
    # Corpus: one doc per searchable node, with per-node tool context + prior.
    corpus = []   # (filepath, o, prior)
    tf_pairs = [] # (tf_full_token_list, tf_brief_token_list) — token LISTS, counted later
    dedup_seen = set()
    for filepath, ir in results:
        section_ts = {
            o.get("_sec"): o.get("_event_timestamp")
            for o in ir
            if o.get("type") == "meta_header" and o.get("_sec") is not None
        }
        cur_tool = None
        for o in ir:
            # Track current tool name from meta headers
            if o["type"] == "meta":
                first = (o.get("content") or [""])[0]
                if first.startswith(">>>tool_call "):
                    cur_tool = first[len(">>>tool_call "):].split(":")[0]
                continue
            if o["type"] == "meta_header":
                first = (o.get("content") or [""])[0]
                m = re.match(r"^\[(?:tool|tool_error)\] (\S+):", first)
                if m:
                    cur_tool = m.group(1)
                continue
            if not o.get("searchable"):
                continue
            full_src = o.get("content") or []
            brief_src = o.get("content_brief") or []
            if not full_src and not brief_src:
                continue
            text_full = "\n".join(full_src)
            # Dedup: (tool, first meaningful line) — collapse repeated identical calls
            dkey = None
            if cur_tool == "bash":
                norm = re.sub(r"\s+", " ", text_full)[:80]
                dkey = ("bash", norm)
            elif cur_tool:
                dkey = (cur_tool, (full_src[0] if full_src else "")[:80])
            if dkey is not None:
                if dkey in dedup_seen:
                    continue
                dedup_seen.add(dkey)
            prior = _node_prior(o, cur_tool, full_src)
            corpus.append((filepath, o, prior, section_ts.get(o.get("_sec"))))
            # Skip brief field when it duplicates full content (common case) —
            # double-counting inflates scores ~2x.
            brief_toks = []
            if brief_src and brief_src != full_src:
                brief_toks = _analyze("\n".join(brief_src))
            tf_pairs.append((_analyze(text_full), brief_toks))

    if not corpus:
        print("No searchable blocks in session.")
        return

    n = len(corpus)
    if fusion:
        # RRF: fuse four ranked lists (BM25F + BM25L, each full/brief), k=60
        score_lists = (
            _bm25f_scores(tf_pairs, query_terms, {'full': 1.0, 'brief': 0.0}),
            _bm25f_scores(tf_pairs, query_terms, {'full': 0.0, 'brief': 1.4}),
            _bm25l_scores(tf_pairs, query_terms, {'full': 1.0, 'brief': 0.0}),
            _bm25l_scores(tf_pairs, query_terms, {'full': 0.0, 'brief': 1.4}),
        )
        rrf = {i: 0.0 for i in range(n)}
        for scores in score_lists:
            for rank, i in enumerate(sorted(range(n), key=lambda j: scores[j], reverse=True)):
                if scores[i] <= 0:
                    continue
                rrf[i] += 1.0 / (60 + rank + 1)
        ranked = sorted(range(n), key=lambda i: rrf[i], reverse=True)
        def _score_of(i):
            return rrf[i]
    else:
        # BM25F single pass: full 1.0 + brief 1.4
        if bm25l:
            scores = _bm25l_scores(tf_pairs, query_terms, {'full': 1.0, 'brief': 1.4})
        else:
            scores = _bm25f_scores(tf_pairs, query_terms, {'full': 1.0, 'brief': 1.4})
        ranked = sorted(range(n), key=lambda i: scores[i] + _PRIOR_SCALE * corpus[i][2], reverse=True)
        def _score_of(i):
            return scores[i] + _PRIOR_SCALE * corpus[i][2]

    count = 0
    first = True
    for i in ranked:
        sc = _score_of(i)
        if sc <= 0:
            continue
        filepath, o, prior, ts = corpus[i]
        short = _rel_path(filepath)
        ts_suffix = f" event={ts}" if ts else ""
        start = o.get("start_line", 0) + 1
        if not first:
            print()
        first = False
        print(f"({short}:{start}-{start}) [{o['type']}] score={sc:.2f}{ts_suffix}")
        src = o.get("content_brief") if brief else o.get("content")
        for line in (src or [])[:8]:
            print(f"   {line}")
        count += 1
        if limit and count >= limit:
            return


# ── compile ──

def compile_pass(input_path, output_dir=None, truncate=128, truncate_user=256,
            grep_pattern=None, quiet=False, grep_limit=0, grep_brief=False):
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_path)) or "."
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]

    recs = merge_chunks(lex(input_path))
    chains = split_chains(recs)
    if not chains:
        if not quiet:
            print("No conversation chains found.")
        return []

    results, paths = [], []

    for i, chain in enumerate(chains):
        sfx = f"_{i+1}" if len(chains) > 1 else ""
        ffn = f"{base}{sfx}.txt"
        mfn = f"{base}{sfx}.min.txt"
        vfn = f"{base}{sfx}.view.txt"
        fp = os.path.join(output_dir, ffn)
        mp = os.path.join(output_dir, mfn)
        vp = os.path.join(output_dir, vfn)
        data_ctr = [0]

        ir = parse(chain, output_dir, f"{base}{sfx}", data_ctr)
        assign_lines(ir)
        lower_brief(ir, truncate, ffn, truncate_user)

        full = emit(ir, "content")
        brief = emit(ir, "content_brief")

        stats_footer = _collect_stats(chain)
        if stats_footer:
            full.extend([""] + stats_footer)

        with open(fp, "w", encoding="utf-8") as f: f.write("\n".join(full))
        with open(mp, "w", encoding="utf-8") as f: f.write("\n".join(brief))

        if grep_pattern:
            lower_view(ir, ffn, grep_pattern, grep_limit, grep_brief)
            view = emit(ir, "content_view")
            with open(vp, "w", encoding="utf-8") as f: f.write("\n".join(view))

        ft, bt = "\n".join(full), "\n".join(brief)
        _cnt = lambda s: sum(1 for t in _tokenize(s) if t.strip())
        results.append((fp, ir))
        paths.append((fp, mp, vp if grep_pattern else None,
                       len(full), _cnt(ft), len(brief), _cnt(bt)))

    if not quiet:
        for fp, _, _, fl, fw, _, _ in paths:
            print(f"  {fp}  ({fl} lines, {fw} words)")
        for _, mp, _, _, _, bl, bw in paths:
            print(f"  {mp}  ({bl} lines, {bw} words)")
        if grep_pattern:
            for _, _, vp, _, _, _, _ in paths:
                if vp:
                    print(f"  {vp}")

    return results

# ── main ──

def _expand_inputs(raw):
    files = []
    for r in raw:
        expanded = globmod.glob(r, recursive=True)
        expanded.sort(key=lambda f: os.path.getmtime(f))
        files.extend(expanded if expanded else [r])
    return files

def main():
    p = argparse.ArgumentParser(description="VCC - View-oriented Conversation Compiler")
    p.add_argument("input", nargs="+")
    p.add_argument("-o", "--output-dir")
    p.add_argument("-t", "--truncate", nargs="?", type=int, const=128, default=128, metavar="N")
    p.add_argument("-tu", "--truncate-user", nargs="?", type=int, const=256, default=256, metavar="N")
    p.add_argument("--grep", metavar="PATTERN")
    p.add_argument("--limit", type=int, metavar="N", default=0,
                   help="Max block matches to report per file (0 = unlimited)")
    p.add_argument("--brief", action="store_true",
                   help="Search brief (min) view content instead of full content")
    p.add_argument("--order", choices=["newest", "oldest"], default="newest",
                   help="grep output order: newest (default) or oldest (chronological)")
    p.add_argument("--search", metavar="QUERY",
                   help="BM25 text search (instead of regex grep)")
    p.add_argument("--fusion", action="store_true",
                   help="RRF-fuse full and brief BM25F ranked lists (k=60)")
    p.add_argument("--bm25l", action="store_true",
                   help="use BM25L scoring instead of BM25F (no fusion)")
    a = p.parse_args()
    try:
        a.grep = re.compile(a.grep) if a.grep else None
    except re.error as e:
        p.error(f"invalid regex for --grep: {e}")
    if a.grep and a.search:
        p.error("use --grep OR --search, not both")
    all_results = []
    for f in _expand_inputs(a.input):
        res = compile_pass(f, a.output_dir, a.truncate, a.truncate_user,
                      a.grep, quiet=bool(a.grep or a.search),
                      grep_limit=a.limit, grep_brief=a.brief)
        all_results.extend(res)
    if a.grep:
        grep_search(all_results, a.grep, a.limit, a.brief, a.order)
    if a.search:
        bm25_search(all_results, a.search, a.limit, a.brief, a.fusion, a.bm25l)

if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    main()
