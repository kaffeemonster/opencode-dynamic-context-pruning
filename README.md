# Dynamic Context Pruning Plugin

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/dansmolsky)
[![npm version](https://img.shields.io/npm/v/@tarquinen/opencode-dcp.svg)](https://www.npmjs.com/package/@tarquinen/opencode-dcp)

> [!IMPORTANT]
> **This is a community fork, not the original DCP.** Upstream development of [opencode-dcp](https://github.com/Opencode-DCP/opencode-dynamic-context-pruning) has slowed — new context-management work moved to the [Sleev](https://sleev.ai) cloud service. This fork keeps DCP alive as a local, self-hosted plugin and adds features the original does not have.
>
> **Main new feature: [VCC](https://github.com/lllyasviel/VCC) integration.** Every compression is archived as a lossless, searchable transcript. Nothing that gets compacted is lost — the model can grep old context (including compressed blocks and reasoning) on demand via a `view` tool, and *never* has to unpack a block to recover it.
>
> **Other additions:** pre-compression sweep (`compress.preSweep`), caveman/wenyan compression styles (`compress.summaryStyle`), configurable reasoning pruning with hysteresis (`strategies.purgeReasoning`), block metainfo footer with token counts, and logrotate-style VCC view rotation (`view.rotateKeep`).

Automatically reduces token usage in OpenCode by managing conversation context.

![DCP in action](assets/images/dcp-demo9.png)

## Installation

Install from the CLI:

```bash
opencode plugin @tarquinen/opencode-dcp@latest --global
```

This installs the package and adds it to your global OpenCode config.

## Project Status

Development on DCP has slowed because most new context-management work has moved to [Sleev](https://sleev.ai) and the `sleev` CLI. Sleev is a local proxy for Claude Code, Codex, and OpenCode that builds on DCP's core ideas with newer context-management features and will work with any harness/client.

DCP remains available for OpenCode plugin users, but new features are landing in Sleev first. If you are starting fresh, we recommend trying Sleev:

```bash
npm i -g sleev
sleev
```

## How It Works

DCP reduces context size through a compress tool and automatic cleanup. Your session history is never modified — DCP replaces pruned content with placeholders before sending requests to your LLM.

### Compress

Compress is a tool exposed to your model that replaces closed, stale conversation content with high-fidelity technical summaries. You can think of this as a much smarter version of Opencode's compaction process. Instead of triggering statically when your session reaches its maximum context and on the entire coding session, Compress allows the model to pick when to activate based on task completion, and to only compress the specific messages that are no longer needed verbatim.

DCP supports two compression modes:

- `range` mode compresses contiguous spans of conversation into one or more summaries.
- `message` mode (experimental) compresses individual raw messages independently, letting the model manage context much more surgically.

In `range` mode, when a new compression overlaps an earlier one, the earlier summary is nested inside the new one so information is preserved through layers of compression rather than diluted away. In both modes, protected tool outputs (such as subagents and skills) and protected file patterns are kept in compression summaries, ensuring that the most important information is never lost. You can also enable `protectUserMessages` to preserve your messages verbatim during compression, though note that large prompts (e.g. copy-pasting log files in the prompt) will then never be compressed away.

### Deduplication

Identifies repeated tool calls (same tool, same arguments) and keeps only the most recent output. Recalculated when the compress tool runs, so prompt cache is only impacted alongside compression.

### Purge Errors

Prunes inputs from errored tool calls after a configurable number of turns (default: 4). Error messages are preserved; only the potentially large input content is removed. Recalculated on compress tool use.

## Configuration

DCP uses its own config file, searched in order:

1. Global: `~/.config/opencode/dcp.jsonc` (or `dcp.json`), created automatically on first run
2. Custom config directory: `$OPENCODE_CONFIG_DIR/dcp.jsonc` (or `dcp.json`), if `OPENCODE_CONFIG_DIR` is set
3. Project: `.opencode/dcp.jsonc` (or `dcp.json`) in your project's `.opencode` directory

Each level overrides the previous, so project settings take priority over global. Restart OpenCode after making config changes.

> [!NOTE]
> If you use models with smaller context windows, such as GitHub Copilot models or local models, lower `compress.minContextLimit` and `compress.maxContextLimit` in your configuration to match the available context.

> [!IMPORTANT]
> Defaults are applied automatically. Expand this if you want to review or override settings.

<details>
<summary><strong>Default Configuration</strong> (click to expand)</summary>

```jsonc
{
    "$schema": "https://raw.githubusercontent.com/Opencode-DCP/opencode-dynamic-context-pruning/master/dcp.schema.json",
    // Enable or disable the plugin
    "enabled": true,
    // Automatically update npm-installed DCP when a newer npm latest is available.
    // Version-locked plugin specs are not updated.
    "autoUpdate": true,
    // Enable debug logging to ~/.config/opencode/logs/dcp/
    "debug": false,
    // Notification display: "off", "minimal", or "detailed"
    "pruneNotification": "detailed",
    // Notification type: "chat" (in-conversation) or "toast" (system toast)
    "pruneNotificationType": "chat",
    // Slash commands configuration
    "commands": {
        "enabled": true,
        // Additional tools to protect from pruning via commands (e.g., /dcp sweep)
        "protectedTools": [],
    },
    // Manual mode: disables autonomous context management,
    // tools only run when explicitly triggered via /dcp commands
    "manualMode": {
        "enabled": false,
        // When true, automatic cleanup (deduplication, purgeErrors)
        // still runs even in manual mode
        "automaticStrategies": true,
    },
    // Protect from pruning for <turns> message turns past tool invocation
    "turnProtection": {
        "enabled": false,
        "turns": 4,
    },
    // Experimental settings
    "experimental": {
        // Allow DCP processing in subagent sessions
        "allowSubAgents": false,
        // Enable user-editable prompt overrides under dcp-prompts directories
        // When false (default), prompt override files/directories are ignored
        "customPrompts": false,
    },
    // Protect file operations from pruning via glob patterns
    // Patterns match tool parameters.filePath (e.g. read/write/edit)
    "protectedFilePatterns": [],
    // Unified context compression tool and behavior settings
    "compress": {
        // Compression mode: "range" (compress spans into block summaries)
        // or experimental "message" (compress individual raw messages)
        "mode": "range",
        // Permission mode: "allow" (no prompt), "ask" (prompt), "deny" (tool not registered)
        "permission": "allow",
        // Show compression content in a chat notification
        "showCompression": false,
        // Let active summary tokens extend the effective maxContextLimit
        "summaryBuffer": true,
        // Soft upper threshold: above this, DCP keeps injecting strong
        // compression nudges (based on nudgeFrequency), so compression is
        // much more likely. Accepts: number or "X%" of model context window.
        "maxContextLimit": 100000,
        // Soft lower threshold for reminder nudges: below this, turn/iteration
        // reminders are off (compression less likely). At/above this, reminders
        // are on. Accepts: number or "X%" of model context window.
        "minContextLimit": 50000,
        // Optional per-model override for maxContextLimit by providerID/modelID.
        // If present, this wins over the global maxContextLimit.
        // Accepts: number or "X%".
        // Example:
        // "modelMaxLimits": {
        //     "openai/gpt-5.3-codex": 120000,
        //     "anthropic/claude-sonnet-4.6": "80%"
        // },
        // Optional per-model override for minContextLimit.
        // If present, this wins over the global minContextLimit.
        // "modelMinLimits": {
        //     "openai/gpt-5.3-codex": 50000,
        //     "anthropic/claude-sonnet-4.6": "25%"
        // },
        // How often the context-limit nudge fires (1 = every fetch, 5 = every 5th)
        "nudgeFrequency": 5,
        // Start adding compression reminders after this many
        // messages have happened since the last user message
        "iterationNudgeThreshold": 15,
        // Controls how likely compression is after user messages
        // ("strong" = more likely, "soft" = less likely)
        "nudgeForce": "soft",
        // Tool names whose completed outputs are appended to the compression
        "protectedTools": [],
        // Preserve text wrapped in <protect>...</protect> when compressed
        "protectTags": false,
        // Preserve your messages during compression.
        // Warning: large copy-pasted prompts will never be compressed away
        "protectUserMessages": false,
        // Writing style for compression summaries:
        // "detailed" = prose explanation
        // "terse" = dense caveman-style (substance preserved, fluff stripped)
        // "wenyan" = Classical Chinese 文言文, 80-90% character reduction
        // (technical payload always verbatim regardless of style)
        "summaryStyle": "detailed",
        // Run a sweep/prune pass before model-triggered compression.
        // Removes stale tool outputs so compression summaries focus on
        // meaningful content instead of tool chatter.
        "preSweep": {
            // Prune tool outputs before every automatic compress execution
            "enabled": false,
            // Number of most recent tool calls to prune (0 = all tools
            // since the previous user message)
            "count": 100,
        },
    },
    // Automatic pruning strategies
    "strategies": {
        // Remove duplicate tool calls (same tool with same arguments)
        "deduplication": {
            "enabled": true,
            // Additional tools to protect from pruning
            "protectedTools": [],
        },
        // Prune tool inputs for errored tools after X turns
        "purgeErrors": {
            "enabled": true,
            // Number of turns before errored tool inputs are pruned
            "turns": 4,
            // Additional tools to protect from pruning
            "protectedTools": [],
        },
        // Strip reasoning (thinking) parts from old assistant messages.
        // Reasoning is scratch work - once the final result exists the
        // verbose chains carry no information, but they count against the
        // context budget. Uses hysteresis: reasoning accumulates until it
        // exceeds "highWater" parts, then the oldest messages' reasoning is
        // stripped down to "lowWater" parts. Batching pruning into bursts
        // is cache-friendlier for providers that dislike mid-stream
        // reasoning mutation.
        // OFF by default - providers with caching semantics (e.g. Anthropic)
        // may penalize reasoning pruning. Enable for long-running tasks.
        "purgeReasoning": {
            "enabled": false,
            // Minimum turn age before a message's reasoning is eligible
            "turns": 8,
            // Prune once reasoning parts exceed this count
            "highWater": 60,
            // Prune down to this many reasoning parts
            "lowWater": 20,
            // Messages that called one of these tools keep their reasoning
            "protectedTools": [],
        },
    },
}
```

</details>

### Commands

DCP provides a TUI panel and one prompt-producing slash command:

- `/dcp` — Opens the DCP panel with context, stats, and manual-mode controls.
- `/dcp-compress [focus]` — Asks the model to run one compression pass. Optional focus text directs what content to compress, following the active `compress.mode`.

### Prompt Overrides

DCP exposes six editable prompts:

- `system`
- `compress-range`
- `compress-message`
- `context-limit-nudge`
- `turn-nudge`
- `iteration-nudge`

This feature is disabled by default. Set `experimental.customPrompts` to `true` in your DCP config to activate it.

When enabled, managed defaults are written to `~/.config/opencode/dcp-prompts/defaults/` as plain-text prompt files. A single `README.md` in that directory explains each prompt and how to create overrides.

To customize behavior, add a file with the same name under an overrides directory and edit it as plain text.

To reset an override, delete the matching file from your overrides directory.

### Protected Tools

By default, these tools are always protected from pruning:
`task`, `skill`, `todowrite`, `todoread`, `compress`, `batch`, `plan_enter`, `plan_exit`, `write`, `edit`

The `protectedTools` arrays in `commands` and `strategies` add to this default list.

For the `compress` tool, `compress.protectedTools` ensures specific tool outputs are appended to the compressed summary. By default it includes `task`, `skill`, `todowrite`, and `todoread`.

### Conversation View (VCC)

DCP can archive and search the session transcript using the [Conversation Compiler](https://github.com/lllyasviel/VCC) (VCC). The full transcript — including compressed blocks and reasoning — survives in a lossless `.txt` view, a brief `.min.txt` view, and a grep-able `.view.txt`.

To enable, set `view.enabled: true` and point `view.scriptPath` at your `VCC.py`:

```jsonc
"view": {
    "enabled": true,
    "pythonPath": "python3",
    "scriptPath": "/path/to/VCC.py",
    "exportDir": "",            // default: <data-dir>/plugin/dcp/vcc
    "autoExport": false,        // re-export + compile after each compression
    "tokenTruncation": 128,
    "userTokenLimit": 256,
    "postMode": "notice",       // "off" | "notice" | "fullminview"
    "rotateKeep": 3,
    "maxReturnChars": 49152,    // max chars of VCC grep/search output returned to the model
    "semantic": {               // optional semantic search (see below)
        "enabled": false,
        "provider": "api",      // "api" (OpenAI-compatible /v1/embeddings) | "onnx" (local MiniLM)
        "apiUrl": "http://127.0.0.1:8012/v1/embeddings",
        "apiKey": "",
        "model": "harrier-oss-v1-0.6B-Embed"   // api: model id; onnx: path to model.onnx
    }
}
```

`view.maxReturnChars` caps the characters of VCC grep/search output returned to the model — a technical safeguard against context bloat. The policy is setup-specific (local vs provider-tiered context, long-context models); raise cautiously. It truncates the search result text only; full views remain on disk. Default is `48 * 1024` (48KB).

When enabled, the `view` tool is registered: the model can grep the archived transcript on demand (`view` with a regex `pattern`) and get line-range references into the full transcript — no decompression needed.

The `view` tool supports result navigation and anchoring (both grep and BM25 search):
- `limit` / `offset` / `order: "newest" | "oldest"` — pagination over match lists.
- `fromLine: N` — restrict results to transcript lines `>= N` (re-anchor beyond the result limit).
- `context: N` — widen each hit with `N` surrounding lines.
- `ref: "<message-id>"` — jump straight to the section of a specific message (ids are stamped as `message.id` in the export).
- `query: "<text>"` — natural-language BM25 search instead of regex.

### Semantic Search (optional sidecar)

Beyond keyword/BM25 search, DCP ships an optional semantic search sidecar (`scripts/vcc-semantic.py`). It embeds each record's text and ranks by cosine similarity, so natural-language queries ("why did the auth flow break?") find relevant messages even when no keyword matches.

- `provider: "api"` — POSTs to any OpenAI-compatible `/v1/embeddings` endpoint (e.g. a local llama.cpp server) using only the stdlib — zero pip dependencies.
- `provider: "onnx"` — runs `all-MiniLM-L6-v2` locally (384-dim) via `onnxruntime` + `tokenizers` + `numpy`. Model + tokenizer auto-download on first use to `$XDG_CACHE_HOME/vcc/all-MiniLM-L6-v2/` (default `~/.cache/vcc/...`).
- Embeddings are cached next to the export (`<export>.emb.json`), incrementally: re-runs embed only new records.
- **Failure isolation**: if semantic search fails (server down, missing deps, bad model), the tool logs a warning and falls back to BM25 with a one-line note. Grep, ref and anchor paths are never affected.

Commands:
- `/dcp view-export` — write the session snapshot to the VCC jsonl format.
- `/dcp view-compile [pattern]` — export, compile with VCC, and (if `postMode` allows) post the result.

### Compressed Block Meta Footer

Every compressed block carries a metadata footer that the model sees in context:

```
topic: Auth System | contains: (b1) (b2) | ~45.6K tokens | recoverable via view tool /dcp-compress
```

- `topic` — the compression's short label.
- `contains: (bN)` — directly-embedded sub-blocks (only present when a super-block was built from older blocks).
- `~N tokens` — the size of the compressed content (what an unpack would restore).
- `recoverable via view tool /dcp-compress` — explicit note that unpacking is never required; the content is reachable through the VCC view.

Block numbers are monotonically increasing and never reused or renumbered, so the `(bN)` markers are stable for the life of a session.

## Impact on Prompt Caching

LLM providers cache prompts based on exact prefix matching. When DCP prunes content, it changes messages, which invalidates cached prefixes from that point forward.

**Trade-off:** You lose some cache reads but gain token savings from reduced context size and fewer hallucinations from stale context. In most cases, especially in long sessions, the savings outweigh the cache miss cost.

> [!NOTE]
> In testing, cache hit rates were approximately 85% with DCP vs 90% without.

**No impact for:**

- **Request-based billing** — Some providers charge per request, not tokens.
- **Uniform token pricing** — Providers like Cerebras that bill cached and uncached tokens at the same rate.

## License

AGPL-3.0-or-later
