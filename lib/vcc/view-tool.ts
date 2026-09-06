import { tool } from "@opencode-ai/plugin"
import { execFile } from "child_process"
import * as fs from "fs/promises"
import { dirname, join } from "path"
import { STORAGE_DIR } from "../state/persistence"
import { rotateViewFiles } from "./rotate"
import { partsToVccContent } from "./parts"
import type { ToolContext } from "../compress/types"
import { filterMessages } from "../messages/shape"

function truncateOutput(s: string, extra: string, maxChars: number): string {
    if (s.length <= maxChars) return s
    return s.slice(0, maxChars) + "\n\n…[truncated] " + extra
}

// ── hybrid fusion helpers ──

interface FusionEntry {
    txt: string
    start: number
    end: number
    score: number
    source: "bm25" | "sem"
    preview: string[]
}

const FUSION_LINE_RE = /\((.*?):(\d+)-(\d+)\) \[[^\]]+\] score=([\d.]+)/

function parseFusionOutput(out: string, source: "bm25" | "sem"): FusionEntry[] {
    const entries: FusionEntry[] = []
    let current: FusionEntry | null = null
    for (const line of out.split("\n")) {
        const m = FUSION_LINE_RE.exec(line)
        if (m) {
            current = {
                txt: m[1].split(/[\\/]/).pop() || m[1],
                start: +m[2],
                end: +m[3],
                score: +m[4],
                source,
                preview: [],
            }
            entries.push(current)
        } else if (current && /^\s+\S/.test(line)) {
            current.preview.push(line.trimStart())
        }
    }
    return entries
}

function fuseEntries(
    lists: [FusionEntry[], FusionEntry[]],
): Array<{
    entry: FusionEntry
    rrf: number
    bm25: number | null
    sem: number | null
    semPreview: string[]
}> {
    const K = 60
    const merged = new Map<string, {
        entry: FusionEntry
        rrf: number
        bm25: number | null
        sem: number | null
        semPreview: string[]
    }>()
    const keyFor = (e: FusionEntry) => `${e.txt}:${e.start}-${e.end}`
    lists.forEach((list, li) => {
        const src = li === 0 ? "bm25" : "sem"
        list.forEach((e, r) => {
            const key = keyFor(e)
            let m = merged.get(key)
            if (!m) {
                m = { entry: e, rrf: 0, bm25: null, sem: null, semPreview: [] }
                merged.set(key, m)
            }
            if (src === "bm25") {
                m.bm25 = e.score
            } else {
                m.sem = e.score
                m.semPreview = e.preview
            }
            m.rrf += 1 / (K + r + 1)
        })
    })
    return [...merged.values()].sort((a, b) => b.rrf - a.rrf)
}

export function createViewTool(ctx: ToolContext): ReturnType<typeof tool> {
    const viewConfig = ctx.config.view
    const maxReturnChars = viewConfig.maxReturnChars ?? 48 * 1024

    return tool({
        description:
            "Search the VCC conversation view. Compiles the session into a grep view and returns all blocks/lines matching the regex pattern, with line-range references into the full transcript. Set sessions=true to list all searchable sessions, or session='<id>' to search a specific one. Use brief=true to search the min view first, limit=N to cap results, offset=N to skip this many matches before reporting, order='newest'|'oldest' for grep output order, fromLine=N to anchor at .txt line N, context=N to widen hits. Pass query instead of pattern for natural-language BM25 text search (ranked by relevance). ref='<message-id>' to locate a message by id.",
        args: {
            pattern: tool.schema
                .string()
                .describe("Regex pattern to search in the conversation view (e.g. 'ISA.*flag')"),
            sessions: tool.schema
                .boolean()
                .optional()
                .describe("List searchable sessions (id, title, last updated) instead of searching"),
            session: tool.schema
                .string()
                .optional()
                .describe("Session id to search (defaults to current session)"),
            limit: tool.schema
                .number()
                .optional()
                .describe("Max matches to return (default 40)"),
            brief: tool.schema
                .boolean()
                .optional()
                .describe("Search the brief/min view instead of full content"),
            order: tool.schema
                .string()
                .optional()
                .describe("grep output order: newest (default) or oldest (chronological)"),
            offset: tool.schema
                .number()
                .optional()
                .describe("Skip this many matches before reporting (0 = none)"),
            fromLine: tool.schema
                .number()
                .optional()
                .describe("Only report blocks whose .txt start line is >= N"),
            context: tool.schema
                .number()
                .optional()
                .describe("Include N context lines around each match (grep) or show up to N lines (search)"),
            query: tool.schema
                .string()
                .optional()
                .describe("Natural-language text search (BM25 ranking) — alternative to regex pattern"),
            ref: tool.schema
                .string()
                .optional()
                .describe("Locate a message by its id (message.id in export)"),
            fusion: tool.schema
                .boolean()
                .optional()
                .describe("hybrid semantic+BM25 RRF fusion, requires query; runs both rankers in parallel at per-message span and merges by rank (k=60). limit applies post-merge; offset is not passed to rankers."),
        },
        async execute(args, toolCtx) {
            const { pattern, sessions, session, limit, brief, query, ref, order, offset, fromLine, context, fusion } = args as {
                pattern: string
                sessions?: boolean
                session?: string
                limit?: number
                brief?: boolean
                query?: string
                ref?: string
                order?: string
                offset?: number
                fromLine?: number
                context?: number
                fusion?: boolean
            }

            if (!viewConfig.enabled) {
                throw new Error(
                    "View feature disabled. Set view.enabled=true in dcp.jsonc",
                )
            }

            if (!viewConfig.scriptPath) {
                throw new Error(
                    "VCC script path not configured. Set view.scriptPath in dcp.jsonc",
                )
            }

            if (fusion && !query) {
                return (
                    "fusion requires query (a natural-language search). " +
                    "Pass query together with fusion=true; ref/pattern are not fusible."
                )
            }

            const exportDir = viewConfig.exportDir || join(STORAGE_DIR, "vcc")

            if (sessions === true) {
                return listSearchableSessions(ctx, exportDir)
            }

            const currentSessionId = toolCtx.sessionID || ctx.state.sessionId
            const targetSessionId = session || currentSessionId
            if (!targetSessionId) {
                throw new Error("No active session")
            }

            // Fetch session messages
            const messagesResponse = await ctx.client.session.messages({
                path: { id: targetSessionId },
            })
            const messages = filterMessages(messagesResponse.data || messagesResponse)

            // Export session snapshot to VCC format
            const exportPath = join(exportDir, `${targetSessionId}_export.jsonl`)

            const records: Record<string, any>[] = [
                {
                    type: "system",
                    timestamp: new Date().toISOString(),
                    message: {
                        content: [
                            { type: "text", text: `View export for session ${targetSessionId}` },
                        ],
                    },
                },
            ]

            if (targetSessionId === currentSessionId) {
                for (const block of ctx.state.prune.messages.blocksById.values()) {
                    records.push({
                        type: "system",
                        timestamp: new Date(block.createdAt).toISOString(),
                        message: {
                            content: [
                                {
                                    type: "text",
                                    text: `[compressed block ${block.blockId}] topic: ${block.topic}\nsummary: ${block.summary}`,
                                },
                            ],
                        },
                    })
                }
            }

            for (const msg of messages) {
                const content = partsToVccContent(
                    msg.parts as unknown as Array<{ type: string } & Record<string, any>>,
                )
                records.push({
                    type:
                        msg.info.role === "user"
                            ? "user"
                            : msg.info.role === "assistant"
                              ? "assistant"
                              : "system",
                    timestamp:
                        typeof msg.info.time?.created === "number"
                            ? new Date(msg.info.time.created).toISOString()
                            : new Date().toISOString(),
                    message: { id: msg.info.id, content: content.length ? content : [] },
                })
            }

            await fs.mkdir(exportDir, { recursive: true })
            await fs.writeFile(
                exportPath,
                records.map((r) => JSON.stringify(r)).join("\n") + "\n",
                "utf-8",
            )

            // Rotate previous view files before VCC overwrites them
            await rotateViewFiles(exportPath, viewConfig.rotateKeep ?? 3)

            // Semantic search first (optional sidecar), fall back to BM25
            let semanticNote: string | null = null
            let semanticOutput: string | null = null
            if (query && viewConfig.semantic?.enabled) {
                const semanticScript =
                    viewConfig.semantic.scriptPath ||
                    join(dirname(viewConfig.scriptPath), "vcc-semantic.py")
                const semanticArgs = [
                    semanticScript,
                    exportPath,
                    "--query",
                    query,
                    "--limit",
                    String(limit ?? 5),
                    "--provider",
                    viewConfig.semantic.provider ?? "api",
                    ...(viewConfig.semantic.apiUrl
                        ? ["--api-url", viewConfig.semantic.apiUrl]
                        : []),
                    ...(viewConfig.semantic.apiKey
                        ? ["--api-key", viewConfig.semantic.apiKey]
                        : []),
                    ...(viewConfig.semantic.model
                        ? ["--model", viewConfig.semantic.model]
                        : []),
                    ...(fusion ? ["--blocks"] : []),
                ]
                try {
                    semanticOutput = await new Promise<string>((resolve, reject) => {
                        execFile(
                            viewConfig.pythonPath || "python3",
                            semanticArgs,
                            { maxBuffer: 20 * 1024 * 1024 },
                            (error: any, stdout: string, stderr: string) => {
                                if (error) {
                                    reject(
                                        new Error(
                                            stderr?.split("\n")[0] || error.message,
                                        ),
                                    )
                                } else {
                                    resolve(stdout || "")
                                }
                            },
                        )
                    })
                    if (!fusion) {
                        return truncateOutput(
                            `**VCC semantic matches for \`${query}\`:**\n\n` +
                                semanticOutput +
                                `\n\nExport: ${exportPath}`,
                            "more matches — narrow the query or read the export",
                            maxReturnChars,
                        )
                    }
                } catch (err: any) {
                    console.warn(
                        `[vcc-semantic] unavailable, falling back to BM25: ${err?.message}`,
                    )
                    semanticNote = `(semantic unavailable — ${err?.message}; showing BM25)`
                }
            } else if (query && fusion) {
                semanticNote = "(hybrid unavailable — semantic search disabled; showing BM25)"
            }

            // Run VCC grep
            const vccArgs = [
                viewConfig.scriptPath,
                exportPath,
                ...(ref ? ["--ref", ref] : query ? ["--search", query] : ["--grep", pattern]),
                "--limit",
                String(limit ?? 40),
                ...(fusion ? ["--sec-level"] : ["--offset", String(offset ?? 0)]),
                ...(query || ref ? [] : ["--order", String(order ?? "newest")]),
                ...(brief === true ? ["--brief"] : []),
                "--from-line",
                String(fromLine ?? 0),
                "--context",
                String(context ?? 0),
            ]
            const output = await new Promise<string>((resolve, reject) => {
                execFile(viewConfig.pythonPath || "python3", vccArgs, { maxBuffer: 20 * 1024 * 1024 }, (error: any, stdout: string, stderr: string) => {
                    if (error) {
                        reject(
                            new Error(
                                `VCC grep failed: ${error.message}${stderr ? "\n" + stderr : ""}`,
                            ),
                        )
                    } else {
                        resolve(stdout || "")
                    }
                })
            }).catch((err: any) => {
                if (!fusion) throw err
                console.warn(`[vcc] hybrid child failed, falling back to semantic: ${err?.message}`)
                return ""
            })

            if (!query && !ref) {
                // Read the .view.txt if it exists (grep mode only — BM25 search
                // never writes a view file)
                const viewPath = exportPath.replace(/\.jsonl$/, ".view.txt")
                let viewContent = ""
                try {
                    viewContent = await fs.readFile(viewPath, "utf-8")
                } catch {
                    // .view.txt only written when matches exist
                }

                if (viewContent.trim()) {
                    const resultText =
                        `**VCC grep matches for \`${pattern}\`:**\n\n` +
                        viewContent +
                        `\n\nFull transcript: ${exportPath.replace(/\.jsonl$/, ".txt")}`
                    return truncateOutput(
                        resultText,
                        "more matches — refine pattern (add .* or narrow terms) or read Full transcript:" +
                            exportPath.replace(/\.jsonl$/, ".txt"),
                        maxReturnChars,
                    )
                }

                return (
                    `VCC grep for \`${pattern}\` found no matches in the current session view.\n` +
                    `Full transcript: ${exportPath.replace(/\.jsonl$/, ".txt")}\n` +
                    `Brief view: ${exportPath.replace(/\.jsonl$/, ".min.txt")}\n` +
                    `Compiler output:\n${truncateOutput(output, "compiler output truncated", maxReturnChars)}`
                )
            }

            // Hybrid fusion: merge BM25 (sec-level) + semantic (blocks) by RRF
            if (query && fusion) {
                const bm25Entries = output.trim() ? parseFusionOutput(output, "bm25") : []
                const semEntries = semanticOutput ? parseFusionOutput(semanticOutput, "sem") : []
                if (semanticNote && bm25Entries.length === 0 && semEntries.length === 0) {
                    return (
                        `VCC hybrid search for \`${query}\` found no matches in the current session view.\n` +
                        `${semanticNote}\n` +
                        `Full transcript: ${exportPath.replace(/\.jsonl$/, ".txt")}\n` +
                        `Brief view: ${exportPath.replace(/\.jsonl$/, ".min.txt")}`
                    )
                }
                const fused = fuseEntries([bm25Entries, semEntries])
                const fusedLimit = limit ?? 40
                const lines: string[] = [`**VCC hybrid search for \`${query}\`:**`]
                if (semanticNote) lines.push("", semanticNote)
                if (fused.length === 0) {
                    lines.push("", "no matching blocks.")
                }
                for (const hit of fused.slice(0, fusedLimit)) {
                    lines.push("")
                    const b25 = hit.bm25 != null ? hit.bm25.toFixed(2) : "-"
                    const sem = hit.sem != null ? hit.sem.toFixed(2) : "-"
                    lines.push(
                        `(${hit.entry.txt}:${hit.entry.start}-${hit.entry.end}) [hybrid] rrf=${hit.rrf.toFixed(3)} bm25=${b25} sem=${sem}`,
                    )
                    if (hit.sem != null) {
                        for (const pl of (hit.semPreview || []).slice(0, 2)) {
                            lines.push("  " + pl)
                        }
                    }
                }
                lines.push("", `Full transcript: ${exportPath.replace(/\.jsonl$/, ".txt")}`)
                return truncateOutput(
                    lines.join("\n"),
                    "more matches — narrow the query or read the export",
                    maxReturnChars,
                )
            }

            // BM25 search mode: stdout IS the ranked result
            if (output.trim()) {
                return (
                    `**VCC ${ref ? "ref" : "search"} matches for \`${ref || query}\`:**\n\n` +
                    (semanticNote ? semanticNote + "\n\n" : "") +
                    output +
                    `\n\nFull transcript: ${exportPath.replace(/\.jsonl$/, ".txt")}`
                )
            }

            return (
                `VCC ${ref ? "ref" : "search"} for \`${ref || query}\` found no matches in the current session view.\n` +
                (semanticNote ? semanticNote + "\n" : "") +
                `Full transcript: ${exportPath.replace(/\.jsonl$/, ".txt")}\n` +
                `Brief view: ${exportPath.replace(/\.jsonl$/, ".min.txt")}`
            )
        },
    })
}

async function listSearchableSessions(
    ctx: ToolContext,
    exportDir: string,
): Promise<string> {
    let files: string[] = []
    try {
        files = await fs.readdir(exportDir)
    } catch {
        // exportDir may not exist yet
    }

    const ids = files
        .filter((f) => /_export\.jsonl$/.test(f))
        .map((f) => f.replace(/_export\.jsonl$/, ""))

    if (ids.length === 0) {
        return (
            "No VCC exports found. Run /dcp view-compile or trigger a compress to create one.\n" +
            "To search a specific session: view(pattern, {session: '<id>'})"
        )
    }

    const titles = new Map<string, { title: string; created: number; updated: number }>()
    try {
        const listRes = await ctx.client.session.list()
        const sessions = listRes.data || listRes
        if (Array.isArray(sessions)) {
            for (const s of sessions) {
                titles.set(s.id, {
                    title: s.title || "(untitled)",
                    created: s.time?.created ?? 0,
                    updated: s.time?.updated ?? 0,
                })
            }
        }
    } catch {
        // fall back to id-only entries
    }

    const fallbackTitles = new Map<string, string>()
    for (const id of ids) {
        const meta = titles.get(id)
        if (meta && meta.title !== "(untitled)") continue
        const t = await firstUserMessage(exportDir, id)
        if (t) fallbackTitles.set(id, t)
    }

    ids.sort((a, b) => (titles.get(b)?.updated ?? 0) - (titles.get(a)?.updated ?? 0))

    const lines = ids.map((id) => {
        const meta = titles.get(id)
        const fb = fallbackTitles.get(id)
        const shownTitle = meta && meta.title !== "(untitled)"
            ? meta.title
            : fb
            ? `"${fb}"`
            : null
        const suffix = shownTitle
            ? ` — ${shownTitle}${meta ? ` — started ${formatTime(meta.created)} — updated ${formatTime(meta.updated)}` : ""}`
            : ""
        return `- \`${id}\`${suffix}`
    })

    return (
        "**Searchable VCC sessions:**\n\n" +
        lines.join("\n") +
        "\n\nTo search a specific session: view(pattern, {session: '<id>'})"
    )
}

async function firstUserMessage(exportDir: string, sessionId: string): Promise<string> {
    try {
        const raw = await fs.readFile(join(exportDir, `${sessionId}_export.jsonl`), "utf-8")
        for (const line of raw.split("\n")) {
            if (!line.trim()) continue
            const rec = JSON.parse(line)
            if (rec?.type === "user" && rec?.message?.content) {
                const parts = Array.isArray(rec.message.content)
                    ? rec.message.content
                    : [rec.message.content]
                const text = parts
                    .filter((p: any) => typeof p?.text === "string")
                    .map((p: any) => p.text)
                    .join(" ")
                    .trim()
                if (text) {
                    const oneLine = text.replace(/\s+/g, " ").slice(0, 60)
                    return oneLine + (text.length > 60 ? "…" : "")
                }
            }
        }
        return ""
    } catch {
        return ""
    }
}

function formatTime(ts: number): string {
    const d = new Date(ts)
    if (isNaN(d.getTime())) return "unknown"
    return d.toISOString()
}
