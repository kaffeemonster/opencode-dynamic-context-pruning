import { tool } from "@opencode-ai/plugin"
import { execFile } from "child_process"
import * as fs from "fs/promises"
import { join } from "path"
import { STORAGE_DIR } from "../state/persistence"
import { rotateViewFiles } from "./rotate"
import { partsToVccContent } from "./parts"
import type { ToolContext } from "../compress/types"
import { filterMessages } from "../messages/shape"

const MAX_RETURN = 32 * 1024
function truncateOutput(s: string, extra: string): string {
    if (s.length <= MAX_RETURN) return s
    return s.slice(0, MAX_RETURN) + "\n\n…[truncated] " + extra
}

export function createViewTool(ctx: ToolContext): ReturnType<typeof tool> {
    const viewConfig = ctx.config.view

    return tool({
        description:
            "Search the VCC conversation view. Compiles the session into a grep view and returns all blocks/lines matching the regex pattern, with line-range references into the full transcript. Set sessions=true to list all searchable sessions, or session='<id>' to search a specific one. Use brief=true to search the min view first, limit=N to cap results.",
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
        },
        async execute(args, toolCtx) {
            const { pattern, sessions, session, limit, brief } = args as {
                pattern: string
                sessions?: boolean
                session?: string
                limit?: number
                brief?: boolean
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
                    timestamp: new Date().toISOString(),
                    message: { content: content.length ? content : [] },
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

            // Run VCC grep
            const vccArgs = [
                viewConfig.scriptPath,
                exportPath,
                "--grep",
                pattern,
                "--limit",
                String(limit ?? 40),
                ...(brief === true ? ["--brief"] : []),
            ]
            const output = await new Promise<string>((resolve, reject) => {
                execFile(viewConfig.pythonPath || "python", vccArgs, { maxBuffer: 20 * 1024 * 1024 }, (error: any, stdout: string, stderr: string) => {
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
            })

            // Read the .view.txt if it exists
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
                )
            }

            return (
                `VCC grep for \`${pattern}\` found no matches in the current session view.\n` +
                `Full transcript: ${exportPath.replace(/\.jsonl$/, ".txt")}\n` +
                `Brief view: ${exportPath.replace(/\.jsonl$/, ".min.txt")}\n` +
                `Compiler output:\n${truncateOutput(output, "compiler output truncated")}`
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

    const titles = new Map<string, { title: string; updated: number }>()
    try {
        const listRes = await ctx.client.session.list()
        const sessions = listRes.data || listRes
        if (Array.isArray(sessions)) {
            for (const s of sessions) {
                titles.set(s.id, {
                    title: s.title || "(untitled)",
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

    const lines = ids.map((id) => {
        const meta = titles.get(id)
        const fb = fallbackTitles.get(id)
        const shownTitle = meta && meta.title !== "(untitled)"
            ? meta.title
            : fb
            ? `"${fb}"`
            : null
        const suffix = shownTitle
            ? ` — ${shownTitle}${meta ? ` — updated ${formatTime(meta.updated)}` : ""}`
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
    const pad = (n: number) => String(n).padStart(2, "0")
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}
