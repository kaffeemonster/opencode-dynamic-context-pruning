import { join } from "path"
import { execFile } from "child_process"
import * as fs from "fs/promises"
import { existsSync, mkdirSync } from "fs"
import type { SessionState, WithParts } from "./state"
import type { Logger } from "./logger"
import type { PluginConfig } from "./config"
import { STORAGE_DIR } from "./state/persistence"
import { rotateViewFiles } from "./vcc/rotate"
import { partsToVccContent } from "./vcc/parts"
import { assignMessageRefs } from "./message-ids"
import {
    buildPriorityMap,
    buildToolIdList,
    injectCompressNudges,
    injectExtendedSubAgentResults,
    injectMessageIds,
    prune,
    stripHallucinations,
    stripHallucinationsFromString,
    stripStaleMetadata,
    syncCompressionBlocks,
} from "./messages"
import { renderSystemPrompt, type PromptStore } from "./prompts"
import { buildProtectedToolsExtension } from "./prompts/extensions/system"
import {
    applyPendingCompressionDurations,
    buildCompressionTimingKey,
    consumeCompressionStart,
    resolveCompressionDuration,
} from "./compress/timing"
import { filterMessages, filterMessagesInPlace } from "./messages/shape"
import {
    applyPendingManualTrigger,
    handleContextCommand,
    handleDecompressCommand,
    handleHelpCommand,
    handleManualToggleCommand,
    handleManualTriggerCommand,
    handleRecompressCommand,
    handleStatsCommand,
    handleSweepCommand,
} from "./commands"
import { type HostPermissionSnapshot } from "./host-permissions"
import { compressPermission, syncCompressPermissionState } from "./compress-permission"
import {
    checkSession,
    deleteSessionState,
    ensureSessionInitialized,
    saveSessionState,
    syncToolCache,
} from "./state"
import { cacheSystemPromptTokens } from "./ui/utils"

const INTERNAL_AGENT_SIGNATURES = [
    "You are a title generator",
    "You are a helpful AI assistant tasked with summarizing conversations",
    "You are an anchored context summarization assistant for coding sessions",
    "Summarize what was done in this conversation",
]

export function createSystemPromptHandler(
    state: SessionState,
    logger: Logger,
    config: PluginConfig,
    prompts: PromptStore,
) {
    return async (
        input: { sessionID?: string; model: { limit: { context: number } } },
        output: { system: string[] },
    ) => {
        if (input.model?.limit?.context) {
            state.modelContextLimit = input.model.limit.context
            logger.debug("Cached model context limit", { limit: state.modelContextLimit })
        }

        if (state.isSubAgent && !config.experimental.allowSubAgents) {
            return
        }

        const systemText = output.system.join("\n")
        if (INTERNAL_AGENT_SIGNATURES.some((sig) => systemText.includes(sig))) {
            logger.info("Skipping DCP system prompt injection for internal agent")
            return
        }

        const effectivePermission =
            input.sessionID && state.sessionId === input.sessionID
                ? compressPermission(state, config)
                : config.compress.permission

        if (effectivePermission === "deny") {
            return
        }

        prompts.reload()
        const runtimePrompts = prompts.getRuntimePrompts()
        const newPrompt = renderSystemPrompt(
            runtimePrompts,
            buildProtectedToolsExtension(config.compress.protectedTools),
            !!state.manualMode,
            state.isSubAgent && config.experimental.allowSubAgents,
        )
        if (output.system.length > 0) {
            output.system[output.system.length - 1] += "\n\n" + newPrompt
        } else {
            output.system.push(newPrompt)
        }
    }
}

export function createChatMessageTransformHandler(
    client: any,
    state: SessionState,
    logger: Logger,
    config: PluginConfig,
    prompts: PromptStore,
    hostPermissions: HostPermissionSnapshot,
) {
    return async (input: {}, output: { messages: WithParts[] }) => {
        const receivedMessages = Array.isArray(output.messages) ? output.messages.length : 0
        const messages = filterMessagesInPlace(output.messages)
        if (messages.length !== receivedMessages) {
            logger.warn("Skipping messages with unexpected shape during chat transform", {
                received: receivedMessages,
                usable: messages.length,
            })
        }

        await checkSession(client, state, logger, output.messages, config.manualMode.enabled)

        syncCompressPermissionState(state, config, hostPermissions, output.messages)

        if (state.isSubAgent && !config.experimental.allowSubAgents) {
            return
        }

        stripHallucinations(output.messages)
        cacheSystemPromptTokens(state, output.messages)
        assignMessageRefs(state, output.messages)
        syncCompressionBlocks(state, logger, output.messages)
        syncToolCache(state, config, logger, output.messages)
        buildToolIdList(state, output.messages)
        prune(state, logger, config, output.messages)
        await injectExtendedSubAgentResults(
            client,
            state,
            logger,
            output.messages,
            config.experimental.allowSubAgents,
        )
        const compressionPriorities = buildPriorityMap(config, state, output.messages)
        prompts.reload()
        injectCompressNudges(
            state,
            config,
            logger,
            output.messages,
            prompts.getRuntimePrompts(),
            compressionPriorities,
        )
        injectMessageIds(state, config, output.messages, compressionPriorities)
        applyPendingManualTrigger(state, output.messages, logger)
        stripStaleMetadata(output.messages)

        if (state.sessionId) {
            await logger.saveContext(state.sessionId, output.messages)
        }
    }
}

export function createCommandExecuteHandler(
    client: any,
    state: SessionState,
    logger: Logger,
    config: PluginConfig,
    workingDirectory: string,
    hostPermissions: HostPermissionSnapshot,
) {
    return async (
        input: { command: string; sessionID: string; arguments: string },
        output: { parts: any[] },
    ) => {
        if (!config.commands.enabled) {
            return
        }

        if (input.command === "dcp" || input.command === "dcp-compress") {
            const messagesResponse = await client.session.messages({
                path: { id: input.sessionID },
            })
            const messages = filterMessages(messagesResponse.data || messagesResponse)

            await ensureSessionInitialized(
                client,
                state,
                input.sessionID,
                logger,
                messages,
                config.manualMode.enabled,
            )

            syncCompressPermissionState(state, config, hostPermissions, messages)

            const effectivePermission = compressPermission(state, config)
            if (effectivePermission === "deny") {
                return
            }

            const args = (input.arguments || "").trim().split(/\s+/).filter(Boolean)
            const isCompressCommand = input.command === "dcp-compress"
            const subcommand = isCompressCommand ? "compress" : args[0]?.toLowerCase() || ""
            const subArgs = isCompressCommand ? args : args.slice(1)

            const commandCtx = {
                client,
                state,
                config,
                logger,
                sessionId: input.sessionID,
                messages,
            }

            if (subcommand === "context") {
                await handleContextCommand(commandCtx)
                return
            }

            if (subcommand === "stats") {
                await handleStatsCommand(commandCtx)
                return
            }

            if (subcommand === "sweep") {
                await handleSweepCommand({
                    ...commandCtx,
                    args: subArgs,
                    workingDirectory,
                })
                return
            }

            if (subcommand === "manual") {
                await handleManualToggleCommand(commandCtx, subArgs[0]?.toLowerCase())
                return
            }

            if (subcommand === "compress") {
                const userFocus = subArgs.join(" ").trim()
                const prompt = await handleManualTriggerCommand(commandCtx, "compress", userFocus)
                if (!prompt) {
                    throw new Error("__DCP_MANUAL_TRIGGER_BLOCKED__")
                }

                state.manualMode = "compress-pending"
                state.pendingManualTrigger = {
                    sessionId: input.sessionID,
                    prompt,
                }
                const rawArgs = (input.arguments || "").trim()
                output.parts.length = 0
                output.parts.push({
                    type: "text",
                    text: isCompressCommand
                        ? rawArgs
                            ? `/dcp-compress ${rawArgs}`
                            : "/dcp-compress"
                        : rawArgs
                          ? `/dcp ${rawArgs}`
                          : `/dcp ${subcommand}`,
                })
                return
            }

            if (subcommand === "decompress") {
                await handleDecompressCommand({
                    ...commandCtx,
                    args: subArgs,
                })
                return
            }

            if (subcommand === "recompress") {
                await handleRecompressCommand({
                    ...commandCtx,
                    args: subArgs,
                })
                return
            }

            if (subcommand === "view-export") {
                await handleViewExportCommand(commandCtx)
                return
            }

            if (subcommand === "view-compile") {
                await handleViewCompileCommand(commandCtx, subArgs)
                return
            }

            await handleHelpCommand(commandCtx)
            return
        }
    }
}

export function createTextCompleteHandler() {
    return async (
        _input: { sessionID: string; messageID: string; partID: string },
        output: { text: string },
    ) => {
        output.text = stripHallucinationsFromString(output.text)
    }
}

export function createEventHandler(
    state: SessionState,
    logger: Logger,
    config?: PluginConfig,
    client?: any,
) {
    return async (input: { event: any }) => {
        const cfg = config ?? { view: { enabled: false } } as PluginConfig
        const clientRef = client ?? {} as any
        const eventTime =
            typeof input.event?.time === "number" && Number.isFinite(input.event.time)
                ? input.event.time
                : typeof input.event?.properties?.time === "number" &&
                    Number.isFinite(input.event.properties.time)
                  ? input.event.properties.time
                  : undefined

        if (input.event.type === "session.deleted") {
            const sessionId = input.event.properties?.sessionID || input.event.properties?.id
            if (typeof sessionId !== "string" || !sessionId) {
                return
            }

            const deleted = await deleteSessionState(sessionId)

            // Clean up VCC exports for the deleted session
            let removedViews = 0
            const vccDir = join(STORAGE_DIR, "vcc")
            if (existsSync(vccDir)) {
                const files = await fs.readdir(vccDir)
                for (const f of files) {
                    if (f.startsWith(`${sessionId}_export`)) {
                        try {
                            await fs.unlink(join(vccDir, f))
                            removedViews++
                        } catch {}
                    }
                }
            }

            logger.info("Handled session deletion", {
                sessionId,
                stateFileDeleted: deleted,
                vccFilesRemoved: removedViews,
            })
            return
        }

        if (input.event.type !== "message.part.updated") {
            return
        }

        const part = input.event.properties?.part
        if (part?.type !== "tool" || part.tool !== "compress") {
            return
        }

        if (part.state.status === "pending") {
            if (typeof part.callID !== "string" || typeof part.messageID !== "string") {
                return
            }

            const startedAt = eventTime ?? Date.now()
            const key = buildCompressionTimingKey(part.messageID, part.callID)
            if (state.compressionTiming.startsByCallId.has(key)) {
                return
            }
            state.compressionTiming.startsByCallId.set(key, startedAt)
            logger.debug("Recorded compression start", {
                messageID: part.messageID,
                callID: part.callID,
                startedAt,
            })
            return
        }

        if (part.state.status === "completed") {
            if (typeof part.callID !== "string" || typeof part.messageID !== "string") {
                return
            }

            const key = buildCompressionTimingKey(part.messageID, part.callID)
            const start = consumeCompressionStart(state, part.messageID, part.callID)
            const durationMs = resolveCompressionDuration(start, eventTime, part.state.time)
            if (typeof durationMs !== "number") {
                return
            }

            state.compressionTiming.pendingByCallId.set(key, {
                messageId: part.messageID,
                callId: part.callID,
                durationMs,
            })

            const updates = applyPendingCompressionDurations(state)
            if (updates === 0) {
                return
            }

            await saveSessionState(state, logger)

            logger.info("Attached compression time to blocks", {
                messageID: part.messageID,
                callID: part.callID,
                blocks: updates,
                durationMs,
            })

            // Auto-export + compile on compression completion
            if (cfg.view?.enabled && cfg.view.autoExport) {
                runAutoVccPipeline(state, cfg, clientRef, logger).catch((err) => {
                    logger.warn("Auto VCC pipeline failed", { error: err?.message })
                })
            }
            return
        }

        if (part.state.status === "running") {
            return
        }

        if (typeof part.callID === "string" && typeof part.messageID === "string") {
            state.compressionTiming.startsByCallId.delete(
                buildCompressionTimingKey(part.messageID, part.callID),
            )
        }
    }
}

async function exportSessionForVcc(
    state: SessionState,
    messages: WithParts[],
    logger: Logger,
    exportDirOverride?: string,
): Promise<string> {
    const EXPORT_DIR = exportDirOverride || join(STORAGE_DIR, "vcc")
    if (!existsSync(EXPORT_DIR)) {
        mkdirSync(EXPORT_DIR, { recursive: true })
    }

    if (!state.sessionId) {
        throw new Error("No active session")
    }

    const exportPath = join(EXPORT_DIR, `${state.sessionId}_export.jsonl`)

    interface VccRecord {
        type: string
        timestamp?: string
        message?: {
            content?: any
            usage?: Record<string, number>
            model?: string
        }
        content?: string
        blockId?: number
        summary?: string
        metadata?: Record<string, any>
    }

    const records: VccRecord[] = []

    records.push({
        type: "system",
        timestamp: new Date().toISOString(),
        message: {
            content: [{ type: "text", text: `View export for session ${state.sessionId}` }],
        },
    })

    const blockRefs: Record<string, any> = {}

    for (const block of state.prune.messages.blocksById.values()) {
        const blockKey = `[block:${block.blockId}]`
        blockRefs[blockKey] = block
        records.push({
            type: "system",
            timestamp: new Date(block.createdAt).toISOString(),
            message: {
                content: [{
                    type: "text",
                    text: `[compressed block ${block.blockId}] topic: ${block.topic}\nsummary: ${block.summary}`,
                }],
            },
        })
    }

    for (const msg of messages) {
        const content = partsToVccContent(
            msg.parts as unknown as Array<{ type: string } & Record<string, any>>,
        )

        const tokenCount = state.prune.messages.byMessageId.get(msg.info.id)?.tokenCount || 0

        records.push({
            type: msg.info.role === "user" ? "user" : msg.info.role === "assistant" ? "assistant" : "system",
            timestamp: typeof msg.info.time === "number" ? new Date(msg.info.time).toISOString() : new Date().toISOString(),
            message: {
                content: content.length ? content : [],
            },
            metadata: {
                messageId: msg.info.id,
                tokenCount,
                role: msg.info.role || "assistant",
            },
        })
    }

    const jsonlContent = records.map((r) => JSON.stringify(r)).join("\n") + "\n"
    await fs.writeFile(exportPath, jsonlContent, "utf-8")

    logger.info("Exported session for VCC", {
        sessionId: state.sessionId,
        outputPath: exportPath,
        records: records.length,
    })

    return exportPath
}

async function handleViewExportCommand(cmdCtx: {
    client: any
    state: SessionState
    config: PluginConfig
    logger: Logger
    sessionId: string
    messages: WithParts[]
}): Promise<void> {
    const exportPath = await exportSessionForVcc(cmdCtx.state, cmdCtx.messages, cmdCtx.logger, cmdCtx.config.view?.exportDir)

    await cmdCtx.client.session.prompt({
        path: { id: cmdCtx.sessionId },
        body: {
            noReply: true,
            parts: [{ type: "text", text: `VCC export written to:\n\n${exportPath}\n\nRun:\npython <path-to-VCC.py> "${exportPath}"` }],
        },
    })
}

async function handleViewCompileCommand(cmdCtx: {
    client: any
    state: SessionState
    config: PluginConfig
    logger: Logger
    sessionId: string
    messages: WithParts[]
}, args: string[]): Promise<void> {
    const viewConfig = cmdCtx.config.view
    if (!viewConfig.enabled) {
        sendResponse(cmdCtx.client, cmdCtx.sessionId, "View feature disabled. Set view.enabled=true in dcp.jsonc")
        return
    }

    const scriptPath = viewConfig.scriptPath
    if (!scriptPath) {
        sendResponse(cmdCtx.client, cmdCtx.sessionId, "VCC script path not configured. Set view.scriptPath in dcp.jsonc")
        return
    }

    const grepPattern = args.length > 0 ? args.join(" ") : undefined

    const exportPath = await exportSessionForVcc(cmdCtx.state, cmdCtx.messages, cmdCtx.logger, cmdCtx.config.view?.exportDir)

    runVccCompile(
        viewConfig.pythonPath || "python",
        scriptPath,
        exportPath,
        grepPattern,
        viewConfig.rotateKeep,
    ).then((output) => {
        sendResponse(cmdCtx.client, cmdCtx.sessionId, String(output))
    }).catch((err) => {
        sendResponse(cmdCtx.client, cmdCtx.sessionId, `Error: ${String(err.message || err)}`)
    })
}

function sendResponse(client: any, sessionId: string, text: string): void {
    client.session.prompt({
        path: { id: sessionId },
        body: {
            noReply: true,
            parts: [{ type: "text", text }],
        },
    }).catch(() => {})
}

async function runVccCompile(
    pythonPath: string,
    scriptPath: string,
    exportPath: string,
    grepPattern?: string,
    rotateKeep?: number,
): Promise<string> {
    // Rotate previous view files before VCC overwrites them
    await rotateViewFiles(exportPath, rotateKeep ?? 3)

    const args = [scriptPath, exportPath]
    if (grepPattern) {
        args.push("--grep", grepPattern)
    }

    return new Promise((resolve, reject) => {
        execFile(pythonPath, args, (error, stdout, stderr) => {
            if (error) {
                reject(new Error(`VCC failed: ${error.message}${stderr ? "\n" + stderr : ""}`))
            } else {
                resolve(stdout || "")
            }
        })
    })
}

async function runAutoVccPipeline(
    state: SessionState,
    config: PluginConfig,
    client: any,
    logger: Logger,
): Promise<void> {
    const viewConfig = config.view
    if (!viewConfig.enabled || !viewConfig.autoExport) {
        return
    }

    const sessionId = state.sessionId
    if (!sessionId) {
        return
    }

    const scriptPath = viewConfig.scriptPath
    if (!scriptPath) {
        logger.warn("view.scriptPath not configured; skipping auto VCC compile")
        return
    }

    // Fetch current session messages
    const messagesResponse = await client.session.messages({ path: { id: sessionId } })
    const messages = filterMessages(messagesResponse.data || messagesResponse)

    // Export session snapshot to VCC format
    const exportPath = await exportSessionForVcc(state, messages, logger, viewConfig.exportDir)

    // Compile with VCC
    const output = await runVccCompile(
        viewConfig.pythonPath || "python",
        scriptPath,
        exportPath,
        undefined,
        viewConfig.rotateKeep,
    )

    // Read the .min.txt brief view
    const minPath = exportPath.replace(/\.jsonl$/, ".min.txt")
    const fullPath = minPath.replace(/\.min\.txt$/, ".txt")

    if (viewConfig.postMode === "off") {
        return
    }

    if (viewConfig.postMode === "notice") {
        sendResponse(
            client,
            sessionId,
            [
                `**VCC views updated** (session ${sessionId})`,
                ``,
                `Brief view (structure + tool call line refs): \`${minPath}\``,
                `Full view (lossless transcript): \`${fullPath}\``,
                ``,
                `You can search past context by running \`/dcp view-compile <pattern>\`, or using the \`view\` tool with a \`pattern\`.`,
            ].join("\n"),
        )
        return
    }

    // postMode === "fullminview": read and post the brief view
    let briefContent = ""
    try {
        briefContent = await fs.readFile(minPath, "utf-8")
    } catch {
        logger.warn("Could not read VCC .min.txt output", { minPath })
    }

    const resultText = [
        `**VCC view updated** (session ${sessionId})`,
        ``,
        briefContent || output,
        ``,
        `Full view: ${fullPath}`,
        `Brief view: ${minPath}`,
    ].join("\n")

    sendResponse(client, sessionId, resultText)
}
