import type { SessionState, WithParts } from "../state"
import type { Logger } from "../logger"
import type { PluginConfig } from "../config"
import { isMessageCompacted } from "../state/utils"
import { createSyntheticUserMessage, replaceBlockIdsWithBlocked } from "./utils"
import { getLastUserMessage } from "./query"
import { isToolNameProtected } from "../protected-patterns"
import type { UserMessage } from "@opencode-ai/sdk/v2"

const PRUNED_TOOL_OUTPUT_REPLACEMENT =
    "[Output removed to save context - information superseded or no longer needed]"
const PRUNED_TOOL_ERROR_INPUT_REPLACEMENT = "[input removed due to failed tool call]"
const PRUNED_QUESTION_INPUT_REPLACEMENT = "[questions removed - see output for user's answers]"

export const prune = (
    state: SessionState,
    logger: Logger,
    config: PluginConfig,
    messages: WithParts[],
): void => {
    filterCompressedRanges(state, logger, config, messages)
    // pruneFullTool(state, logger, messages)
    pruneToolOutputs(state, logger, messages)
    pruneToolInputs(state, logger, messages)
    pruneToolErrors(state, logger, messages)
    purgeReasoning(state, logger, config, messages)
}

const pruneFullTool = (state: SessionState, logger: Logger, messages: WithParts[]): void => {
    const messagesToRemove: string[] = []

    for (const msg of messages) {
        if (isMessageCompacted(state, msg)) {
            continue
        }

        const parts = Array.isArray(msg.parts) ? msg.parts : []
        const partsToRemove: string[] = []

        for (const part of parts) {
            if (part.type !== "tool") {
                continue
            }

            if (!state.prune.tools.has(part.callID)) {
                continue
            }
            if (part.tool !== "edit" && part.tool !== "write") {
                continue
            }

            partsToRemove.push(part.callID)
        }

        if (partsToRemove.length === 0) {
            continue
        }

        msg.parts = parts.filter(
            (part) => part.type !== "tool" || !partsToRemove.includes(part.callID),
        )

        if (msg.parts.length === 0) {
            messagesToRemove.push(msg.info.id)
        }
    }

    if (messagesToRemove.length > 0) {
        const result = messages.filter((msg) => !messagesToRemove.includes(msg.info.id))
        messages.length = 0
        messages.push(...result)
    }
}

const pruneToolOutputs = (state: SessionState, logger: Logger, messages: WithParts[]): void => {
    for (const msg of messages) {
        if (isMessageCompacted(state, msg)) {
            continue
        }

        const parts = Array.isArray(msg.parts) ? msg.parts : []
        for (const part of parts) {
            if (part.type !== "tool") {
                continue
            }
            if (!state.prune.tools.has(part.callID)) {
                continue
            }
            if (part.state.status !== "completed") {
                continue
            }
            if (part.tool === "question" || part.tool === "edit" || part.tool === "write") {
                continue
            }

            part.state.output = PRUNED_TOOL_OUTPUT_REPLACEMENT
        }
    }
}

const pruneToolInputs = (state: SessionState, logger: Logger, messages: WithParts[]): void => {
    for (const msg of messages) {
        if (isMessageCompacted(state, msg)) {
            continue
        }

        const parts = Array.isArray(msg.parts) ? msg.parts : []
        for (const part of parts) {
            if (part.type !== "tool") {
                continue
            }

            if (!state.prune.tools.has(part.callID)) {
                continue
            }
            if (part.state.status !== "completed") {
                continue
            }
            if (part.tool !== "question") {
                continue
            }

            if (part.state.input?.questions !== undefined) {
                part.state.input.questions = PRUNED_QUESTION_INPUT_REPLACEMENT
            }
        }
    }
}

const pruneToolErrors = (state: SessionState, logger: Logger, messages: WithParts[]): void => {
    for (const msg of messages) {
        if (isMessageCompacted(state, msg)) {
            continue
        }

        const parts = Array.isArray(msg.parts) ? msg.parts : []
        for (const part of parts) {
            if (part.type !== "tool") {
                continue
            }
            if (!state.prune.tools.has(part.callID)) {
                continue
            }
            if (part.state.status !== "error") {
                continue
            }

            // Prune all string inputs for errored tools
            const input = part.state.input
            if (input && typeof input === "object") {
                for (const key of Object.keys(input)) {
                    if (typeof input[key] === "string") {
                        input[key] = PRUNED_TOOL_ERROR_INPUT_REPLACEMENT
                    }
                }
            }
        }
    }
}

const filterCompressedRanges = (
    state: SessionState,
    logger: Logger,
    config: PluginConfig,
    messages: WithParts[],
): void => {
    if (
        state.prune.messages.byMessageId.size === 0 &&
        state.prune.messages.activeByAnchorMessageId.size === 0
    ) {
        return
    }

    const result: WithParts[] = []

    for (const msg of messages) {
        const msgId = msg.info.id

        // Check if there's a summary to inject at this anchor point
        const blockId = state.prune.messages.activeByAnchorMessageId.get(msgId)
        const summary =
            blockId !== undefined ? state.prune.messages.blocksById.get(blockId) : undefined
        if (summary) {
            const rawSummaryContent = (summary as { summary?: unknown }).summary
            if (
                summary.active !== true ||
                typeof rawSummaryContent !== "string" ||
                rawSummaryContent.length === 0
            ) {
                logger.warn("Skipping malformed compress summary", {
                    anchorMessageId: msgId,
                    blockId: (summary as { blockId?: unknown }).blockId,
                })
            } else {
                // Find user message for variant and as base for synthetic message
                const msgIndex = messages.indexOf(msg)
                const userMessage = getLastUserMessage(messages, msgIndex)

                if (userMessage) {
                    const userInfo = userMessage.info as UserMessage
                    const summaryContent =
                        config.compress.mode === "message"
                            ? replaceBlockIdsWithBlocked(rawSummaryContent)
                            : rawSummaryContent
                    const summarySeed = `${summary.blockId}:${summary.anchorMessageId}`
                    result.push(
                        createSyntheticUserMessage(userMessage, summaryContent, summarySeed),
                    )

                    logger.info("Injected compress summary", {
                        anchorMessageId: msgId,
                        summaryLength: summaryContent.length,
                    })
                } else {
                    logger.warn("No user message found for compress summary", {
                        anchorMessageId: msgId,
                    })
                }
            }
        }

        // Skip messages that are in the prune list
        const pruneEntry = state.prune.messages.byMessageId.get(msgId)
        if (pruneEntry && pruneEntry.activeBlockIds.length > 0) {
            continue
        }

        // Normal message, include it
        result.push(msg)
    }

    // Replace messages array contents
    messages.length = 0
    messages.push(...result)
}

/**
 * Deterministic reasoning purge with hysteresis.
 *
 * Reasoning traces are scratch work: after the final result is produced the
 * "wait, let me" / "first I have to" chains carry no information, but they are
 * counted against the context budget and model-triggered compression never
 * reaches them because it focuses on the newest messages.
 *
 * Hysteresis: reasoning parts accumulate freely until the total exceeds
 * `highWater` (e.g. 30 parts). Then the oldest messages' reasoning is stripped
 * until only `lowWater` (e.g. 10 parts) remain. This batches the pruning into
 * infrequent bursts instead of stripping one part per turn, which matters for
 * providers that dislike mid-stream reasoning mutation (cache-friendliness).
 *
 * Runs in the request transform on every turn, independent of any compression.
 * Off by default - reasoning carries no value once the result exists, but
 * providers with caching semantics may penalize pruning it.
 *
 * Messages that call a protected tool keep their reasoning so the model can
 * still reconstruct what the protected tool was asked to do.
 */
const purgeReasoning = (
    state: SessionState,
    logger: Logger,
    config: PluginConfig,
    messages: WithParts[],
): void => {
    const strategy = config.strategies.purgeReasoning
    if (!strategy?.enabled) {
        return
    }
    if (state.manualMode && !config.manualMode.automaticStrategies) {
        return
    }

    const turnThreshold = Math.max(1, strategy.turns)
    const protectedTools = strategy.protectedTools

    // Reference turn = total step-starts in the current request messages.
    // state.currentTurn is only refreshed on session init, so computing it
    // here keeps the purge correct even after the session has grown.
    let referenceTurn = 0
    for (const msg of messages) {
        if (isMessageCompacted(state, msg)) {
            continue
        }
        const parts = Array.isArray(msg.parts) ? msg.parts : []
        for (const part of parts) {
            if (part.type === "step-start") {
                referenceTurn++
            }
        }
    }

    // Hysteresis accounting: collect each assistant message's reasoning
    // along with its turn, oldest first.
    const reasoningByMessage: Array<{ msg: WithParts; turn: number; count: number }> = []
    let totalReasoning = 0

    let turnCounter = 0
    for (const msg of messages) {
        if (isMessageCompacted(state, msg)) {
            continue
        }

        const parts = Array.isArray(msg.parts) ? msg.parts : []
        let reasoningCount = 0
        for (const part of parts) {
            if (part.type === "step-start") {
                turnCounter++
            }
            if (part.type === "reasoning") {
                reasoningCount++
            }
        }

        if (msg.info.role !== "assistant" || reasoningCount === 0) {
            continue
        }

        reasoningByMessage.push({ msg, turn: turnCounter, count: reasoningCount })
        totalReasoning += reasoningCount
    }

    const highWater = Math.max(1, strategy.highWater)
    const lowWater = Math.max(0, strategy.lowWater)

    // Hysteresis gate: only act once reasoning exceeds highWater.
    if (totalReasoning <= highWater) {
        return
    }

    // How many reasoning parts to strip to reach the low-water mark.
    const pruneTarget = totalReasoning - lowWater

    let pruned = 0
    for (const entry of reasoningByMessage) {
        if (pruned >= pruneTarget) {
            break
        }

        const turnAge = referenceTurn - entry.turn
        if (turnAge < turnThreshold) {
            continue
        }

        // Skip messages that called a protected tool
        const callsProtectedTool = entry.msg.parts.some(
            (part) =>
                part.type === "tool" &&
                part.tool &&
                isToolNameProtected(part.tool, protectedTools),
        )
        if (callsProtectedTool) {
            continue
        }

        entry.msg.parts = entry.msg.parts.filter((part) => part.type !== "reasoning")
        pruned += entry.count
        logger.debug(
            `Purged ${entry.count} reasoning part(s) from assistant message ${entry.msg.info.id} (turn age ${turnAge})`,
        )
    }

    if (pruned > 0) {
        logger.info("Reasoning purge applied (hysteresis)", {
            totalReasoning,
            highWater,
            lowWater,
            pruned,
            remaining: totalReasoning - pruned,
        })
    }
}
