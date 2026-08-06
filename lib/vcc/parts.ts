import type { ReasoningPart, ToolPart, TextPart } from "@opencode-ai/sdk/v2"

export type VccContentBlock = Record<string, any>

/**
 * Convert opencode message parts into VCC (Claude Code format) content blocks.
 *
 * Mapping:
 *   text      -> { type: "text", text }                       (VCC.py:378)
 *   reasoning -> { type: "thinking", thinking }               (VCC.py:361)
 *   tool      -> { type: "tool_use", ... }                    (VCC.py:385)
 *
 * VCC renders thinking blocks in the full view (.txt) and hides them in the
 * brief view (.min.txt) / truncation (VCC.py:769), so reasoning survives the
 * archive without polluting the compact view.
 */
export function partsToVccContent(
    parts: Array<Record<string, any>>,
): VccContentBlock[] {
    const blocks: VccContentBlock[] = []

    for (const part of parts) {
        switch (part?.type) {
            case "text": {
                const text = (part as TextPart).text ?? ""
                if (text.trim()) {
                    blocks.push({ type: "text", text })
                }
                break
            }

            case "reasoning": {
                const text = (part as ReasoningPart).text ?? ""
                if (text.trim()) {
                    blocks.push({ type: "thinking", thinking: text })
                }
                break
            }

            case "tool": {
                const toolPart = part as ToolPart
                const input = toolPart.state?.input
                const output = (toolPart.state as { output?: unknown })?.output

                if (input !== undefined && input !== null) {
                    blocks.push({
                        type: "tool_use",
                        id: toolPart.callID,
                        name: toolPart.tool,
                        input: typeof input === "object" ? input : { value: String(input) },
                    })
                }
                if (output !== undefined && output !== null) {
                    blocks.push({
                        type: "tool_result",
                        tool_use_id: toolPart.callID,
                        content:
                            typeof output === "object"
                                ? JSON.stringify(output)
                                : String(output),
                    })
                }
                break
            }
        }
    }

    return blocks
}
