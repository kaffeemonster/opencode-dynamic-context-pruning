// These format schemas are kept separate from the editable compress prompts
// so they cannot be modified via custom prompt overrides. The schemas must
// match the tool's input validation and are not safe to change independently.

export const RANGE_FORMAT_EXTENSION = `
THE FORMAT OF COMPRESS

\`\`\`
{
  topic: string,           // Short label (3-5 words) - e.g., "Auth System Exploration"
  content: [               // One or more ranges to compress
    {
      startId: string,     // Boundary ID at range start: mNNNN or bN
      endId: string,       // Boundary ID at range end: mNNNN or bN
      summary: string      // Complete technical summary replacing all content in range
    }
  ]
}
\`\`\``

export const MESSAGE_FORMAT_EXTENSION = `
THE FORMAT OF COMPRESS

\`\`\`
{
  topic: string,           // Short label (3-5 words) for the overall batch
  content: [               // One or more messages to compress independently
    {
      messageId: string,   // Raw message ID only: mNNNN (ignore metadata attributes like priority)
      topic: string,       // Short label (3-5 words) for this one message summary
      summary: string      // Complete technical summary replacing that one message
    }
  ]
}
\`\`\``

/**
 * Appended to compress tool descriptions when compress.summaryStyle is
 * "terse". Instructs the model to write summaries in dense caveman style:
 * all technical substance preserved, all prose fluff stripped.
 */
export const TERSE_STYLE_EXTENSION = `
SUMMARY STYLE: TERSE

Write the summary in dense telegraphic style. All technical substance must survive - nothing below is optional.

Rules:
- Drop: articles (a/an/the), filler words (just/really/basically), pleasantries, hedging.
- Fragments OK. Short synonyms. Technical terms kept exact. Code unchanged.
- Pattern: [thing] [action] [reason]. [next step].
- Not: "The user asked about authentication flow implementation details."
- Yes: "Auth flow. JWT refresh bug. Fix in refresh-token.ts: expiry check off-by-one."

- All file paths, function signatures, decisions, constraints, user intent, and tool outcomes must appear - just in dense form, never omitted.
- Preserve direct quotes of short user instructions verbatim when they best capture intent.
- A summary shorter than the content it replaces is the goal. Only include prose where omitting it would lose meaning.`

/**
 * Appended to compress tool descriptions when compress.summaryStyle is
 * "wenyan". Maximum classical terseness: 80-90% character reduction via
 * Classical Chinese grammar, while ALL technical substance stays verbatim.
 * The reader model is the same model that wrote the summary, so dense
 * classical scaffolding retains fidelity.
 */
export const WENYAN_STYLE_EXTENSION = `
SUMMARY STYLE: WENYAN

Write the summary in Classical Chinese (文言文). Maximum classical terseness, 80-90% character reduction. Technical substance must survive fully - nothing below is optional.

Rules:
- Classical sentence patterns. Verbs precede objects. Subjects often omitted. Classical particles (之/乃/為/其).
- Prose scaffolding in wenyan; technical payload verbatim. File paths, function signatures, code, API names, CLI commands, tool names, error strings: NEVER translate, NEVER abbreviate, keep exact.
- Decisions, constraints, user intent, and tool outcomes must appear - dense, never omitted.
- Preserve short user instructions as verbatim quotes when they best capture intent.
- Prefer wenyan characters that map 1:1 to meaning over longer constructions.

Example — "Why React component re-render?"
- wenyan: "每繪新生對象參照，故重繪；以 useMemo 包之則免。"

Never output normal English prose. If a technical term has no classical rendering, keep it as-is.`
