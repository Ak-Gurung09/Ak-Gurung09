import type {
  EngineInterface,
  Register,
  SessionMessage,
  ToolUseSummary,
} from 'claude-code'

// endless-memory: long-term memory that runs on its own, every turn.
//
//  1. After every finished turn, a small model reads that turn and folds
//     anything worth keeping into two sets of notes, saved across sessions:
//     one about the user (shared by every project), one per project folder.
//  2. Those notes ride along in the conversation's opening context, which the
//     engine rebuilds after every compaction and in every new session, so
//     nothing important is lost when old messages get summarized away.
//  3. When the context passes COMPACT_AT_PERCENT, the chat is compacted early
//     (right after a turn, while you read the reply) to keep it fast.
//
// There are no commands: nothing to type, nothing to remember to run.

const MEMORY_MODEL = 'haiku'
const COMPACT_AT_PERCENT = 50
const MAX_SECTION_CHARS = 4000
const MAX_TURN_CHARS = 16000
const MAX_PIECE_CHARS = 4000
const MAX_BACKLOG_CHARS = 24000
const BLOCK_NAME = 'endlessMemory'
const GLOBAL_KEY = 'memory:global'
const projectKey = (cwd: string) => `memory:project:${cwd}`

const COMPACT_NOTE =
  'Durable facts, preferences and project notes are kept separately by the ' +
  'endless-memory mod and re-attached after this compaction. Focus the summary ' +
  'on the current task: what is in progress, exact file paths and identifiers ' +
  'being worked on, recent decisions, errors still open, and the next steps.'

type Memory = { global: string; project: string }

// Module state. A reload starts these over, which only costs the turn text
// waiting in the backlog (if a memory update had failed just before).
let lastPrompt = ''
let backlog = ''
let pending: Promise<void> = Promise.resolve()

const clip = (text: string, max: number) =>
  text.length <= max ? text : `${text.slice(0, max)}\n…(cut)`

async function readMemory($: EngineInterface, cwd: string): Promise<Memory> {
  const [g, p] = await Promise.all([
    $.store.get(GLOBAL_KEY),
    $.store.get(projectKey(cwd)),
  ])
  return {
    global: typeof g === 'string' ? g : '',
    project: typeof p === 'string' ? p : '',
  }
}

function renderMemory(memory: Memory, cwd: string): string | undefined {
  if (!memory.global.trim() && !memory.project.trim()) return undefined
  return [
    'Long-term memory, kept automatically across compactions and sessions by the',
    'endless-memory mod. Treat it as background notes from earlier work: the live',
    'conversation wins where they disagree. No need to mention it unless asked.',
    '',
    '## About the user (all projects)',
    memory.global.trim() || '(nothing yet)',
    '',
    `## This project (${cwd})`,
    memory.project.trim() || '(nothing yet)',
  ].join('\n')
}

function describeTool(use: ToolUseSummary): string {
  let input = ''
  try {
    input = JSON.stringify(use.input ?? {})
  } catch {
    input = '{…}'
  }
  return `[${use.tool} ${clip(input, 300)}${use.isError ? ' -> error' : ''}]`
}

// The text of the turn that just ended: the prompt, the assistant's replies
// and a one-line note per tool call, newest last, within MAX_TURN_CHARS.
function lastTurnText(rows: readonly SessionMessage[], prompt: string): string {
  const pieces: string[] = []
  let used = 0
  let sawPrompt = false
  const probe = prompt.trim().slice(0, 200)

  for (let i = rows.length - 1; i >= 0 && used < MAX_TURN_CHARS; i--) {
    const row = rows[i]
    if (!row) continue
    const text = row.text.trim()
    const isPrompt =
      row.role === 'user' && !row.toolResults?.length && text !== ''
    let piece = ''
    if (row.role === 'assistant') {
      piece = [text && `ASSISTANT: ${text}`, ...row.toolUses.map(describeTool)]
        .filter(Boolean)
        .join('\n')
    } else if (isPrompt) {
      piece = `USER: ${text}`
    }
    if (piece) {
      piece = clip(piece, Math.min(MAX_PIECE_CHARS, MAX_TURN_CHARS - used))
      pieces.push(piece)
      used += piece.length
    }
    if (isPrompt && (probe === '' || text.includes(probe))) {
      sawPrompt = true
      break
    }
  }
  if (!sawPrompt && prompt.trim()) {
    pieces.push(`USER: ${clip(prompt.trim(), MAX_PIECE_CHARS)}`)
  }
  return pieces.reverse().join('\n\n')
}

const EXTRACT_SYSTEM =
  'You maintain the long-term memory notes of an AI coding assistant (Claude ' +
  'Code). Its conversations are periodically compacted and new sessions start ' +
  'from scratch, so these notes are the only thing that survives. Keep them ' +
  'accurate, terse and current.'

function extractPrompt(memory: Memory, exchange: string, cwd: string): string {
  return `<current_memory>
<global>
${memory.global.trim() || '(empty)'}
</global>
<project path="${cwd}">
${memory.project.trim() || '(empty)'}
</project>
</current_memory>

<latest_exchange>
${exchange}
</latest_exchange>

Update the memory with anything from the latest exchange worth knowing later.

- global: durable facts about the user that hold in any project: preferences, working style, background, how they like answers and code.
- project: this project and its work: goals, decisions and the reasons for them, key files, commands and conventions, current status, open tasks and next steps, pitfalls found.

Rules:
- Terse "- " bullet points. Merge duplicates, correct stale items, drop finished or obsolete ones; keep the most important when space is short.
- Keep each section under about 350 words.
- Never store secrets (API keys, tokens, passwords, private keys) or code longer than one line.
- Skip small talk and anything trivially re-read from the files.
- If nothing worth remembering changed, reply exactly: NO_CHANGE
- Otherwise reply with exactly these two tags holding the FULL updated notes, nothing before or after:
<global>
...
</global>
<project>
...
</project>`
}

function parseSection(reply: string, tag: string): string | undefined {
  const match = reply.match(new RegExp(`<${tag}[^>]*>([\\s\\S]*?)</${tag}>`))
  if (!match) return undefined
  const body = (match[1] ?? '').trim()
  return body === '(empty)' ? '' : clip(body, MAX_SECTION_CHARS)
}

async function updateMemory($: EngineInterface, prompt: string): Promise<void> {
  const cwd = await $.session.cwd()
  const rows = await $.session.messages()
  const turn = lastTurnText(rows, prompt)
  if (!turn.trim()) return

  const exchange = clip(
    backlog ? `${backlog}\n\n---\n\n${turn}` : turn,
    MAX_BACKLOG_CHARS,
  )
  const memory = await readMemory($, cwd)
  const reply = await $.model.complete({
    model: MEMORY_MODEL,
    system: EXTRACT_SYSTEM,
    prompt: extractPrompt(memory, exchange, cwd),
    maxTokens: 3000,
    effort: 'low',
    timeoutMs: 90_000,
  })

  if (!reply.isAnswered) {
    // Keep the turn and fold it in with the next one.
    backlog = exchange.slice(-MAX_BACKLOG_CHARS)
    $.ui.status('memory: will retry next turn')
    return
  }
  backlog = ''

  const text = reply.text.trim()
  if (text.startsWith('NO_CHANGE')) return
  const global = parseSection(text, 'global')
  const project = parseSection(text, 'project')
  if (global === undefined || project === undefined) return

  if (global !== memory.global) await $.store.set(GLOBAL_KEY, global)
  if (project !== memory.project) await $.store.set(projectKey(cwd), project)
  const notes = (global + '\n' + project)
    .split('\n')
    .filter(line => line.trim().startsWith('-')).length
  $.ui.status(`memory: ${notes} notes saved`)
}

// How full the context was right after the last early compaction. If a
// compaction cannot get below the threshold (a lot had to be kept), wait for
// real growth before compacting again instead of compacting every turn.
let compactFloor = 0

async function compactIfFull($: EngineInterface): Promise<void> {
  const { context } = await $.session.usage()
  const percent = context.percent ?? 0
  if (percent < Math.max(COMPACT_AT_PERCENT, compactFloor + 15)) return
  $.ui.toast(`Context ${percent}% full: compacting early. Memory is saved.`)
  try {
    const done = await $.session.compact({ instructions: COMPACT_NOTE })
    const after = done.skip === undefined ? done.tokensAfter : undefined
    compactFloor =
      after && context.window ? Math.round((after / context.window) * 100) : 0
  } catch {
    // A new turn started first; the next turn's end tries again.
  }
}

export const register: Register = on => {
  // Remember what was asked, so the turn's text can be found afterwards.
  on('prompt.submit', ($, e, next) => {
    if (!e.isMeta && !e.command && e.text.trim()) lastPrompt = e.text
    return next(e)
  }).catch(($, e, next) => next(e)) // never get in the way of a prompt

  // After every finished turn of the main conversation: save memory, then
  // compact early if the context is getting full. Runs on a timer so the
  // turn itself is never held up.
  on('turn.complete', async ($, e, next) => {
    const result = await next(e)
    if (e.agentId === undefined && e.reason === 'answer') {
      const prompt = lastPrompt
      $.clock.after(1, () => {
        const update = pending
          .then(() => updateMemory($, prompt))
          .catch(() => $.ui.status('memory: update failed'))
        pending = update
        void update.then(() => compactIfFull($)).catch(() => {})
      })
    }
    return result
  })

  // Every conversation's opening context (a new session, /clear, and again
  // after each compaction) carries the saved memory.
  on('prompt.context', async ($, e, next) => {
    const result = await next(e)
    const cwd = await $.session.cwd()
    const text = renderMemory(await readMemory($, cwd), cwd)
    if (!text) return result
    return {
      ...result,
      blocks: [
        ...result.blocks.filter(block => block.name !== BLOCK_NAME),
        { name: BLOCK_NAME, text },
      ],
    }
  })

  // Any compaction (automatic, /compact, or ours): tell the summarizer the
  // long-term notes are kept elsewhere, so the summary can go to the task.
  on('session.compact', ($, e, next) => {
    if (e.agentId !== undefined || e.trigger === 'precompute') return next(e)
    const instructions = e.instructions
      ? `${e.instructions}\n\n${COMPACT_NOTE}`
      : COMPACT_NOTE
    return next({ ...e, instructions })
  }).catch(($, e, next) => next(e)) // never get in the way of a compaction
}
