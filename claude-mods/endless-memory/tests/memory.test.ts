import type { On } from 'claude-code'
import { expect, mock, test } from 'claude-code/testing'

const USAGE = {
  input_tokens: 1,
  output_tokens: 1,
  cache_creation_input_tokens: 0,
  cache_read_input_tokens: 0,
}

const MEMORY_REPLY = [
  '<global>',
  '- Prefers pnpm over npm',
  '</global>',
  '<project>',
  '- The API lives in src/api.ts',
  '</project>',
].join('\n')

// The engine beneath the plugin: a project at /proj with one finished turn.
function engine(on: On, percent: number) {
  const clock = mock.clock(on)
  mock.store(on)
  const asked: string[] = []
  const compactions: string[] = []
  on('session.cwd', () => ({ value: '/proj' }))
  on('session.messages', () => ({
    value: [
      { role: 'user', text: 'Use pnpm. Put the API in src/api.ts', toolUses: [] },
      { role: 'assistant', text: 'Done: created src/api.ts', toolUses: [] },
    ],
  }))
  on('model.complete', (_, e) => {
    asked.push(e.prompt)
    return {
      value: { isAnswered: true, text: MEMORY_REPLY, usage: USAGE },
    }
  })
  on('session.usage', () => ({
    value: { startedAt: 0, context: { window: 200_000, percent }, rateLimits: [] },
  }))
  on('session.compact', (_, e) => {
    compactions.push(e.instructions ?? '')
    return {
      messages: [{ role: 'user' as const, text: 'Summary so far', toolUses: [] }],
      tokensAfter: 20_000,
    }
  })
  on('ui.status', () => ({ value: undefined }))
  on('ui.toast', () => ({ value: undefined }))
  on('prompt.submit', (_, e) => ({ text: e.text }))
  on('turn.complete', (_, e) => ({ text: e.answer }))
  on('prompt.context', (_, e) => ({ blocks: e.blocks }))
  return { clock, asked, compactions }
}

const prompt = {
  text: 'Use pnpm. Put the API in src/api.ts',
  wait: false,
  origin: { kind: 'composer' },
} as const

const finishTurn = {
  answer: 'Done: created src/api.ts',
  durationMs: 1000,
  isAborted: false,
  turnId: 't1',
  reason: 'answer',
} as const

test('a finished turn is saved and comes back in the next conversation', async ($, on) => {
  const { clock, asked, compactions } = engine(on, 10)

  await $.prompt.submit(prompt)
  await $.turn.complete(finishTurn)
  await clock.advance(10)

  expect(asked.length).toBe(1)
  expect(asked[0]).toContain('Use pnpm. Put the API in src/api.ts')
  expect(compactions.length).toBe(0)

  const { blocks } = await $.prompt.context({ blocks: [] })
  const memory = blocks.find(block => block.name === 'endlessMemory')
  expect(memory?.text).toContain('Prefers pnpm over npm')
  expect(memory?.text).toContain('The API lives in src/api.ts')
  expect(memory?.text).toContain('This project (/proj)')
})

test('a context past the threshold is compacted right after the turn', async ($, on) => {
  const { clock, compactions } = engine(on, 62)

  await $.prompt.submit(prompt)
  await $.turn.complete(finishTurn)
  await clock.advance(10)

  expect(compactions.length).toBe(1)
  expect(compactions[0]).toContain('endless-memory')
})

test('a subagent turn or an interrupted one saves nothing', async ($, on) => {
  const { clock, asked } = engine(on, 10)

  await $.turn.complete({ ...finishTurn, agentId: 'a1' })
  await $.turn.complete({ ...finishTurn, reason: 'aborted', isAborted: true })
  await clock.advance(10)

  expect(asked.length).toBe(0)
})

test('nothing saved yet adds no block', async ($, on) => {
  engine(on, 10)
  const { blocks } = await $.prompt.context({ blocks: [] })
  expect(blocks.length).toBe(0)
})
