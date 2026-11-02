# endless-memory

A Claude Code mod that gives Claude long-term memory, so a conversation can go
on indefinitely without Claude losing track of what matters. It runs on its
own: there are no commands to type and nothing to remember to run.

## What it does, every turn

1. **Saves memory after every turn.** When a turn finishes, a small, cheap
   model (Haiku) reads that turn and folds anything worth keeping into two
   sets of notes:
   - **About you**: preferences, working style, how you like answers. Shared
     by every project.
   - **This project**: goals, decisions and why, key files and commands,
     current status, open tasks, next steps. One set per project folder.
2. **Feeds it back.** The notes ride along in the conversation's opening
   context. Claude Code rebuilds that after every compaction, after `/clear`
   and in every new session, so the important things survive when old
   messages get summarized away.
3. **Compacts early.** When the context passes 50% full, the chat is
   compacted right after a turn, while you read the reply. Turns stay fast
   and Claude stays focused, and the summary is told to concentrate on the
   current task because the long-term notes are kept separately.

The status line shows `memory: N notes saved` after each update.

## Install (once per computer)

At the prompt of a Claude Code terminal session:

```
/plugin install endless-memory --marketplace ak-gurung09/ak-gurung09
```

Answer `y` to add the marketplace, then choose the **user** scope. That makes
it active in every Claude Code session on that computer from then on,
including the desktop app's local sessions.

From a shell instead:

```bash
claude plugin install endless-memory --marketplace ak-gurung09/ak-gurung09
```

In the desktop app, where `/plugin` does not run: click + next to the prompt
box, then Plugins, Add marketplace with `ak-gurung09/ak-gurung09`, then Add
plugin.

Cloud sessions (claude.ai/code) started on the `ak-gurung09/ak-gurung09`
repository pick it up from that repository's `.claude/settings.json`.

## Good to know

- **Cost:** one small Haiku call per finished turn, billed like your other
  Claude Code usage. Early compaction also makes one summarization call each
  time it runs.
- **Where memory lives:** in Claude Code's plugin storage on the computer the
  session runs on. A cloud session's container is temporary, so there the
  memory lasts as long as that session's container does.
- **What it never saves:** the extraction prompt forbids API keys, tokens,
  passwords and code longer than a line.
- **Subagents** and interrupted turns are skipped.
- **Tuning:** the constants at the top of `hooks/register.ts`
  (`COMPACT_AT_PERCENT`, `MAX_SECTION_CHARS`, `MEMORY_MODEL`).
- **Limits:** Claude still reads the conversation it has. This mod keeps that
  conversation small and makes sure the important parts survive in notes. Old
  details live on as notes, not word for word.

## Development

```bash
claude plugin validate claude-mods/endless-memory
```

```bash
claude plugin test claude-mods/endless-memory
```
