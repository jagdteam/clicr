#!/usr/bin/env node
// B3, PreToolUse (Bash): when the command is a single plain test run, rewrite it to run through
// shrink-run.mjs, which prints failures only (plus the summary). Anything else passes untouched.
// A trailing `2>&1` is dropped (the runner merges both streams). Escape hatch: put SHRINK_OFF=1 in front of the command. Fails open. Node stdlib only.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const TEST_CMD = /^(?:npm (?:run )?test|npx vitest(?: run)?|vitest(?: run)?|pytest|python3? -m pytest|uv run pytest)(?:\s+[^;&|<>`$()]*)?$/;

try {
  const input = JSON.parse(fs.readFileSync(0, 'utf8'));
  const command = ((input.tool_input && input.tool_input.command) || '').trim().replace(/\s+2>&1$/, '');
  if (!TEST_CMD.test(command)) process.exit(0);
  const runner = path.join(path.dirname(fileURLToPath(import.meta.url)), 'shrink-run.mjs');
  const quoted = `'${command.replace(/'/g, `'\\''`)}'`;
  process.stdout.write(JSON.stringify({
    hookSpecificOutput: {
      hookEventName: 'PreToolUse',
      updatedInput: { ...input.tool_input, command: `node '${runner}' ${quoted}` },
    },
  }));
} catch (e) {
  process.stderr.write(`shrink-test-output hook skipped: ${e.message}\n`);
}
process.exit(0);
