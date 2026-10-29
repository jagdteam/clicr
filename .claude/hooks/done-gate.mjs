#!/usr/bin/env node
// V1, Stop hook: block a "done" claim unless the same turn shows a passing verify run
// after the last edit. Exit 2 + stderr = Claude must keep working. Exit 0 = allow.
// Honours stop_hook_active: if we already blocked once this turn, let it stop (no loop).
// "Same turn" = every transcript entry after the last real user prompt.
// A verify run = a Bash call containing the repo's verify command (.claude/cloud-kit.json) whose result is not an error.
// No cloud-kit.json: prints one line and allows (never blocks on a missing config).
// The final message comes from the payload's last_assistant_message, else the transcript.
// Fails open on any internal error. Node stdlib only.
import fs from 'node:fs';
import path from 'node:path';

const CLAIM = /\b(done|complete[d]?|finished|all (?:tests? )?(?:pass(?:ing|ed)?|green)|tests? (?:pass(?:ing|ed)?|green)|ready (?:to|for)|fixed|implemented)\b/i;
const EDITS = new Set(['Edit', 'Write', 'MultiEdit', 'NotebookEdit']);

// Per-repo verify command, written by the cloud kit installer to .claude/cloud-kit.json.
function loadVerify(root) {
  try {
    const v = JSON.parse(fs.readFileSync(path.join(root, '.claude', 'cloud-kit.json'), 'utf8')).verify;
    return typeof v === 'string' && v.trim() ? v.trim() : null;
  } catch { return null; }
}

// True only when the command IS the verify command: after peeling leading `cd <x> &&` and VAR=val
// prefixes it must start with the verify string, and what follows may not use ||, ;, or | (which
// would hide a failure). `echo 'npm test'` and `npm test || true` do not count.
function isVerifyRun(command, verify) {
  let s = String(command).trim();
  // shrink-test-output rewrites a plain test command to: node '<dir>/shrink-run.mjs' '<command>'. Check the inner command.
  const wrapped = s.match(/^node\s+(?:'[^']*shrink-run\.mjs'|"[^"]*shrink-run\.mjs")\s+(?:'([^]*)'|"([^]*)")\s*$/);
  if (wrapped) s = wrapped[1] !== undefined ? wrapped[1].replace(/'\\''/g, "'") : wrapped[2].replace(/\\(["\$`])/g, '$1');
  for (;;) {
    const next = s.replace(/^cd\s+(?:"[^"]*"|'[^']*'|\S+)\s*&&\s*/, '').replace(/^[A-Za-z_]\w*=\S*\s+/, '');
    if (next === s) break;
    s = next;
  }
  if (!s.startsWith(verify)) return false;
  const rest = s.slice(verify.length);
  if (rest && !/^(\s|&&)/.test(rest)) return false; // "npm test:watch" is another command
  const beforeAnd = rest.replace(/2>&1/g, '').split('&&')[0];
  return !/[;|&`\n]/.test(beforeAnd);
}

const blocks = (m) => (Array.isArray(m && m.content) ? m.content : typeof (m && m.content) === 'string' ? [{ type: 'text', text: m.content }] : []);

try {
  const verifyCmd = loadVerify(process.env.CLAUDE_PROJECT_DIR || process.cwd());
  if (!verifyCmd) { process.stderr.write('done-gate: no .claude/cloud-kit.json verify command, skipping\n'); process.exit(0); }
  const input = JSON.parse(fs.readFileSync(0, 'utf8'));
  if (input.stop_hook_active) process.exit(0);
  const lines = fs.readFileSync(input.transcript_path, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));

  // Turn start = last user entry that is a prompt (has text, not just tool_result blocks).
  let start = 0;
  lines.forEach((e, i) => {
    if (e.type === 'user' && !e.isMeta && blocks(e.message).some((b) => b.type === 'text')) start = i + 1;
  });
  const turn = lines.slice(start);

  // The Stop payload carries the final message; the transcript may not have it flushed yet.
  const lastText = (typeof input.last_assistant_message === 'string' && input.last_assistant_message) || [...turn].reverse().filter((e) => e.type === 'assistant')
    .map((e) => blocks(e.message).filter((b) => b.type === 'text').map((b) => b.text).join('\n')).find(Boolean) || '';
  if (!CLAIM.test(lastText)) process.exit(0);

  // Walk the turn: remember the last edit and whether a passing verify came after it.
  const pending = new Map(); // tool_use id -> 'verify' | 'edit'
  let passingVerifyAfterEdit = false;
  let sawEdit = false;
  for (const e of turn) {
    for (const b of blocks(e.message)) {
      if (e.type === 'assistant' && b.type === 'tool_use') {
        if (EDITS.has(b.name)) { sawEdit = true; passingVerifyAfterEdit = false; }
        else if (b.name === 'Bash' && isVerifyRun((b.input && b.input.command) || '', verifyCmd)) pending.set(b.id, 'verify');
      } else if (e.type === 'user' && b.type === 'tool_result' && pending.get(b.tool_use_id) === 'verify') {
        const body = typeof b.content === 'string' ? b.content : JSON.stringify(b.content);
        if (!b.is_error && !/^\s*Exit code [1-9]/.test(body)) passingVerifyAfterEdit = true;
        else passingVerifyAfterEdit = false;
      }
    }
  }
  if (passingVerifyAfterEdit) process.exit(0);
  process.stderr.write(
    `Done-gate: your last message claims the work is done, but this turn has no passing verify run${sawEdit ? ' after your last edit' : ''}. ` +
    `Run the project's verify command (${verifyCmd}), fix any failure, then report the result. ` +
    `If the work is genuinely not finished, say so instead of claiming done.\n`);
  process.exit(2);
} catch (e) {
  process.stderr.write(`done-gate hook skipped: ${e.message}\n`);
  process.exit(0);
}
