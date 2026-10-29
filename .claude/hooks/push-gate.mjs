#!/usr/bin/env node
// PreToolUse hook: refuses `git push` until the repo's verify command (.claude/cloud-kit.json) passes.
// No cloud-kit.json: one stderr line, allow.
// Exit 0 = allow, exit 2 = block (stderr goes to Claude). Fails open on internal error.
// Escape hatch for hook tests only: PUSH_GATE_SKIP=1. Node stdlib only.
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const HEREDOC = /<<-?\s*(['"]?)(\w+)\1([^\n]*)\n[\s\S]*?\n[ \t]*\2(?=\s|$)/g;
const QUOTED = /"(?:[^"\\]|\\[\s\S])*"|'[^']*'/g;
const SEPARATORS = /[;&|(`\n\r]/;
const PREFIX = /^(?:sudo|env|[A-Za-z_]\w*=\S*)\s+/i;
const GIT_PUSH = /^git(?:\s+(?:-[Cc]\s+\S+|--\S+))*\s+push(?:\s+(.*))?$/;

const root = process.env.CLAUDE_PROJECT_DIR || process.cwd();
// Per-repo verify command, written by the cloud kit installer to .claude/cloud-kit.json.
function loadVerify(root) {
  try {
    const v = JSON.parse(fs.readFileSync(path.join(root, '.claude', 'cloud-kit.json'), 'utf8')).verify;
    return typeof v === 'string' && v.trim() ? v.trim() : null;
  } catch { return null; }
}

function isPush(command) {
  const flat = command.replace(HEREDOC, (_m, _q, _t, rest) => rest).replace(QUOTED, '""');
  return flat.split(SEPARATORS).some((seg) => {
    let s = seg.trim();
    while (PREFIX.test(s)) s = s.replace(PREFIX, '');
    return GIT_PUSH.test(s);
  });
}

function tail(text, n = 40) {
  return text.trim().split('\n').slice(-n).join('\n');
}

try {
  const input = JSON.parse(fs.readFileSync(0, 'utf8'));
  if (input.tool_name !== 'Bash') process.exit(0);
  const command = input.tool_input && input.tool_input.command;
  if (typeof command !== 'string' || !isPush(command)) process.exit(0);
  if (process.env.PUSH_GATE_SKIP === '1') process.exit(0);

  const verify = loadVerify(root);
  if (!verify) { fs.writeSync(2, 'push-gate: no .claude/cloud-kit.json verify command, skipping\n'); process.exit(0); }
  const r = spawnSync(verify, { cwd: root, shell: true, encoding: 'utf8', timeout: 240000 });
  if (r.error || r.status !== 0) {
    const out = tail(`${r.stdout || ''}\n${r.stderr || ''}${r.error ? '\n' + r.error.message : ''}`);
    fs.writeSync(2, `Push refused by the push gate: verify failed (${verify}). Fix it and push again.\n${out}\n`);
    process.exit(2);
  }
  process.exit(0);
} catch (e) {
  try { fs.writeSync(2, `push-gate: internal error, failing open: ${e && e.message ? e.message : e}\n`); } catch { /* nothing more */ }
  process.exit(0);
}
