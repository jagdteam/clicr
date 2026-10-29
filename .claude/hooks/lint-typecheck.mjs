#!/usr/bin/env node
// PostToolUse (Edit|Write|MultiEdit): lint and type-check the file Claude just changed.
// Exit 0 = clean or not applicable. Exit 2 = errors; output tail goes to stderr for Claude to fix.
// If .claude/cloud-kit.json has a "lint" command, run that (60 s timeout) after a source-file edit.
// Otherwise fall back to detection on the edited file: eslint then tsc --noEmit (package.json),
// or ruff check and ruff format --check (pyproject.toml). Never runs "verify" (too slow per edit).
// No cloud-kit.json at all: one stderr line, exit 0. Fails open if tooling cannot run. Node stdlib only.
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const root = process.env.CLAUDE_PROJECT_DIR || process.cwd();
const MAX = 3000;
const SOURCE = /\.(tsx?|jsx?|mjs|cjs|py|go|rs|rb|java|kt|cs|php|sh)$/;

// Output that means the tool itself is not installed (npx --no-install, uv run): skip, never block.
const ABSENT = /canceled|not found|No matching|Failed to spawn|could not determine executable/i;
let absentNote = null;

function run(cmd, args) {
  const r = spawnSync(cmd, args, { cwd: root, encoding: 'utf8', timeout: 90000, shell: process.platform === 'win32' });
  if (r.error) throw r.error;
  const out = `${r.stdout || ''}${r.stderr || ''}`.trim();
  if (r.status !== 0 && ABSENT.test(out)) { absentNote = `${cmd} ${args.join(' ')}`; return { ok: true, out: '' }; }
  return { ok: r.status === 0, out };
}

// Reads .claude/cloud-kit.json; returns null when the file is missing or unreadable.
function loadConfig(dir) {
  try {
    return JSON.parse(fs.readFileSync(path.join(dir, '.claude', 'cloud-kit.json'), 'utf8')) || {};
  } catch { return null; }
}

function fail(problems) {
  const text = problems.join('\n\n');
  process.stderr.write(`Lint/type errors after your edit. Fix them before continuing.\n${text.slice(-MAX)}${text.length > MAX ? '\n[truncated, tail]' : ''}\n`);
  process.exit(2);
}

try {
  const cfg = loadConfig(root);
  if (!cfg) { process.stderr.write('lint-typecheck: no .claude/cloud-kit.json, skipping\n'); process.exit(0); }
  const input = JSON.parse(fs.readFileSync(0, 'utf8'));
  const file = input.tool_input && input.tool_input.file_path;
  if (!file) process.exit(0);
  const rel = path.relative(root, path.resolve(root, file));
  if (rel.startsWith('..') || rel.includes('node_modules') || rel.startsWith('.next')) process.exit(0);

  const lint = typeof cfg.lint === 'string' ? cfg.lint.trim() : '';
  if (lint) {
    if (!SOURCE.test(rel)) process.exit(0);
    const r = spawnSync(lint, { cwd: root, shell: true, encoding: 'utf8', timeout: 60000 });
    if (r.error) throw r.error;
    if (r.status !== 0) fail([`${lint} (after editing ${rel}):\n${`${r.stdout || ''}${r.stderr || ''}`.trim()}`]);
    process.exit(0);
  }

  const problems = [];
  if (fs.existsSync(path.join(root, 'package.json')) && /\.(tsx?|jsx?|mjs)$/.test(rel)) {
    const eslint = run('npx', ['--no-install', 'eslint', rel]);
    if (!eslint.ok) problems.push(`eslint ${rel}:\n${eslint.out}`);
    if (/\.tsx?$/.test(rel)) {
      const tsc = run('npx', ['--no-install', 'tsc', '--noEmit']);
      if (!tsc.ok) problems.push(`tsc --noEmit:\n${tsc.out}`);
    }
  } else if (fs.existsSync(path.join(root, 'pyproject.toml')) && rel.endsWith('.py')) {
    const ruff = run('uv', ['run', '--quiet', 'ruff', 'check', rel]);
    if (!ruff.ok) problems.push(`ruff check ${rel}:\n${ruff.out}`);
    const fmt = run('uv', ['run', '--quiet', 'ruff', 'format', '--check', '--diff', rel]);
    if (!fmt.ok) problems.push(`ruff format --check ${rel} (run \`uv run ruff format ${rel}\`):\n${fmt.out}`);
  }
  if (absentNote) process.stderr.write(`lint-typecheck: tool not installed (${absentNote}), skipped\n`);
  if (problems.length) fail(problems);
} catch (e) {
  process.stderr.write(`lint-typecheck hook skipped: ${e.message}\n`);
}
process.exit(0);
