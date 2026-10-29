#!/usr/bin/env node
// PreToolUse hook (fleet cloud kit): for repos WITHOUT GitHub branch protection.
// Blocks `git push` to main/master, force pushes, mass pushes and branch deletes.
// Exit 0 = allow, exit 2 = block (stderr goes to Claude). Fails open on any
// internal error, with one stderr warning line. Node stdlib only, POSIX friendly.
// Exception (the owner's own private projects): when the ONLY problem is the protected
// target (main/master) and the session has a main-go marker (fleet main-go.mjs grant),
// the push is allowed. Force, delete, mirror and all are blocked regardless.
// Self-contained on purpose: this file is copied into repos' .claude/hooks/.
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

// Same approach as the fleet lease-gate: drop heredoc bodies, blank out quoted
// strings (so `echo "git push origin main"` does not count), split on command
// separators, then peel leading sudo / env / VAR=value prefixes.
const HEREDOC = /<<-?\s*(['"]?)(\w+)\1([^\n]*)\n[\s\S]*?\n[ \t]*\2(?=\s|$)/g;
const QUOTED = /"(?:[^"\\]|\\[\s\S])*"|'[^']*'/g;
const SEPARATORS = /[;&|(`\n\r]/;
const PREFIX = /^(?:sudo|env|[A-Za-z_]\w*=\S*)\s+/i;
// `git push`, allowing git's global options in between (-C dir, -c k=v, --no-pager ...).
const GIT_PUSH = /^git(?:\s+(?:-[Cc]\s+\S+|--\S+))*\s+push(?:\s+(.*))?$/;
const PROTECTED = new Set(['main', 'master']);

// Shell one-liners whose string argument is itself a command line: bash -c '...', pwsh -Command "..."
const SHELL_C = /\b(?:bash|sh|zsh|dash|pwsh|powershell)(?:\.exe)?\s+(?:-[A-Za-z]+\s+)*?(?:-c|-Command|-command)\s+(["'])((?:\\[\s\S]|(?!\1)[^\\])*)\1/g;

// Returns the argument string (quotes removed) of every `git push` found in the command,
// including ones inside bash -c / sh -c / pwsh -c / powershell -Command strings.
// Quoted text is blanked (same length) while splitting, so `echo "git push origin main"` does not
// count, but the real arguments of a push are read from the original text with quotes dropped.
function pushArgs(command, depth = 0) {
  const text = command.replace(HEREDOC, (_m, _q, _tag, rest) => rest);
  const flat = text.replace(QUOTED, (m) => 'x'.repeat(m.length));
  const out = [];
  let pos = 0;
  for (const piece of flat.split(SEPARATORS)) {
    const start = pos;
    pos += piece.length + 1;
    let f = piece;
    let o = text.slice(start, start + piece.length);
    const lead = f.length - f.trimStart().length;
    f = f.slice(lead).trimEnd();
    o = o.slice(lead, lead + f.length);
    let m;
    while ((m = f.match(PREFIX))) { f = f.slice(m[0].length); o = o.slice(m[0].length); }
    const g = f.match(GIT_PUSH);
    if (g) out.push(o.slice(f.length - (g[1] || '').length).replace(/["']/g, ''));
  }
  if (depth < 3) {
    for (const m of text.matchAll(SHELL_C)) {
      const inner = m[1] === '"' ? m[2].replace(/\\(["\\$`])/g, '$1') : m[2];
      out.push(...pushArgs(inner, depth + 1));
    }
  }
  return out;
}

function currentBranch() {
  const cwd = process.env.CLAUDE_PROJECT_DIR || process.cwd();
  const r = spawnSync('git', ['rev-parse', '--abbrev-ref', 'HEAD'], { cwd, encoding: 'utf8' });
  if (r.error || r.status !== 0) throw new Error(`cannot read current branch: ${(r.stderr || (r.error && r.error.message) || '').trim()}`);
  return r.stdout.trim();
}

// ---- main-go marker (inline copy of fleet/main-go.mjs check; keep in step) ----
const SESSION_RE = /^[A-Za-z0-9-]{8,80}$/;
// Set FLEET_DIR (the fleet folder on the laptop) to enable the main-go exception; unset = guard stays strict.
const FLEET_DIR = process.env.FLEET_DIR || '';
const stateDir = () => process.env.FLEET_STATE_DIR || (FLEET_DIR ? path.join(FLEET_DIR, 'state') : '');
const projectDir = () => process.env.CLAUDE_PROJECT_DIR || process.cwd();

function normaliseRemote(url) {
  let s = String(url || '').trim();
  s = s.replace(/^[a-z][a-z0-9+.-]*:\/\//i, '');
  s = s.replace(/^[^@/]*@/, '');
  s = s.replace(/^([^/:]+):(?!\d+\/)/, '$1/');
  s = s.replace(/^([^/:]+):\d+\//, '$1/');
  s = s.replace(/\/+$/, '').replace(/\.git$/i, '').replace(/\/+$/, '');
  return s.toLowerCase();
}

const sameDir = (a, b) => {
  const x = path.resolve(a);
  const y = path.resolve(b);
  return process.platform === 'win32' ? x.toLowerCase() === y.toLowerCase() : x === y;
};

function stateDirExists() {
  if (!stateDir()) return false;
  try { return fs.statSync(stateDir()).isDirectory(); } catch { return false; }
}

// True only if a marker exists for this session + this repo and the repo's origin is still an own project.
function mainGoGranted(session) {
  try {
    if (!FLEET_DIR || !SESSION_RE.test(session || '')) return false;
    const dir = projectDir();
    const r = spawnSync('git', ['-C', dir, 'config', '--get', 'remote.origin.url'], { encoding: 'utf8' });
    if (r.error || r.status !== 0 || !r.stdout.trim()) return false;
    const remote = normaliseRemote(r.stdout);
    const m = JSON.parse(fs.readFileSync(path.join(stateDir(), 'main-go', `${session}.json`), 'utf8'));
    if (m.session !== session || !sameDir(m.repo, dir) || m.remote !== remote) return false;
    const own = JSON.parse(fs.readFileSync(path.join(FLEET_DIR, 'own-projects.json'), 'utf8'));
    return own.some((p) => p.remote === remote);
  } catch {
    return false;
  }
}

const bare = (ref) => ref.replace(/^\+/, '').replace(/^refs\/heads\//, '');

// Returns { reason, protectedOnly } if this push must be blocked, else null.
// protectedOnly = the sole problem is a protected target (main/master / current branch).
function checkPush(argString) {
  const hard = (reason) => ({ reason, protectedOnly: false });
  const tokens = (argString || '').split(/\s+/).filter(Boolean);
  const positionals = [];
  for (const t of tokens) {
    if (t.startsWith('--')) {
      if (/^--(force[\w-]*|mirror|all|delete)(=|$)/.test(t)) return hard(`${t.split('=')[0]} is not allowed`);
    } else if (t.startsWith('-') && t.length > 1) {
      if (/[fd]/.test(t.slice(1))) return hard(`${t} (force or delete) is not allowed`);
    } else {
      positionals.push(t);
    }
  }
  const refspecs = positionals.slice(1); // first positional is the remote
  for (const spec of positionals) {
    if (spec.startsWith('+')) return hard(`force refspec "${spec}" is not allowed`);
    if (spec.startsWith(':')) return hard(`delete refspec "${spec}" is not allowed`);
  }
  const isHead = (r) => r === 'HEAD' || r === '@';
  const needBranch = refspecs.length === 0 || refspecs.some((s) => isHead(s.includes(':') ? s.split(':').pop() : s));
  for (const spec of positionals) {
    const dst = bare(spec.includes(':') ? spec.split(':').pop() : spec);
    if (PROTECTED.has(dst)) return { reason: `pushing to "${dst}" is not allowed`, protectedOnly: true };
  }
  if (needBranch) {
    const branch = currentBranch();
    if (PROTECTED.has(branch)) return { reason: `this push would update "${branch}" (the current branch)`, protectedOnly: true };
  }
  return null;
}

try {
  const input = JSON.parse(fs.readFileSync(0, 'utf8'));
  if (input.tool_name !== 'Bash') process.exit(0);
  const command = input.tool_input && input.tool_input.command;
  if (typeof command !== 'string' || !command) process.exit(0);
  for (const args of pushArgs(command)) {
    const found = checkPush(args);
    if (found) {
      if (found.protectedOnly && mainGoGranted(input.session_id)) continue; // the owner's go for this session
      let hint = '';
      if (found.protectedOnly && SESSION_RE.test(input.session_id || '') && stateDirExists()) {
        hint = ` If the owner gave this session the go for main (own private project only), set FLEET_DIR and grant with main-go.mjs on the laptop (session ${input.session_id}, repo ${projectDir()}).`;
      }
      fs.writeSync(2,
        `Push blocked by the fleet push guard: ${found.reason}. This repo may lack GitHub branch protection, so pushes to main/master, force-pushes and branch deletes are forbidden. Push a claude/ branch instead and open a PR. Do not try to work around this.${hint}\n`);
      process.exit(2);
    }
  }
  process.exit(0);
} catch (e) {
  try { fs.writeSync(2, `guard-push: internal error, failing open: ${e && e.message ? e.message : e}\n`); } catch { /* nothing more */ }
  process.exit(0);
}
