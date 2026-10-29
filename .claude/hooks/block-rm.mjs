#!/usr/bin/env node
// PreToolUse (Bash): block recursive+force rm variants that the permission deny list misses
// (`rm -r -f`, `rm --recursive --force`, `rm -rf` after `cd x &&`). Adopted from the Trail of Bits
// rm regex (harness-lab c10-tob.md row 6). Push checks live in guard-push.mjs, not here.
// Exit 2 blocks, stderr goes to Claude. Not a security boundary; fails open. Node stdlib only.
import fs from 'node:fs';

// True if any command segment is `rm` with both a recursive flag (-r, -R, --recursive, also inside
// bundles like -rfv) and a force flag (-f, --force). `git rm` is not matched (tokens start with git).
function isRecursiveForceRm(cmd) {
  for (const seg of cmd.split(/[;&|(`\n\r]/)) {
    const tokens = seg.trim().split(/\s+/);
    while (tokens.length && (/^(?:sudo|env)$/i.test(tokens[0]) || /^[A-Za-z_]\w*=\S*$/.test(tokens[0]))) tokens.shift();
    if (tokens[0] !== 'rm') continue;
    const args = tokens.slice(1);
    const rec = args.some((a) => a === '--recursive' || /^-[a-zA-Z]*[rR][a-zA-Z]*$/.test(a));
    const force = args.some((a) => a === '--force' || /^-[a-zA-Z]*f[a-zA-Z]*$/.test(a));
    if (rec && force) return true;
  }
  return false;
}

try {
  const input = JSON.parse(fs.readFileSync(0, 'utf8'));
  const cmd = (input.tool_input && input.tool_input.command) || '';
  if (isRecursiveForceRm(cmd)) {
    fs.writeSync(2, 'Blocked: recursive force rm. Delete specific files, or ask the user.\n');
    process.exit(2);
  }
  process.exit(0);
} catch (e) {
  fs.writeSync(2, `block-rm: internal error, failing open: ${e.message}\n`);
  process.exit(0);
}
