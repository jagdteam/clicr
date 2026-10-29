#!/usr/bin/env node
// Runs the given shell command, keeps its exit code, prints only what matters:
// failures and the summary. Passing-test lines, progress dots and blanks are dropped.
// A footer shows bytes before and after so the saving is measurable.
import { spawnSync } from 'node:child_process';

const command = process.argv[2];
const r = spawnSync('sh', ['-c', command], { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
const raw = `${r.stdout || ''}${r.stderr || ''}`.replace(/\x1b\[[0-9;]*m/g, '');
const NOISE = /^\s*$|^\s*(?:✓|√|✔)|\bPASSED\b|^\s*[.sxXEF]*\.[.sxXEF]*\s*(?:\[\s*\d+%\])?$|^\s*RUN\b|^=+ test session starts|^platform |^rootdir:|^configfile:|^plugins:|^collected \d+ items|^\s*$/;
const kept = raw.split('\n').filter((l) => !NOISE.test(l)).join('\n').trim();
const status = r.status === null ? 1 : r.status;
process.stdout.write(`${kept}\n[shrink: ${Buffer.byteLength(raw)} -> ${Buffer.byteLength(kept)} bytes, exit ${status}]\n`);
process.exit(status);
