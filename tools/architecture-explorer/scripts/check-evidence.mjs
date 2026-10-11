#!/usr/bin/env node
// npm run check:evidence        validate schema, cross-links, cited paths/line ranges, drift (read-only)
// npm run lock:evidence         record current hashes as the reviewed baseline (writes content/evidence.lock.json only)
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { checkEvidence, computeLock } from '../shared/evidence.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, '../../..');
const contentDir = path.resolve(here, '../content');

if (process.argv.includes('--write-lock')) {
  const { errors } = checkEvidence(repoRoot, contentDir);
  const blocking = errors.filter((e) => !e.startsWith('drift'));
  if (blocking.length) { console.error(blocking.map((e) => `ERROR ${e}`).join('\n') + '\nRefusing to record a baseline while errors remain.'); process.exit(1); }
  const model = JSON.parse(fs.readFileSync(path.join(contentDir, 'model.json'), 'utf8'));
  const lock = { generated: new Date().toISOString().slice(0, 10), ...computeLock(repoRoot, model) };
  fs.writeFileSync(path.join(contentDir, 'evidence.lock.json'), JSON.stringify(lock, null, 2) + '\n');
  console.log(`Recorded ${Object.keys(lock.evidence).length} evidence hashes at revision ${lock.revision}.`);
  process.exit(0);
}
const { errors, warnings, info } = checkEvidence(repoRoot, contentDir);
for (const m of errors) console.log(`ERROR   ${m}`);
for (const m of warnings) console.log(`WARN    ${m}`);
for (const m of info) console.log(`INFO    ${m}`);
console.log(errors.length ? `\n${errors.length} error(s), ${warnings.length} warning(s).` : `\nOK: no errors, ${warnings.length} warning(s).`);
process.exit(errors.length ? 1 : 0);
