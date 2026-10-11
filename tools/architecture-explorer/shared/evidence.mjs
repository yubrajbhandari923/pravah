// Read-only drift/consistency check of the curated map against the repository. Never rewrites content.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { validateModel, validateIdeas, validateDecisions, danglingRelated } from './validate.mjs';

const sha = (s) => crypto.createHash('sha256').update(s).digest('hex').slice(0, 16);
const readJson = (f) => JSON.parse(fs.readFileSync(f, 'utf8'));

export function gitRevision(repoRoot) {
  try { return execFileSync('git', ['rev-parse', '--short=10', 'HEAD'], { cwd: repoRoot, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim(); } catch { return 'unknown'; }
}

/** Hash of exactly what an evidence entry cites (its line range, else the whole file); null if unreadable. */
export function evidenceHash(repoRoot, e) {
  const abs = path.resolve(repoRoot, e.path);
  if (!abs.startsWith(path.resolve(repoRoot) + path.sep) || !fs.existsSync(abs) || !fs.statSync(abs).isFile()) return { missing: true };
  const text = fs.readFileSync(abs, 'utf8');
  const lines = text.split('\n');
  if (!e.lines) return { hash: sha(text), total: lines.length };
  if (e.lines[1] > lines.length) return { beyond: true, total: lines.length };
  return { hash: sha(lines.slice(e.lines[0] - 1, e.lines[1]).join('\n')), total: lines.length };
}

export function computeLock(repoRoot, model) {
  const evidence = {};
  for (const e of model.evidence) { const h = evidenceHash(repoRoot, e); if (h.hash) evidence[e.id] = h.hash; }
  return { revision: gitRevision(repoRoot), evidence };
}

/** @returns {{errors:string[], warnings:string[], info:string[]}} */
export function checkEvidence(repoRoot, contentDir) {
  const errors = [], warnings = [], info = [];
  let model;
  try { model = readJson(path.join(contentDir, 'model.json')); } catch (e) { return { errors: [`content/model.json unreadable: ${e.message}`], warnings, info }; }
  errors.push(...validateModel(model).map((m) => `schema: ${m}`));
  for (const [file, validate] of [['ideas.json', validateIdeas], ['decisions.json', validateDecisions]]) {
    const f = path.join(contentDir, file);
    if (!fs.existsSync(f)) { warnings.push(`${file} is missing (it will be created on first save)`); continue; }
    try {
      const items = readJson(f);
      errors.push(...validate(items).map((m) => `${file}: ${m}`));
      warnings.push(...danglingRelated(items, model).map((m) => `${file}: ${m} - re-link or keep as history; the explorer never deletes it`));
    } catch (e) { errors.push(`${file}: ${e.message}`); }
  }
  let lock = null;
  const lockPath = path.join(contentDir, 'evidence.lock.json');
  if (fs.existsSync(lockPath)) lock = readJson(lockPath);
  else warnings.push('no content/evidence.lock.json: run `npm run lock:evidence` after reviewing the map to record the baseline');
  const used = new Set(model.nodes.flatMap((n) => n.evidence ?? []));
  for (const e of model.evidence) {
    if (!used.has(e.id)) warnings.push(`evidence ${e.id} is not cited by any node`);
    const h = evidenceHash(repoRoot, e);
    const where = e.lines ? `${e.path}:${e.lines[0]}-${e.lines[1]}` : e.path;
    if (h.missing) { errors.push(`stale path: evidence ${e.id} -> ${e.path} does not exist`); continue; }
    if (h.beyond) { errors.push(`stale line range: evidence ${e.id} -> ${where} but the file has ${h.total} lines`); continue; }
    if (lock) {
      const old = lock.evidence?.[e.id];
      if (!old) warnings.push(`evidence ${e.id} (${where}) has no recorded hash; review it, then run lock:evidence`);
      else if (old !== h.hash) warnings.push(`drift: ${where} changed since revision ${lock.revision} (evidence ${e.id}) - re-inspect the nodes citing it; a change is a signal, not proof the claim is false`);
    }
  }
  if (lock) for (const id of Object.keys(lock.evidence ?? {})) if (!model.evidence.some((e) => e.id === id)) warnings.push(`lock lists removed evidence ${id}`);
  for (const n of model.nodes) {
    if (n.lifecycle === 'planned' && n.review === 'verified-scan') errors.push(`unsupported claim: ${n.id} is planned but marked verified-scan`);
    if (n.review === 'needs-inspection') warnings.push(`manual review requested: ${n.id}`);
  }
  info.push(`${model.nodes.length} nodes, ${model.edges.length} edges, ${model.evidence.length} evidence entries; lock revision ${lock?.revision ?? 'none'}, HEAD ${gitRevision(repoRoot)}`);
  return { errors, warnings, info };
}
