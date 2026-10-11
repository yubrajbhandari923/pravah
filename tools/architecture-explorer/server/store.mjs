// File-backed store for the two explorer-owned annotation files. Nothing else is ever written by the explorer.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { validateIdeas, validateDecisions } from '../shared/validate.mjs';

export const FILES = {
  ideas: { file: 'ideas.json', validate: validateIdeas },
  decisions: { file: 'decisions.json', validate: validateDecisions },
};
export const MAX_BYTES = 2_000_000;

const rev = (text) => crypto.createHash('sha256').update(text).digest('hex').slice(0, 16);

export function readStore(dir, name) {
  const spec = FILES[name];
  if (!spec) throw httpError(404, 'unknown collection');
  const text = fs.existsSync(path.join(dir, spec.file)) ? fs.readFileSync(path.join(dir, spec.file), 'utf8') : '[]\n';
  return { items: JSON.parse(text), rev: rev(text) };
}

/** Replace the collection. `baseRev` must equal the current file revision, so an edit made outside the browser
 *  (e.g. by a refresh session) is never overwritten silently. Written atomically: temp file in the same dir + rename. */
export function writeStore(dir, name, items, baseRev) {
  const spec = FILES[name];
  if (!spec) throw httpError(404, 'unknown collection');
  const problems = spec.validate(items);
  if (problems.length) throw httpError(400, problems.slice(0, 5).join('; '));
  const cur = readStore(dir, name);
  if (baseRev !== cur.rev) throw httpError(409, 'file changed on disk since it was loaded; reload before saving');
  const text = JSON.stringify(items, null, 2) + '\n';
  if (Buffer.byteLength(text) > MAX_BYTES) throw httpError(413, 'too large');
  const target = path.join(dir, spec.file);
  const tmp = `${target}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, text);
  fs.renameSync(tmp, target);
  return { items, rev: rev(text) };
}

export function httpError(status, message) {
  return Object.assign(new Error(message), { status });
}

/** Loopback-only guard: Host and (when present) Origin must be this server on 127.0.0.1/localhost, and writes need a
 *  custom header, which a cross-site page cannot send without a CORS preflight that this server never grants. */
export function checkLocal(req, write) {
  const host = String(req.headers.host ?? '');
  const ok = /^(127\.0\.0\.1|localhost)(:\d+)?$/;
  if (!ok.test(host)) throw httpError(403, 'bad host');
  const origin = req.headers.origin;
  if (origin) {
    let h = '';
    try { h = new URL(origin).host; } catch { /* falls through */ }
    if (!ok.test(h) || h !== host) throw httpError(403, 'bad origin');
  }
  if (write && req.headers['x-pravah-explorer'] !== '1') throw httpError(403, 'missing header');
}

/** Read-only excerpt for a CURATED evidence entry (looked up by id; no path from the client). */
export function excerpt(repoRoot, model, id, context = 0) {
  const e = model.evidence.find((x) => x.id === id);
  if (!e) throw httpError(404, 'unknown evidence id');
  const abs = path.resolve(repoRoot, e.path);
  if (!abs.startsWith(path.resolve(repoRoot) + path.sep)) throw httpError(400, 'outside repository');
  if (!fs.existsSync(abs)) return { id, path: e.path, missing: true };
  const lines = fs.readFileSync(abs, 'utf8').split('\n');
  const [a, b] = e.lines ?? [1, Math.min(lines.length, 40)];
  const from = Math.max(1, a - context), to = Math.min(lines.length, Math.min(b, a + 79) + context);
  return { id, path: e.path, start: from, hlStart: a, hlEnd: Math.min(b, to), total: lines.length, text: lines.slice(from - 1, to).join('\n') };
}
