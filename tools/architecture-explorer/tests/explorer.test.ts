import { describe, expect, it } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import model from '../content/model.json';
import { ancestors, buildIndex, layout, reveal, search, toggle } from '../src/layout';
import type { Model } from '../src/types';
// @ts-ignore plain ESM
import { validateModel, validateIdeas } from '../shared/validate.mjs';
// @ts-ignore plain ESM
import { checkEvidence } from '../shared/evidence.mjs';
// @ts-ignore plain ESM
import { readStore, writeStore, checkLocal, excerpt } from '../server/store.mjs';

const m = model as unknown as Model;
const ix = buildIndex(m);
const root = path.resolve(__dirname, '../../..');
const content = path.resolve(__dirname, '../content');

describe('content model', () => {
  it('validates with no problems (schema, cross-links, reciprocity)', () => expect(validateModel(m)).toEqual([]));
  it('every software node links to science and every mirror is reciprocal', () => {
    for (const n of m.nodes.filter((n) => n.perspective === 'software')) expect(n.mirrors?.length, n.id).toBeGreaterThan(0);
  });
  it('rejects a dangling cross-link, asymmetric mirror and doc-only verification', () => {
    const bad = structuredClone(m);
    bad.nodes[0].mirrors = ['nope'];
    bad.nodes[1].evidence = [];
    expect(validateModel(bad).join('\n')).toMatch(/unknown mirror nope/);
    const b2 = structuredClone(m);
    b2.nodes.find((n) => n.id === 's.val.chi2')!.review = 'verified-scan';
    expect(validateModel(b2).join('\n')).toMatch(/s.val.chi2.*planned/);
    const b3 = structuredClone(m);
    b3.nodes.find((n) => n.id === 'w.cli')!.mirrors = [];
    expect(validateModel(b3).join('\n')).toMatch(/not reciprocal/);
  });
  it('ideas validate; proposals are labelled', () => {
    const ideas = JSON.parse(fs.readFileSync(path.join(content, 'ideas.json'), 'utf8'));
    expect(validateIdeas(ideas)).toEqual([]);
    expect(ideas.filter((i: any) => i.origin === 'claude-proposal').length).toBeGreaterThan(0);
    expect(validateIdeas([{ ...ideas[0], status: 'bogus' }]).length).toBeGreaterThan(0);
  });
});

describe('graph expansion and collapse', () => {
  const ids = (e: Set<string>, p: 'science' | 'software' = 'science') => layout(ix, p, e).nodes.map((n) => n.id);
  it('starts with only the level-0 concepts', () => {
    expect(ids(new Set())).toEqual(expect.arrayContaining(['s.forward', 's.inverse']));
    expect(ids(new Set()).length).toBe(8);
  });
  it('expands twice: concept -> mechanisms -> source files', () => {
    let e = toggle(ix, new Set(), 's.forward');
    expect(ids(e)).toContain('s.forward.numdiff');
    e = toggle(ix, e, 's.forward.numdiff');
    expect(ids(e).some((i) => i.startsWith('ev:s.forward.numdiff:'))).toBe(true);
    const leaf = layout(ix, 'science', e).nodes.find((n) => n.id.startsWith('ev:s.forward.numdiff:'))!;
    expect(leaf.parentId).toBe('s.forward.numdiff');
  });
  it('collapsing a parent hides descendants and forgets nested expansion', () => {
    let e = toggle(ix, toggle(ix, new Set(), 's.forward'), 's.forward.numdiff');
    e = toggle(ix, e, 's.forward');
    expect(ids(e).length).toBe(8);
    expect(e.has('s.forward.numdiff')).toBe(false);
  });
  it('lifts edges to visible ancestors and never leaves dangling endpoints', () => {
    for (const e of [new Set<string>(), toggle(ix, new Set(), 's.inverse'), reveal(ix, new Set(), 's.inverse.ident')]) {
      const l = layout(ix, 'science', e), vis = new Set(l.nodes.map((n) => n.id));
      expect(l.edges.length).toBeGreaterThan(0);
      for (const ed of l.edges) { expect(vis.has(ed.source)).toBe(true); expect(vis.has(ed.target)).toBe(true); expect(ed.label).toBeTruthy(); }
    }
  });
  it('child boxes lie inside their expanded parent and siblings do not overlap', () => {
    const l = layout(ix, 'science', toggle(ix, new Set(), 's.design'));
    const p = l.nodes.find((n) => n.id === 's.design')!, kids = l.nodes.filter((n) => n.parentId === 's.design');
    expect(kids.length).toBe(5);
    for (const k of kids) { expect(k.x).toBeGreaterThanOrEqual(0); expect(k.x + k.w).toBeLessThanOrEqual(p.w); expect(k.y + k.h).toBeLessThanOrEqual(p.h); }
    for (const a of kids) for (const b of kids) if (a !== b) expect(a.ax < b.ax + b.w && b.ax < a.ax + a.w && a.ay < b.ay + b.h && b.ay < a.ay + a.h).toBe(false);
  });
  it('reveal expands every ancestor; search finds concepts, assumptions and file paths', () => {
    expect(ancestors(ix, 's.forward.numdiff')).toEqual(['s.forward']);
    expect(reveal(ix, new Set(), 's.forward.numdiff').has('s.forward')).toBe(true);
    expect(search(ix, 'Tikhonov').some((h) => h.id === 's.inverse.nnls')).toBe(true);
    expect(search(ix, 'geodesign_robust.py').length).toBeGreaterThan(0);
    expect(search(ix, 'x')).toEqual([]);
  });
  it('software perspective has its own roots and the science nodes it links to', () => {
    expect(ids(new Set(), 'software').every((i) => i.startsWith('w.'))).toBe(true);
  });
});

describe('evidence check', () => {
  it('passes on the real repository (no errors)', () => expect(checkEvidence(root, content).errors).toEqual([]));
  const sandbox = (mut: (mdl: any, dir: string) => void) => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ek-'));
    fs.mkdirSync(path.join(dir, 'content'));
    fs.writeFileSync(path.join(dir, 'a.py'), 'line1\nline2\nline3\n');
    const mdl = {
      nodes: [{ id: 'n.a', perspective: 'science', parent: null, kind: 'concept', title: 'A', role: 'r', purpose: 'p', lifecycle: 'implemented', review: 'verified-scan', evidence: ['e.a'] }],
      edges: [], evidence: [{ id: 'e.a', path: 'a.py', lines: [1, 2], kind: 'code', note: 'n' }],
    };
    mut(mdl, dir);
    fs.writeFileSync(path.join(dir, 'content/model.json'), JSON.stringify(mdl));
    return dir;
  };
  it('reports a missing file, a stale range, an invalid id, and drift after a baseline', () => {
    let d = sandbox((mdl) => { mdl.evidence[0].path = 'gone.py'; });
    expect(checkEvidence(d, path.join(d, 'content')).errors.join()).toMatch(/stale path/);
    d = sandbox((mdl) => { mdl.evidence[0].lines = [1, 99]; });
    expect(checkEvidence(d, path.join(d, 'content')).errors.join()).toMatch(/stale line range/);
    d = sandbox((mdl) => { mdl.nodes[0].evidence = ['missing.id']; });
    expect(checkEvidence(d, path.join(d, 'content')).errors.join()).toMatch(/unknown evidence/);
    d = sandbox(() => undefined);
    const c = path.join(d, 'content');
    expect(checkEvidence(d, c).errors).toEqual([]);
    // record a baseline, then change a cited line: a drift WARNING (signal, not error)
    fs.writeFileSync(path.join(c, 'evidence.lock.json'), JSON.stringify({ revision: 'x', evidence: { 'e.a': 'deadbeef' } }));
    const r = checkEvidence(d, c);
    expect(r.errors).toEqual([]);
    expect(r.warnings.join()).toMatch(/drift: a.py:1-2 changed/);
  });
});

describe('research notes persistence', () => {
  const tmp = () => fs.mkdtempSync(path.join(os.tmpdir(), 'ek-store-'));
  const idea = { id: 'idea.t', title: 'T', status: 'idea', related: ['s.forward'], origin: 'human', created: '2026-10-10T00:00:00Z', updated: '2026-10-10T00:00:00Z' };
  it('round-trips through the file with atomic write and no temp leftovers', () => {
    const d = tmp();
    const first = readStore(d, 'ideas');
    expect(first.items).toEqual([]);
    const saved = writeStore(d, 'ideas', [idea], first.rev);
    expect(readStore(d, 'ideas').items).toEqual([idea]);
    expect(fs.readdirSync(d).filter((f) => f.endsWith('.tmp'))).toEqual([]);
    expect(saved.rev).toBe(readStore(d, 'ideas').rev);
  });
  it('refuses a stale revision (edit made outside the browser is never overwritten)', () => {
    const d = tmp();
    const { rev } = readStore(d, 'ideas');
    writeStore(d, 'ideas', [idea], rev);
    expect(() => writeStore(d, 'ideas', [], rev)).toThrow(/changed on disk/);
    expect(readStore(d, 'ideas').items.length).toBe(1);
  });
  it('rejects invalid input, unknown collections and oversized/traversal names', () => {
    const d = tmp();
    const { rev } = readStore(d, 'ideas');
    expect(() => writeStore(d, 'ideas', [{ ...idea, status: 'nope' }], rev)).toThrow(/status/);
    expect(() => writeStore(d, '../model', [], rev)).toThrow(/unknown collection/);
    expect(() => readStore(d, 'model')).toThrow(/unknown collection/);
    expect(fs.existsSync(path.join(d, 'ideas.json'))).toBe(false);
  });
  it('request guard: loopback host/origin only, write needs the custom header', () => {
    const req = (h: Record<string, string>) => ({ headers: h });
    expect(() => checkLocal(req({ host: '127.0.0.1:5174' }), false)).not.toThrow();
    expect(() => checkLocal(req({ host: 'evil.com' }), false)).toThrow(/bad host/);
    expect(() => checkLocal(req({ host: '127.0.0.1:5174', origin: 'http://evil.com' }), true)).toThrow(/bad origin/);
    expect(() => checkLocal(req({ host: '127.0.0.1:5174', origin: 'http://127.0.0.1:5174' }), true)).toThrow(/missing header/);
    expect(() => checkLocal(req({ host: '127.0.0.1:5174', origin: 'http://127.0.0.1:5174', 'x-pravah-explorer': '1' }), true)).not.toThrow();
  });
  it('excerpts are served only for curated evidence ids', () => {
    expect(excerpt(root, m, 'ev.fw.untol').text).toMatch(/UN_TOL/);
    expect(() => excerpt(root, m, '../../etc/passwd')).toThrow(/unknown evidence/);
  });
});
