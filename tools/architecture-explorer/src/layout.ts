// Pure layout / visibility logic for the expandable graph. No React here, so it is unit-tested directly.
import type { EdgeKind, Evidence, MEdge, MNode, Model, Persp } from './types';

export const CARD = { w: 252, h: 128 };
export const LEAF = { w: 252, h: 66 };
const GAP_X = 56, GAP_Y = 84, PAD = 26, HEAD = 58;

export const leafId = (node: string, ev: string) => `ev:${node}:${ev}`;
export const isLeaf = (id: string) => id.startsWith('ev:');

export interface Idx {
  nodes: Map<string, MNode>; ev: Map<string, Evidence>; kids: Map<string, string[]>;
  roots: Record<Persp, string[]>; edges: MEdge[]; parentOf: Map<string, string | null>;
}

export function buildIndex(m: Model): Idx {
  const nodes = new Map(m.nodes.map((n) => [n.id, n]));
  const kids = new Map<string, string[]>();
  const roots: Record<Persp, string[]> = { science: [], software: [] };
  const parentOf = new Map<string, string | null>();
  for (const n of m.nodes) {
    parentOf.set(n.id, n.parent);
    if (n.parent) kids.set(n.parent, [...(kids.get(n.parent) ?? []), n.id]);
    else roots[n.perspective].push(n.id);
  }
  // evidence leaves (implementation level) hang under nodes that have no modelled children
  const ev = new Map(m.evidence.map((e) => [e.id, e]));
  for (const n of m.nodes) {
    if (!kids.has(n.id) && n.parent && n.evidence?.length) {
      const ids = n.evidence.map((e) => leafId(n.id, e));
      kids.set(n.id, ids);
      ids.forEach((i) => parentOf.set(i, n.id));
    }
  }
  return { nodes, ev, kids, roots, edges: m.edges, parentOf };
}

export const hasKids = (ix: Idx, id: string) => (ix.kids.get(id)?.length ?? 0) > 0;

/** Nearest-first chain of ancestors (works for evidence leaves too). */
export function ancestors(ix: Idx, id: string): string[] {
  const out: string[] = [];
  for (let p = ix.parentOf.get(id); p; p = ix.parentOf.get(p)) out.push(p);
  return out;
}

export function descendants(ix: Idx, id: string): string[] {
  return (ix.kids.get(id) ?? []).flatMap((k) => [k, ...descendants(ix, k)]);
}

/** Expand ids so that `id` becomes visible (all its ancestors expanded). */
export function reveal(ix: Idx, expanded: Set<string>, id: string): Set<string> {
  const next = new Set(expanded);
  ancestors(ix, id).forEach((a) => next.add(a));
  return next;
}

/** Collapsing a node also forgets expanded descendants, so re-expanding starts from the overview of that node. */
export function toggle(ix: Idx, expanded: Set<string>, id: string): Set<string> {
  const next = new Set(expanded);
  if (next.has(id)) { next.delete(id); descendants(ix, id).forEach((d) => next.delete(d)); } else if (hasKids(ix, id)) next.add(id);
  return next;
}

export interface LNode { id: string; x: number; y: number; w: number; h: number; parentId?: string; expanded: boolean; leaf: boolean; ax: number; ay: number }
export interface LEdge { id: string; source: string; target: string; label: string; kind: EdgeKind; lifted: boolean; sh: string; th: string }
export interface Layout { nodes: LNode[]; edges: LEdge[] }

interface Placed { w: number; h: number; pos: Map<string, { x: number; y: number }> }

export function layout(ix: Idx, persp: Persp, expanded: Set<string>): Layout {
  const edges = ix.edges.filter((e) => e.perspective === persp);
  const sizes = new Map<string, { w: number; h: number }>();
  const inner = new Map<string, Placed>();

  const isOpen = (id: string) => expanded.has(id) && hasKids(ix, id);

  function liftTo(id: string, members: Set<string>): string | null {
    for (let c: string | null | undefined = id; c; c = ix.parentOf.get(c)) if (members.has(c)) return c;
    return null;
  }

  function place(members: string[]): Placed {
    members.forEach(size);
    const set = new Set(members);
    const pairs = new Map<string, [string, string]>();
    for (const e of edges) {
      const a = liftTo(e.from, set), b = liftTo(e.to, set);
      if (a && b && a !== b) pairs.set(`${a}>${b}`, [a, b]);
    }
    const out = new Map<string, string[]>(members.map((m) => [m, []]));
    for (const [a, b] of pairs.values()) out.get(a)!.push(b);
    // DFS in curated order: edges that close a cycle are "back" edges and do not constrain the layering
    const state = new Map<string, 0 | 1 | 2>(), back = new Set<string>(), post: string[] = [];
    const dfs = (v: string) => {
      state.set(v, 1);
      for (const w of out.get(v)!) {
        if (state.get(w) === 1) back.add(`${v}>${w}`);
        else if (!state.get(w)) dfs(w);
      }
      state.set(v, 2); post.push(v);
    };
    members.forEach((m) => { if (!state.get(m)) dfs(m); });
    const layer = new Map<string, number>(members.map((m) => [m, 0]));
    const preds = new Map<string, string[]>(members.map((m) => [m, []]));
    for (const v of [...post].reverse())
      for (const w of out.get(v)!) if (!back.has(`${v}>${w}`)) { layer.set(w, Math.max(layer.get(w)!, layer.get(v)! + 1)); preds.get(w)!.push(v); }
    if (pairs.size === 0) {
      const per = members.every(isLeaf) ? 2 : 3;
      members.forEach((m, i) => layer.set(m, Math.floor(i / per)));
    }
    const gx = members.every(isLeaf) ? 24 : GAP_X, gy = members.every(isLeaf) ? 22 : GAP_Y;
    // top-to-bottom: a layer is a row; items inside a row are ordered left to right
    const nLayers = Math.max(0, ...layer.values()) + 1;
    const rows: string[][] = Array.from({ length: nLayers }, () => []);
    members.forEach((m) => rows[layer.get(m)!].push(m));
    const pos = new Map<string, { x: number; y: number }>();
    const rowW: number[] = [], rowH: number[] = [];
    const xc = new Map<string, number>();
    rows.forEach((row, li) => {
      const bary = (m: string) => { const p = preds.get(m)!.filter((q) => xc.has(q)); return p.length ? p.reduce((t, q) => t + xc.get(q)!, 0) / p.length : null; };
      row.sort((a, b) => { const oa = bary(a), ob = bary(b); return oa == null || ob == null ? members.indexOf(a) - members.indexOf(b) : oa - ob; });
      let x = 0;
      for (const m of row) { pos.set(m, { x, y: 0 }); xc.set(m, x + sizes.get(m)!.w / 2); x += sizes.get(m)!.w + gx; }
      rowW[li] = Math.max(0, x - gx); rowH[li] = Math.max(0, ...row.map((m) => sizes.get(m)!.h));
    });
    const Wmax = Math.max(0, ...rowW);
    let y = 0;
    rows.forEach((row, li) => {
      const dx = (Wmax - rowW[li]) / 2;
      for (const m of row) { const p = pos.get(m)!; p.x += dx; p.y = y + (rowH[li] - sizes.get(m)!.h) / 2; }
      y += rowH[li] + gy;
    });
    return { w: Wmax, h: Math.max(0, y - gy), pos };
  }

  function size(id: string) {
    if (sizes.has(id)) return;
    if (isLeaf(id)) { sizes.set(id, LEAF); return; }
    if (!isOpen(id)) { sizes.set(id, CARD); return; }
    const p = place(ix.kids.get(id)!);
    inner.set(id, p);
    sizes.set(id, { w: Math.max(CARD.w + 2 * PAD, p.w + 2 * PAD), h: HEAD + p.h + PAD });
  }

  const top = place(ix.roots[persp]);
  const nodes: LNode[] = [];
  const emit = (id: string, x: number, y: number, parentId: string | undefined, ox: number, oy: number) => {
    const s = sizes.get(id)!;
    nodes.push({ id, x, y, w: s.w, h: s.h, parentId, expanded: isOpen(id), leaf: isLeaf(id), ax: ox + x, ay: oy + y });
    const p = inner.get(id);
    if (p) for (const k of ix.kids.get(id)!) { const q = p.pos.get(k)!; emit(k, q.x + PAD, q.y + HEAD, id, ox + x, oy + y); }
  };
  for (const id of ix.roots[persp]) { const q = top.pos.get(id)!; emit(id, q.x, q.y, undefined, 0, 0); }

  const vis = new Map(nodes.map((n) => [n.id, n]));
  const lift = (id: string) => { for (let c: string | null | undefined = id; c; c = ix.parentOf.get(c)) if (vis.has(c)) return c; return null; };
  const seen = new Set<string>(), outEdges: LEdge[] = [];
  for (const e of edges) {
    const a = lift(e.from), b = lift(e.to);
    if (!a || !b || a === b || ancestors(ix, a).includes(b) || ancestors(ix, b).includes(a)) continue;
    const key = `${a}>${b}>${e.label}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const A = vis.get(a)!, B = vis.get(b)!;
    let sh = 'sb', th = 'tt';
    if (B.ay < A.ay + A.h - 1) {
      if (B.ay + B.h <= A.ay + 1) { sh = 'st'; th = 'tb'; } else if (B.ax >= A.ax + A.w - 1) { sh = 'sr'; th = 'tl'; } else { sh = 'sl'; th = 'tr'; }
    }
    outEdges.push({ id: e.id, source: a, target: b, label: e.label, kind: e.kind, lifted: a !== e.from || b !== e.to, sh, th });
  }
  return { nodes, edges: outEdges };
}

/** Which nodes match a predicate themselves or through a descendant (so containers stay lit). */
export function matchSet(ix: Idx, pred: (n: MNode) => boolean): Set<string> {
  const ok = new Set<string>();
  for (const n of ix.nodes.values()) if (pred(n)) { ok.add(n.id); ancestors(ix, n.id).forEach((a) => ok.add(a)); }
  return ok;
}

export interface Hit { id: string; title: string; where: string; persp: Persp }
export function search(ix: Idx, q: string, limit = 14): Hit[] {
  const t = q.trim().toLowerCase();
  if (t.length < 2) return [];
  const hits: (Hit & { score: number })[] = [];
  for (const n of ix.nodes.values()) {
    const ev = (n.evidence ?? []).map((e) => ix.ev.get(e)?.path ?? '');
    const fields: [string, string][] = [
      ['title', n.title], ['tags', (n.tags ?? []).join(' ')], ['summary', n.role], ['purpose', n.purpose], ['method', n.technical],
      ['assumption', (n.assumptions ?? []).join(' ')], ['limitation', (n.limitations ?? []).map((l) => l.text).join(' ')],
      ['decision', `${n.chosen ?? ''} ${n.why ?? ''} ${(n.alternatives ?? []).join(' ')}`], ['file', ev.join(' ')], ['equation', n.equation ?? ''],
      ['io', [...(n.inputs ?? []), ...(n.outputs ?? [])].join(' ')],
    ];
    const f = fields.find(([, v]) => v.toLowerCase().includes(t));
    if (f) hits.push({ id: n.id, title: n.title, where: f[0] === 'file' ? `file: ${ev.find((p) => p.toLowerCase().includes(t))}` : f[0], persp: n.perspective, score: f[0] === 'title' ? 0 : f[0] === 'tags' ? 1 : f[0] === 'file' ? 2 : 3 });
  }
  return hits.sort((a, b) => a.score - b.score || a.title.localeCompare(b.title)).slice(0, limit);
}
