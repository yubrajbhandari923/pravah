import type { Decision, Idea } from './types';

const H = { 'content-type': 'application/json', 'x-pravah-explorer': '1' };
export type Kind = 'ideas' | 'decisions';

export async function load<T extends Idea | Decision>(kind: Kind): Promise<{ items: T[]; rev: string }> {
  const r = await fetch(`/api/${kind}`, { headers: { 'x-pravah-explorer': '1' } });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error ?? `HTTP ${r.status}`);
  return r.json();
}

export async function save<T extends Idea | Decision>(kind: Kind, items: T[], rev: string): Promise<{ items: T[]; rev: string }> {
  const r = await fetch(`/api/${kind}`, { method: 'PUT', headers: H, body: JSON.stringify({ items, rev }) });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw Object.assign(new Error(body.error ?? `HTTP ${r.status}`), { status: r.status });
  return body;
}

export interface Excerpt { id: string; path: string; missing?: boolean; start?: number; hlStart?: number; hlEnd?: number; total?: number; text?: string }
export async function excerpt(id: string): Promise<Excerpt> {
  const r = await fetch(`/api/excerpt?id=${encodeURIComponent(id)}`, { headers: { 'x-pravah-explorer': '1' } });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}
