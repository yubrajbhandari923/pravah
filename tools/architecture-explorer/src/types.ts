export type Persp = 'science' | 'software';
export type Lifecycle = 'implemented' | 'experimental' | 'planned' | 'unknown';
export type Review = 'verified-scan' | 'docs-only' | 'needs-inspection';
export type Severity = 'high' | 'medium' | 'low';
export type EdgeKind = 'data' | 'control' | 'evaluates' | 'alternative' | 'feedback';

export interface Limitation { text: string; severity: Severity; impact: string }
export interface MNode {
  id: string; perspective: Persp; parent: string | null;
  kind: 'concept' | 'mechanism' | 'decision' | 'module' | 'component';
  title: string; role: string; purpose: string; technical: string;
  equation?: string; inputs?: string[]; outputs?: string[];
  chosen?: string; why?: string; alternatives?: string[]; assumptions?: string[];
  limitations?: Limitation[]; lifecycle: Lifecycle; review: Review; reviewNote?: string;
  evidence?: string[]; mirrors?: string[]; tags?: string[];
}
export interface MEdge { id: string; perspective: Persp; from: string; to: string; label: string; kind: EdgeKind }
export interface Evidence {
  id: string; path: string; lines?: [number, number]; kind: 'code' | 'test' | 'config' | 'docs' | 'notebook' | 'result';
  validation?: string; note: string;
}
export interface Model {
  schemaVersion: number; title: string; curatedAt: string; reviewLegend: Record<Review, string>;
  nodes: MNode[]; edges: MEdge[]; evidence: Evidence[];
}
export interface Idea {
  id: string; title: string; status: 'idea' | 'investigating' | 'tested' | 'accepted' | 'rejected';
  question: string; related: string[]; motivation: string; method: string; references: string[]; expected: string; notes: string;
  origin: 'human' | 'claude-proposal'; created: string; updated: string;
}
export interface Decision {
  id: string; title: string; status: 'proposed' | 'decided' | 'superseded'; choice: string; rationale: string;
  alternatives: string[]; evidence: string; related: string[]; notes: string;
  origin: 'human' | 'claude-proposal'; created: string; updated: string;
}
