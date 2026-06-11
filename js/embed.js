// Embedding engine: TF-IDF vectors, cosine similarity, and a 2D projection
// (classical MDS / kernel PCA on the cosine-similarity matrix) so chunks and
// queries can be drawn as points in "semantic space".

import { contentTokens } from "./text.js";

/**
 * Build a TF-IDF index over chunks.
 * Each vector is a Map(term -> weight), L2-normalized.
 */
export function buildIndex(chunks) {
  const docFreq = new Map();
  const termLists = chunks.map(c => contentTokens(c.text));

  for (const terms of termLists) {
    for (const t of new Set(terms)) docFreq.set(t, (docFreq.get(t) || 0) + 1);
  }

  const n = chunks.length;
  const idf = new Map();
  for (const [t, df] of docFreq) idf.set(t, Math.log(1 + n / df));

  const vectors = termLists.map(terms => {
    const tf = new Map();
    for (const t of terms) tf.set(t, (tf.get(t) || 0) + 1);
    const vec = new Map();
    for (const [t, f] of tf) vec.set(t, (1 + Math.log(f)) * idf.get(t));
    return l2normalize(vec);
  });

  return { idf, vectors };
}

/** Embed a query with the index's idf table (unseen words get a default idf). */
export function embedQuery(text, index) {
  const terms = contentTokens(text);
  const tf = new Map();
  for (const t of terms) tf.set(t, (tf.get(t) || 0) + 1);
  const vec = new Map();
  for (const [t, f] of tf) {
    const idf = index.idf.get(t) ?? 1.0; // unknown word: neutral weight
    vec.set(t, (1 + Math.log(f)) * idf);
  }
  return l2normalize(vec);
}

function l2normalize(vec) {
  let norm = 0;
  for (const w of vec.values()) norm += w * w;
  norm = Math.sqrt(norm);
  if (norm === 0) return vec;
  const out = new Map();
  for (const [t, w] of vec) out.set(t, w / norm);
  return out;
}

/** Cosine similarity between two sparse, L2-normalized vectors. */
export function cosine(a, b) {
  if (!a || !b) return 0;
  const [small, big] = a.size <= b.size ? [a, b] : [b, a];
  let dot = 0;
  for (const [t, w] of small) {
    const v = big.get(t);
    if (v !== undefined) dot += w * v;
  }
  return Math.max(0, Math.min(1, dot));
}

/** Top weighted terms of a vector — what the embedding "noticed". */
export function topTerms(vec, k = 6) {
  return [...vec.entries()].sort((x, y) => y[1] - x[1]).slice(0, k);
}

/** Terms two vectors share, sorted by combined weight — used to explain matches. */
export function sharedTerms(a, b, k = 5) {
  const out = [];
  for (const [t, w] of a) {
    const v = b.get(t);
    if (v !== undefined) out.push([t, w * v]);
  }
  return out.sort((x, y) => y[1] - x[1]).slice(0, k).map(([t]) => t);
}

/**
 * Project chunk vectors to 2D via classical MDS on the cosine kernel.
 * Returns { points: [[x,y],...], placeQuery(qvec) -> [x,y] }.
 * Coordinates are unscaled; callers normalize to their viewport.
 */
export function project2D(vectors) {
  const n = vectors.length;
  if (n === 0) return { points: [], placeQuery: () => [0, 0] };
  if (n === 1) return { points: [[0, 0]], placeQuery: q => [1 - cosine(q, vectors[0]), 0] };

  // Cosine kernel matrix
  const K = vectors.map(a => vectors.map(b => cosine(a, b)));

  // Double-center: Kc = (I - 1/n) K (I - 1/n)
  const rowMean = K.map(row => mean(row));
  const grand = mean(rowMean);
  const Kc = K.map((row, i) => row.map((v, j) => v - rowMean[i] - rowMean[j] + grand));

  // Top-2 eigenpairs by power iteration with deflation
  const [u1, l1] = powerIteration(Kc);
  const Kc2 = Kc.map((row, i) => row.map((v, j) => v - l1 * u1[i] * u1[j]));
  const [u2, l2] = powerIteration(Kc2);

  const s1 = Math.sqrt(Math.max(l1, 1e-9));
  const s2 = Math.sqrt(Math.max(l2, 1e-9));
  const points = u1.map((_, i) => [u1[i] * s1, u2[i] * s2]);

  // Out-of-sample (Nyström) placement for the query point
  const placeQuery = (qvec) => {
    const k = vectors.map(v => cosine(qvec, v));
    const kMean = mean(k);
    const kc = k.map((v, i) => v - kMean - rowMean[i] + grand);
    return [dot(u1, kc) / Math.max(s1, 1e-9), dot(u2, kc) / Math.max(s2, 1e-9)];
  };

  return { points, placeQuery };
}

function mean(arr) { return arr.reduce((a, b) => a + b, 0) / arr.length; }
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

function powerIteration(M, iters = 200) {
  const n = M.length;
  // Deterministic-ish init so layouts are stable across re-renders
  let v = Array.from({ length: n }, (_, i) => Math.sin(i * 12.9898 + 4.1414) * 0.5 + 0.5);
  let lambda = 0;
  for (let it = 0; it < iters; it++) {
    const w = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      const row = M[i];
      for (let j = 0; j < n; j++) w[i] += row[j] * v[j];
    }
    lambda = Math.sqrt(dot(w, w));
    if (lambda < 1e-12) return [v, 0];
    v = w.map(x => x / lambda);
  }
  // Rayleigh quotient for a signed eigenvalue
  const Mv = M.map(row => dot(row, v));
  return [v, dot(v, Mv)];
}
