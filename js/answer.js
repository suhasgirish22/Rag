// Simulated answer generation. With no LLM in the browser, we approximate
// the "G" in RAG with extractive QA: score every sentence in the retrieved
// chunks against the query and stitch the best ones into an answer. This
// keeps the lesson honest — the answer is visibly grounded in (and only in)
// what retrieval surfaced, which is exactly the property we want to teach.

import { splitSentences, tokenize } from "./text.js";
import { embedQuery, cosine } from "./embed.js";

const MIN_SENTENCE_SCORE = 0.12;

/**
 * Build an extractive answer from retrieved chunks.
 * Returns { sentences: [{ text, chunk, score }], failed, failReason }.
 */
export function generateAnswer(question, retrieved, index) {
  if (retrieved.length === 0) {
    return { sentences: [], failed: true, failReason: "Nothing was retrieved — the database has no chunks left to search." };
  }
  const qvec = embedQuery(question, index);

  const candidates = [];
  for (const { chunk } of retrieved) {
    for (const sent of splitSentences(chunk.text)) {
      const svec = embedQuery(sent, index);
      candidates.push({ text: sent, chunk, score: cosine(qvec, svec) });
    }
  }
  candidates.sort((a, b) => b.score - a.score);

  // Pick top sentences, skipping near-duplicates (chunk overlap repeats text)
  const picked = [];
  for (const c of candidates) {
    if (picked.length >= 3 || c.score < MIN_SENTENCE_SCORE) break;
    if (!picked.some(p => wordOverlap(p.text, c.text) > 0.5)) picked.push(c);
  }
  if (picked.length === 0) {
    return {
      sentences: [],
      failed: true,
      failReason: "The retrieved chunks don't contain text relevant to this question, so a well-behaved model should say: \"The provided context does not contain enough information to answer this question.\"",
    };
  }

  // Keep sentences in their original document order for readability
  const order = new Map(candidates.map((c, i) => [c, i]));
  picked.sort((a, b) => {
    if (a.chunk !== b.chunk) return a.chunk.id < b.chunk.id ? -1 : 1;
    return order.get(a) - order.get(b);
  });

  return { sentences: picked, failed: false };
}

/** Fraction of the smaller sentence's words shared with the other sentence. */
function wordOverlap(a, b) {
  const wa = new Set(tokenize(a)), wb = new Set(tokenize(b));
  if (wa.size === 0 || wb.size === 0) return 0;
  let shared = 0;
  for (const w of wa) if (wb.has(w)) shared++;
  return shared / Math.min(wa.size, wb.size);
}

/**
 * Crude answer-quality heuristic for the experiments panel: average of the
 * best sentence score and the best retrieval score, mapped to 0–100.
 */
export function answerQuality(answer, retrieved) {
  if (answer.failed || retrieved.length === 0) return 0;
  const bestSentence = Math.max(...answer.sentences.map(s => s.score));
  const bestRetrieval = Math.max(...retrieved.map(r => r.score));
  return Math.round(Math.min(1, (bestSentence + bestRetrieval) / 2 / 0.6) * 100);
}
