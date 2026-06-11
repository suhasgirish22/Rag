// Text utilities: tokenization, sentence splitting, chunking, token estimation.

export const STOPWORDS = new Set([
  "a","an","the","and","or","but","if","then","else","of","to","in","on","at",
  "for","from","by","with","about","as","into","through","is","are","was","were",
  "be","been","being","it","its","this","that","these","those","they","them",
  "their","he","she","his","her","we","our","you","your","i","me","my","do",
  "does","did","done","can","could","will","would","shall","should","may","might",
  "must","have","has","had","not","no","nor","so","than","too","very","just",
  "also","only","own","same","such","both","each","few","more","most","other",
  "some","any","all","there","here","when","where","why","how","what","which",
  "who","whom","while","during","before","after","above","below","up","down",
  "out","off","over","under","again","further","once","because","until","against"
]);

/** Lowercase word tokens, punctuation stripped. */
export function tokenize(text) {
  return (text.toLowerCase().match(/[a-z0-9']+/g) || [])
    .map(w => w.replace(/^'+|'+$/g, ""))
    .filter(w => w.length > 0);
}

/** Content words only (no stopwords, length > 2). */
export function contentTokens(text) {
  return tokenize(text).filter(w => !STOPWORDS.has(w) && w.length > 2);
}

/** Split text into sentences (simple heuristic, good enough for prose). */
export function splitSentences(text) {
  const parts = text
    .replace(/\s+/g, " ")
    .match(/[^.!?]+[.!?]+["')\]]*|[^.!?]+$/g) || [];
  return parts.map(s => s.trim()).filter(s => s.length > 0);
}

/**
 * Split a document into chunks of ~chunkSize words with overlap.
 * Returns [{ id, label, text, wordCount, overlapWords }].
 * The first `overlapWords` words of each chunk (except the first) are
 * repeated from the previous chunk.
 */
export function chunkText(text, chunkSize, overlap) {
  const words = text.replace(/\s+/g, " ").trim().split(" ").filter(Boolean);
  if (words.length === 0) return [];
  overlap = Math.min(overlap, Math.max(0, chunkSize - 5));
  const stride = chunkSize - overlap;

  const chunks = [];
  for (let start = 0, i = 0; start < words.length; start += stride, i++) {
    const slice = words.slice(start, start + chunkSize);
    // Avoid a tiny trailing chunk that's entirely overlap of the previous one
    if (i > 0 && slice.length <= overlap) break;
    chunks.push({
      id: `c${i}`,
      label: `Chunk ${i + 1}`,
      text: slice.join(" "),
      wordCount: slice.length,
      overlapWords: i === 0 ? 0 : Math.min(overlap, slice.length),
      isDistractor: false,
    });
    if (start + chunkSize >= words.length) break;
  }
  return chunks;
}

/** Rough token estimate (~4 characters per token, like GPT/Claude tokenizers). */
export function estimateTokens(text) {
  if (!text) return 0;
  return Math.max(1, Math.round(text.length / 4));
}
