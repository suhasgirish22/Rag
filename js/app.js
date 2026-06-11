// "How RAG Actually Works" — main orchestrator.
// One reactive pipeline: any control change recomputes
// chunking → embedding → projection → retrieval → prompt → answer,
// then re-renders every unlocked step.

import { chunkText, estimateTokens } from "./text.js";
import { buildIndex, embedQuery, cosine, topTerms, sharedTerms, project2D } from "./embed.js";
import { generateAnswer, answerQuality } from "./answer.js";
import { renderQuiz } from "./quiz.js";
import { renderSpace, chunkColor } from "./viz.js";
import { SAMPLE_DOC, SAMPLE_QUESTIONS, DISTRACTOR_CHUNKS, QUIZZES } from "./content.js";

const $ = id => document.getElementById(id);

const STEP_NAMES = ["Document", "Chunks", "Embeddings", "Question", "Query point", "Retrieval", "Prompt", "Answer", "Experiments"];

const state = {
  docText: "",
  chunkSize: 45,
  overlap: 10,
  topK: 3,
  question: "",
  removed: new Set(),     // chunk ids deleted in experiments
  distractors: false,     // off-topic chunks injected?
  unlocked: 1,            // highest visible step
  quizShown: new Set(),
  expDone: new Set(),     // completed experiment tasks
  selectedChunk: null,    // inspector selection in step 3
};

// Derived pipeline outputs, refreshed by recompute()
let D = {};

function recompute() {
  const base = chunkText(state.docText, state.chunkSize, state.overlap);
  const extras = state.distractors
    ? DISTRACTOR_CHUNKS.map((text, i) => ({
        id: `x${i}`, label: `Off-topic ${i + 1}`, text,
        wordCount: text.split(/\s+/).length, overlapWords: 0, isDistractor: true,
      }))
    : [];
  const all = [...base, ...extras];
  all.forEach((c, i) => { c.color = chunkColor(i, c.isDistractor); });

  const active = all.filter(c => !state.removed.has(c.id));
  const index = buildIndex(active);
  const projection = project2D(index.vectors);
  active.forEach((c, i) => {
    c.vec = index.vectors[i];
    [c.px, c.py] = projection.points[i];
  });

  let qvec = null, qpos = null, scored = [], retrieved = [], answer = null;
  if (state.question) {
    qvec = embedQuery(state.question, index);
    qpos = projection.placeQuery(qvec);
    scored = active
      .map(c => ({ chunk: c, score: cosine(qvec, c.vec) }))
      .sort((a, b) => b.score - a.score);
    retrieved = scored.slice(0, Math.min(state.topK, scored.length));
    answer = generateAnswer(state.question, retrieved, index);
  }

  D = { all, active, index, qvec, qpos, scored, retrieved, answer };
}

function rerender() {
  recompute();
  renderNav();
  if (state.unlocked >= 2) renderChunksStep();
  if (state.unlocked >= 3) renderEmbeddingStep();
  if (state.unlocked >= 4) renderQuestionStep();
  if (state.question && state.unlocked >= 5) renderQuerySpaceStep();
  if (state.question && state.unlocked >= 6) renderRetrievalStep();
  if (state.question && state.unlocked >= 7) renderPromptStep();
  if (state.question && state.unlocked >= 8) renderAnswerStep();
  if (state.question && state.unlocked >= 9) renderExperimentsStep();
}

/* ---------- Step unlocking, nav, quizzes ---------- */

function unlock(step) {
  if (step <= state.unlocked) return;
  state.unlocked = step;
  $(`step-${step}`).classList.remove("locked");
  rerender();
  setTimeout(() => $(`step-${step}`).scrollIntoView({ behavior: "smooth", block: "start" }), 60);
}

function renderNav() {
  const nav = $("progress-nav");
  nav.innerHTML = "";
  STEP_NAMES.forEach((name, i) => {
    const step = i + 1;
    const btn = document.createElement("button");
    btn.textContent = `${step}. ${name}`;
    btn.disabled = step > state.unlocked;
    if (step < state.unlocked) btn.classList.add("done");
    if (step === state.unlocked) btn.classList.add("current");
    btn.addEventListener("click", () =>
      $(`step-${step}`).scrollIntoView({ behavior: "smooth", block: "start" }));
    nav.appendChild(btn);
  });
}

function showQuiz(step, onContinue) {
  if (state.quizShown.has(step)) return;
  state.quizShown.add(step);
  renderQuiz($(`quiz-${step}`), QUIZZES[step], onContinue);
}

/* ---------- Step 1: document ---------- */

const MAX_CHARS = 12000;

function setupStep1() {
  const docInput = $("doc-input");
  const confirmBtn = $("confirm-doc-btn");

  const refreshMeta = () => {
    const text = docInput.value.trim();
    const words = text ? text.split(/\s+/).length : 0;
    $("doc-meta").textContent = text
      ? `${words.toLocaleString()} words · ~${estimateTokens(text).toLocaleString()} tokens`
      : "";
    confirmBtn.disabled = words < 40;
    if (text && words < 40) $("doc-meta").textContent += " — add a bit more text (at least ~40 words) so chunking is interesting";
  };

  docInput.addEventListener("input", refreshMeta);

  $("use-sample-btn").addEventListener("click", () => {
    docInput.value = SAMPLE_DOC;
    refreshMeta();
    docInput.classList.add("flash");
    setTimeout(() => docInput.classList.remove("flash"), 1000);
  });

  $("file-input").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    let text = await file.text();
    if (text.length > MAX_CHARS) {
      text = text.slice(0, MAX_CHARS);
      $("doc-meta").textContent = `File truncated to ${MAX_CHARS.toLocaleString()} characters to keep the demo snappy.`;
    }
    docInput.value = text;
    refreshMeta();
  });

  confirmBtn.addEventListener("click", () => {
    state.docText = docInput.value.trim().slice(0, MAX_CHARS);
    state.removed.clear();
    state.question = "";
    showQuiz(1, () => unlock(2));
    rerender();
    $("quiz-1").scrollIntoView({ behavior: "smooth", block: "center" });
  });

  $("start-btn").addEventListener("click", () =>
    $("step-1").scrollIntoView({ behavior: "smooth", block: "start" }));
}

/* ---------- Step 2: chunking ---------- */

function setupStep2() {
  const size = $("chunk-size"), overlap = $("chunk-overlap");
  size.value = state.chunkSize;
  overlap.value = state.overlap;
  size.addEventListener("input", () => {
    state.chunkSize = +size.value;
    state.removed.clear(); // chunk ids change when re-chunking
    markExp("size");
    rerender();
  });
  overlap.addEventListener("input", () => {
    state.overlap = +overlap.value;
    state.removed.clear();
    rerender();
  });
}

function renderChunksStep() {
  $("chunk-size-label").textContent = state.chunkSize;
  $("chunk-overlap-label").textContent = state.overlap;
  $("chunk-size").value = state.chunkSize;
  $("chunk-overlap").value = state.overlap;

  const base = D.all.filter(c => !c.isDistractor);
  const avgTok = base.length
    ? Math.round(base.reduce((s, c) => s + estimateTokens(c.text), 0) / base.length)
    : 0;
  $("chunk-stats").innerHTML = `
    <span><strong>${base.length}</strong> chunks</span>
    <span>~<strong>${avgTok}</strong> tokens each</span>
    <span>overlap: <strong>${state.overlap}</strong> words</span>`;

  const grid = $("chunk-grid");
  grid.innerHTML = "";
  base.forEach((c, i) => {
    const card = document.createElement("div");
    card.className = "chunk-card";
    card.style.borderLeftColor = c.color;
    card.style.animationDelay = `${i * 50}ms`;
    const words = c.text.split(" ");
    const ovl = words.slice(0, c.overlapWords).join(" ");
    const rest = words.slice(c.overlapWords).join(" ");
    card.innerHTML = `
      <span class="chunk-tag" style="color:${c.color}">${c.label}</span><br>
      ${ovl ? `<span class="overlap-mark" title="Overlap repeated from the previous chunk">${esc(ovl)}</span> ` : ""}${esc(rest)}`;
    grid.appendChild(card);
  });

  showQuiz(2, () => unlock(3));
}

/* ---------- Step 3: embeddings ---------- */

function renderEmbeddingStep() {
  renderSpace($("embed-space"), {
    items: spaceItems(),
    onClick: id => { state.selectedChunk = id; renderEmbeddingStep(); },
    selectedId: state.selectedChunk,
  });
  renderInspector();
  showQuiz(3, () => unlock(4));
}

function spaceItems() {
  return D.active.map(c => ({
    id: c.id, label: c.label.replace("Chunk ", "C").replace("Off-topic ", "X"),
    color: c.color, x: c.px, y: c.py,
  }));
}

function renderInspector() {
  const box = $("chunk-inspector");
  const c = D.active.find(c => c.id === state.selectedChunk);
  if (!c) {
    box.innerHTML = `<p class="inspector-empty">Click a dot to inspect that chunk's embedding.</p>`;
    return;
  }
  const terms = topTerms(c.vec, 7);
  const maxW = terms.length ? terms[0][1] : 1;
  box.innerHTML = `
    <h4 style="color:${c.color}">${c.label}</h4>
    <p class="insp-text">“${esc(truncate(c.text, 220))}”</p>
    <p><strong>What its embedding "noticed"</strong><br>
    <span class="mini-hint">The words that most define this chunk's position. Rare, distinctive words count more than common ones.</span></p>
    ${terms.map(([t, w]) => `
      <div class="term-bar-row">
        <span class="term">${esc(t)}</span>
        <div class="term-bar-track"><div class="term-bar-fill" style="width:${Math.round(w / maxW * 100)}%"></div></div>
      </div>`).join("")}`;
}

/* ---------- Step 4: question ---------- */

function setupStep4() {
  const input = $("question-input");
  const ask = () => {
    const q = input.value.trim();
    if (!q) return;
    state.question = q;
    if (state.unlocked < 5) {
      rerender();
      showQuiz(4, () => unlock(5));
      $("quiz-4").scrollIntoView({ behavior: "smooth", block: "center" });
    } else {
      rerender();
    }
  };
  $("ask-btn").addEventListener("click", ask);
  input.addEventListener("keydown", e => { if (e.key === "Enter") ask(); });
}

function renderQuestionStep() {
  const wrap = $("suggested-questions");
  if (wrap.childElementCount === 0 && state.docText === SAMPLE_DOC) {
    SAMPLE_QUESTIONS.forEach(q => {
      const b = document.createElement("button");
      b.className = "btn btn-ghost";
      b.textContent = `“${q}”`;
      b.addEventListener("click", () => {
        $("question-input").value = q;
        $("ask-btn").click();
      });
      wrap.appendChild(b);
    });
  }
}

/* ---------- Step 5: query embedding ---------- */

function renderQuerySpaceStep() {
  renderSpace($("query-space"), {
    items: spaceItems(),
    query: { x: D.qpos[0], y: D.qpos[1] },
  });
  const nearest = D.scored[0];
  $("query-caption").innerHTML = nearest
    ? `The question <em>“${esc(truncate(state.question, 70))}”</em> landed nearest to
       <strong style="color:${nearest.chunk.color}">${nearest.chunk.label}</strong> —
       the embedding thinks that's the most related text. Next step: make that official.`
    : "";
  showQuiz(5, () => unlock(6));
}

/* ---------- Step 6: retrieval ---------- */

function setupStep6() {
  const slider = $("topk-slider");
  slider.value = state.topK;
  slider.addEventListener("input", () => {
    state.topK = +slider.value;
    markExp("topk");
    rerender();
  });
}

function renderRetrievalStep() {
  $("topk-label").textContent = state.topK;
  $("topk-slider").value = state.topK;

  const retrievedIds = new Set(D.retrieved.map(r => r.chunk.id));
  renderSpace($("retrieval-space"), {
    items: D.active.map(c => ({
      id: c.id, label: c.label.replace("Chunk ", "C").replace("Off-topic ", "X"),
      color: c.color, x: c.px, y: c.py,
      dim: !retrievedIds.has(c.id), ring: retrievedIds.has(c.id),
    })),
    query: { x: D.qpos[0], y: D.qpos[1] },
    links: D.retrieved.map(r => ({ toId: r.chunk.id, score: r.score })),
  });

  renderRetrievalList($("retrieval-list"), D.retrieved, true);

  const rejected = D.scored.slice(state.topK);
  if (rejected.length) {
    const note = document.createElement("p");
    note.className = "rejected-note";
    note.textContent = `…${rejected.length} other chunk${rejected.length > 1 ? "s" : ""} scored lower (best rejected: ${rejected[0].chunk.label} at ${rejected[0].score.toFixed(2)}) and won't be sent to the model.`;
    $("retrieval-list").appendChild(note);
  }

  showQuiz(6, () => unlock(7));
}

function renderRetrievalList(el, retrieved, withWhy) {
  el.innerHTML = "";
  retrieved.forEach((r, i) => {
    const item = document.createElement("div");
    item.className = "retrieval-item";
    item.style.borderLeftColor = r.chunk.color;
    item.style.animationDelay = `${i * 100}ms`;
    const shared = sharedTerms(D.qvec, r.chunk.vec, 4);
    const why = shared.length
      ? `Selected because the question and this chunk share the distinctive word${shared.length > 1 ? "s" : ""}
         ${shared.map(t => `<span class="shared-term">${esc(t)}</span>`).join(" ")}`
      : `Selected by overall similarity — it's the closest remaining point, though it shares no distinctive words with the question (a weak match).`;
    item.innerHTML = `
      <div class="ri-head">
        <span class="ri-rank" style="color:${r.chunk.color}">#${i + 1} · ${r.chunk.label}</span>
        <div class="ri-score-track"><div class="ri-score-fill" style="width:0; background:${r.chunk.color}"></div></div>
        <span class="ri-score-num">${r.score.toFixed(3)}</span>
      </div>
      <p class="ri-text">“${esc(truncate(r.chunk.text, 180))}”</p>
      ${withWhy ? `<p class="ri-why">💡 ${why}</p>` : ""}`;
    el.appendChild(item);
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        item.querySelector(".ri-score-fill").style.width = `${Math.round(r.score * 100)}%`;
      }));
  });
  if (retrieved.length === 0) {
    el.innerHTML = `<p class="rejected-note">Nothing to retrieve — the database is empty.</p>`;
  }
}

/* ---------- Step 7: prompt construction ---------- */

const SYSTEM_TEXT = `You are a helpful assistant. Answer the user's question using ONLY the context below. If the context does not contain the answer, say "I don't have enough information to answer that."

Context:
`;

function promptParts() {
  const parts = [{ kind: "system", label: "Instructions", color: "#9aa6b8", text: SYSTEM_TEXT }];
  for (const r of D.retrieved) {
    parts.push({ kind: "chunk", label: r.chunk.label, color: r.chunk.color, text: `[${r.chunk.label}]\n${r.chunk.text}\n\n` });
  }
  parts.push({ kind: "question", label: "Question", color: "#ffb454", text: `Question: ${state.question}\nAnswer:` });
  return parts;
}

function renderPromptStep() {
  const parts = promptParts();
  const counts = parts.map(p => estimateTokens(p.text));
  const total = counts.reduce((a, b) => a + b, 0);

  const bar = $("token-bar");
  bar.innerHTML = "";
  parts.forEach((p, i) => {
    const seg = document.createElement("div");
    seg.className = "tb-seg";
    seg.style.width = `${(counts[i] / total) * 100}%`;
    seg.style.background = p.color;
    seg.title = `${p.label}: ~${counts[i]} tokens`;
    if (counts[i] / total > 0.08) seg.textContent = counts[i];
    bar.appendChild(seg);
  });
  $("token-legend").innerHTML = parts.map((p, i) =>
    `<span><span class="tl-swatch" style="background:${p.color}"></span>${p.label}: ~${counts[i]} tok</span>`
  ).join("") + `<span><strong>Total: ~${total} tokens</strong> (≈4 characters per token — real tokenizers vary slightly)</span>`;

  const view = $("prompt-view");
  view.innerHTML = parts.map((p, i) => {
    const cls = p.kind === "system" ? "p-system" : p.kind === "question" ? "p-question" : "p-chunk";
    const style = p.kind === "chunk" ? ` style="border-left-color:${p.color}; animation-delay:${i * 150}ms"` : ` style="animation-delay:${i * 150}ms"`;
    return `<span class="${cls} prompt-seg"${style}>${esc(p.text)}</span>`;
  }).join("");

  showQuiz(7, () => unlock(8));
}

/* ---------- Step 8: answer generation ---------- */

function renderAnswerStep() {
  renderAnswerInto($("answer-box"), $("answer-sources"), true);
  showQuiz(8, () => unlock(9));
}

function renderAnswerInto(box, sourcesEl, withSources) {
  const a = D.answer;
  box.innerHTML = "";
  if (!a || a.failed) {
    box.innerHTML = `<span class="answer-fallback typing">⚠️ ${esc(a ? a.failReason : "Ask a question first.")}</span>`;
    if (sourcesEl) sourcesEl.innerHTML = "";
    return;
  }
  a.sentences.forEach((s, i) => {
    const span = document.createElement("span");
    span.className = "answer-sentence typing";
    span.style.borderBottomColor = s.chunk.color;
    span.style.animationDelay = `${i * 600}ms`;
    span.title = `Supported by ${s.chunk.label} (similarity ${s.score.toFixed(2)})`;
    span.textContent = s.text;
    box.appendChild(span);
    box.appendChild(document.createTextNode(" "));
  });
  if (withSources && sourcesEl) {
    const byChunk = new Map();
    for (const s of a.sentences) {
      if (!byChunk.has(s.chunk)) byChunk.set(s.chunk, []);
      byChunk.get(s.chunk).push(s);
    }
    sourcesEl.innerHTML = [...byChunk.entries()].map(([chunk, sents]) => `
      <div class="source-row" style="border-left-color:${chunk.color}">
        <strong>${chunk.label}</strong> supports ${sents.length} sentence${sents.length > 1 ? "s" : ""} of this answer
        — “${esc(truncate(chunk.text, 140))}”
      </div>`).join("");
  }
}

/* ---------- Step 9: experiments ---------- */

const EXP_TASKS = [
  { id: "size", label: "✂️ Change the chunk size" },
  { id: "topk", label: "🔢 Change top-k" },
  { id: "remove", label: "🗑️ Remove the top-scoring chunk" },
  { id: "distract", label: "🍕 Add irrelevant chunks" },
];

function markExp(id) {
  if (state.unlocked < 9 || state.expDone.has(id)) return;
  state.expDone.add(id);
}

function setupStep9() {
  const size = $("exp-chunk-size"), topk = $("exp-topk");
  size.value = state.chunkSize;
  topk.value = state.topK;
  size.addEventListener("input", () => {
    state.chunkSize = +size.value;
    state.removed.clear();
    markExp("size");
    rerender();
  });
  topk.addEventListener("input", () => {
    state.topK = +topk.value;
    markExp("topk");
    rerender();
  });
  $("exp-distractors").addEventListener("change", (e) => {
    state.distractors = e.target.checked;
    markExp("distract");
    rerender();
  });
  $("restart-btn").addEventListener("click", () => location.reload());
}

function renderExperimentsStep() {
  // Checklist
  $("experiment-checklist").innerHTML = EXP_TASKS.map(t =>
    `<div class="exp-task ${state.expDone.has(t.id) ? "done" : ""}">${state.expDone.has(t.id) ? "✓ " : ""}${t.label}</div>`
  ).join("");

  // Controls (kept in sync with steps 2 & 6)
  $("exp-chunk-size").value = state.chunkSize;
  $("exp-chunk-size-label").textContent = state.chunkSize;
  $("exp-chunk-count").textContent = `${D.all.filter(c => !c.isDistractor).length} chunks`;
  $("exp-topk").value = state.topK;
  $("exp-topk-label").textContent = state.topK;
  $("exp-distractors").checked = state.distractors;

  // Chunk include/exclude toggles
  const toggles = $("exp-chunk-toggles");
  toggles.innerHTML = "";
  const topId = D.scored[0]?.chunk.id;
  for (const c of D.all) {
    const off = state.removed.has(c.id);
    const lab = document.createElement("label");
    lab.className = off ? "off" : "";
    lab.style.borderColor = c.color;
    lab.innerHTML = `<input type="checkbox" ${off ? "" : "checked"} hidden>
      <span style="color:${c.color}">●</span> ${c.label}${c.id === topId ? " ★top" : ""}`;
    lab.querySelector("input").addEventListener("change", (e) => {
      if (e.target.checked) state.removed.delete(c.id);
      else {
        state.removed.add(c.id);
        if (c.id === topId) markExp("remove");
      }
      rerender();
    });
    toggles.appendChild(lab);
  }

  // Live result
  $("exp-question").textContent = `“${state.question}”`;
  renderRetrievalList($("exp-retrieval"), D.retrieved, false);
  renderAnswerInto($("exp-answer-box"), null, false);

  const q = answerQuality(D.answer, D.retrieved);
  const color = q > 65 ? "var(--good)" : q > 35 ? "var(--accent)" : "var(--warn)";
  const note =
    q > 65 ? "Strong grounding — retrieval found highly relevant chunks and the answer sticks to them."
    : q > 35 ? "Partial grounding — the retrieved chunks are only loosely related, so the answer is shaky."
    : "Poor grounding — the chunks that contain the answer aren't being retrieved. No model could answer well from this context.";
  $("quality-meter").innerHTML = `
    <strong>Answer grounding quality: ${q}/100</strong>
    <div class="qm-track"><div class="qm-fill" style="width:${q}%; background:${color}"></div></div>
    <p class="qm-note">${note}</p>`;

  showQuiz(9, () => {
    $("finale").hidden = false;
    $("finale").scrollIntoView({ behavior: "smooth", block: "center" });
  });
}

/* ---------- Helpers & boot ---------- */

function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function truncate(s, n) {
  return s.length <= n ? s : s.slice(0, n).replace(/\s+\S*$/, "") + "…";
}

setupStep1();
setupStep2();
setupStep4();
setupStep6();
setupStep9();
renderNav();
