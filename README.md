# How RAG Actually Works

An interactive, single-page lesson that teaches **Retrieval-Augmented Generation** from
zero — by letting the learner run a real RAG pipeline on their own document, step by step.

## Run it

No build step, no dependencies. ES modules require a web server, so from this directory:

```bash
python3 -m http.server 8000
# or: npx http-server -p 8000
```

…then open <http://localhost:8000>. Everything runs locally in the browser; no document
or question ever leaves the machine.

## The lesson

| Step | What the learner does |
|---|---|
| 1. Document | Uploads / pastes a document (or uses the built-in honeybee sample) |
| 2. Chunks | Watches the document split into chunks; adjusts size & overlap live |
| 3. Embeddings | Sees chunks as points in 2D semantic space; clicks dots to inspect what each embedding "noticed" |
| 4. Question | Asks a question (suggested questions provided for the sample doc) |
| 5. Query point | Watches the question land in the same space as the chunks |
| 6. Retrieval | Sees top-k selection with cosine similarity scores and per-chunk "why selected" explanations |
| 7. Prompt | Reads the exact prompt text that would be sent, with a token-count breakdown bar |
| 8. Answer | Reads a grounded answer where every sentence is color-coded to its supporting chunk |
| 9. Experiments | Breaks the pipeline: changes chunk size / top-k, deletes key chunks, injects off-topic chunks — and watches answer quality respond live |

Each step ends with a quick-check quiz, and every concept is explained in plain English
assuming no prior knowledge.

## How it works under the hood

- **Chunking** — fixed-size word windows with configurable overlap (`js/text.js`).
- **Embeddings** — TF-IDF vectors with cosine similarity (`js/embed.js`). A real system
  would use neural embeddings; TF-IDF keeps the demo dependency-free while preserving the
  property being taught (similar meaning ⇒ nearby points), and the app says so explicitly.
- **2D semantic space** — classical MDS / kernel PCA on the cosine-similarity matrix,
  with Nyström out-of-sample placement for the query point.
- **Answer generation** — extractive QA over the retrieved chunks (`js/answer.js`):
  the best-matching sentences are stitched together, which makes per-sentence source
  attribution exact. The app labels this clearly as a simulation of the "G" in RAG.
- **Token counts** — estimated at ~4 characters per token, labeled as an estimate.

`js/app.js` wires it all together as one reactive pipeline: any control change re-runs
chunking → embedding → projection → retrieval → prompt → answer and re-renders every
unlocked step, so cause and effect are always visible.
