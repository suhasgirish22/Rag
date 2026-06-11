// SVG scatter-plot rendering for the "semantic space" views.

const W = 640, H = 480, PAD = 60;

export function chunkColor(i, isDistractor = false) {
  if (isDistractor) return `hsl(${356 + i * 8}, 72%, 62%)`;
  return `hsl(${(200 + i * 47) % 360}, 65%, 62%)`;
}

/**
 * Render points (and optionally a query star + retrieval links) into an SVG.
 * items: [{ id, label, color, x, y, dim, ring }]  — x/y in raw projection coords
 * query: { x, y } | null
 * links: [{ toId, score }] — drawn from query to the item with that id
 */
export function renderSpace(svg, { items, query = null, links = [], onClick = null, selectedId = null }) {
  svg.innerHTML = "";
  if (items.length === 0) return;

  // Fit all coordinates (including the query) into the viewBox
  const xs = items.map(p => p.x), ys = items.map(p => p.y);
  if (query) { xs.push(query.x); ys.push(query.y); }
  let minX = Math.min(...xs), maxX = Math.max(...xs);
  let minY = Math.min(...ys), maxY = Math.max(...ys);
  const spanX = Math.max(maxX - minX, 1e-6), spanY = Math.max(maxY - minY, 1e-6);
  const sx = v => PAD + ((v - minX) / spanX) * (W - 2 * PAD);
  const sy = v => PAD + ((v - minY) / spanY) * (H - 2 * PAD);

  // Degenerate case (all points identical): spread on a circle so dots are visible
  const degenerate = spanX < 1e-5 && spanY < 1e-5;
  const pos = new Map();
  items.forEach((p, i) => {
    if (degenerate) {
      const a = (i / items.length) * 2 * Math.PI;
      pos.set(p.id, [W / 2 + Math.cos(a) * 120, H / 2 + Math.sin(a) * 120]);
    } else {
      pos.set(p.id, [sx(p.x), sy(p.y)]);
    }
  });
  const qpos = query ? (degenerate ? [W / 2, H / 2] : [sx(query.x), sy(query.y)]) : null;

  const NS = "http://www.w3.org/2000/svg";
  const make = (tag, attrs) => {
    const el = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
  };

  // Retrieval links (under the dots)
  if (qpos) {
    for (const link of links) {
      const item = items.find(p => p.id === link.toId);
      if (!item) continue;
      const [x2, y2] = pos.get(item.id);
      const line = make("line", {
        x1: qpos[0], y1: qpos[1], x2, y2,
        stroke: item.color, "stroke-width": 1.5 + link.score * 5,
        "stroke-opacity": 0.75, class: "retrieval-line",
      });
      svg.appendChild(line);
      // Score label at the midpoint
      const label = make("text", {
        x: (qpos[0] + x2) / 2, y: (qpos[1] + y2) / 2 - 6,
        class: "dot-label", "text-anchor": "middle", "font-size": "12",
        fill: item.color,
      });
      label.textContent = link.score.toFixed(2);
      svg.appendChild(label);
    }
  }

  // Chunk dots
  items.forEach((p, i) => {
    const [cx, cy] = pos.get(p.id);
    const g = make("g", { class: "dot-enter", style: `animation-delay:${i * 60}ms` });
    if (p.ring || p.id === selectedId) {
      g.appendChild(make("circle", {
        cx, cy, r: 17, fill: "none",
        stroke: p.id === selectedId ? "#fff" : p.color,
        "stroke-width": 2, "stroke-dasharray": "4 3",
      }));
    }
    const dot = make("circle", {
      cx, cy, r: 11, fill: p.color, class: "embed-dot",
      "fill-opacity": p.dim ? 0.25 : 0.95,
      stroke: "#0e1117", "stroke-width": 2,
    });
    if (onClick) dot.addEventListener("click", () => onClick(p.id));
    g.appendChild(dot);
    const txt = make("text", { x: cx, y: cy - 17, class: "dot-label", "text-anchor": "middle" });
    txt.setAttribute("fill", p.dim ? "#5c6878" : "#e8ecf3");
    txt.textContent = p.label;
    g.appendChild(txt);
    svg.appendChild(g);
  });

  // Query star
  if (qpos) {
    const g = make("g", { class: "query-star" });
    g.appendChild(make("path", {
      d: starPath(qpos[0], qpos[1], 15, 7),
      fill: "#ffb454", stroke: "#0e1117", "stroke-width": 2,
    }));
    const txt = make("text", { x: qpos[0], y: qpos[1] - 22, class: "dot-label", "text-anchor": "middle", "font-weight": "700" });
    txt.setAttribute("fill", "#ffb454");
    txt.textContent = "Your question";
    g.appendChild(txt);
    svg.appendChild(g);
  }
}

function starPath(cx, cy, outer, inner, points = 5) {
  let d = "";
  for (let i = 0; i < points * 2; i++) {
    const r = i % 2 === 0 ? outer : inner;
    const a = (i / (points * 2)) * 2 * Math.PI - Math.PI / 2;
    d += `${i === 0 ? "M" : "L"}${(cx + r * Math.cos(a)).toFixed(1)},${(cy + r * Math.sin(a)).toFixed(1)}`;
  }
  return d + "Z";
}
